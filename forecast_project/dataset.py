import bisect
import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
import zarr
from scipy.ndimage import zoom
from torch.utils.data import Dataset


WAVELENGTHS_9 = ['94A', '131A', '171A', '193A', '211A', '304A', '335A', '1600A', '1700A']

_RECHUNKED = '/home/gpatane/Dataset/zarr_file_magnetogram_1024_rechunked.zarr'
_ORIGINAL  = '/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr'
ZARR_PATH  = _RECHUNKED if __import__('os').path.exists(_RECHUNKED) else _ORIGINAL

_STATS_PATH = '/home/gpatane/Dataset/statistiche_globali.json'
with open(_STATS_PATH) as _f:
    _CHANNEL_STATS = json.load(_f)


class SDO_TemporalDataset(Dataset):
    """
    Dataset for solar image temporal forecasting.

    For each sample returns:
        input  : [9, H, W] multi-channel image at time t
        target : [9, H, W] multi-channel image at time t + Δt
        delta_t_idx : int index into delta_t_hours list
        delta_t_h   : float actual Δt in hours (for logging)

    Timestamps are read from zarr T_OBS attributes.
    ~2 images/day → Δt=12h is the minimum step.
    """

    DEFAULT_DELTA_T = [12, 24, 36, 48, 168]  # hours

    def __init__(self, zarr_path, list_year, wavelengths,
                 target_size=1024,
                 delta_t_hours=None,
                 max_gap_hours=3.0,
                 transform=None,
                 skip_invalid=True):
        self.z = zarr.open(zarr_path, mode='r')
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.delta_t_hours = delta_t_hours or self.DEFAULT_DELTA_T
        self.max_gap_sec = max_gap_hours * 3600.0
        self.transform = transform
        self.skip_invalid = skip_invalid

        # Build sorted global timeline: [(year_str, local_idx, unix_ts)]
        self.timeline = []
        for year in [str(y) for y in list_year]:
            if year not in self.z:
                continue
            t_obs_raw = list(self.z[year]['131A'].attrs.get('T_OBS', []))
            n = min(self.z[year][wl].shape[0] for wl in self.wavelengths)
            for local_idx in range(min(n, len(t_obs_raw))):
                ts = self._parse_ts(t_obs_raw[local_idx])
                if ts is not None:
                    self.timeline.append((year, local_idx, ts))

        self.timeline.sort(key=lambda x: x[2])
        self._unix = [x[2] for x in self.timeline]  # fast lookup list

        # Build (src_global_idx, tgt_global_idx, delta_t_h) pairs
        self.pairs = []
        for src_i, (_, _, ts) in enumerate(self.timeline):
            for dt_h in self.delta_t_hours:
                target_ts = ts + dt_h * 3600.0
                tgt_i = self._nearest_after(target_ts, src_i + 1)
                if tgt_i is not None:
                    gap = abs(self._unix[tgt_i] - target_ts)
                    if gap <= self.max_gap_sec:
                        self.pairs.append((src_i, tgt_i, dt_h))

        self.dt_to_idx = {dt: i for i, dt in enumerate(self.delta_t_hours)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ts(t_str):
        if not isinstance(t_str, str):
            return None
        s = t_str.rstrip('Z').strip()
        for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
        return None

    def _nearest_after(self, target_ts, lo):
        pos = bisect.bisect_left(self._unix, target_ts, lo=lo)
        if pos >= len(self._unix):
            return None
        candidates = [pos]
        if pos > lo:
            candidates.append(pos - 1)
        return min(candidates, key=lambda i: abs(self._unix[i] - target_ts))

    def _load_img(self, year, local_idx):
        imgs = []
        for wl in self.wavelengths:
            img = self.z[year][wl][local_idx].astype(np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            # Same normalization as MAE pre-training (statistiche_globali.json)
            stats = _CHANNEL_STATS[wl]
            img = np.clip(img, 0, None)
            img = np.log1p(img * 0.01)
            p_max = stats['p_max_log']
            img = np.clip(img, 0.0, p_max) / p_max  # → [0, 1]

            if img.shape[0] != self.target_size:
                scale = self.target_size / img.shape[0]
                img = zoom(img, scale, order=1)

            imgs.append(img)
        return torch.from_numpy(np.stack(imgs, axis=0)).float()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        for _ in range(5):
            try:
                src_i, tgt_i, dt_h = self.pairs[idx]
                src_year, src_loc, _ = self.timeline[src_i]
                tgt_year, tgt_loc, _ = self.timeline[tgt_i]

                src_img = self._load_img(src_year, src_loc)
                tgt_img = self._load_img(tgt_year, tgt_loc)

                return {
                    'input': src_img,                                          # [9, H, W]
                    'target': tgt_img,                                         # [9, H, W]
                    'delta_t_h': torch.tensor(dt_h, dtype=torch.float32),
                    'delta_t_idx': torch.tensor(self.dt_to_idx[dt_h], dtype=torch.long),
                }
            except Exception:
                if not self.skip_invalid:
                    raise
                idx = random.randrange(len(self.pairs))

        raise RuntimeError('SDO_TemporalDataset: failed to load sample after 5 retries')
