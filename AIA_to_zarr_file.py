import os
import re
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
from sunpy.io._file_tools import read_file_header 
from sunpy.map import Map
import skimage.transform
from astropy import units as u
import astropy.time
from aiapy.calibrate import degradation
from aiapy.calibrate.util import get_correction_table
from astropy.io import fits

warnings.filterwarnings("ignore")
CHANNELS = [131,171,193,211,304,335,94,1600, 1700]
trgtAS = 976.0  

def resize_to_match_solar_radius(image_data, current_radius, target_radius, padding_value=0):
    """
    Resize image to match target solar radius and add padding to maintain original dimensions
    """
    scale_factor = target_radius / current_radius
    
    # Store original dimensions
    original_height, original_width = image_data.shape
    
    # Resize the image
    new_height = int(image_data.shape[0] * scale_factor)
    new_width = int(image_data.shape[1] * scale_factor)
    
    resized_data = skimage.transform.resize(
        image_data,
        (new_height, new_width),
        preserve_range=True,
        anti_aliasing=True
    )
    
    # Add padding to restore original dimensions
    if new_height <= original_height and new_width <= original_width:
        # Create array filled with padding_value at original size
        padded_data = np.full((original_height, original_width), padding_value, dtype=resized_data.dtype)
        
        # Calculate padding offsets to center the resized image
        pad_height = (original_height - new_height) // 2
        pad_width = (original_width - new_width) // 2
        
        # Place resized image in center of padded array
        padded_data[pad_height:pad_height + new_height, 
                   pad_width:pad_width + new_width] = resized_data
        
        return padded_data, scale_factor
    else:
        # If resized image is larger than original, crop to original size
        start_height = (new_height - original_height) // 2
        start_width = (new_width - original_width) // 2
        
        cropped_data = resized_data[start_height:start_height + original_height,
                                   start_width:start_width + original_width]
        
        return cropped_data, scale_factor

def estrai_wavelenght(file_name):
    stringa = file_name[23:26]  
    num = re.findall(r'\d+', stringa)
    return num[0] if num else ""

def create_synchronized_filelists(aia_folder_path, M_folder_path, ic_nl_folder_path, year, args):
    """Create lists of files that have matching observation dates across all three data sources"""
    year_str = str(year)
    
    # Initialize dictionaries to store files by date
    aia_files_by_date = {wavelength: {} for wavelength in ['131', '171', '193', '211', '304', '335', '94', '1600', '1700']}
    M_files_by_date = {}
    ic_nl_files_by_date = {}
    
    # Process AIA files
    for file in sorted(os.listdir(aia_folder_path)):
        if file[1:5] == year_str:
            wavelength = estrai_wavelenght(file)
            if wavelength:
                if wavelength in ["160", "170"]:
                    wavelength = wavelength + "0"
                try:
                    Xh = read_file_header(os.path.join(aia_folder_path, file))
                    if Xh[1]['QUALITY'] == 0:
                        date_str = file[1:11]
                        if date_str not in aia_files_by_date[wavelength]:
                            aia_files_by_date[wavelength][date_str] = []
                        aia_files_by_date[wavelength][date_str].append(os.path.join(aia_folder_path, file))
                except:
                    print(f"FILE CORRUPTED: {file}")
                    continue
    if args.do_M_list:
        # Process Magnetogram files
        for file in sorted(os.listdir(M_folder_path)):
            if file[11:15] == year_str:
                date_str = file[11:21].replace(".","-")
                if date_str not in M_files_by_date:
                    M_files_by_date[date_str] = []
                M_files_by_date[date_str].append(os.path.join(M_folder_path, file))
    
    # Process Ic_noLimbDark files
    for file in sorted(os.listdir(ic_nl_folder_path)):
        if file[23:27] == year_str:
            date_str = file[23:33].replace(".","-")
            if date_str not in ic_nl_files_by_date:
                ic_nl_files_by_date[date_str] = []
            ic_nl_files_by_date[date_str].append(os.path.join(ic_nl_folder_path, file))
    
    # Find common dates
    if all(aia_files_by_date[wavelength] for wavelength in aia_files_by_date):
        wavelengths = list(aia_files_by_date.keys())
        common_aia_dates = set(aia_files_by_date[wavelengths[0]].keys())
        
        for wavelength in wavelengths[1:]:
            common_aia_dates = common_aia_dates.intersection(set(aia_files_by_date[wavelength].keys()))
        if args.do_M_list:
            # Then find intersection with Ic and Ic_noLimbDark
            common_dates = common_aia_dates.intersection(set(M_files_by_date.keys()), set(ic_nl_files_by_date.keys()))
        else:
            common_dates = common_aia_dates.intersection(set(ic_nl_files_by_date.keys()))
    else:
        print("Warning: Some wavelengths have no data")
        common_dates = set()
    
    # Filter files to only include common dates
    filelist_131 = [f for date in common_dates for f in aia_files_by_date['131'].get(date, [])]
    filelist_171 = [f for date in common_dates for f in aia_files_by_date['171'].get(date, [])]
    filelist_193 = [f for date in common_dates for f in aia_files_by_date['193'].get(date, [])]
    filelist_211 = [f for date in common_dates for f in aia_files_by_date['211'].get(date, [])]
    filelist_304 = [f for date in common_dates for f in aia_files_by_date['304'].get(date, [])]
    filelist_335 = [f for date in common_dates for f in aia_files_by_date['335'].get(date, [])]
    filelist_94 = [f for date in common_dates for f in aia_files_by_date['94'].get(date, [])]
    filelist_1600 = [f for date in common_dates for f in aia_files_by_date['1600'].get(date, [])]
    filelist_1700 = [f for date in common_dates for f in aia_files_by_date['1700'].get(date, [])]
    M_list = [f for date in common_dates for f in M_files_by_date.get(date, [])]
    Ic_noLimbDark = [f for date in common_dates for f in ic_nl_files_by_date.get(date, [])]
    
    return M_list, Ic_noLimbDark, filelist_131, filelist_171, filelist_193, filelist_211, filelist_304, filelist_335, filelist_94, filelist_1600, filelist_1700

def get_avg_degrad(anno):
    correction_table = get_correction_table(source="JSOC")
    aia_channels = [131,171,193,211,304,335,94,1700, 1600] * u.angstrom
    CHANNELS = [131,171,193,211,304,335,94,1700, 1600]
    anno = str(anno)
    inizio = anno + "-01-01T00:00:11"
    fine = anno + "-12-31T00:00:11"
    start_time = astropy.time.Time(inizio, scale="utc")
    now = astropy.time.Time(fine, scale="utc")

    time_range = start_time + np.arange(0, (now - start_time).to(u.day).value, 3) * u.day
    degradations = {
        channel: degradation(channel, time_range, correction_table=correction_table) for channel in aia_channels
    }
    degrads={}
    for i, wl in enumerate(CHANNELS):
        degrads[wl]=round(float(np.mean(degradations[aia_channels[i]])),4)
    return degrads

def parse_args():
    parser = argparse.ArgumentParser(description="Process AIA files and generate Zarr datasets.")
    parser.add_argument("--folder_path", type=str, default="/home/gpatane/Dataset/AIA_folder", help="Path to the folder containing AIA files")
    parser.add_argument("--M_folder_path", type=str, default="/home/gpatane/Dataset/Magnetogram", help="Path to the folder containing IC files")
    parser.add_argument("--IC_folder_path_nl", type=str, default="/home/gpatane/Dataset/Ic_noLimbDark", help="Path to the folder containing IC files no Limb")
    parser.add_argument('--output_folder', type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram_4096.zarr")
    parser.add_argument('--overwrite', action='store_true', default=False, help='Allow existing output folder without raising an error')
    parser.add_argument('--start_period', type=int, required=False, default=2024)
    parser.add_argument('--end_period', type=int, required=False, default=2026)
    parser.add_argument('--image_size', type=int, required=False, default=4096, help='Size of the images to process')
    parser.add_argument('--do_M_list', action='store_true', default=True, help='Whether to process Magnetogram files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder_path
    years = list(range(args.start_period, args.end_period))
    M_folder_path = args.M_folder_path
    IC_folder_path_nl = args.IC_folder_path_nl
    output_folder = args.output_folder
    
    os.makedirs(output_folder, exist_ok=True)
    store = zarr.storage.DirectoryStore(output_folder)
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=store, overwrite=args.overwrite)
    # Target radius from AIA data (approximately)
    radius_ratio = 0.3865
    TARGET_RADIUS = args.image_size * radius_ratio
    for year in years:
        print(f"Processing year {year}...")
        if args.do_M_list:
            M_list, Ic_noLimbDark, filelist_131, filelist_171, filelist_193, filelist_211, filelist_304, filelist_335, filelist_94, filelist_1600, filelist_1700 = create_synchronized_filelists(folder_path, M_folder_path, IC_folder_path_nl, year, args)
        else:
            _, Ic_noLimbDark, filelist_131, filelist_171, filelist_193, filelist_211, filelist_304, filelist_335, filelist_94, filelist_1600, filelist_1700 = create_synchronized_filelists(folder_path, M_folder_path, IC_folder_path_nl, year, args)
        
        if not Ic_noLimbDark:
            print(f"No data found for year {year}, skipping...")
            continue
            
        degrads = get_avg_degrad(year)
        
        print(f"Year {year}:")
        if args.do_M_list:
            print(f"  Magnetogram files: {len(M_list)}")
        print(f"  Ic_noLimbDark files: {len(Ic_noLimbDark)}")
        print(f"  AIA files: {len(filelist_131)}")

        year_group = root.create_group(str(year))
        scale = args.image_size
        divideFactor = int(4096 / scale)
        #divideFactor = int(1024 / scale)
        
        # Create datasets
        if args.do_M_list:
            M = year_group.create_dataset('Magnetogram', shape=(len(M_list), scale, scale), chunks=(15, None, None), dtype='f4', compressor=compressor)
        else:
            print("Skipping Magnetogram dataset creation as do_M_list is False")
        if Ic_noLimbDark:
            Ic_nl = year_group.create_dataset('Ic_noLimbDark', shape=(len(Ic_noLimbDark), scale, scale), chunks=(15, None, None), dtype='f4', compressor=compressor)
        
        # Create AIA datasets
        aia_datasets = {}
        filelists = {
            '94A': filelist_94, '171A': filelist_171, '193A': filelist_193,
            '211A': filelist_211, '304A': filelist_304, '335A': filelist_335,
            '131A': filelist_131, '1600A': filelist_1600, '1700A': filelist_1700
        }
        
        for dataset_name, file_list in filelists.items():
            if file_list:
                aia_datasets[dataset_name] = year_group.create_dataset(
                    dataset_name, shape=(len(file_list), scale, scale),
                    chunks=(15, None, None), dtype='f4', compressor=compressor
                )
        

        
        # Process Ic_noLimbDark files with resizing
        if Ic_noLimbDark:
            print("Processing Ic_noLimbDark files...")
            metadata_dict = {}
            
            for fn, file in tqdm(enumerate(Ic_noLimbDark), desc="Processing Ic_noLimbDark"):
                try:
                    data, header = fits.getdata(file, header=True)
                    data = np.nan_to_num(data, nan=0)
                    data = np.rot90(data, 2)
                    
                    header['cunit1'] = 'arcsec'
                    header['cunit2'] = 'arcsec'
                    header_string = header.tostring()
                    header = fits.Header.fromstring(header_string)
                    Xd = Map(data, header)
                    
                    # Store metadata
                    Xd.meta.pop('license', None)
                    for key in Xd.meta:
                        if key not in ['keycomments', 'simple']:
                            if key not in metadata_dict:
                                metadata_dict[key] = []
                            metadata_dict[key].append(Xd.meta[key])
                    
                    # Calculate current radius and resize
                    current_radius = Xd.meta['RSUN_OBS'] / Xd.meta['CDELT1']
                    X, scale_factor = resize_to_match_solar_radius(
                        Xd.data, current_radius, TARGET_RADIUS, padding_value=0
                    )
                    
                    # Apply existing transformations
                    validMask = 1.0 * (X > 0)
                    X[np.where(X <= 0.0)] = 0.0
                    
                    # Apply similarity transform (using TARGET_RADIUS as reference)
                    rad = Xd.meta['RSUN_OBS']
                    scale_factor = trgtAS / rad
                    t = (X.shape[0]/2.0) - scale_factor * (X.shape[0]/2.0)
                    XForm = skimage.transform.SimilarityTransform(scale=scale_factor, translation=(t,t))
                    Xr = skimage.transform.warp(X, XForm.inverse, preserve_range=True, mode='edge', output_shape=(X.shape[0], X.shape[0]))
                    Xm = skimage.transform.warp(validMask, XForm.inverse, preserve_range=True, mode='edge', output_shape=(X.shape[0], X.shape[0]))
                    Xr = np.divide(Xr, (Xm + 1e-8))
                    Xr = skimage.transform.downscale_local_mean(Xr, (divideFactor, divideFactor))
                    Xr = Xr.astype('float32')
                    Ic_nl[fn,:,:] = Xr
                    
                except Exception as e:
                    print(f"Error processing Ic_noLimbDark file {file}: {e}")
                    continue
                
        # Process Magnetogram files with resizing
        if args.do_M_list:
            print("Processing Magnetogram files...")
            metadata_dict = {}
            
            for fn, file in tqdm(enumerate(M_list), desc="Processing Magnetograms"):
                try:
                    data, header = fits.getdata(file, header=True)
                    data = np.nan_to_num(data, nan=-5000)
                    data = np.rot90(data, 2)
                    
                    header['cunit1'] = 'arcsec'
                    header['cunit2'] = 'arcsec'
                    header_string = header.tostring()
                    header = fits.Header.fromstring(header_string)
                    Xd = Map(data, header)
                    
                    # Store metadata
                    Xd.meta.pop('license', None)
                    for key in Xd.meta:
                        if key not in ['keycomments', 'simple']:
                            if key not in metadata_dict:
                                metadata_dict[key] = []
                            metadata_dict[key].append(Xd.meta[key])
                    
                    # Calculate current radius and resize
                    current_radius = Xd.meta['RSUN_OBS'] / Xd.meta['CDELT1']
                    X, scale_factor = resize_to_match_solar_radius(
                        Xd.data, current_radius, TARGET_RADIUS, padding_value=-5000
                    )
                    
                    # Apply existing transformations
                    validMask = np.isfinite(X) 
                    validMask = validMask.astype(float)
                    
                    padding_value = -5000  
                    X_processed = X.copy()
                    X_processed[~validMask.astype(bool)] = padding_value
                    
                    rad = Xd.meta['RSUN_OBS']
                    scale_factor = trgtAS / rad
                    t = (X_processed.shape[0]/2.0) - scale_factor * (X_processed.shape[0]/2.0)
                    XForm = skimage.transform.SimilarityTransform(scale=scale_factor, translation=(t,t))
                    
                    Xr = skimage.transform.warp(X_processed, XForm.inverse, preserve_range=True, 
                                            mode='edge', output_shape=(X_processed.shape[0], X_processed.shape[0]))
                    Xm = skimage.transform.warp(validMask, XForm.inverse, preserve_range=True, 
                                            mode='edge', output_shape=(X_processed.shape[0], X_processed.shape[0]))
                    mask_threshold = 0.1
                    valid_correction = Xm > mask_threshold
                    
                    Xr_corrected = Xr.copy()
                    Xr_corrected[valid_correction] = Xr[valid_correction] / Xm[valid_correction]
                    
                    Xr_corrected[~valid_correction] = padding_value
                    
                    Xr_corrected = skimage.transform.downscale_local_mean(Xr_corrected, (divideFactor, divideFactor))
                    Xr_corrected = Xr_corrected.astype('float32')
                    M[fn,:,:] = Xr_corrected
                    
                except Exception as e:
                    print(f"Error processing magnetogram file {file}: {e}")
                    continue
            else:
                print("No Magnetogram files found for this year.")
                
                
        # Process AIA files (existing logic)
        for dataset_name, file_list in filelists.items():
            if file_list:
                print(f"Processing {dataset_name}...")
                metadata_dict = {}
                deg_cor = []
                pixlunit = []
                
                # Initialize metadata from first file
                Xd = Map(file_list[0])
                for key in Xd.meta:
                    if key not in ['keycomments', 'simple']:
                        metadata_dict[key] = []
                
                for fn, file in tqdm(enumerate(file_list), desc=f"Processing {dataset_name}"):
                    try:
                        Xd = Map(file)

                        fn2 = file[37:]
                        if dataset_name == '94A':
                            wavelength = int(fn2[-24:].split("]")[0])  
                        else:
                            wavelength = int(fn2[-25:].split("]")[0])
                        correction = degrads[wavelength]
                        
                        # Store metadata
                        Xd.meta.pop('license', None)
                        for key in Xd.meta:
                            if key not in ['keycomments', 'simple']:
                                if key in metadata_dict:
                                    metadata_dict[key].append(Xd.meta[key])
                        
                        deg_cor.append(correction)
                        pixlunit.append('DN/s')
                        
                        # Process image data
                        X = Xd.data
                        validMask = 1.0 * (X > 0)
                        X[np.where(X <= 0.0)] = 0.0
                        expTime = max(Xd.meta['EXPTIME'], 1e-2)
                        rad = Xd.meta['RSUN_OBS']
                        scale_factor = trgtAS / rad
                        t = (X.shape[0]/2.0) - scale_factor * (X.shape[0]/2.0)
                        XForm = skimage.transform.SimilarityTransform(scale=scale_factor, translation=(t,t))
                        Xr = skimage.transform.warp(X, XForm.inverse, preserve_range=True, mode='edge', output_shape=(X.shape[0], X.shape[0]))
                        Xm = skimage.transform.warp(validMask, XForm.inverse, preserve_range=True, mode='edge', output_shape=(X.shape[0], X.shape[0]))
                        Xr = np.divide(Xr, (Xm + 1e-8))
                        Xr = Xr / (expTime * correction)
                        Xr = skimage.transform.downscale_local_mean(Xr, (divideFactor, divideFactor))
                        Xr = Xr.astype('float32')
                        aia_datasets[dataset_name][fn,:,:] = Xr

                    except Exception as e:
                        print(f"Error processing AIA file {file}: {e}")
                        continue
                # Store attributes for AIA datasets
                dataset = aia_datasets[dataset_name]
                for key, values in metadata_dict.items():
                    if values:
                        try:
                            if isinstance(values[0], (str, int, float, bool)) or values[0] is None:
                                dataset.attrs[key.upper()] = values
                            else:
                                dataset.attrs[key.upper()] = [str(v) for v in values]
                        except Exception as e:
                            print(f"Skipping attribute {key}: {e}")
                
                # Store processed attributes
                if 'naxis1' in metadata_dict and metadata_dict['naxis1']:
                    dataset.attrs['NAXIS1'] = list(np.asarray(metadata_dict['naxis1'], dtype=np.float64) / divideFactor)
                    dataset.attrs['NAXIS2'] = list(np.asarray(metadata_dict['naxis2'], dtype=np.float64) / divideFactor)
                    dataset.attrs['CDELT1'] = list(np.asarray(metadata_dict['cdelt1'], dtype=np.float64) * divideFactor * metadata_dict['rsun_obs'][0] / trgtAS)
                    dataset.attrs['CDELT2'] = list(np.asarray(metadata_dict['cdelt2'], dtype=np.float64) * divideFactor * metadata_dict['rsun_obs'][0] / trgtAS)
                    dataset.attrs['CRPIX1'] = list(np.asarray(metadata_dict['crpix1'], dtype=np.float64) / divideFactor)
                    dataset.attrs['CRPIX2'] = list(np.asarray(metadata_dict['crpix2'], dtype=np.float64) / divideFactor)
                
                dataset.attrs['DEG_COR'] = list(np.asarray(deg_cor, dtype=np.float64))
                dataset.attrs['PIXLUNIT'] = list(pixlunit)
        
        
        print(f"  Completed processing first file from each dataset for testing")
