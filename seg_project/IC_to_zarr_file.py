import os
from sunpy.map import Map
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from astropy.io import fits
import numpy as np
import skimage.transform
import warnings
import argparse
warnings.filterwarnings("ignore")
trgtAS = 976.0  


def create_filelists(folder_path, year):
    year = str(year)
    Ic_noLimbDark=[]
    lista_file = os.listdir(folder_path)
    lista_file.sort()
    for file in lista_file:
        if file[23:27] == year:
            Ic_noLimbDark.append(os.path.join(folder_path,file))
    return Ic_noLimbDark

folder_path = "/home/gpatane/Dataset/Ic_noLimbDark/"
output_folder = '/home/gpatane/Dataset/IC_zarr_file.zarr'
def parse_args():
    
    parser = argparse.ArgumentParser(description='Create Zarr file, a compressed archive to resize images and reduce the file size of individual FITS files. ')
    parser.add_argument('--folder_path', type=str, default="/home/gpatane/Dataset/Ic_noLimbDark/",
                        help='Path to the Zarr dataset file.')
    parser.add_argument('--output_folder', type=str, default="/home/gpatane/Dataset/IC_zarr_file.zarr",
                        help='Path to the Zarr dataset file.')
    parser.add_argument('--overwrite', action='store_true', help='Allow existing output folder without raising an error')
    parser.add_argument('--start_period', type=int, required=False, default=2010)
    parser.add_argument('--end_period', type=int, required=False, default=2026)
    
    
    args = parser.parse_args()
    return args
args = parse_args()

os.makedirs(args.output_folder, exist_ok=True)
store = zarr.storage.DirectoryStore(args.output_folder)
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
root = zarr.group(store=store,overwrite=args.overwrite)   

years = list(range(args.start_period,args.end_period))
for year in years:

    
    Ic_noLimbDark = create_filelists(args.folder_path, year)

    Y = root.create_group(year)                                               #crea un gruppo dell'anno selezionato
    scale = 1024
    divideFactor = int(1024 / scale)
    
    Ic = Y.create_dataset('Ic_noLimbDark',shape=(np.shape(Ic_noLimbDark)[0],scale,scale),chunks=(15,None,None),dtype='f4',compressor=compressor)
    
    deg_cor = []
    pixlunit = []
    for fn,file in tqdm(enumerate(Ic_noLimbDark)):
        data, header = fits.getdata(file, header=True)
        data = np.nan_to_num(data, nan=0)
        data = np.rot90(data, 2)
        header['cunit1'] = 'arcsec'
        header['cunit2'] = 'arcsec'
        header_string = header.tostring()
        header = fits.Header.fromstring(header_string)
        Xd = Map(data, header)
        fn2 = file[36:]        # it extract exactly [2022-03-23T00:00:11Z][131]aia_lev1_euv_12s.fits
        datestring = f'{fn2[23:27]}_{fn2[28:30]}_{fn2[31:33]}'
        
        Xd.meta.pop('license', None)
        for key in Xd.meta:
            if key != 'keycomments' and key != 'simple':
                if key not in vars():
                    vars()[key] = []  # Inizializza se non esiste
                vars()[key].append(Xd.meta[key])
        
        pixlunit.append('DN/s')

        X = Xd.data
        validMask = 1.0 * (X > 0)
        X[np.where(X<=0.0)] = 0.0
        
        #rad = Xd.meta['RSUN_OBS']
        rad = 949.590558
        scale_factor = trgtAS/rad
        t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
        XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
        Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
        Xr = np.divide(Xr,(Xm+1e-8))
        Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor*4,divideFactor*4))
        Xr = Xr.astype('float32')
        Ic[fn,:,:]=Xr
        
    Xd.meta.pop('license', None)  
    for key in Xd.meta:
        if key != 'keycomments' and key != 'simple':
            Ic.attrs[key.upper()] = vars()[key]
            
    Ic.attrs['NAXIS1'] = list(np.asarray(naxis1, dtype=np.float64) / divideFactor)
    Ic.attrs['NAXIS2'] = list(np.asarray(naxis2, dtype=np.float64) / divideFactor)
