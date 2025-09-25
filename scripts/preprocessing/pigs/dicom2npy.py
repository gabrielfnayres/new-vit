from pathlib import Path 
import logging  
import pandas as pd 
from multiprocessing import Pool

import numpy as np 
import pydicom
import pydicom.datadict
import pydicom.dataelem
import pydicom.sequence
import pydicom.valuerep
from tqdm import tqdm
import SimpleITK as sitk 


# Logging 
logger = logging.getLogger(__name__)


def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }


def series2npy(args):
    series_info, path_root_in_str, path_root_out_data_str = args
    seq_name, path_series = series_info 
    
    path_root_in = Path(path_root_in_str)
    path_root_out_data = Path(path_root_out_data_str)
    
    path_series = path_root_in/Path(path_series)
    
    # If path points to a file, get its parent directory (the series directory)
    if path_series.is_file():
        path_series = path_series.parent
    elif not path_series.is_dir():
        logger.warning(f"Path does not exist: {path_series}")
        return 
    
    try:
        # Create local reader for this process
        local_reader = sitk.ImageSeriesReader()
        
        # Read DICOM using SimpleITK
        dicom_names = local_reader.GetGDCMSeriesFileNames(str(path_series))
        local_reader.SetFileNames(dicom_names) 
        img_sitk = local_reader.Execute()
        
        # Convert SimpleITK image to numpy array
        img_array = sitk.GetArrayFromImage(img_sitk)

        # Read Metadata 
        ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
        metadata = dataset2dict(ds)
        
        # Create output folder 
        path_out_dir = path_root_out_data/path_series.parts[-3]
        path_out_dir.mkdir(exist_ok=True, parents=True)

        # Write numpy array
        filename = seq_name
        logger.info(f"Writing file: {filename}:")
        path_file = path_out_dir/f'{seq_name}.npy'
        np.save(path_file, img_array)

        metadata['_path_file'] = str(path_file.relative_to(path_root_out_data))
        metadata['_array_shape'] = img_array.shape
        metadata['_array_dtype'] = str(img_array.dtype)
        return metadata

    except Exception as e:
        logger.warning(f"Error in: {path_series}")
        logger.warning(str(e))


if __name__ == "__main__":
    # Setting 
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI') # Path to the Duke Breast Cancer MRI dataset

    path_root_in = path_root/'manifest-1654812109500'
    path_root_out = path_root/'preprocessed-v1'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
   
    # Init reader 
    reader = sitk.ImageSeriesReader()

    # Note: Contains path to every single dicom file 
    # WARNING: reading this .xlsx file takes some time 
    df_path2name = pd.read_excel(path_root/'Breast-Cancer-MRI-filepath_filename-mapping.xlsx') 
    
    df_path2name = df_path2name[df_path2name.columns[:4]].copy()
    seq_paths = df_path2name['original_path_and_filename'].str.split('/')
    df_path2name['PatientID'] = seq_paths.apply(lambda x:int(x[1].rsplit('_', 1)[1]))
    df_path2name['SequenceName'] = seq_paths.apply(lambda x:x[2])
    df_path2name['classic_path'] = df_path2name['classic_path'].str.rsplit('/', n=1).str[0] # remove xx.dcm 
    df_path2name = df_path2name.drop_duplicates(subset=['PatientID', 'SequenceName'], keep='first')
    df_path2name.to_csv(path_root_out/'Breast-Cancer-MRI-filepath_filename-mapping.csv', index=False)
    df_path2name = pd.read_csv(path_root_out/'Breast-Cancer-MRI-filepath_filename-mapping.csv')
    series = list(zip(df_path2name['SequenceName'], df_path2name['classic_path']))

    # Validate 
    print("Number Series: ", len(series), "of 5034 (5034+127=5161) ")


    # Option 1: Multi-CPU 
    metadata_list = []
    # Prepare arguments for multiprocessing
    args_list = [(series_info, str(path_root_in), str(path_root_out_data)) for series_info in series]
    
    with Pool() as pool:
        for meta in tqdm(pool.imap_unordered(series2npy, args_list), total=len(series)):
            metadata_list.append(meta)


    # Option 2: Single-CPU (if you need a coffee break)
    # metadata_list = []
    # for series_info in tqdm(series):
    #     args = (series_info, str(path_root_in), str(path_root_out_data))
    #     meta = series2npy(args)
    #     metadata_list.append(meta)

    
    df = pd.DataFrame(metadata_list)
    df.to_csv(path_root_out/'metadata.csv', index=False)

    # Check export 
    num_series = len([path for path in path_root_out_data.rglob('*.npy')])
    print("Number Series: ", num_series, "of 5034 (5034+127=5161) ")