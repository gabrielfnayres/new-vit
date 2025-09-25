import os 
import numpy as np 
import pydicom
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import functools
from tqdm import tqdm
import SimpleITK as sitk
def bias_field_correction(img_path):
    
    dicom_files = [pydicom.dcmread(os.path.join(img_path, f)) for f in os.listdir(img_path)]
    volume = np.stack([ds.pixel_array for ds in dicom_files])
    
    # Convert to float32 for processing
    volume = volume.astype(np.float32)
    
    # Create SimpleITK image from numpy array
    sitk_image = sitk.GetImageFromArray(volume)
    
    # Set spacing (Duke dataset spacing)
    spacing = (0.7, 0.7, 3.0)
    sitk_image.SetSpacing(spacing)
    
    # Convert to float32 pixel type for N4 correction
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    
    # Apply N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Reduce iterations for faster processing
    corrector.SetMaximumNumberOfIterations([20, 20, 20, 20])
    
    corrected_image = corrector.Execute(sitk_image)
    corrected_image = sitk.Cast(corrected_image, sitk.sitkUInt16)
    corrected_volume = sitk.GetArrayFromImage(corrected_image)
    
    return corrected_volume

def process_series(row, path_root_in_str, path_root_out_data_str):
    """Worker function for multiprocessing"""
    try:
        patient_id = row['PatientID']
        sequence_name = row['SequenceName']
        classic_path = row['classic_path']
        
        path_root_in = Path(path_root_in_str)
        path_root_out_data = Path(path_root_out_data_str)
        
        dicom_dir = path_root_in / classic_path
        
        if not dicom_dir.exists():
            return f"Directory not found: {dicom_dir}"
        
        output_dir = path_root_out_data / f"Breast_MRI_{patient_id:03d}" / sequence_name
        output_file = output_dir / f"{sequence_name}.npy"
        
        if output_file.exists():
            return f"Skipped (already exists): Breast_MRI_{patient_id:03d} - {sequence_name}"
        
        corrected_volume = bias_field_correction(str(dicom_dir))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_file, corrected_volume)
        
        return f"Completed: Breast_MRI_{patient_id:03d} - {sequence_name}"
        
    except Exception as e:
        return f"Error processing {row['PatientID']}-{row['SequenceName']}: {e}"

if __name__ == "__main__":
    # Setting 
    path_root = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI') # Path to the Duke Breast Cancer MRI dataset
    
    path_root_in = path_root/'manifest-1654812109500'
    path_root_out = path_root/'preprocessed-v1'
    path_root_out_data = path_root_out/'data'
    path_root_out_data.mkdir(parents=True, exist_ok=True)
    
    print("Loading mapping files...")
    df_path2name = pd.read_excel(path_root/'Breast-Cancer-MRI-filepath_filename-mapping.xlsx') 
    aux_df = pd.read_excel(path_root/'my-mapping.xlsx')

    df_path2name['SequenceName'] = aux_df['SequenceName']
    df_path2name['classic_path'] = aux_df['classic_path']

    seg_mask = df_path2name['SequenceName'] == 'Segmentation'
    other_df = df_path2name[~seg_mask].copy()

    seq_paths_split = other_df['original_path_and_filename'].str.split('/')
    other_df['PatientID'] = seq_paths_split.apply(lambda x: int(x[1].rsplit('_', 1)[1]))
    other_df['SequenceName'] = seq_paths_split.apply(lambda x: x[2]) # Derive from path
    other_df['classic_path'] = other_df['classic_path'].apply(lambda p: str(Path(p).parent))

    final_df = other_df[['PatientID', 'SequenceName', 'classic_path', 'original_path_and_filename']].copy()
    final_df = final_df.drop_duplicates(subset=['PatientID', 'SequenceName'], keep='first')
    final_df = other_df[['PatientID', 'SequenceName', 'classic_path', 'original_path_and_filename']].copy()
    final_df = final_df.drop_duplicates(subset=['PatientID', 'SequenceName'], keep='first')

    print(f"Found {len(final_df)} series to process")
    
    num_processes = os.cpu_count()
    print(f"Using SimpleITK N4 bias field correction with {num_processes} CPU processes")
    
    worker_func = functools.partial(process_series, 
                                  path_root_in_str=str(path_root_in),
                                  path_root_out_data_str=str(path_root_out_data))
    
    rows_to_process = [row for _, row in final_df.iterrows()]
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_func, rows_to_process), 
                           total=len(rows_to_process),
                           desc="Processing series"))
    
    completed = sum(1 for r in results if r.startswith("Completed"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    errors = sum(1 for r in results if r.startswith("Error"))
    
    print(f"\nBias field correction complete!")
    print(f"Completed: {completed}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total processed: {len(results)}")