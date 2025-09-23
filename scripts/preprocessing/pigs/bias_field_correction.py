import SimpleITK as sitk 
import os 
import numpy as np 
import pydicom

def bias_field_correction(img_path):
    
    dicom_files = [pydicom.dcmread(os.path.join(img_path, f)) for f in os.listdir(img_path)]
    volume = np.stack([ds.pixel_array for ds in dicom_files])
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    sitk_image = sitk.GetImageFromArray(volume)
    
    #duke spacing 
    spacing = (0.7, 0.7, 3)
    sitk_image.SetSpacing(spacing)
    
    corrected_image = corrector.Execute(sitk_image)
    corrected_volume = sitk.GetArrayFromImage(corrected_image)
    return corrected_volume

if __name == '__main__':
    