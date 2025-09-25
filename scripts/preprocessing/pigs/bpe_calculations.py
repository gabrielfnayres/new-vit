import numpy as np

def calculate_bpe_mask(pre_img, post_img, fgt_mask, enhancement_threshold=1.0):
    """
    Calculate BPE mask with proper shape handling and realistic enhancement values
    """
    # Ensure all inputs have same shape
    assert pre_img.shape == post_img.shape == fgt_mask.shape, \
        f"Shape mismatch: Pre={pre_img.shape}, Post={post_img.shape}, Mask={fgt_mask.shape}"
    
    fgt_indices = fgt_mask > 0
    
    enhancement = np.zeros_like(post_img, dtype=np.float32)
    
    if np.any(fgt_indices):
        pre_fgt = pre_img[fgt_indices]
        post_fgt = post_img[fgt_indices]
        
        valid_pre = pre_fgt > 1.0 
        
        if np.any(valid_pre):
            epsilon = 1e-6
            pre_valid = pre_fgt[valid_pre]
            post_valid = post_fgt[valid_pre]
            
            enhancement_valid = (post_valid - pre_valid) / (pre_valid + epsilon) * 100.0
            
         
            fgt_coords = np.where(fgt_indices)
            valid_coords = tuple(coord[valid_pre] for coord in fgt_coords)
            enhancement[valid_coords] = enhancement_valid
    
    bpe_mask = (fgt_mask > 0) & (enhancement > enhancement_threshold)
   # bpe_mask = ((fgt_mask)*post_img)
    print("="*12)
    print(bpe_mask.shape)
    print(bpe_mask.max())
    print(bpe_mask.min())
    print(bpe_mask)
    print("="*12)
    return bpe_mask.astype(np.uint16)
def calculate_relative_enhancement(pre_img, post_img, mask):
    """
    Calculates mean and median relative (percent) enhancement in the fibroglandular mask.
    RE = ((SI_post - SI_pre) / SI_pre) * 100
    """
    pre_vals = pre_img[mask > 0]
    post_vals = post_img[mask > 0]

    epsilon = 1e-6
    re_vals = (post_vals - pre_vals) / (pre_vals + epsilon) * 100.0

    mean_re = np.mean(re_vals)
    median_re = np.median(re_vals)
    std_re = np.std(re_vals)
    
    return mean_re, median_re, std_re

def calculate_volumetric_bpe(pre_img, post_img, mask, voxel_spacing=(0,0,0), enhancement_threshold=20.0):  # voxel (x,y,z)
    """
    Calculates:
      - BPE Volume (in cm³)
      - BPE Fraction (fraction of FGT above threshold)
    
    threshold is percent enhancement (e.g. 50%).
    voxel_spacing = (row_spacing, col_spacing, slice_thickness) in mm.
    """
    pre_vals = pre_img[mask > 0]
    post_vals = post_img[mask > 0]

    epsilon = 1e-6
    re_vals = (post_vals - pre_vals) / (pre_vals + epsilon) * 100.0

    bpe_voxels = np.sum(re_vals > enhancement_threshold)
    
    bpe_mask = np.zeros(post_img.shape, dtype=bool)
    mask_coords = np.where(mask > 0)
    enhanced_coords = tuple(coord[re_vals > enhancement_threshold] for coord in mask_coords)
    if len(enhanced_coords[0]) > 0:
        bpe_mask[enhanced_coords] = True

    total_fgt_voxels = len(pre_vals)

    if total_fgt_voxels == 0:
        bpe_fraction = 0.0
    else:
        bpe_fraction = bpe_voxels / total_fgt_voxels
        
    # Compute voxel volume in mm³ if spacing is known
    row_spacing, col_spacing, slice_thickness = voxel_spacing
    voxel_volume_cm3 = (row_spacing * col_spacing * slice_thickness) / 1000
    bpe_volume_cm3 = bpe_voxels * voxel_volume_cm3
    
    return bpe_volume_cm3, bpe_fraction, bpe_mask
