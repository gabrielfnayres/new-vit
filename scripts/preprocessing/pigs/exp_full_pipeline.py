import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from itertools import permutations
import torchio as tio
import torch
import os
from pathlib import Path
import nibabel as nib
from bpe_calculations import calculate_bpe_mask
from normalize import normalize_mean_std

def shape_correction(img, target_shape):
    """
    Find the correct transpose order to match target_shape
    """
    # Generate all possible axis permutations
    for axes in permutations(range(len(img.shape))):
        # Apply transpose and check if resulting shape matches target
        transposed_shape = tuple(img.shape[i] for i in axes)
        if transposed_shape == target_shape:
            return np.transpose(img, axes)
    return img
        

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

def calculate_volumetric_bpe(pre_img, post_img, mask, voxel_spacing=(1.0, 1.0, 1.0), enhancement_threshold=20.0):
    """
    Calculates:
      - BPE Volume (in cm³)
      - BPE Fraction (fraction of FGT above threshold)
      - Enhanced area mask (FIXED VERSION)
    
    threshold is percent enhancement (e.g. 20%).
    voxel_spacing = (row_spacing, col_spacing, slice_thickness) in mm.
    """
    # Get values only within the fibroglandular tissue mask
    pre_vals = pre_img[mask > 0]
    post_vals = post_img[mask > 0]

    # Calculate relative enhancement
    epsilon = 1e-6
    re_vals = (post_vals - pre_vals) / (pre_vals + epsilon) * 100.0

    # Find voxels above enhancement threshold
    enhanced_indices = re_vals > enhancement_threshold
    bpe_voxels = np.sum(enhanced_indices)
    
    # Create BPE mask in full image space - FIXED VERSION
    bpe_mask = np.zeros(post_img.shape, dtype=bool)
    
    # Get coordinates of mask voxels
    mask_coords = np.where(mask > 0)
    
    # Set enhanced voxels to True in the BPE mask
    if len(mask_coords[0]) > 0 and np.any(enhanced_indices):
        enhanced_coords = tuple(coord[enhanced_indices] for coord in mask_coords)
        bpe_mask[enhanced_coords] = True
    
    # Calculate BPE fraction
    total_fgt_voxels = len(pre_vals)
    if total_fgt_voxels == 0:
        bpe_fraction = 0.0
    else:
        bpe_fraction = bpe_voxels / total_fgt_voxels
        
    # Compute voxel volume in cm³
    row_spacing, col_spacing, slice_thickness = voxel_spacing
    voxel_volume_cm3 = (row_spacing * col_spacing * slice_thickness) / 1000.0
    bpe_volume_cm3 = bpe_voxels * voxel_volume_cm3
    
    return bpe_volume_cm3, bpe_fraction, bpe_mask



def get_enhanced_area_mask_simple(pre_img, post_img, mask, enhancement_threshold=20.0):
    """
    Simple function to get enhanced area mask with debugging
    """
    print(f"Input shapes - Pre: {pre_img.shape}, Post: {post_img.shape}, Mask: {mask.shape}")
    print(f"Mask statistics - Min: {mask.min()}, Max: {mask.max()}, Non-zero: {np.sum(mask > 0)}")
    
    # Calculate enhancement map
    enhancement_map = calculate_enhancement_map(pre_img, post_img)
    print(f"Enhancement map statistics - Min: {enhancement_map.min():.2f}, Max: {enhancement_map.max():.2f}, Mean: {enhancement_map.mean():.2f}")
    
    # Apply FGT mask
    masked_enhancement = enhancement_map * (mask > 0)
    print(f"Masked enhancement statistics - Min: {masked_enhancement.min():.2f}, Max: {masked_enhancement.max():.2f}")
    
    # Create enhanced area mask
    enhanced_mask = (masked_enhancement > enhancement_threshold) & (mask > 0)
    print(f"Enhanced area mask - Total enhanced voxels: {np.sum(enhanced_mask)}")
    print(f"Enhancement threshold: {enhancement_threshold}%")
    
    if np.sum(enhanced_mask) == 0:
        print("WARNING: No voxels found above enhancement threshold!")
        print(f"Consider lowering threshold. Current max enhancement: {masked_enhancement.max():.2f}%")
    
    return enhanced_mask, enhancement_map

def validate_image_data(pre_img, post_img, mask):
    """
    Validate input data and provide diagnostics
    """
    print("\n=== DATA VALIDATION ===")
    print(f"Pre-image: shape={pre_img.shape}, dtype={pre_img.dtype}")
    print(f"  Range: [{pre_img.min():.3f}, {pre_img.max():.3f}], Mean: {pre_img.mean():.3f}")
    
    print(f"Post-image: shape={post_img.shape}, dtype={post_img.dtype}")
    print(f"  Range: [{post_img.min():.3f}, {post_img.max():.3f}], Mean: {post_img.mean():.3f}")
    
    print(f"Mask: shape={mask.shape}, dtype={mask.dtype}")
    print(f"  Range: [{mask.min():.3f}, {mask.max():.3f}], Non-zero voxels: {np.sum(mask > 0)}")
    
    # Check for potential issues
    if np.sum(mask > 0) == 0:
        print("ERROR: Mask contains no positive voxels!")
        return False
    
    if pre_img.max() <= 0:
        print("ERROR: Pre-image has no positive values!")
        return False
        
    if post_img.max() <= 0:
        print("ERROR: Post-image has no positive values!")
        return False
    
    # Check if post > pre in masked regions
    pre_masked = pre_img[mask > 0]
    post_masked = post_img[mask > 0]
    
    enhancement_raw = post_masked - pre_masked
    print(f"Raw enhancement in masked region: [{enhancement_raw.min():.3f}, {enhancement_raw.max():.3f}]")
    
    positive_enhancement = np.sum(enhancement_raw > 0)
    print(f"Voxels with positive enhancement: {positive_enhancement}/{len(enhancement_raw)} ({100*positive_enhancement/len(enhancement_raw):.1f}%)")
    
    return True

def get_slices_check(volume):
    """Extract slices at 75%, 50%, and 25% depth"""
    depth = volume.shape[0]  # First dimension is slice/depth
    slices = np.array([
        volume[int(depth * 0.75), :, :],
        volume[depth // 2, :, :],
        volume[int(depth * 0.25), :, :]
    ])
    return slices

def plot_bpe_grid(pre_volume, post_volume, fgt_mask_volume, breast_mask_volume=None, 
                  enhancement_threshold=20.0, save_path="bpe_comparison.png"):
    """Plot BPE analysis across multiple slices in a 2x3 grid"""
    
    # Get slices at different depths
    pre_slices = get_slices_check(pre_volume)
    post_slices = get_slices_check(post_volume) 
    fgt_slices = get_slices_check(fgt_mask_volume)
    
    if breast_mask_volume is not None:
        breast_slices = get_slices_check(breast_mask_volume)
    else:
        breast_slices = None
    
    # Calculate BPE masks for each slice
    bpe_masks = []
    for i in range(3):
        pre_slice = pre_slices[i]
        post_slice = post_slices[i]
        fgt_slice = fgt_slices[i]
        
        # Apply breast mask if available
        if breast_slices is not None:
            breast_slice = breast_slices[i]
            pre_slice = pre_slice * breast_slice
            post_slice = post_slice * breast_slice
            fgt_slice = fgt_slice * breast_slice
        
        # Calculate BPE mask
        bpe_mask = calculate_bpe_mask(pre_slice, post_slice, fgt_slice, enhancement_threshold)
        bpe_masks.append(bpe_mask)
    
    bpe_masks = np.array(bpe_masks)
    
    # Create 2x3 plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Post-contrast images
    axes[0, 0].imshow(post_slices[0], cmap='gray')
    axes[0, 0].set_title("Post-contrast (75%)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(post_slices[1], cmap='gray')
    axes[0, 1].set_title("Post-contrast (50%)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(post_slices[2], cmap='gray')
    axes[0, 2].set_title("Post-contrast (25%)")
    axes[0, 2].axis('off')

    # Bottom row: BPE masks
    axes[1, 0].imshow(bpe_masks[0], cmap='gray')
    axes[1, 0].set_title(f"BPE Mask (75%)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(bpe_masks[1], cmap='gray')
    axes[1, 1].set_title(f"BPE Mask (50%)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(bpe_masks[2], cmap='gray')
    axes[1, 2].set_title(f"BPE Mask (25%)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"BPE grid plot saved to: {save_path}")

def plot_enhancement_analysis(pre_img, post_img, mask, enhanced_mask, enhancement_map, slice_title=""):
    """
    Comprehensive visualization of enhancement analysis
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Top row
    axes[0, 0].imshow(pre_img, cmap='gray')
    axes[0, 0].set_title(f'Pre-contrast\n{slice_title}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(post_img, cmap='gray')  
    axes[0, 1].set_title(f'Post-contrast\n{slice_title}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask, cmap='gray_r')
    axes[0, 2].set_title('FGT Mask')
    axes[0, 2].axis('off')
    
    # Enhancement map with color scale
    im1 = axes[0, 3].imshow(enhancement_map, cmap='hot', vmin=0, vmax=200)
    axes[0, 3].set_title('Enhancement Map (%)')
    axes[0, 3].axis('off')
    plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)
    
    # Bottom row
    masked_enhancement = enhancement_map * (mask > 0)
    im2 = axes[1, 0].imshow(masked_enhancement, cmap='hot', vmin=0, vmax=200)
    axes[1, 0].set_title('Masked Enhancement')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    axes[1, 1].imshow(enhanced_mask, cmap='gray_r')
    axes[1, 1].set_title(f'Enhanced Area Mask\n({np.sum(enhanced_mask)} voxels)')
    axes[1, 1].axis('off')
    
    # Overlay enhanced mask on original image
    overlay = pre_img.copy()
    overlay_colored = np.stack([overlay, overlay, overlay], axis=-1)
    overlay_colored[enhanced_mask, 0] = np.maximum(overlay_colored[enhanced_mask, 0], 0.8 * overlay.max())
    axes[1, 2].imshow(overlay_colored)
    axes[1, 2].set_title('Enhanced Areas Overlay')
    axes[1, 2].axis('off')
    
    # Histogram of enhancement values in FGT
    enhancement_values = enhancement_map[mask > 0]
    if len(enhancement_values) > 0:
        # Flatten arrays to ensure 1D for histogram
        enhancement_values_flat = enhancement_values.flatten()
        axes[1, 3].hist(enhancement_values_flat, bins=50, alpha=0.7, color='blue', label='All FGT')
        enhanced_values = enhancement_map[enhanced_mask > 0]
        if len(enhanced_values) > 0:
            enhanced_values_flat = enhanced_values.flatten()
            axes[1, 3].hist(enhanced_values_flat, bins=50, alpha=0.7, color='red', label='Enhanced')
        axes[1, 3].axvline(x=20, color='green', linestyle='--', label='Threshold')
        axes[1, 3].set_xlabel('Enhancement (%)')
        axes[1, 3].set_ylabel('Frequency')
        axes[1, 3].set_title('Enhancement Distribution')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)
    else:
        axes[1, 3].text(0.5, 0.5, 'No FGT data', ha='center', va='center', transform=axes[1, 3].transAxes)
        axes[1, 3].set_title('Enhancement Distribution')
    
    plt.tight_layout()
    plt.show()

def debug_bpe_calculation(pre_slice, post_slice, mask_slice, enhancement_threshold=20.0):
    """
    Debug function to analyze BPE calculation step by step
    """
    print("\n=== BPE CALCULATION DEBUG ===")
    
    # Validate data
    is_valid = validate_image_data(pre_slice, post_slice, mask_slice)
    if not is_valid:
        return None, None
    
    # Calculate enhanced area mask
    enhanced_mask, enhancement_map = get_enhanced_area_mask_simple(
        pre_slice, post_slice, mask_slice, enhancement_threshold
    )
    
    # Calculate BPE metrics using the fixed function
    if np.sum(mask_slice > 0) > 0:
        mean_re, median_re, std_re = calculate_relative_enhancement(pre_slice, post_slice, mask_slice)
        print(f"\nBPE Metrics:")
        print(f"  Mean enhancement: {mean_re:.2f}%")
        print(f"  Median enhancement: {median_re:.2f}%")
        print(f"  Std enhancement: {std_re:.2f}%")
        
        bpe_volume, bpe_fraction, bpe_mask_alt = calculate_volumetric_bpe(
            pre_slice, post_slice, mask_slice, enhancement_threshold=enhancement_threshold
        )
        print(f"  BPE fraction: {bpe_fraction:.3f} ({bpe_fraction*100:.1f}%)")
        print(f"  Enhanced voxels (method 1): {np.sum(enhanced_mask)}")
        print(f"  Enhanced voxels (method 2): {np.sum(bpe_mask_alt)}")
        
        # Check if methods agree
        if np.array_equal(enhanced_mask, bpe_mask_alt):
            print("✓ Both methods produce identical enhanced masks")
        else:
            print("⚠ Methods produce different enhanced masks")
            
    return enhanced_mask, enhancement_map

# Modified process_bpe_pipeline function with debugging
def process_bpe_pipeline_with_debug(pre_image_path, post_image_path, fgt_mask_path, 
                                   breast_mask_path=None, slice_idx=None, 
                                   visualize_debug=True):
    """
    BPE pipeline with enhanced debugging capabilities
    """
    print("Loading images and masks...")
    pre_img = np.load(pre_image_path)
    post_img = np.load(post_image_path)
    fgt_mask = np.load(fgt_mask_path)
    
    print(f"Pre-image shape: {pre_img.shape}")
    print(f"Post-image shape: {post_img.shape}")
    print(f"FGT mask shape: {fgt_mask.shape}")
    
    if pre_img.shape != fgt_mask.shape:
        print("WARNING: Shape mismatch between pre-image and FGT mask")
        print("Attempting to match orientations...")
        
        if len(fgt_mask.shape) == 4 and len(pre_img.shape) == 3:
            for channel in range(fgt_mask.shape[0]):
                channel_mask = fgt_mask[channel]
                corrected_mask = shape_correction(channel_mask, pre_img.shape)
                if corrected_mask.shape == pre_img.shape:
                    fgt_mask = corrected_mask
                    print(f"Successfully matched using channel {channel}")
                    print(f"FGT mask shape after matching: {fgt_mask.shape}")
                    break
        else:
            fgt_mask = shape_correction(fgt_mask, pre_img.shape)
            print(f"FGT mask shape after correction: {fgt_mask.shape}")
    
    pre_img, _, _ = normalize_mean_std(pre_img)
    post_img, _, _ = normalize_mean_std(post_img)
    
    if breast_mask_path is not None:
        breast_mask = np.load(breast_mask_path)
        print(f"Breast mask shape: {breast_mask.shape}")
        
        if len(breast_mask.shape) == 4:
            breast_mask = breast_mask[0]  # Use first channel
            print(f"Using first channel, breast mask shape: {breast_mask.shape}")
        
        if breast_mask.shape != pre_img.shape:
            print("WARNING: Breast mask shape mismatch, attempting correction...")
            breast_mask = shape_correction(breast_mask, pre_img.shape)
            print(f"Breast mask shape after correction: {breast_mask.shape}")
        
        pre_img = pre_img * breast_mask
        post_img = post_img * breast_mask
        fgt_mask = fgt_mask * breast_mask
        print("Applied breast mask")
    
    enhanced_mask = calculate_bpe_mask(pre_img, post_img, fgt_mask)
    
    if enhanced_mask is None:
        print("ERROR: Failed to calculate enhanced area mask")
        return None

    plot_bpe_grid(pre_img, post_img, fgt_mask, 
                     breast_mask if 'breast_mask' in locals() else None,
                     save_path=f"bpe_grid_comparison_threshold_1.png")
    print(enhanced_mask.shape)
    results = {
        'enhanced_mask': enhanced_mask,
        'pre_img': pre_img,
        'post_img': post_img,
        'mask': fgt_mask,
    }
    
    return results


if __name__ == "__main__":
    # Test with real data
    pre_image_path = r"\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed-v1\data\Breast_MRI_001\pre.npy"
    post_image_path = r"\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed-v1\data\Breast_MRI_001\post_1.npy" 
    fgt_mask_path = r"D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\duke_output\fgt\Breast_MRI_001.npy"
    breast_mask_path = r"D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\duke_output\breast\Breast_MRI_001.npy"
    
    # Process with debug pipeline
    results = process_bpe_pipeline_with_debug(
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
        fgt_mask_path=fgt_mask_path,
        breast_mask_path=breast_mask_path,
        visualize_debug=True
    )
    # INSTANT DRAMATIC TUMOR VISUALIZATION
    bpe_mask = results['enhanced_mask']
    pre_img = results['pre_img']
    post_img = results['post_img']
    mask = results['mask']
    print("="*12)
    
    post_img = post_img + bpe_mask
    # Find best slice (most enhancement)
    if len(bpe_mask.shape) == 3:
        bpe_counts = np.sum(bpe_mask, axis=(0,1))
        best_slice = np.argmax(bpe_counts)
        bpe_2d = bpe_mask[:,:,best_slice]
        pre_2d = pre_img[:,:,best_slice]
        post_2d = post_img[:,:,best_slice]
        mask_2d = mask[:,:,best_slice]
    else:
        bpe_2d, pre_2d, post_2d, mask_2d = bpe_mask, pre_img, post_img, mask

    # DRAMATIC BEFORE/AFTER
    plt.figure(figsize=(16, 8), facecolor='black')

    plt.subplot(1,2,1)
    plt.imshow(pre_2d, cmap='gray')
    plt.contour(mask_2d, colors='cyan', linewidths=3)
    plt.title('BEFORE', color='white', fontsize=20, fontweight='bold')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(post_2d, cmap='gray', alpha=0.7)
    # MAKE BPE REGIONS EXPLODE WITH COLOR
    bpe_overlay = np.ma.masked_where(bpe_2d == 0, bpe_2d)
    plt.imshow(bpe_overlay, cmap='hot', alpha=1.0)
    plt.contour(mask_2d, colors='white', linewidths=3)
    plt.title('AFTER - ENHANCED!', color='white', fontsize=20, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    
    # Save BPE mask as NIfTI if results are valid
    if results is not None and 'enhanced_mask' in results:
        # Save enhanced mask as NIfTI
        import nibabel as nib
        
        # Create basic affine matrix
        voxel_spacing = (0.7, 0.7, 3.0)
        affine = np.eye(4)
        affine[0, 0] = voxel_spacing[0]
        affine[1, 1] = voxel_spacing[1] 
        affine[2, 2] = voxel_spacing[2]
        
        enhanced_mask_3d = results['enhanced_mask'].astype(np.uint16)
        
        nii_img = nib.Nifti1Image(post_img.astype(np.uint16), affine)
        output_path = f"bpe_enhanced_mask.nii.gz"
        nib.save(nii_img, output_path)
        print(f"Enhanced mask saved as NIfTI: {output_path}")
        
        np.save(f"bpe_enhanced_mask.npy", results['enhanced_mask'])
        print(f"Enhanced mask saved as numpy: bpe_enhanced_mask.npy")