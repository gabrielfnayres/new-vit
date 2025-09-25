import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import einops
from itertools import permutations
def crop_breast_height(image, margin_top=10):
    "Crop height to 256 and try to cover breast based on intensity localization"
    # threshold = int(image.data.float().quantile(0.9))
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data>threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return  tio.Crop((0,0, bottom, top, 0, 0))

def auto_match_orientation(target_array, source_array):

    target_shape = target_array.shape
    source_shape = source_array.shape
    
    if target_shape == source_shape:
        return source_array, (0, 1, 2)
    
    if len(target_shape) != len(source_shape):
        return None, None
    
    for perm in permutations(range(len(source_shape))):
        transposed_shape = tuple(source_shape[i] for i in perm)
        if transposed_shape == target_shape:
            transposed_array = np.transpose(source_array, perm)
            return transposed_array, perm
    
    return None, None

def image_k_space(image):
    return np.fft.fftshift(np.fft.fft2(image))

def k_space_energy(k_space_data, axis=0):
    energy = np.sum(np.abs(k_space_data)**2, axis=axis)
    return energy

def find_breast_center_kspace(kspace, method='valley'):

    horizontal_profile = k_space_energy(kspace, axis=0)
    horizontal_profile = horizontal_profile / np.max(horizontal_profile)
    
    if method == 'valley':
        smoothed = gaussian_filter1d(horizontal_profile, sigma=2)
        
        peaks, _ = find_peaks(smoothed, height=0.3, distance=20)
        
        if len(peaks) >= 2:
            peak_heights = smoothed[peaks]
            top_peaks_idx = np.argsort(peak_heights)[-2:]
            left_peak = peaks[min(top_peaks_idx)]
            right_peak = peaks[max(top_peaks_idx)]
            
            valley_region = smoothed[left_peak:right_peak+1]
            valley_idx = np.argmin(valley_region) + left_peak
            
            return valley_idx, horizontal_profile, peaks
        
    return len(horizontal_profile) // 2, horizontal_profile, []

def find_breast_center_intensity(image):
    horizontal_profile = np.sum(image, axis=0)
    horizontal_profile = horizontal_profile / np.max(horizontal_profile)
    
    smoothed = gaussian_filter1d(horizontal_profile, sigma=3)
    
    peaks, _ = find_peaks(smoothed, height=0.3, distance=20)
    
    if len(peaks) >= 2:
        peak_heights = smoothed[peaks]
        top_peaks_idx = np.argsort(peak_heights)[-2:]
        left_peak = peaks[min(top_peaks_idx)]
        right_peak = peaks[max(top_peaks_idx)]
        
        valley_region = smoothed[left_peak:right_peak+1]
        valley_idx = np.argmin(valley_region) + left_peak
        
        return valley_idx, horizontal_profile, peaks
    return len(horizontal_profile) // 2, horizontal_profile, []


def crop_breasts(image, center_x):
    height, width = image.shape
    left_breast = image[:, :center_x]
    right_breast = image[:, center_x:]
    
    return left_breast, right_breast

def plot_analysis(image, profile, center, peaks=None, method_name=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.imshow(image, cmap='gray')
    ax1.axvline(x=center, color='red', linestyle='--', linewidth=2, label=f'Center: {center}')
    ax1.set_title(f'MRI Image with Detected Center ({method_name})')
    ax1.legend()
    
    ax2.plot(profile, 'b-', linewidth=1.5, label='Profile')
    ax2.axvline(x=center, color='red', linestyle='--', linewidth=2, label=f'Center: {center}')
    
    if peaks is not None and len(peaks) > 0:
        ax2.plot(peaks, profile[peaks], 'ro', markersize=8, label='Detected Peaks')
    
    ax2.set_title(f'Horizontal Profile ({method_name})')
    ax2.set_xlabel('Position (pixels)')
    ax2.set_ylabel('Normalized Intensity/Energy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_cropped_results(left_breast, right_breast):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(left_breast, cmap='gray')
    ax1.set_title('Left Breast')
    ax1.axis('off')
    
    ax2.imshow(right_breast, cmap='gray')
    ax2.set_title('Right Breast')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = np.load(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed-v1\data\Breast_MRI_001\pre.npy')
    mask = np.load(r'D:\Users\UFPB\gabriel ayres\3D-Breast-FGT-and-Blood-Vessel-Segmentation\duke_output\breast\Breast_MRI_001.npy')
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Automatically match orientations
    matched_mask, transpose_order = auto_match_orientation(image, mask)
    
    if matched_mask is not None:
        mask = matched_mask
        print(f"Successfully matched mask orientation using transpose order: {transpose_order}")
        print(f"Adjusted mask shape: {mask.shape}")
    else:
        print(f"Warning: Could not automatically match mask orientation to image shape")
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    
    slice_image_original = image[80, :, :]
    
    kspace = image_k_space(slice_image_original)
    center_kspace, profile_kspace, peaks_kspace = find_breast_center_kspace(kspace, method='valley')
    print(f"K-space center: {center_kspace}")
    
    center_intensity, profile_intensity, peaks_intensity = find_breast_center_intensity(slice_image_original)
    chosen_center = center_intensity
    
    print(f"Cropping coordinates - Left: 0 to {chosen_center}, Right: {chosen_center} to {slice_image_original.shape[1]}")
    
    if image.shape == mask.shape:
        masked_image = image * mask
        slice_image_masked = masked_image[80, :, :]
    else:
        print(f"Warning: Could not match shapes. Using image without mask.")
        slice_image_masked = slice_image_original
    
    left_breast, right_breast = crop_breasts(slice_image_masked, chosen_center)
    
    plot_analysis(slice_image_original, profile_kspace, center_kspace, peaks_kspace, "K-space Valley (Original)")
    plot_analysis(slice_image_original, profile_intensity, center_intensity, peaks_intensity, "Intensity (Original)")
    
    plot_cropped_results(left_breast, right_breast)
    
    print(f"Original image shape: {slice_image_original.shape}")
    print(f"Masked image shape: {slice_image_masked.shape}")
    print(f"Left breast shape: {left_breast.shape}")
    print(f"Right breast shape: {right_breast.shape}")