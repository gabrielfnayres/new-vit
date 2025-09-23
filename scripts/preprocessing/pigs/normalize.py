import numpy as np
import pydicom


def normalize_mean_std(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    normalized_volume = (volume - mean) / (std + 1e-6)
    return normalized_volume, mean, std
