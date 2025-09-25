"""
Debug script to inspect the structure of the HDF5 file and see what patient IDs exist.
"""

import h5py
from pathlib import Path

def inspect_h5_structure(h5_path):
    """Inspect the structure of an HDF5 file."""
    print(f"Inspecting HDF5 file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nTotal number of patient groups: {len(f.keys())}")
        print("\nFirst 20 patient IDs in the HDF5 file:")
        patient_ids = list(f.keys())
        for i, patient_id in enumerate(patient_ids[:20]):
            print(f"  {i+1}: {patient_id}")
        
        if len(patient_ids) > 20:
            print(f"  ... and {len(patient_ids) - 20} more")
        
        # Check if the problematic patient ID exists
        problematic_id = 'Breast_MRI_001_left'
        if problematic_id in patient_ids:
            print(f"\n✓ Found problematic patient ID: {problematic_id}")
        else:
            print(f"\n✗ Problematic patient ID NOT found: {problematic_id}")
            
            # Look for similar IDs
            similar_ids = [pid for pid in patient_ids if '001' in pid and 'left' in pid]
            if similar_ids:
                print(f"Similar IDs found: {similar_ids}")
            
            # Check for different patterns
            left_ids = [pid for pid in patient_ids if 'left' in pid][:5]
            if left_ids:
                print(f"Sample 'left' IDs: {left_ids}")
        
        # Inspect the structure of the first patient
        if patient_ids:
            first_patient = patient_ids[0]
            print(f"\nStructure of first patient '{first_patient}':")
            patient_group = f[first_patient]
            for key in patient_group.keys():
                item = patient_group[key]
                if hasattr(item, 'shape'):
                    print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"  {key}: {type(item)}")

if __name__ == "__main__":
    # Path to the HDF5 file
    h5_path = Path('/mnt/d/Users/UFPB/gabriel ayres/test/MST/data_compressed.h5')
    
    if h5_path.exists():
        inspect_h5_structure(h5_path)
    else:
        print(f"HDF5 file not found at: {h5_path}