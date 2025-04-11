"""
Examine the structure of the HDF5 files containing simulation data
"""

import h5py
import sys

def print_h5_structure(file_path):
    """Print the structure of an HDF5 file"""
    print(f"Examining HDF5 file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("\nFile attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            
            print("\nGroups and datasets:")
            def print_group(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                    if len(obj.attrs) > 0:
                        print(f"    Attributes:")
                        for key, value in obj.attrs.items():
                            print(f"      {key}: {value}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")
                    if len(obj.attrs) > 0:
                        print(f"    Attributes:")
                        for key, value in obj.attrs.items():
                            print(f"      {key}: {value}")
            
            f.visititems(print_group)
            
    except Exception as e:
        print(f"Error examining file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examine_h5.py <h5_file_path>")
        sys.exit(1)
    
    print_h5_structure(sys.argv[1])
