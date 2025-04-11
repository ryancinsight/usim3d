"""
Ultrasound Simulation Utilities

This module provides common utilities for ultrasound simulation and analysis,
including data loading, visualization, and statistics calculation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from matplotlib.colors import Normalize

def load_simulation_data(file_path):
    """
    Load simulation data from HDF5 file
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
        
    Returns:
    --------
    tuple
        (max_pressure, x, y, z, metadata)
        max_pressure : ndarray
            Maximum pressure field
        x, y, z : ndarray
            Coordinate arrays
        metadata : dict
            Simulation metadata
    """
    with h5py.File(file_path, 'r') as f:
        # Extract pressure data
        # The pressure data is stored in the 'p' dataset
        # It has shape (1, Nt, Nx*Ny*Nz)
        pressure_data = np.array(f['p'])
        
        # Get grid dimensions
        Nx = int(np.array(f['Nx'])[0, 0, 0])
        Ny = int(np.array(f['Ny'])[0, 0, 0])
        Nz = int(np.array(f['Nz'])[0, 0, 0])
        
        # Get grid spacing
        dx = float(np.array(f['dx'])[0, 0, 0])
        dy = float(np.array(f['dy'])[0, 0, 0])
        dz = float(np.array(f['dz'])[0, 0, 0])
        
        # Get PML size
        pml_x_size = int(np.array(f['pml_x_size'])[0, 0, 0])
        
        # Calculate coordinates (in mm)
        x = np.linspace(-Nx/2 * dx * 1000, Nx/2 * dx * 1000, Nx)
        y = np.linspace(-Ny/2 * dy * 1000, Ny/2 * dy * 1000, Ny)
        
        # Calculate z coordinates with transducer at z=0 and beam propagating upward
        # In the k-wave simulation, the transducer is at the bottom of the grid
        # We need to flip the z-axis so that z=0 is at the transducer
        z = np.linspace(0, Nz * dz * 1000, Nz)
        
        # Extract metadata
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
            
        # Calculate max pressure over time
        # The pressure data has shape (1, Nt, Nx*Ny*Nz)
        # We need to reshape it to (Nt, Nx, Ny, Nz) and take the max over time
        Nt = pressure_data.shape[1]
        
        # Reshape to (Nt, Nx*Ny*Nz)
        pressure_data = pressure_data[0, :, :]
        
        # Calculate max pressure over time
        max_pressure_flat = np.max(np.abs(pressure_data), axis=0)
        
        # Calculate the actual grid dimensions without PML
        Nx_nopml = Nx - 2 * pml_x_size
        Ny_nopml = Ny - 2 * pml_x_size
        Nz_nopml = Nz - 2 * pml_x_size
        
        # Check if the size matches
        if len(max_pressure_flat) == Nx_nopml * Ny_nopml * Nz_nopml:
            print(f"Reshaping to grid without PML: {Nx_nopml} x {Ny_nopml} x {Nz_nopml}")
            # Reshape to 3D grid without PML
            max_pressure = max_pressure_flat.reshape(Nx_nopml, Ny_nopml, Nz_nopml, order='F')
        else:
            print(f"Reshaping to full grid: {Nx} x {Ny} x {Nz}")
            # Reshape to full 3D grid
            max_pressure = max_pressure_flat.reshape(Nx, Ny, Nz, order='F')
        
        # Scale to 0.81 MPa
        max_val = np.max(max_pressure)
        scaling_factor = 810000 / max_val
        max_pressure = max_pressure * scaling_factor
        
        # Flip the z-axis so that the transducer is at z=0 and the beam propagates upward
        max_pressure = np.flip(max_pressure, axis=2)
            
    return max_pressure, x, y, z, metadata

def calculate_mechanical_index(pressure, frequency):
    """
    Calculate the Mechanical Index (MI) based on the pressure field
    
    MI = p_neg / sqrt(f)
    where p_neg is the peak negative pressure in MPa and f is the frequency in MHz
    
    Parameters:
    -----------
    pressure : ndarray
        Pressure field in Pa
    frequency : float
        Frequency in Hz
        
    Returns:
    --------
    ndarray
        Mechanical Index values with the same shape as pressure
    """
    # Convert frequency to MHz
    freq_mhz = frequency / 1e6
    
    # Convert pressure to MPa
    pressure_mpa = pressure / 1e6
    
    # Calculate MI
    mi = pressure_mpa / np.sqrt(freq_mhz)
    
    return mi

def calculate_cavitation_probability(mi, frequency):
    """
    Calculate the cavitation probability based on the Mechanical Index (MI)

    Uses a sigmoid function to model probability based on literature values:
    P(cavitation) = 1 / (1 + exp(-k * (MI - MI_threshold)))

    Based on literature review, cavitation can occur at pressures as low as 0.2-0.3 MPa,
    especially at lower frequencies like 180 kHz. The threshold is frequency-dependent.

    Parameters:
    -----------
    mi : ndarray
        Mechanical Index values
    frequency : float
        Frequency in Hz

    Returns:
    --------
    ndarray
        Cavitation probability values between 0 and 1 with the same shape as mi
    """
    # Get frequency in MHz for threshold calculation
    freq_mhz = frequency / 1e6

    # Calculate frequency-dependent threshold
    # At lower frequencies, cavitation occurs at lower MI values
    # Literature suggests cavitation can occur at 0.2-0.3 MPa at low frequencies
    # For 180 kHz (0.18 MHz), an MI of ~0.2-0.3 corresponds to this pressure range
    if freq_mhz < 0.5:  # For low frequencies (< 500 kHz)
        mi_threshold = 0.25  # Lower threshold for low frequencies
    else:  # For higher frequencies
        mi_threshold = 0.35  # Higher threshold for higher frequencies

    # Steepness parameter - higher value makes transition sharper
    k = 8.0

    # Calculate probability using sigmoid function
    prob = 1.0 / (1.0 + np.exp(-k * (mi - mi_threshold)))

    print(f"Frequency: {freq_mhz:.3f} MHz")
    print(f"MI threshold for 50% cavitation probability: {mi_threshold:.3f}")
    print(f"Max MI: {np.max(mi):.3f}")
    print(f"Max cavitation probability: {np.max(prob):.3f}")

    return prob

def create_roi_mask(data_shape, z, roi_start, roi_end, exclude_interface=False, interface_depth=5):
    """
    Create a mask for the region of interest (ROI)
    
    Parameters:
    -----------
    data_shape : tuple
        Shape of the data array
    z : ndarray
        Z-coordinate array
    roi_start : float
        Start of ROI in mm
    roi_end : float
        End of ROI in mm
    exclude_interface : bool
        Whether to exclude the transducer interface region
    interface_depth : float
        Depth of the interface region to exclude in mm
        
    Returns:
    --------
    ndarray
        Boolean mask with the same shape as data
    """
    # Find indices corresponding to ROI in z direction
    roi_start_idx = np.argmin(np.abs(z - roi_start))
    roi_end_idx = np.argmin(np.abs(z - roi_end))
    
    # Create mask for ROI
    mask = np.zeros(data_shape, dtype=bool)
    mask[:, :, roi_start_idx:roi_end_idx+1] = True
    
    # If excluding interface, create interface mask
    if exclude_interface:
        interface_idx = np.argmin(np.abs(z - interface_depth))
        mask[:, :, 0:interface_idx+1] = False
    
    return mask

def calculate_statistics(data1, data2, mask=None):
    """
    Calculate statistics for the difference between two datasets
    
    Parameters:
    -----------
    data1 : ndarray
        First dataset
    data2 : ndarray
        Second dataset
    mask : ndarray or None
        Boolean mask to apply (if None, use the entire dataset)
        
    Returns:
    --------
    tuple
        (difference, abs_diff, max_diff, mean_diff, std_diff, relative_diff)
    """
    # Calculate difference
    difference = data2 - data1
    abs_diff = np.abs(difference)
    
    # Apply mask if provided
    if mask is not None:
        roi_abs_diff = abs_diff[mask]
        roi_data1 = data1[mask]
        roi_difference = difference[mask]
        
        # Calculate statistics in ROI
        max_diff = np.max(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
        mean_diff = np.mean(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
        std_diff = np.std(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
        relative_diff = np.max(roi_abs_diff) / np.max(roi_data1) * 100 if np.max(roi_data1) > 0 else 0
    else:
        # Calculate statistics for the entire dataset
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        std_diff = np.std(abs_diff)
        relative_diff = np.max(abs_diff) / np.max(data1) * 100 if np.max(data1) > 0 else 0
    
    return difference, abs_diff, max_diff, mean_diff, std_diff, relative_diff

def add_roi_markers(ax, roi_start, roi_end, exclude_interface=False, interface_depth=5, focal_depth=None):
    """
    Add ROI markers to a plot
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    roi_start : float
        Start of ROI in mm
    roi_end : float
        End of ROI in mm
    exclude_interface : bool
        Whether to exclude the transducer interface region
    interface_depth : float
        Depth of the interface region to exclude in mm
    focal_depth : float or None
        Focal depth in mm (if provided, add a marker)
    """
    # Add ROI markers
    ax.axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
    ax.axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)
    
    # Add interface marker if excluding interface
    if exclude_interface:
        ax.axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)
    
    # Add focal point marker if provided
    if focal_depth is not None:
        ax.axhline(y=focal_depth, color='r', linestyle='--', alpha=0.5)
    
    # Add center line
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

def get_roi_text(roi_start, roi_end, exclude_interface=False, interface_depth=5):
    """
    Get text describing the ROI
    
    Parameters:
    -----------
    roi_start : float
        Start of ROI in mm
    roi_end : float
        End of ROI in mm
    exclude_interface : bool
        Whether to exclude the transducer interface region
    interface_depth : float
        Depth of the interface region to exclude in mm
        
    Returns:
    --------
    str
        Text describing the ROI
    """
    roi_text = f"ROI: z = {roi_start}-{roi_end} mm"
    if exclude_interface:
        roi_text += f", excl. interface (0-{interface_depth} mm)"
    return roi_text

def create_output_dirs(focal_depths, signal_types=None):
    """
    Create output directories for simulation results
    
    Parameters:
    -----------
    focal_depths : list
        List of focal depths in mm
    signal_types : list or None
        List of signal types (if None, use default list)
    """
    if signal_types is None:
        signal_types = ['single_freq', 'sweep_freq', 'dual_freq', 'dual_sweep_freq']
    
    for depth in focal_depths:
        # Convert to integer to remove decimal point
        depth_int = int(depth)
        os.makedirs(f'sim_results/focal_{depth_int}mm', exist_ok=True)
        os.makedirs(f'sim_results/focal_{depth_int}mm/3d_sim_runner', exist_ok=True)

        # Create all signal type subfolders
        for signal_type in signal_types:
            os.makedirs(f'sim_results/focal_{depth_int}mm/3d_sim_runner/{signal_type}', exist_ok=True)

        # Create comparison folder
        os.makedirs(f'sim_results/focal_{depth_int}mm/3d_sim_runner/comparison', exist_ok=True)
