<<<<<<< HEAD
"""
Compare Mechanical Index (MI) across Different Signal Types

This script loads the simulation data from different signal types (single frequency,
sweep frequency, dual frequency, dual sweep frequency), calculates the Mechanical Index (MI)
for each, and creates visualizations showing the differences.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from matplotlib.colors import Normalize
import scipy.io as sio

def load_simulation_data(file_path):
    """Load simulation data from HDF5 file"""
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


def compare_mechanical_index(focal_depth_mm, medium_type='bone', roi_start=10, roi_end=None,
                        exclude_interface=False, interface_depth=5):
    """
    Compare Mechanical Index (MI) across different signal types

    Parameters:
    -----------
    focal_depth_mm : int
        Focal depth in mm
    medium_type : str
        Medium type ('water', 'bone', or 'soft_tissue')
    roi_start : int
        Start of region of interest in Z direction (mm)
    roi_end : int or None
        End of region of interest in Z direction (mm)
        If None, set to focal_depth_mm + 10
    exclude_interface : bool
        Whether to exclude transducer interface region from statistics calculation
    interface_depth : int
        Depth of transducer interface region to exclude (mm)

    Returns:
    --------
    dict
        Dictionary containing MI values for each signal type
    """
    # Define file paths for different signal types
    single_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/single_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    sweep_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/sweep_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    dual_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/dual_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    dual_sweep_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/dual_sweep_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'

    # Check if files exist
    file_paths = {
        'single_freq': single_freq_path,
        'sweep_freq': sweep_freq_path,
        'dual_freq': dual_freq_path,
        'dual_sweep_freq': dual_sweep_freq_path
    }

    # Dictionary to store data
    data = {}

    # Load data for each signal type
    for signal_type, file_path in file_paths.items():
        if os.path.exists(file_path):
            print(f"Loading {signal_type} simulation data from {file_path}")
            pressure, x, y, z, metadata = load_simulation_data(file_path)

            # Calculate MI
            # Use 180 kHz as the base frequency for all signal types
            frequency = 180e3
            mi = calculate_mechanical_index(pressure, frequency)

            # Store data
            data[signal_type] = {
                'pressure': pressure,
                'mi': mi,
                'x': x,
                'y': y,
                'z': z,
                'metadata': metadata
            }
        else:
            print(f"Warning: {signal_type} simulation data not found at {file_path}")

    # Create output directory
    output_dir = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Create comparison visualizations if we have at least two signal types
    if len(data) >= 2:
        # Create pairwise comparisons
        pairs = [
            ('single_freq', 'sweep_freq', 'Single vs Sweep'),
            ('single_freq', 'dual_freq', 'Single vs Dual'),
            ('sweep_freq', 'dual_sweep_freq', 'Sweep vs Dual Sweep'),
            ('dual_freq', 'dual_sweep_freq', 'Dual vs Dual Sweep')
        ]

        for type1, type2, title in pairs:
            if type1 in data and type2 in data:
                create_mi_comparison_plot(data[type1], data[type2], type1, type2, title, focal_depth_mm, medium_type, output_dir,
                                        roi_start=roi_start, roi_end=roi_end, exclude_interface=exclude_interface, interface_depth=interface_depth)

    # Create combined visualization with all signal types
    if len(data) >= 2:
        create_combined_mi_plot(data, focal_depth_mm, medium_type, output_dir,
                               roi_start=roi_start, roi_end=roi_end, exclude_interface=exclude_interface, interface_depth=interface_depth)

    return data


def create_mi_comparison_plot(data1, data2, type1, type2, title, focal_depth_mm, medium_type, output_dir,
                        roi_start=10, roi_end=None, exclude_interface=False, interface_depth=5):
    """
    Create comparison plot for Mechanical Index (MI) between two signal types

    Parameters:
    -----------
    data1 : dict
        Data for first signal type
    data2 : dict
        Data for second signal type
    type1 : str
        Name of first signal type
    type2 : str
        Name of second signal type
    title : str
        Title for the plot
    focal_depth_mm : int
        Focal depth in mm
    medium_type : str
        Medium type
    output_dir : str
        Output directory
    roi_start : int
        Start of region of interest in Z direction (mm)
    roi_end : int or None
        End of region of interest in Z direction (mm)
        If None, set to focal_depth_mm + 10
    exclude_interface : bool
        Whether to exclude transducer interface region from statistics calculation
    interface_depth : int
        Depth of transducer interface region to exclude (mm)
    """
    # Get data
    mi1 = data1['mi']
    mi2 = data2['mi']
    x = data1['x']
    y = data1['y']
    z = data1['z']

    # Check if dimensions match
    if mi1.shape != mi2.shape:
        print(f"Error: MI field dimensions don't match: {mi1.shape} vs {mi2.shape}")
        return

    # Set default roi_end if not provided
    if roi_end is None:
        roi_end = focal_depth_mm + 10

    # Calculate difference
    difference = mi2 - mi1
    abs_diff = np.abs(difference)

    # Find indices corresponding to ROI in z direction
    roi_start_idx = np.argmin(np.abs(z - roi_start))
    roi_end_idx = np.argmin(np.abs(z - roi_end))

    # Create mask for ROI
    mask = np.zeros_like(mi1, dtype=bool)
    mask[:, :, roi_start_idx:roi_end_idx+1] = True

    # If excluding interface, create interface mask
    if exclude_interface:
        interface_idx = np.argmin(np.abs(z - interface_depth))
        mask[:, :, 0:interface_idx+1] = False

    # Apply mask to calculate statistics only in ROI
    roi_abs_diff = abs_diff[mask]
    roi_mi1 = mi1[mask]
    roi_difference = difference[mask]

    # Calculate statistics in ROI
    max_diff = np.max(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
    mean_diff = np.mean(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
    std_diff = np.std(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
    relative_diff = np.max(roi_abs_diff) / np.max(roi_mi1) * 100 if np.max(roi_mi1) > 0 else 0

    print(f"ROI: z = {roi_start} mm to {roi_end} mm")
    if exclude_interface:
        print(f"Excluding interface region: z = 0 mm to {interface_depth} mm")
    print(f"Maximum absolute MI difference in ROI: {max_diff:.4f}")
    print(f"Mean absolute MI difference in ROI: {mean_diff:.4f}")
    print(f"Standard deviation of MI difference in ROI: {std_diff:.4f}")
    print(f"Maximum relative MI difference in ROI: {relative_diff:.2f}%")

    # Get the middle indices for each dimension
    nx, ny, nz = mi1.shape
    mid_x = nx // 2
    mid_y = ny // 2

    # Find the index closest to the focal point
    focal_z_idx = np.argmin(np.abs(z - focal_depth_mm))

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot first signal type X-slice (YZ plane at center of x)
    im1 = axes[0, 0].imshow(mi1[mid_x, :, :].T,
                           extent=[y.min(), y.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 0].set_title(f'{type1.replace("_", " ").title()} MI X-Slice')
    axes[0, 0].set_xlabel('Y (mm)')
    axes[0, 0].set_ylabel('Z (mm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Mechanical Index')

    # Plot second signal type X-slice (YZ plane at center of x)
    im2 = axes[0, 1].imshow(mi2[mid_x, :, :].T,
                           extent=[y.min(), y.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 1].set_title(f'{type2.replace("_", " ").title()} MI X-Slice')
    axes[0, 1].set_xlabel('Y (mm)')
    axes[0, 1].set_ylabel('Z (mm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Mechanical Index')

    # Plot difference X-slice (YZ plane at center of x)
    im3 = axes[0, 2].imshow(difference[mid_x, :, :].T,
                           extent=[y.min(), y.max(), z.min(), z.max()],
                           origin='lower', cmap='RdBu_r',
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[0, 2].set_title(f'Difference X-Slice ({type2.replace("_", " ").title()} - {type1.replace("_", " ").title()})')
    axes[0, 2].set_xlabel('Y (mm)')
    axes[0, 2].set_ylabel('Z (mm)')
    plt.colorbar(im3, ax=axes[0, 2], label='MI Difference')

    # Plot first signal type Y-slice (XZ plane at center of y)
    im4 = axes[1, 0].imshow(mi1[:, mid_y, :].T,
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[1, 0].set_title(f'{type1.replace("_", " ").title()} MI Y-Slice')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Z (mm)')
    plt.colorbar(im4, ax=axes[1, 0], label='Mechanical Index')

    # Plot second signal type Y-slice (XZ plane at center of y)
    im5 = axes[1, 1].imshow(mi2[:, mid_y, :].T,
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[1, 1].set_title(f'{type2.replace("_", " ").title()} MI Y-Slice')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Z (mm)')
    plt.colorbar(im5, ax=axes[1, 1], label='Mechanical Index')

    # Plot difference Y-slice (XZ plane at center of y)
    im6 = axes[1, 2].imshow(difference[:, mid_y, :].T,
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           origin='lower', cmap='RdBu_r',
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[1, 2].set_title(f'Difference Y-Slice ({type2.replace("_", " ").title()} - {type1.replace("_", " ").title()})')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Z (mm)')
    plt.colorbar(im6, ax=axes[1, 2], label='MI Difference')

    # Add focal point marker and ROI markers to all plots
    for ax in axes.flatten():
        # Add focal point marker
        ax.axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add ROI markers for YZ and XZ plots (first 2 rows)
        if ax in axes[:, :].flatten():
            # Add ROI start and end lines
            ax.axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
            ax.axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)

            # Add interface marker if excluding interface
            if exclude_interface:
                ax.axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)

    # Add statistics as text
    roi_text = f"ROI: z = {roi_start}-{roi_end} mm"
    if exclude_interface:
        roi_text += f", excluding interface (0-{interface_depth} mm)"

    plt.figtext(0.5, 0.01,
                f"ROI: {roi_text}\nMax Diff: {max_diff:.4f}, Mean Diff: {mean_diff:.4f}, Std Dev: {std_diff:.4f}, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Add title
    plt.suptitle(f'Mechanical Index Comparison: {title} ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    output_file = f'{output_dir}/mi_comparison_{type1}_vs_{type2}_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved MI comparison plot to {output_file}")

    # Create Z-slice comparison at focal point (XY plane at focal depth)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot first signal type Z-slice (XY plane at focal depth)
    im1 = axes[0].imshow(mi1[:, :, focal_z_idx],
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[0].set_title(f'{type1.replace("_", " ").title()} MI Z-Slice at Z={z[focal_z_idx]:.2f} mm')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0], label='Mechanical Index')

    # Plot second signal type Z-slice (XY plane at focal depth)
    im2 = axes[1].imshow(mi2[:, :, focal_z_idx],
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[1].set_title(f'{type2.replace("_", " ").title()} MI Z-Slice at Z={z[focal_z_idx]:.2f} mm')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[1], label='Mechanical Index')

    # Plot difference Z-slice (XY plane at focal depth)
    im3 = axes[2].imshow(difference[:, :, focal_z_idx],
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='RdBu_r',
                        norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[2].set_title(f'Difference Z-Slice at Z={z[focal_z_idx]:.2f} mm')
    axes[2].set_xlabel('X (mm)')
    axes[2].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[2], label='MI Difference')

    # Add focal point marker to all plots
    for ax in axes.flatten():
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Add statistics as text
    plt.figtext(0.5, 0.01,
                f"ROI: {roi_text}\nMax Diff: {max_diff:.4f}, Mean Diff: {mean_diff:.4f}, Std Dev: {std_diff:.4f}, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Add title
    plt.suptitle(f'Mechanical Index Comparison at Focal Plane: {title} ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    output_file = f'{output_dir}/mi_comparison_{type1}_vs_{type2}_z_slice_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved Z-slice MI comparison plot to {output_file}")

    return difference, max_diff, mean_diff, std_diff, relative_diff


def create_combined_mi_plot(data, focal_depth_mm, medium_type, output_dir,
                       roi_start=10, roi_end=None, exclude_interface=False, interface_depth=5):
    """
    Create combined plot for Mechanical Index (MI) across all signal types

    Parameters:
    -----------
    data : dict
        Dictionary containing data for all signal types
    focal_depth_mm : int
        Focal depth in mm
    medium_type : str
        Medium type
    output_dir : str
        Output directory
    roi_start : int
        Start of region of interest in Z direction (mm)
    roi_end : int or None
        End of region of interest in Z direction (mm)
        If None, set to focal_depth_mm + 10
    exclude_interface : bool
        Whether to exclude transducer interface region from statistics calculation
    interface_depth : int
        Depth of transducer interface region to exclude (mm)
    """
    # Check which signal types we have
    signal_types = list(data.keys())
    n_types = len(signal_types)

    if n_types < 2:
        print("Need at least 2 signal types for comparison")
        return

    # Set default roi_end if not provided
    if roi_end is None:
        roi_end = focal_depth_mm + 10

    # Print ROI information
    print(f"ROI: z = {roi_start} mm to {roi_end} mm")
    if exclude_interface:
        print(f"Excluding interface region: z = 0 mm to {interface_depth} mm")

    # Get coordinates from the first signal type
    first_type = signal_types[0]
    x = data[first_type]['x']
    y = data[first_type]['y']
    z = data[first_type]['z']

    # Find the index closest to the focal point
    focal_z_idx = np.argmin(np.abs(z - focal_depth_mm))

    # Get the middle indices for each dimension
    mi_shape = data[first_type]['mi'].shape
    mid_x = mi_shape[0] // 2
    mid_y = mi_shape[1] // 2

    # Create figure for X-slices (YZ plane)
    fig_x, axes_x = plt.subplots(1, n_types, figsize=(5*n_types, 5))

    # Create figure for Y-slices (XZ plane)
    fig_y, axes_y = plt.subplots(1, n_types, figsize=(5*n_types, 5))

    # Create figure for Z-slices (XY plane at focal depth)
    fig_z, axes_z = plt.subplots(1, n_types, figsize=(5*n_types, 5))

    # If we only have 2 signal types, axes will be a 1D array
    if n_types == 2:
        axes_x = [axes_x[0], axes_x[1]]
        axes_y = [axes_y[0], axes_y[1]]
        axes_z = [axes_z[0], axes_z[1]]

    # Plot each signal type
    for i, signal_type in enumerate(signal_types):
        mi = data[signal_type]['mi']

        # Plot X-slice (YZ plane at center of x)
        im_x = axes_x[i].imshow(mi[mid_x, :, :].T,
                              extent=[y.min(), y.max(), z.min(), z.max()],
                              origin='lower', cmap='viridis', vmin=0, vmax=2.0)
        axes_x[i].set_title(f'{signal_type.replace("_", " ").title()}')
        axes_x[i].set_xlabel('Y (mm)')
        axes_x[i].set_ylabel('Z (mm)')
        plt.colorbar(im_x, ax=axes_x[i], label='Mechanical Index')

        # Add focal point marker
        axes_x[i].axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        axes_x[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add ROI markers
        axes_x[i].axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
        axes_x[i].axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)

        # Add interface marker if excluding interface
        if exclude_interface:
            axes_x[i].axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)

        # Plot Y-slice (XZ plane at center of y)
        im_y = axes_y[i].imshow(mi[:, mid_y, :].T,
                              extent=[x.min(), x.max(), z.min(), z.max()],
                              origin='lower', cmap='viridis', vmin=0, vmax=2.0)
        axes_y[i].set_title(f'{signal_type.replace("_", " ").title()}')
        axes_y[i].set_xlabel('X (mm)')
        axes_y[i].set_ylabel('Z (mm)')
        plt.colorbar(im_y, ax=axes_y[i], label='Mechanical Index')

        # Add focal point marker
        axes_y[i].axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        axes_y[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add ROI markers
        axes_y[i].axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
        axes_y[i].axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)

        # Add interface marker if excluding interface
        if exclude_interface:
            axes_y[i].axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)

        # Plot Z-slice (XY plane at focal depth)
        im_z = axes_z[i].imshow(mi[:, :, focal_z_idx],
                              extent=[x.min(), x.max(), y.min(), y.max()],
                              origin='lower', cmap='viridis', vmin=0, vmax=2.0)
        axes_z[i].set_title(f'{signal_type.replace("_", " ").title()}')
        axes_z[i].set_xlabel('X (mm)')
        axes_z[i].set_ylabel('Y (mm)')
        plt.colorbar(im_z, ax=axes_z[i], label='Mechanical Index')

        # Add focal point marker
        axes_z[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes_z[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Create ROI text for titles
    roi_text = f"ROI: z = {roi_start}-{roi_end} mm"
    if exclude_interface:
        roi_text += f", excl. interface (0-{interface_depth} mm)"

    # Add titles
    fig_x.suptitle(f'Mechanical Index X-Slice Comparison ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)\n{roi_text}', fontsize=16)
    fig_y.suptitle(f'Mechanical Index Y-Slice Comparison ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)\n{roi_text}', fontsize=16)
    fig_z.suptitle(f'Mechanical Index Z-Slice Comparison at Focal Plane ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)', fontsize=16)

    # Adjust layout
    fig_x.tight_layout(rect=[0, 0, 1, 0.95])
    fig_y.tight_layout(rect=[0, 0, 1, 0.95])
    fig_z.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figures
    output_file_x = f'{output_dir}/mi_combined_x_slice_{medium_type}_{focal_depth_mm}mm.png'
    output_file_y = f'{output_dir}/mi_combined_y_slice_{medium_type}_{focal_depth_mm}mm.png'
    output_file_z = f'{output_dir}/mi_combined_z_slice_{medium_type}_{focal_depth_mm}mm.png'

    fig_x.savefig(output_file_x, dpi=300, bbox_inches='tight')
    fig_y.savefig(output_file_y, dpi=300, bbox_inches='tight')
    fig_z.savefig(output_file_z, dpi=300, bbox_inches='tight')

    print(f"Saved combined MI X-slice plot to {output_file_x}")
    print(f"Saved combined MI Y-slice plot to {output_file_y}")
    print(f"Saved combined MI Z-slice plot to {output_file_z}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare Mechanical Index (MI) across different signal types')
    parser.add_argument('--focal-depth', type=int, default=50,
                       help='Focal depth in mm')
    parser.add_argument('--medium', type=str, default='bone', choices=['water', 'bone', 'soft_tissue'],
                       help='Medium type')
    parser.add_argument('--roi-start', type=int, default=10,
                       help='Start of region of interest in Z direction (mm)')
    parser.add_argument('--roi-end', type=int, default=None,
                       help='End of region of interest in Z direction (mm)')
    parser.add_argument('--exclude-interface', action='store_true',
                       help='Exclude transducer interface region from statistics calculation')
    parser.add_argument('--interface-depth', type=int, default=5,
                       help='Depth of transducer interface region to exclude (mm)')

    args = parser.parse_args()

    # If roi-end is not specified, set it to slightly beyond the focal depth
    if args.roi_end is None:
        args.roi_end = args.focal_depth + 10

    compare_mechanical_index(args.focal_depth, args.medium,
                            roi_start=args.roi_start, roi_end=args.roi_end,
                            exclude_interface=args.exclude_interface,
                            interface_depth=args.interface_depth)
=======
"""
Compare Mechanical Index (MI) across Different Signal Types

This script loads the simulation data from different signal types (single frequency,
sweep frequency, dual frequency, dual sweep frequency), calculates the Mechanical Index (MI)
for each, and creates visualizations showing the differences.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from matplotlib.colors import Normalize
import scipy.io as sio

def load_simulation_data(file_path):
    """Load simulation data from HDF5 file"""
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


def compare_mechanical_index(focal_depth_mm, medium_type='bone', roi_start=10, roi_end=None,
                        exclude_interface=False, interface_depth=5):
    """
    Compare Mechanical Index (MI) across different signal types

    Parameters:
    -----------
    focal_depth_mm : int
        Focal depth in mm
    medium_type : str
        Medium type ('water', 'bone', or 'soft_tissue')
    roi_start : int
        Start of region of interest in Z direction (mm)
    roi_end : int or None
        End of region of interest in Z direction (mm)
        If None, set to focal_depth_mm + 10
    exclude_interface : bool
        Whether to exclude transducer interface region from statistics calculation
    interface_depth : int
        Depth of transducer interface region to exclude (mm)

    Returns:
    --------
    dict
        Dictionary containing MI values for each signal type
    """
    # Define file paths for different signal types
    single_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/single_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    sweep_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/sweep_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    dual_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/dual_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    dual_sweep_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/dual_sweep_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'

    # Check if files exist
    file_paths = {
        'single_freq': single_freq_path,
        'sweep_freq': sweep_freq_path,
        'dual_freq': dual_freq_path,
        'dual_sweep_freq': dual_sweep_freq_path
    }

    # Dictionary to store data
    data = {}

    # Load data for each signal type
    for signal_type, file_path in file_paths.items():
        if os.path.exists(file_path):
            print(f"Loading {signal_type} simulation data from {file_path}")
            pressure, x, y, z, metadata = load_simulation_data(file_path)

            # Calculate MI
            # Use 180 kHz as the base frequency for all signal types
            frequency = 180e3
            mi = calculate_mechanical_index(pressure, frequency)

            # Store data
            data[signal_type] = {
                'pressure': pressure,
                'mi': mi,
                'x': x,
                'y': y,
                'z': z,
                'metadata': metadata
            }
        else:
            print(f"Warning: {signal_type} simulation data not found at {file_path}")

    # Create output directory
    output_dir = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Create comparison visualizations if we have at least two signal types
    if len(data) >= 2:
        # Create pairwise comparisons
        pairs = [
            ('single_freq', 'sweep_freq', 'Single vs Sweep'),
            ('single_freq', 'dual_freq', 'Single vs Dual'),
            ('sweep_freq', 'dual_sweep_freq', 'Sweep vs Dual Sweep'),
            ('dual_freq', 'dual_sweep_freq', 'Dual vs Dual Sweep')
        ]

        for type1, type2, title in pairs:
            if type1 in data and type2 in data:
                create_mi_comparison_plot(data[type1], data[type2], type1, type2, title, focal_depth_mm, medium_type, output_dir,
                                        roi_start=roi_start, roi_end=roi_end, exclude_interface=exclude_interface, interface_depth=interface_depth)

    # Create combined visualization with all signal types
    if len(data) >= 2:
        create_combined_mi_plot(data, focal_depth_mm, medium_type, output_dir,
                               roi_start=roi_start, roi_end=roi_end, exclude_interface=exclude_interface, interface_depth=interface_depth)

    return data


def create_mi_comparison_plot(data1, data2, type1, type2, title, focal_depth_mm, medium_type, output_dir,
                        roi_start=10, roi_end=None, exclude_interface=False, interface_depth=5):
    """
    Create comparison plot for Mechanical Index (MI) between two signal types

    Parameters:
    -----------
    data1 : dict
        Data for first signal type
    data2 : dict
        Data for second signal type
    type1 : str
        Name of first signal type
    type2 : str
        Name of second signal type
    title : str
        Title for the plot
    focal_depth_mm : int
        Focal depth in mm
    medium_type : str
        Medium type
    output_dir : str
        Output directory
    roi_start : int
        Start of region of interest in Z direction (mm)
    roi_end : int or None
        End of region of interest in Z direction (mm)
        If None, set to focal_depth_mm + 10
    exclude_interface : bool
        Whether to exclude transducer interface region from statistics calculation
    interface_depth : int
        Depth of transducer interface region to exclude (mm)
    """
    # Get data
    mi1 = data1['mi']
    mi2 = data2['mi']
    x = data1['x']
    y = data1['y']
    z = data1['z']

    # Check if dimensions match
    if mi1.shape != mi2.shape:
        print(f"Error: MI field dimensions don't match: {mi1.shape} vs {mi2.shape}")
        return

    # Set default roi_end if not provided
    if roi_end is None:
        roi_end = focal_depth_mm + 10

    # Calculate difference
    difference = mi2 - mi1
    abs_diff = np.abs(difference)

    # Find indices corresponding to ROI in z direction
    roi_start_idx = np.argmin(np.abs(z - roi_start))
    roi_end_idx = np.argmin(np.abs(z - roi_end))

    # Create mask for ROI
    mask = np.zeros_like(mi1, dtype=bool)
    mask[:, :, roi_start_idx:roi_end_idx+1] = True

    # If excluding interface, create interface mask
    if exclude_interface:
        interface_idx = np.argmin(np.abs(z - interface_depth))
        mask[:, :, 0:interface_idx+1] = False

    # Apply mask to calculate statistics only in ROI
    roi_abs_diff = abs_diff[mask]
    roi_mi1 = mi1[mask]
    roi_difference = difference[mask]

    # Calculate statistics in ROI
    max_diff = np.max(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
    mean_diff = np.mean(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
    std_diff = np.std(roi_abs_diff) if len(roi_abs_diff) > 0 else 0
    relative_diff = np.max(roi_abs_diff) / np.max(roi_mi1) * 100 if np.max(roi_mi1) > 0 else 0

    print(f"ROI: z = {roi_start} mm to {roi_end} mm")
    if exclude_interface:
        print(f"Excluding interface region: z = 0 mm to {interface_depth} mm")
    print(f"Maximum absolute MI difference in ROI: {max_diff:.4f}")
    print(f"Mean absolute MI difference in ROI: {mean_diff:.4f}")
    print(f"Standard deviation of MI difference in ROI: {std_diff:.4f}")
    print(f"Maximum relative MI difference in ROI: {relative_diff:.2f}%")

    # Get the middle indices for each dimension
    nx, ny, nz = mi1.shape
    mid_x = nx // 2
    mid_y = ny // 2

    # Find the index closest to the focal point
    focal_z_idx = np.argmin(np.abs(z - focal_depth_mm))

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot first signal type X-slice (YZ plane at center of x)
    im1 = axes[0, 0].imshow(mi1[mid_x, :, :].T,
                           extent=[y.min(), y.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 0].set_title(f'{type1.replace("_", " ").title()} MI X-Slice')
    axes[0, 0].set_xlabel('Y (mm)')
    axes[0, 0].set_ylabel('Z (mm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Mechanical Index')

    # Plot second signal type X-slice (YZ plane at center of x)
    im2 = axes[0, 1].imshow(mi2[mid_x, :, :].T,
                           extent=[y.min(), y.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[0, 1].set_title(f'{type2.replace("_", " ").title()} MI X-Slice')
    axes[0, 1].set_xlabel('Y (mm)')
    axes[0, 1].set_ylabel('Z (mm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Mechanical Index')

    # Plot difference X-slice (YZ plane at center of x)
    im3 = axes[0, 2].imshow(difference[mid_x, :, :].T,
                           extent=[y.min(), y.max(), z.min(), z.max()],
                           origin='lower', cmap='RdBu_r',
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[0, 2].set_title(f'Difference X-Slice ({type2.replace("_", " ").title()} - {type1.replace("_", " ").title()})')
    axes[0, 2].set_xlabel('Y (mm)')
    axes[0, 2].set_ylabel('Z (mm)')
    plt.colorbar(im3, ax=axes[0, 2], label='MI Difference')

    # Plot first signal type Y-slice (XZ plane at center of y)
    im4 = axes[1, 0].imshow(mi1[:, mid_y, :].T,
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[1, 0].set_title(f'{type1.replace("_", " ").title()} MI Y-Slice')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Z (mm)')
    plt.colorbar(im4, ax=axes[1, 0], label='Mechanical Index')

    # Plot second signal type Y-slice (XZ plane at center of y)
    im5 = axes[1, 1].imshow(mi2[:, mid_y, :].T,
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[1, 1].set_title(f'{type2.replace("_", " ").title()} MI Y-Slice')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Z (mm)')
    plt.colorbar(im5, ax=axes[1, 1], label='Mechanical Index')

    # Plot difference Y-slice (XZ plane at center of y)
    im6 = axes[1, 2].imshow(difference[:, mid_y, :].T,
                           extent=[x.min(), x.max(), z.min(), z.max()],
                           origin='lower', cmap='RdBu_r',
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[1, 2].set_title(f'Difference Y-Slice ({type2.replace("_", " ").title()} - {type1.replace("_", " ").title()})')
    axes[1, 2].set_xlabel('X (mm)')
    axes[1, 2].set_ylabel('Z (mm)')
    plt.colorbar(im6, ax=axes[1, 2], label='MI Difference')

    # Add focal point marker and ROI markers to all plots
    for ax in axes.flatten():
        # Add focal point marker
        ax.axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add ROI markers for YZ and XZ plots (first 2 rows)
        if ax in axes[:, :].flatten():
            # Add ROI start and end lines
            ax.axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
            ax.axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)

            # Add interface marker if excluding interface
            if exclude_interface:
                ax.axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)

    # Add statistics as text
    roi_text = f"ROI: z = {roi_start}-{roi_end} mm"
    if exclude_interface:
        roi_text += f", excluding interface (0-{interface_depth} mm)"

    plt.figtext(0.5, 0.01,
                f"ROI: {roi_text}\nMax Diff: {max_diff:.4f}, Mean Diff: {mean_diff:.4f}, Std Dev: {std_diff:.4f}, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Add title
    plt.suptitle(f'Mechanical Index Comparison: {title} ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    output_file = f'{output_dir}/mi_comparison_{type1}_vs_{type2}_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved MI comparison plot to {output_file}")

    # Create Z-slice comparison at focal point (XY plane at focal depth)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot first signal type Z-slice (XY plane at focal depth)
    im1 = axes[0].imshow(mi1[:, :, focal_z_idx],
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[0].set_title(f'{type1.replace("_", " ").title()} MI Z-Slice at Z={z[focal_z_idx]:.2f} mm')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0], label='Mechanical Index')

    # Plot second signal type Z-slice (XY plane at focal depth)
    im2 = axes[1].imshow(mi2[:, :, focal_z_idx],
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='viridis', vmin=0, vmax=2.0)
    axes[1].set_title(f'{type2.replace("_", " ").title()} MI Z-Slice at Z={z[focal_z_idx]:.2f} mm')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[1], label='Mechanical Index')

    # Plot difference Z-slice (XY plane at focal depth)
    im3 = axes[2].imshow(difference[:, :, focal_z_idx],
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='RdBu_r',
                        norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[2].set_title(f'Difference Z-Slice at Z={z[focal_z_idx]:.2f} mm')
    axes[2].set_xlabel('X (mm)')
    axes[2].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[2], label='MI Difference')

    # Add focal point marker to all plots
    for ax in axes.flatten():
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Add statistics as text
    plt.figtext(0.5, 0.01,
                f"ROI: {roi_text}\nMax Diff: {max_diff:.4f}, Mean Diff: {mean_diff:.4f}, Std Dev: {std_diff:.4f}, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Add title
    plt.suptitle(f'Mechanical Index Comparison at Focal Plane: {title} ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    output_file = f'{output_dir}/mi_comparison_{type1}_vs_{type2}_z_slice_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved Z-slice MI comparison plot to {output_file}")

    return difference, max_diff, mean_diff, std_diff, relative_diff


def create_combined_mi_plot(data, focal_depth_mm, medium_type, output_dir,
                       roi_start=10, roi_end=None, exclude_interface=False, interface_depth=5):
    """
    Create combined plot for Mechanical Index (MI) across all signal types

    Parameters:
    -----------
    data : dict
        Dictionary containing data for all signal types
    focal_depth_mm : int
        Focal depth in mm
    medium_type : str
        Medium type
    output_dir : str
        Output directory
    roi_start : int
        Start of region of interest in Z direction (mm)
    roi_end : int or None
        End of region of interest in Z direction (mm)
        If None, set to focal_depth_mm + 10
    exclude_interface : bool
        Whether to exclude transducer interface region from statistics calculation
    interface_depth : int
        Depth of transducer interface region to exclude (mm)
    """
    # Check which signal types we have
    signal_types = list(data.keys())
    n_types = len(signal_types)

    if n_types < 2:
        print("Need at least 2 signal types for comparison")
        return

    # Set default roi_end if not provided
    if roi_end is None:
        roi_end = focal_depth_mm + 10

    # Print ROI information
    print(f"ROI: z = {roi_start} mm to {roi_end} mm")
    if exclude_interface:
        print(f"Excluding interface region: z = 0 mm to {interface_depth} mm")

    # Get coordinates from the first signal type
    first_type = signal_types[0]
    x = data[first_type]['x']
    y = data[first_type]['y']
    z = data[first_type]['z']

    # Find the index closest to the focal point
    focal_z_idx = np.argmin(np.abs(z - focal_depth_mm))

    # Get the middle indices for each dimension
    mi_shape = data[first_type]['mi'].shape
    mid_x = mi_shape[0] // 2
    mid_y = mi_shape[1] // 2

    # Create figure for X-slices (YZ plane)
    fig_x, axes_x = plt.subplots(1, n_types, figsize=(5*n_types, 5))

    # Create figure for Y-slices (XZ plane)
    fig_y, axes_y = plt.subplots(1, n_types, figsize=(5*n_types, 5))

    # Create figure for Z-slices (XY plane at focal depth)
    fig_z, axes_z = plt.subplots(1, n_types, figsize=(5*n_types, 5))

    # If we only have 2 signal types, axes will be a 1D array
    if n_types == 2:
        axes_x = [axes_x[0], axes_x[1]]
        axes_y = [axes_y[0], axes_y[1]]
        axes_z = [axes_z[0], axes_z[1]]

    # Plot each signal type
    for i, signal_type in enumerate(signal_types):
        mi = data[signal_type]['mi']

        # Plot X-slice (YZ plane at center of x)
        im_x = axes_x[i].imshow(mi[mid_x, :, :].T,
                              extent=[y.min(), y.max(), z.min(), z.max()],
                              origin='lower', cmap='viridis', vmin=0, vmax=2.0)
        axes_x[i].set_title(f'{signal_type.replace("_", " ").title()}')
        axes_x[i].set_xlabel('Y (mm)')
        axes_x[i].set_ylabel('Z (mm)')
        plt.colorbar(im_x, ax=axes_x[i], label='Mechanical Index')

        # Add focal point marker
        axes_x[i].axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        axes_x[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add ROI markers
        axes_x[i].axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
        axes_x[i].axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)

        # Add interface marker if excluding interface
        if exclude_interface:
            axes_x[i].axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)

        # Plot Y-slice (XZ plane at center of y)
        im_y = axes_y[i].imshow(mi[:, mid_y, :].T,
                              extent=[x.min(), x.max(), z.min(), z.max()],
                              origin='lower', cmap='viridis', vmin=0, vmax=2.0)
        axes_y[i].set_title(f'{signal_type.replace("_", " ").title()}')
        axes_y[i].set_xlabel('X (mm)')
        axes_y[i].set_ylabel('Z (mm)')
        plt.colorbar(im_y, ax=axes_y[i], label='Mechanical Index')

        # Add focal point marker
        axes_y[i].axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        axes_y[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add ROI markers
        axes_y[i].axhline(y=roi_start, color='g', linestyle='-', alpha=0.5)
        axes_y[i].axhline(y=roi_end, color='g', linestyle='-', alpha=0.5)

        # Add interface marker if excluding interface
        if exclude_interface:
            axes_y[i].axhline(y=interface_depth, color='r', linestyle=':', alpha=0.5)

        # Plot Z-slice (XY plane at focal depth)
        im_z = axes_z[i].imshow(mi[:, :, focal_z_idx],
                              extent=[x.min(), x.max(), y.min(), y.max()],
                              origin='lower', cmap='viridis', vmin=0, vmax=2.0)
        axes_z[i].set_title(f'{signal_type.replace("_", " ").title()}')
        axes_z[i].set_xlabel('X (mm)')
        axes_z[i].set_ylabel('Y (mm)')
        plt.colorbar(im_z, ax=axes_z[i], label='Mechanical Index')

        # Add focal point marker
        axes_z[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes_z[i].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Create ROI text for titles
    roi_text = f"ROI: z = {roi_start}-{roi_end} mm"
    if exclude_interface:
        roi_text += f", excl. interface (0-{interface_depth} mm)"

    # Add titles
    fig_x.suptitle(f'Mechanical Index X-Slice Comparison ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)\n{roi_text}', fontsize=16)
    fig_y.suptitle(f'Mechanical Index Y-Slice Comparison ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)\n{roi_text}', fontsize=16)
    fig_z.suptitle(f'Mechanical Index Z-Slice Comparison at Focal Plane ({medium_type.capitalize()}, {focal_depth_mm} mm focal depth)', fontsize=16)

    # Adjust layout
    fig_x.tight_layout(rect=[0, 0, 1, 0.95])
    fig_y.tight_layout(rect=[0, 0, 1, 0.95])
    fig_z.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figures
    output_file_x = f'{output_dir}/mi_combined_x_slice_{medium_type}_{focal_depth_mm}mm.png'
    output_file_y = f'{output_dir}/mi_combined_y_slice_{medium_type}_{focal_depth_mm}mm.png'
    output_file_z = f'{output_dir}/mi_combined_z_slice_{medium_type}_{focal_depth_mm}mm.png'

    fig_x.savefig(output_file_x, dpi=300, bbox_inches='tight')
    fig_y.savefig(output_file_y, dpi=300, bbox_inches='tight')
    fig_z.savefig(output_file_z, dpi=300, bbox_inches='tight')

    print(f"Saved combined MI X-slice plot to {output_file_x}")
    print(f"Saved combined MI Y-slice plot to {output_file_y}")
    print(f"Saved combined MI Z-slice plot to {output_file_z}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare Mechanical Index (MI) across different signal types')
    parser.add_argument('--focal-depth', type=int, default=50,
                       help='Focal depth in mm')
    parser.add_argument('--medium', type=str, default='bone', choices=['water', 'bone', 'soft_tissue'],
                       help='Medium type')
    parser.add_argument('--roi-start', type=int, default=10,
                       help='Start of region of interest in Z direction (mm)')
    parser.add_argument('--roi-end', type=int, default=None,
                       help='End of region of interest in Z direction (mm)')
    parser.add_argument('--exclude-interface', action='store_true',
                       help='Exclude transducer interface region from statistics calculation')
    parser.add_argument('--interface-depth', type=int, default=5,
                       help='Depth of transducer interface region to exclude (mm)')

    args = parser.parse_args()

    # If roi-end is not specified, set it to slightly beyond the focal depth
    if args.roi_end is None:
        args.roi_end = args.focal_depth + 10

    compare_mechanical_index(args.focal_depth, args.medium,
                            roi_start=args.roi_start, roi_end=args.roi_end,
                            exclude_interface=args.exclude_interface,
                            interface_depth=args.interface_depth)
>>>>>>> 793ef48ece9336b7d7632f02e7da7317c59632a2
