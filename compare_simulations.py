"""
Compare Single Frequency and Sweep Frequency Simulations

This script loads the simulation data from single frequency and sweep frequency
simulations, calculates the difference between the pressure fields, and creates
a visualization showing the difference.
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

def create_difference_plot(focal_depth_mm, medium_type='bone'):
    """Create difference plot between single frequency and sweep frequency simulations"""
    # Define file paths
    single_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/single_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    sweep_freq_path = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/sweep_freq/simulation_data_{medium_type}_{focal_depth_mm}mm.h5'
    
    # Check if files exist
    if not os.path.exists(single_freq_path):
        print(f"Error: Single frequency simulation data not found at {single_freq_path}")
        return
    
    if not os.path.exists(sweep_freq_path):
        print(f"Error: Sweep frequency simulation data not found at {sweep_freq_path}")
        return
    
    # Load simulation data
    print(f"Loading single frequency simulation data from {single_freq_path}")
    single_pressure, x_single, y_single, z_single, single_metadata = load_simulation_data(single_freq_path)
    
    print(f"Loading sweep frequency simulation data from {sweep_freq_path}")
    sweep_pressure, x_sweep, y_sweep, z_sweep, sweep_metadata = load_simulation_data(sweep_freq_path)
    
    # Check if dimensions match
    if single_pressure.shape != sweep_pressure.shape:
        print(f"Error: Pressure field dimensions don't match: {single_pressure.shape} vs {sweep_pressure.shape}")
        return
    
    # Calculate difference
    difference = sweep_pressure - single_pressure
    
    # Calculate statistics
    abs_diff = np.abs(difference)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    std_diff = np.std(abs_diff)
    relative_diff = np.max(abs_diff) / np.max(single_pressure) * 100
    
    print(f"Maximum absolute difference: {max_diff:.2f} Pa")
    print(f"Mean absolute difference: {mean_diff:.2f} Pa")
    print(f"Standard deviation of difference: {std_diff:.2f} Pa")
    print(f"Maximum relative difference: {relative_diff:.2f}%")
    
    # Create output directory
    output_dir = f'sim_results/focal_{focal_depth_mm}mm/3d_sim_runner/comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get the middle indices for each dimension
    nx, ny, nz = single_pressure.shape
    mid_x = nx // 2
    mid_y = ny // 2
    
    # Find the index closest to the focal point
    focal_z_idx = np.argmin(np.abs(z_single - focal_depth_mm))
    
    # Plot single frequency X-slice (YZ plane at center of x)
    im1 = axes[0, 0].imshow(single_pressure[mid_x, :, :].T, 
                           extent=[y_single.min(), y_single.max(), z_single.min(), z_single.max()],
                           origin='lower', cmap='viridis')
    axes[0, 0].set_title('Single Frequency X-Slice (YZ plane)')
    axes[0, 0].set_xlabel('Y (mm)')
    axes[0, 0].set_ylabel('Z (mm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Pressure (Pa)')
    
    # Plot sweep frequency X-slice (YZ plane at center of x)
    im2 = axes[0, 1].imshow(sweep_pressure[mid_x, :, :].T, 
                           extent=[y_sweep.min(), y_sweep.max(), z_sweep.min(), z_sweep.max()],
                           origin='lower', cmap='viridis')
    axes[0, 1].set_title('Sweep Frequency X-Slice (YZ plane)')
    axes[0, 1].set_xlabel('Y (mm)')
    axes[0, 1].set_ylabel('Z (mm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Pressure (Pa)')
    
    # Plot difference X-slice (YZ plane at center of x)
    im3 = axes[1, 0].imshow(difference[mid_x, :, :].T, 
                           extent=[y_single.min(), y_single.max(), z_single.min(), z_single.max()],
                           origin='lower', cmap='RdBu_r', 
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[1, 0].set_title('Difference X-Slice (Sweep - Single)')
    axes[1, 0].set_xlabel('Y (mm)')
    axes[1, 0].set_ylabel('Z (mm)')
    plt.colorbar(im3, ax=axes[1, 0], label='Pressure Difference (Pa)')
    
    # Plot absolute difference X-slice (YZ plane at center of x)
    im4 = axes[1, 1].imshow(abs_diff[mid_x, :, :].T, 
                           extent=[y_single.min(), y_single.max(), z_single.min(), z_single.max()],
                           origin='lower', cmap='hot')
    axes[1, 1].set_title('Absolute Difference X-Slice')
    axes[1, 1].set_xlabel('Y (mm)')
    axes[1, 1].set_ylabel('Z (mm)')
    plt.colorbar(im4, ax=axes[1, 1], label='Absolute Pressure Difference (Pa)')
    
    # Add focal point marker to all plots
    for ax in axes.flatten():
        ax.axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Add statistics as text
    plt.figtext(0.5, 0.01, 
                f"Max Diff: {max_diff:.2f} Pa, Mean Diff: {mean_diff:.2f} Pa, Std Dev: {std_diff:.2f} Pa, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add title
    plt.suptitle(f'Comparison of Single Frequency vs Sweep Frequency ({medium_type}, {focal_depth_mm} mm focal depth)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_file = f'{output_dir}/comparison_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_file}")
    
    # Create Y-slice comparison (XZ plane at center of y)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot single frequency Y-slice (XZ plane at center of y)
    im1 = axes[0, 0].imshow(single_pressure[:, mid_y, :].T, 
                           extent=[x_single.min(), x_single.max(), z_single.min(), z_single.max()],
                           origin='lower', cmap='viridis')
    axes[0, 0].set_title('Single Frequency Y-Slice (XZ plane)')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Z (mm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Pressure (Pa)')
    
    # Plot sweep frequency Y-slice (XZ plane at center of y)
    im2 = axes[0, 1].imshow(sweep_pressure[:, mid_y, :].T, 
                           extent=[x_sweep.min(), x_sweep.max(), z_sweep.min(), z_sweep.max()],
                           origin='lower', cmap='viridis')
    axes[0, 1].set_title('Sweep Frequency Y-Slice (XZ plane)')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Z (mm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Pressure (Pa)')
    
    # Plot difference Y-slice (XZ plane at center of y)
    im3 = axes[1, 0].imshow(difference[:, mid_y, :].T, 
                           extent=[x_single.min(), x_single.max(), z_single.min(), z_single.max()],
                           origin='lower', cmap='RdBu_r', 
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[1, 0].set_title('Difference Y-Slice (Sweep - Single)')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Z (mm)')
    plt.colorbar(im3, ax=axes[1, 0], label='Pressure Difference (Pa)')
    
    # Plot absolute difference Y-slice (XZ plane at center of y)
    im4 = axes[1, 1].imshow(abs_diff[:, mid_y, :].T, 
                           extent=[x_single.min(), x_single.max(), z_single.min(), z_single.max()],
                           origin='lower', cmap='hot')
    axes[1, 1].set_title('Absolute Difference Y-Slice')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Z (mm)')
    plt.colorbar(im4, ax=axes[1, 1], label='Absolute Pressure Difference (Pa)')
    
    # Add focal point marker to all plots
    for ax in axes.flatten():
        ax.axhline(y=focal_depth_mm, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Add statistics as text
    plt.figtext(0.5, 0.01, 
                f"Max Diff: {max_diff:.2f} Pa, Mean Diff: {mean_diff:.2f} Pa, Std Dev: {std_diff:.2f} Pa, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add title
    plt.suptitle(f'Y-Slice Comparison of Single Frequency vs Sweep Frequency ({medium_type}, {focal_depth_mm} mm focal depth)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_file = f'{output_dir}/comparison_y_slice_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved Y-slice comparison plot to {output_file}")
    
    # Create Z-slice comparison at focal point (XY plane at focal depth)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot single frequency Z-slice (XY plane at focal depth)
    im1 = axes[0, 0].imshow(single_pressure[:, :, focal_z_idx], 
                           extent=[x_single.min(), x_single.max(), y_single.min(), y_single.max()],
                           origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Single Frequency Z-Slice at Z={z_single[focal_z_idx]:.2f} mm')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Pressure (Pa)')
    
    # Plot sweep frequency Z-slice (XY plane at focal depth)
    im2 = axes[0, 1].imshow(sweep_pressure[:, :, focal_z_idx], 
                           extent=[x_sweep.min(), x_sweep.max(), y_sweep.min(), y_sweep.max()],
                           origin='lower', cmap='viridis')
    axes[0, 1].set_title(f'Sweep Frequency Z-Slice at Z={z_sweep[focal_z_idx]:.2f} mm')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Pressure (Pa)')
    
    # Plot difference Z-slice (XY plane at focal depth)
    im3 = axes[1, 0].imshow(difference[:, :, focal_z_idx], 
                           extent=[x_single.min(), x_single.max(), y_single.min(), y_single.max()],
                           origin='lower', cmap='RdBu_r', 
                           norm=Normalize(vmin=-max_diff, vmax=max_diff))
    axes[1, 0].set_title(f'Difference Z-Slice at Z={z_single[focal_z_idx]:.2f} mm (Sweep - Single)')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[1, 0], label='Pressure Difference (Pa)')
    
    # Plot absolute difference Z-slice (XY plane at focal depth)
    im4 = axes[1, 1].imshow(abs_diff[:, :, focal_z_idx], 
                           extent=[x_single.min(), x_single.max(), y_single.min(), y_single.max()],
                           origin='lower', cmap='hot')
    axes[1, 1].set_title(f'Absolute Difference Z-Slice at Z={z_single[focal_z_idx]:.2f} mm')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    plt.colorbar(im4, ax=axes[1, 1], label='Absolute Pressure Difference (Pa)')
    
    # Add focal point marker to all plots
    for ax in axes.flatten():
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Add statistics as text
    plt.figtext(0.5, 0.01, 
                f"Max Diff: {max_diff:.2f} Pa, Mean Diff: {mean_diff:.2f} Pa, Std Dev: {std_diff:.2f} Pa, Relative Diff: {relative_diff:.2f}%",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add title
    plt.suptitle(f'Z-Slice Comparison at Focal Plane: Single Frequency vs Sweep Frequency ({medium_type}, {focal_depth_mm} mm focal depth)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_file = f'{output_dir}/comparison_z_slice_{medium_type}_{focal_depth_mm}mm.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved Z-slice comparison plot to {output_file}")
    
    return difference, max_diff, mean_diff, std_diff, relative_diff

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare single frequency and sweep frequency simulations')
    parser.add_argument('--focal-depth', type=int, default=50,
                       help='Focal depth in mm')
    parser.add_argument('--medium', type=str, default='bone', choices=['water', 'bone', 'soft_tissue'],
                       help='Medium type')
    
    args = parser.parse_args()
    
    create_difference_plot(args.focal_depth, args.medium)
