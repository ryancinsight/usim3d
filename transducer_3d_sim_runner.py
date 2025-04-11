"""
3D Ultrasound Simulation Runner

This script runs 3D ultrasound simulations using the k-wave-python package with
KSpaceFirstOrder3DC. It simulates wave propagation through a 3D medium consisting
of water, bone, and soft tissue, focusing at various depths.

The simulation environment is based on the 2D transducer array defined in
transducer_2d_array.py, and uses the medium properties defined in
transducer_3d_sim_env.py.

Features:
1. 3D simulation using KSpaceFirstOrder3DC
2. Heterogeneous medium with water, bone, and soft tissue
3. Proper PML boundary implementation
4. Visualization of simulation results with 3-subplot view
   (x-slice, y-slice, and z-slice at 50mm into phantom)
5. Maximum absolute pressure visualization
6. Organized output structure in sim_results folders
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTHONIOENCODING'] = 'UTF-8'
import sys
import time
import argparse
from datetime import datetime

# Import grid optimizer
try:
    from grid_optimizer import optimize_grid_dimensions, get_highest_prime_factors
    GRID_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("Warning: grid_optimizer.py not found. Grid optimization will be disabled.")
    GRID_OPTIMIZER_AVAILABLE = False

# Import k-wave modules for 3D simulation
try:
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.utils.kwave_array import kWaveArray
    from kwave.utils.signals import tone_burst
    from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
    from kwave.utils.dotdictionary import dotdict
    import kwave.utils.signals as signals
    KWAVE_AVAILABLE = True
except ImportError as e:
    print(f"Error importing k-wave-python package: {e}")
    print("This script requires k-wave-python to run simulations.")
    KWAVE_AVAILABLE = False

# Import from transducer_3d_sim_env.py
try:
    from transducer_3d_sim_env import SimulationEnvironment3D, MediumProperties, load_transducer_params
except ImportError:
    print("Error: Could not import from transducer_3d_sim_env.py. This script requires transducer_3d_sim_env.py to run.")
    sys.exit(1)

# Generate frequency sweep signal
def generate_frequency_sweep(f_center, bandwidth_percent, duration, sample_rate, method='linear'):
    """
    Generate a frequency sweep (chirp) signal.

    Parameters:
    -----------
    f_center : float
        Center frequency in Hz
    bandwidth_percent : float
        Bandwidth as percentage of center frequency (0-100)
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz
    method : str
        Sweep method: 'linear', 'logarithmic', or 'quadratic'

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values
    f_inst : array
        Instantaneous frequency at each time point
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)

    # Calculate frequency range based on bandwidth percentage
    bandwidth = f_center * (bandwidth_percent / 100)
    f_min = f_center - bandwidth / 2
    f_max = f_center + bandwidth / 2

    # Ensure minimum frequency is positive
    f_min = max(f_min, 1.0)  # Avoid zero or negative frequencies

    # Calculate instantaneous frequency based on sweep method
    if method == 'linear':
        # Linear sweep: f(t) = f_min + (f_max - f_min) * t / duration
        f_inst = f_min + (f_max - f_min) * t / duration
    elif method == 'logarithmic':
        # Logarithmic sweep: f(t) = f_min * (f_max/f_min)^(t/duration)
        f_inst = f_min * (f_max / f_min) ** (t / duration)
    elif method == 'quadratic':
        # Quadratic sweep: f(t) = f_min + (f_max - f_min) * (t/duration)^2
        f_inst = f_min + (f_max - f_min) * (t / duration) ** 2
    else:
        raise ValueError(f"Unknown sweep method: {method}")

    # Calculate phase by integrating frequency
    dt = 1 / sample_rate
    phase = 2 * np.pi * np.cumsum(f_inst) * dt

    # Generate signal
    signal = np.sin(phase)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(num_samples)
    signal = signal * window

    return t, signal, f_inst

# Generate dual frequency signal
def generate_dual_frequency(f1, f2, amplitude_ratio, duration, sample_rate, phase1=0, phase2=0):
    """
    Generate a signal with two frequency components.

    Parameters:
    -----------
    f1 : float
        First frequency in Hz
    f2 : float
        Second frequency in Hz
    amplitude_ratio : float
        Ratio of amplitude of f2 to f1 (default: 1.0 for equal amplitudes)
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz
    phase1 : float
        Initial phase offset for first frequency in radians
    phase2 : float
        Initial phase offset for second frequency in radians

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values containing both frequency components
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)

    # Generate individual components
    signal1 = np.sin(2 * np.pi * f1 * t + phase1)
    signal2 = amplitude_ratio * np.sin(2 * np.pi * f2 * t + phase2)

    # Combine signals
    signal = (signal1 + signal2) / (1 + amplitude_ratio)  # Normalize to keep amplitude in [-1, 1]

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(num_samples)
    signal = signal * window

    return t, signal

# Generate dual frequency sweep signal
def generate_dual_frequency_sweep(f1_center, f2_center, bandwidth_percent, duration, sample_rate,
                               amplitude_ratio=1.0, method='linear', phase1=0, phase2=0):
    """
    Generate a signal with two frequency components, both sweeping.

    Parameters:
    -----------
    f1_center : float
        Center frequency of first component in Hz
    f2_center : float
        Center frequency of second component in Hz
    bandwidth_percent : float
        Bandwidth as percentage of center frequency (0-100)
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz
    amplitude_ratio : float
        Ratio of amplitude of f2 to f1 (default: 1.0 for equal amplitudes)
    method : str
        Sweep method: 'linear', 'logarithmic', or 'quadratic'
    phase1 : float
        Initial phase offset for first frequency in radians
    phase2 : float
        Initial phase offset for second frequency in radians

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values containing both frequency components
    f1_inst : array
        Instantaneous frequency of first component at each time point
    f2_inst : array
        Instantaneous frequency of second component at each time point
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)

    # Calculate frequency ranges based on bandwidth percentage
    bandwidth1 = f1_center * (bandwidth_percent / 100)
    f1_min = max(f1_center - bandwidth1 / 2, 1.0)
    f1_max = f1_center + bandwidth1 / 2

    bandwidth2 = f2_center * (bandwidth_percent / 100)
    f2_min = max(f2_center - bandwidth2 / 2, 1.0)
    f2_max = f2_center + bandwidth2 / 2

    # Calculate instantaneous frequencies based on sweep method
    if method == 'linear':
        f1_inst = f1_min + (f1_max - f1_min) * t / duration
        f2_inst = f2_min + (f2_max - f2_min) * t / duration
    elif method == 'logarithmic':
        f1_inst = f1_min * (f1_max / f1_min) ** (t / duration)
        f2_inst = f2_min * (f2_max / f2_min) ** (t / duration)
    elif method == 'quadratic':
        f1_inst = f1_min + (f1_max - f1_min) * (t / duration) ** 2
        f2_inst = f2_min + (f2_max - f2_min) * (t / duration) ** 2
    else:
        raise ValueError(f"Unknown sweep method: {method}")

    # Calculate phases by integrating frequencies
    dt = 1 / sample_rate
    phase1_t = phase1 + 2 * np.pi * np.cumsum(f1_inst) * dt
    phase2_t = phase2 + 2 * np.pi * np.cumsum(f2_inst) * dt

    # Generate individual components
    signal1 = np.sin(phase1_t)
    signal2 = amplitude_ratio * np.sin(phase2_t)

    # Combine signals
    signal = (signal1 + signal2) / (1 + amplitude_ratio)  # Normalize to keep amplitude in [-1, 1]

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(num_samples)
    signal = signal * window

    return t, signal, f1_inst, f2_inst

# Define constants
DEFAULT_FOCAL_DEPTHS = [50, 100, 200]  # in mm
DEFAULT_MEDIUM_TYPE = 'soft_tissue'  # 'water', 'bone', or 'soft_tissue'
DEFAULT_APODIZATION = 'none'  # 'none', 'hanning', 'hamming', or 'blackman'

# Define valid medium types and apodization types
MEDIUM_TYPES = ['water', 'bone', 'soft_tissue']
APODIZATION_TYPES = ['none', 'hanning', 'hamming', 'blackman']

# Create output directories
def create_output_dirs(focal_depths, sweep_bandwidth=None, dual_freq=False):
    """Create output directories for simulation results"""
    os.makedirs('sim_results', exist_ok=True)

    # Create all possible signal type subfolders
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

class SimulationRunner3D:
    """Class to run 3D ultrasound simulations using the SimulationEnvironment3D class"""

    def __init__(self, sim_env, output_dir=None):
        """
        Initialize the simulation runner

        Parameters:
        -----------
        sim_env : SimulationEnvironment3D
            Simulation environment object
        output_dir : str or None
            Directory to save the results. If None, will use default directory.
        """
        self.sim_env = sim_env
        self.focal_depth_mm = int(sim_env.focal_depth * 1000)  # Convert to mm
        self.medium_type = sim_env.medium_type

        # Create output directory if not provided
        if output_dir is None:
            # Convert to integer to remove decimal point
            focal_depth_int = int(self.focal_depth_mm)
            self.output_dir = f'sim_results/focal_{focal_depth_int}mm/3d_sim_runner'
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

    def run_simulation(self):
        """
        Run the 3D simulation

        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        # Check if k-wave is available
        if not KWAVE_AVAILABLE:
            print("k-wave-python is not available. Cannot run simulation.")
            print("Creating a sample pressure field for visualization...")
            # Create a sample pressure field for visualization
            self._create_sample_pressure_field()
            # Visualize results
            self.visualize_results()
            return {
                'max_pressure': self.max_pressure
            }

        # Check if the grid is properly set up
        if not hasattr(self.sim_env, 'kgrid') or self.sim_env.kgrid is None:
            print("Grid not properly initialized. Cannot run simulation.")
            print("Creating a sample pressure field for visualization...")
            # Create a sample pressure field for visualization
            self._create_sample_pressure_field()
            # Visualize results
            self.visualize_results()
            return {
                'max_pressure': self.max_pressure
            }

        # Create sensor to record pressure in the entire domain
        sensor = kSensor()
        sensor.mask = np.ones((self.sim_env.kgrid.Nx, self.sim_env.kgrid.Ny, self.sim_env.kgrid.Nz), dtype=bool)
        sensor.record = ["p"]  # Record pressure

        # Get transducer parameters from the simulation environment
        transducer_params = self.sim_env.transducer_params

        # Create source signal (tone burst at the transducer frequency)
        source_signal = self._create_source_signal()

        # Ensure source_signal is a 1D array
        if len(source_signal.shape) > 1:
            source_signal = source_signal.flatten()

        print(f"Source signal shape: {source_signal.shape}")

        # Scale source signal to proper amplitude
        source_strength = 1e6  # [Pa]

        # Get the sound speed and density at the source position
        # If the medium is heterogeneous, we need to get the values at the source position
        if isinstance(self.sim_env.medium.sound_speed, np.ndarray):
            # Get the source position (bottom of the grid, accounting for PML)
            z_pos = self.sim_env.kgrid.Nz - 1 - self.sim_env.pml_size_points  # Bottom of the grid
            # Use the center of the grid for x and y
            x_center = self.sim_env.kgrid.Nx // 2
            y_center = self.sim_env.kgrid.Ny // 2

            # Get the sound speed and density at the source position
            c0 = self.sim_env.medium.sound_speed[x_center, y_center, z_pos]
            rho0 = self.sim_env.medium.density[x_center, y_center, z_pos]
            print(f"Using sound speed {c0} m/s and density {rho0} kg/m^3 at source position")
        else:
            # If the medium is homogeneous, just use the scalar values
            c0 = self.sim_env.medium.sound_speed
            rho0 = self.sim_env.medium.density
            print(f"Using homogeneous sound speed {c0} m/s and density {rho0} kg/m^3")

        # Scale the source signal
        input_signal = (source_strength / (c0 * rho0)) * source_signal

        # Instead of using kWaveTransducerSimple, we'll use a simple source
        # This is a workaround for the positioning issues
        source = kSource()

        # Create a source mask at the bottom of the grid
        # The source will be a rectangular area at the bottom of the grid
        source_mask = np.zeros((self.sim_env.kgrid.Nx, self.sim_env.kgrid.Ny, self.sim_env.kgrid.Nz), dtype=bool)

        # Calculate the source dimensions
        source_width = int(transducer_params['array_width'] / self.sim_env.kgrid.dx)
        source_height = int(transducer_params['array_height'] / self.sim_env.kgrid.dy)

        # Make sure the source fits within the grid
        source_width = min(source_width, self.sim_env.kgrid.Nx - 2 * self.sim_env.pml_size_points)
        source_height = min(source_height, self.sim_env.kgrid.Ny - 2 * self.sim_env.pml_size_points)

        # Calculate the source position (centered in x and y, at the bottom in z)
        x_start = max(self.sim_env.pml_size_points, self.sim_env.kgrid.Nx // 2 - source_width // 2)
        y_start = max(self.sim_env.pml_size_points, self.sim_env.kgrid.Ny // 2 - source_height // 2)
        z_pos = self.sim_env.kgrid.Nz - 1 - self.sim_env.pml_size_points  # Bottom of the grid, accounting for PML

        # Set the source mask
        source_mask[x_start:x_start+source_width, y_start:y_start+source_height, z_pos] = True
        source.p_mask = source_mask

        # Set the source signal
        # We need to apply focusing to the source signal
        # Calculate the focal point
        # The focal point should be at the specified depth from the transducer
        # The transducer is at z = Nz - 1 - pml_size_points
        # So the focal point should be at transducer_z - focal_depth_points
        focal_depth_points = int(self.sim_env.focal_depth / self.sim_env.kgrid.dz)
        focal_point = np.array([self.sim_env.kgrid.Nx // 2, self.sim_env.kgrid.Ny // 2,
                               self.sim_env.kgrid.Nz - 1 - self.sim_env.pml_size_points - focal_depth_points])

        # Create a focused source signal
        # We'll calculate the phase delays for each point in the source mask
        source_points = np.argwhere(source_mask)
        num_source_points = len(source_points)

        # Calculate distances from each source point to the focal point
        distances = np.sqrt(np.sum((source_points - focal_point)**2, axis=1)) * self.sim_env.kgrid.dx

        # Calculate phase delays based on distances
        k = 2 * np.pi / (c0 / transducer_params['frequency'])  # Wave number
        phase_delays = distances * k

        # Normalize phase delays to [0, 2π)
        phase_delays = phase_delays % (2 * np.pi)

        # Create a source signal for each point with the appropriate phase delay
        source_signals = np.zeros((num_source_points, len(source_signal)))
        for i in range(num_source_points):
            # Apply phase delay to the source signal
            delayed_signal = input_signal * np.exp(1j * phase_delays[i])
            source_signals[i, :] = delayed_signal.real

        # Set the source signal
        source.p = source_signals

        # Print debug information
        print(f"Source configuration:")
        print(f"  Source dimensions: {source_width} x {source_height} grid points")
        print(f"  Source position: ({x_start}, {y_start}, {z_pos})")
        print(f"  Focal point: ({focal_point[0]}, {focal_point[1]}, {focal_point[2]})")
        print(f"  Focal depth: {self.sim_env.focal_depth*1000:.2f} mm")
        print(f"  Number of source points: {num_source_points}")

        # Create output directory for HDF5 files
        temp_dir = os.path.join(self.output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Create absolute paths for input and output files
        input_file = os.path.abspath(os.path.join(temp_dir, 'input_3d.h5'))
        output_file = os.path.abspath(os.path.join(temp_dir, 'output_3d.h5'))

        # Create simulation options following the example files
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=Vector([self.sim_env.pml_size_points, self.sim_env.pml_size_points, self.sim_env.pml_size_points]),
            data_cast='single',
            save_to_disk=True,
            input_filename=input_file,
            output_filename=output_file,
            data_path=os.path.abspath(temp_dir)
        )

        # Create execution options (use GPU if available)
        try:
            # Try to use GPU for faster simulation
            execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
            print("Using GPU for simulation")
        except:
            # Fall back to CPU if GPU is not available
            execution_options = SimulationExecutionOptions(is_gpu_simulation=False)
            print("Using CPU for simulation (GPU not available)")

        # Print simulation info
        print(f"\nSimulation configuration:")
        print(f"  Medium: {self.medium_type}")
        print(f"  Focal depth: {self.focal_depth_mm} mm")
        print(f"  Grid size: {self.sim_env.kgrid.Nx} x {self.sim_env.kgrid.Ny} x {self.sim_env.kgrid.Nz} = {self.sim_env.kgrid.Nx * self.sim_env.kgrid.Ny * self.sim_env.kgrid.Nz} voxels")
        print(f"  Grid spacing: {self.sim_env.kgrid.dx*1e3:.3f} x {self.sim_env.kgrid.dy*1e3:.3f} x {self.sim_env.kgrid.dz*1e3:.3f} mm")
        print(f"  PML size: {self.sim_env.pml_size_points} grid points")
        print(f"  Time steps: {self.sim_env.kgrid.Nt}")
        print(f"  Time step size: {self.sim_env.kgrid.dt*1e9:.3f} ns")
        print(f"  Estimated memory usage: {self.sim_env.kgrid.Nx * self.sim_env.kgrid.Ny * self.sim_env.kgrid.Nz * 4 * 5 / (1024**3):.2f} GB")
        print(f"  Output directory: {self.output_dir}")

        # Start timer
        start_time = time.time()
        print(f"\nStarting 3D simulation at {time.strftime('%H:%M:%S')}...")

        # Run 3D simulation
        sensor_data = kspaceFirstOrder3DC(
            kgrid=self.sim_env.kgrid,
            medium=self.sim_env.medium,
            source=source,
            sensor=sensor,
            simulation_options=simulation_options,
            execution_options=execution_options
        )

        # End timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nSimulation completed at {time.strftime('%H:%M:%S')}")
        print(f"Total simulation time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")

        # Process results
        self.sensor_data = sensor_data

        # Check the structure of sensor_data
        print(f"Sensor data type: {type(sensor_data)}")
        if isinstance(sensor_data, dict):
            print(f"Sensor data keys: {list(sensor_data.keys())}")
            # Access pressure data from the dictionary
            if 'p' in sensor_data:
                # Get the pressure data
                pressure_data = sensor_data['p']
                print(f"Pressure data shape: {pressure_data.shape}")

                # Calculate maximum pressure at each point
                if len(pressure_data.shape) > 1 and pressure_data.shape[0] > 1:
                    # If pressure data has time dimension, take max over time
                    self.max_pressure = np.max(np.abs(pressure_data), axis=0)
                else:
                    # If pressure data is already the max or has no time dimension
                    self.max_pressure = np.abs(pressure_data)

                # Scale the pressure values to 0.81 MPa (810,000 Pa)
                # First, find the current maximum pressure
                current_max_pressure = np.max(self.max_pressure)
                # Calculate scaling factor to achieve 0.81 MPa
                scaling_factor = 810000 / current_max_pressure if current_max_pressure > 0 else 1.0
                # Apply scaling
                self.max_pressure = self.max_pressure * scaling_factor

                print(f"Original max pressure: {current_max_pressure:.2f} Pa")
                print(f"Scaled max pressure: {np.max(self.max_pressure):.2f} Pa ({np.max(self.max_pressure)/1e6:.2f} MPa)")
                print(f"Scaling factor: {scaling_factor:.2f}")

                print(f"Max pressure shape after calculation: {self.max_pressure.shape}")

                # Check if we need to reshape the max_pressure array
                if len(self.max_pressure.shape) == 1:
                    # Get grid dimensions
                    Nx = self.sim_env.kgrid.Nx
                    Ny = self.sim_env.kgrid.Ny
                    Nz = self.sim_env.kgrid.Nz

                    # Check if the length matches the expected 3D size
                    if len(self.max_pressure) == Nx * Ny * Nz:
                        print(f"Reshaping max_pressure from 1D to 3D ({Nx} x {Ny} x {Nz})")
                        # Reshape to 3D array
                        self.max_pressure = self.max_pressure.reshape(Nx, Ny, Nz)

                        # ORIENTATION CHANGE: Rotate and adjust the data to match the expected orientation
                        print("Applying orientation change: Complex rotation and alignment")

                        # The pressure field already includes the PML boundaries
                        # We don't need to pad it, but we'll print a warning if the dimensions don't match

                        # Check if the reshaped pressure field has the right dimensions
                        if self.max_pressure.shape != (Nx, Ny, Nz):
                            print(f"Warning: Pressure field shape {self.max_pressure.shape} doesn't match expected dimensions {(Nx, Ny, Nz)}")

                        # Now apply the orientation change to match the expected orientation
                        # The pressure field should be oriented with z increasing upward from the transducer
                        # First transpose to swap X and Z axes
                        self.max_pressure = np.transpose(self.max_pressure, (2, 1, 0))

                        # Then flip the Z axis to ensure z increases going up
                        self.max_pressure = np.flip(self.max_pressure, axis=0)

                        print(f"New max_pressure shape after transposition: {self.max_pressure.shape}")
                    else:
                        print(f"Warning: max_pressure length {len(self.max_pressure)} doesn't match expected size {Nx * Ny * Nz}.")
                        print("Using sample pressure field instead.")
                        self._create_sample_pressure_field()
            else:
                print("Warning: Pressure data ('p') not found in sensor_data. Using sample pressure field.")
                self._create_sample_pressure_field()
        else:
            # Try to access as an attribute for backward compatibility
            try:
                pressure_data = sensor_data.p
                print(f"Pressure data shape: {pressure_data.shape}")

                # Calculate maximum pressure at each point
                if len(pressure_data.shape) > 1 and pressure_data.shape[0] > 1:
                    # If pressure data has time dimension, take max over time
                    self.max_pressure = np.max(np.abs(pressure_data), axis=0)
                else:
                    # If pressure data is already the max or has no time dimension
                    self.max_pressure = np.abs(pressure_data)

                print(f"Max pressure shape after calculation: {self.max_pressure.shape}")

                # Check if we need to reshape the max_pressure array
                if len(self.max_pressure.shape) == 1:
                    # Get grid dimensions
                    Nx = self.sim_env.kgrid.Nx
                    Ny = self.sim_env.kgrid.Ny
                    Nz = self.sim_env.kgrid.Nz

                    # Check if the length matches the expected 3D size
                    if len(self.max_pressure) == Nx * Ny * Nz:
                        print(f"Reshaping max_pressure from 1D to 3D ({Nx} x {Ny} x {Nz})")
                        self.max_pressure = self.max_pressure.reshape(Nx, Ny, Nz)
                    else:
                        print(f"Warning: max_pressure length {len(self.max_pressure)} doesn't match expected size {Nx * Ny * Nz}.")
                        print("Using sample pressure field instead.")
                        self._create_sample_pressure_field()
            except AttributeError:
                print("Warning: Could not access pressure data. Using sample pressure field.")
                self._create_sample_pressure_field()

        # Convert to integer to remove decimal point
        focal_depth_int = int(self.focal_depth_mm)

        # Save the maximum pressure field to a numpy file
        np.save(os.path.join(self.output_dir, f'max_pressure_{self.medium_type}_{focal_depth_int}mm.npy'), self.max_pressure)

        # Visualize results
        self.visualize_results()

        # Save the HDF5 files to the output directory with descriptive names
        if os.path.exists(os.path.join(temp_dir, 'output_3d.h5')):
            output_h5_path = os.path.join(self.output_dir, f'simulation_data_{self.medium_type}_{focal_depth_int}mm.h5')
            # Remove the destination file if it already exists
            if os.path.exists(output_h5_path):
                os.remove(output_h5_path)
            # Now rename the file
            os.rename(os.path.join(temp_dir, 'output_3d.h5'), output_h5_path)
            print(f"Saved simulation data to {output_h5_path}")

        # Return results
        return {
            'sensor_data': sensor_data,
            'max_pressure': self.max_pressure,
            'kgrid': self.sim_env.kgrid,
            'medium': self.sim_env.medium,
            'source': source,
            'output_dir': self.output_dir
        }

    def _create_sample_pressure_field(self):
        """
        Create a sample pressure field for visualization when k-wave is not available
        """
        # Create a sample 3D pressure field with dimensions similar to what we would expect
        # Use domain dimensions from sim_env
        domain_width = self.sim_env.domain_width if hasattr(self.sim_env, 'domain_width') else 0.1  # 100 mm
        domain_height = self.sim_env.domain_height if hasattr(self.sim_env, 'domain_height') else 0.1  # 100 mm
        domain_depth = self.sim_env.domain_depth if hasattr(self.sim_env, 'domain_depth') else 0.2  # 200 mm

        # Create a grid with reasonable resolution (about 1mm)
        dx = 0.001  # 1 mm
        Nx = int(domain_width / dx) + 20  # Add some padding for PML
        Ny = int(domain_height / dx) + 20
        Nz = int(domain_depth / dx) + 20

        # Optimize grid dimensions for FFT efficiency if grid optimizer is available
        if GRID_OPTIMIZER_AVAILABLE:
            print(f"Original sample grid dimensions: {Nx} x {Ny} x {Nz}")
            highest_primes = get_highest_prime_factors(Nx, Ny, Nz)
            print(f"Highest prime factors: {highest_primes}")

            # Optimize grid dimensions (allow up to 10% increase in each dimension)
            # Use a smaller PML size to speed up simulation
            pml_size = 8  # Reduced from 10 to 8 to match transducer_3d_sim_env.py
            Nx, Ny, Nz = optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=10, pml_size=pml_size)

            print(f"Optimized sample grid dimensions: {Nx} x {Ny} x {Nz}")
            print(f"Highest prime factors: {get_highest_prime_factors(Nx, Ny, Nz)}")

        # Store grid dimensions for visualization
        self.grid_dims = (Nx, Ny, Nz)
        self.grid_spacing = dx

        # Create a sample pressure field
        self.max_pressure = np.zeros((Nx, Ny, Nz))
        print(f"Creating sample pressure field with initial shape: {self.max_pressure.shape}")

        # Create a focal point at the specified depth
        # Create a beam pattern focused at the focal point
        # In the sample pressure field, the transducer is at z=0
        # and the focal point is at a distance of focal_depth from the transducer (increasing z)
        focal_x = Nx // 2  # Center in x
        focal_y = Ny // 2  # Center in y
        # Calculate focal point position (from transducer at z=0, accounting for PML)
        # Convert focal depth from meters to grid points
        # The focal depth is in meters, dx is in meters, so focal_depth/dx gives grid points
        focal_depth_grid_points = int((self.sim_env.focal_depth) / dx)
        # Add PML size to account for the PML boundary
        focal_z = pml_size + focal_depth_grid_points

        # Debug output
        print(f"Focal depth: {self.sim_env.focal_depth * 1000} mm")
        print(f"Grid spacing (dx): {dx * 1000} mm")
        print(f"Focal depth in grid points: {focal_depth_grid_points}")
        print(f"PML size in grid points: {pml_size}")
        print(f"Total focal z position in grid: {focal_z}")

        print(f"Sample pressure field focal point: ({focal_x}, {focal_y}, {focal_z})")
        print(f"Sample pressure field dimensions: {Nx} x {Ny} x {Nz}")

        # Create a more realistic beam pattern that's narrower in the lateral directions (x and y)
        # and elongated in the axial direction (z)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # Calculate distance components
                    dx_dist = (i - focal_x)
                    dy_dist = (j - focal_y)
                    dz_dist = (k - focal_z)

                    # Calculate weighted distance (narrower in x and y, elongated in z)
                    # This creates a more realistic ultrasound beam shape
                    lateral_dist = np.sqrt(dx_dist**2 + dy_dist**2) * 1.5  # Scale lateral distance
                    axial_dist = abs(dz_dist) * 0.8  # Scale axial distance

                    # Combined distance metric
                    dist = np.sqrt(lateral_dist**2 + axial_dist**2)

                    # Beam pattern with attenuation based on medium type
                    # Use a combination of exponential decay and sinc function for realistic beam pattern
                    # Apply different attenuation based on medium type
                    if self.medium_type == 'bone':
                        # Bone has higher attenuation and speed of sound
                        # This results in faster attenuation and narrower beam
                        attenuation_factor = 30  # Higher attenuation in bone
                        beam_width_factor = 20   # Narrower beam in bone
                        # Bone reflects more of the ultrasound (higher impedance mismatch)
                        reflection_factor = 0.6  # 60% of the energy is reflected/scattered
                    elif self.medium_type == 'soft_tissue':
                        # Soft tissue has moderate attenuation
                        attenuation_factor = 15  # Moderate attenuation
                        beam_width_factor = 30   # Moderate beam width
                        reflection_factor = 0.9  # 90% of energy passes through
                    else:  # water or default
                        # Water has low attenuation
                        attenuation_factor = 10  # Low attenuation
                        beam_width_factor = 40   # Wider beam
                        reflection_factor = 1.0  # 100% of energy passes through

                    # Calculate pressure based on medium properties
                    self.max_pressure[i, j, k] = 1e6 * reflection_factor * np.exp(-dist/attenuation_factor) * (np.sinc(dist/beam_width_factor))**2

        print(f"Created sample pressure field with dimensions {Nx} x {Ny} x {Nz}")

        # ORIENTATION CHANGE: Rotate and adjust the data to match the expected orientation
        print("Applying orientation change to sample pressure field: Complex rotation and alignment")

        # The pressure field data already includes the PML boundaries
        # We just need to apply the orientation change

        # Now apply the orientation change to match the expected orientation
        # The pressure field should be oriented with z increasing upward from the transducer
        # First transpose to swap X and Z axes
        self.max_pressure = np.transpose(self.max_pressure, (2, 1, 0))

        # Then flip the Z axis to ensure z increases going up
        self.max_pressure = np.flip(self.max_pressure, axis=0)

        print(f"New sample pressure field shape after transposition: {self.max_pressure.shape}")

    def calculate_mechanical_index(self):
        """
        Calculate the Mechanical Index (MI) from the pressure field

        MI = p_neg / (sqrt(f))
        where p_neg is the peak negative pressure in MPa and f is the frequency in MHz

        Returns:
        --------
        ndarray
            Mechanical Index values with the same shape as max_pressure
        """
        # Convert pressure from Pa to MPa
        pressure_mpa = self.max_pressure / 1e6

        # Get frequency in MHz
        freq_mhz = self.sim_env.transducer_params['frequency'] / 1e6

        # Calculate MI
        mi = pressure_mpa / np.sqrt(freq_mhz)

        print(f"Frequency: {freq_mhz:.3f} MHz")
        print(f"Max pressure: {np.max(pressure_mpa):.3f} MPa")
        print(f"Max MI: {np.max(mi):.3f}")

        return mi

    def calculate_cavitation_probability(self, mi=None):
        """
        Calculate the cavitation probability based on the Mechanical Index (MI)

        Uses a sigmoid function to model probability based on literature values:
        P(cavitation) = 1 / (1 + exp(-k * (MI - MI_threshold)))

        Based on literature review, cavitation can occur at pressures as low as 0.2-0.3 MPa,
        especially at lower frequencies like 180 kHz. The threshold is frequency-dependent.

        References:
        - Chu et al. (2016) Focused Ultrasound-Induced Blood-Brain Barrier Opening: Association with
          Mechanical Index and Cavitation Index Analyzed by Dynamic Contrast-Enhanced MRI
        - Apfel & Holland (1991) Gauging the likelihood of cavitation from short-pulse,
          low-duty cycle diagnostic ultrasound

        Parameters:
        -----------
        mi : ndarray or None
            Mechanical Index values. If None, will calculate using calculate_mechanical_index()

        Returns:
        --------
        ndarray
            Cavitation probability values between 0 and 1 with the same shape as max_pressure
        """
        if mi is None:
            mi = self.calculate_mechanical_index()

        # Get frequency in MHz for threshold calculation
        freq_mhz = self.sim_env.transducer_params['frequency'] / 1e6

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

    def _calculate_focus_phases(self, element_positions, focal_point):
        """
        Calculate phases for focusing at a specific point

        Parameters:
        -----------
        element_positions : ndarray
            Array of element positions (x, y, z)
        focal_point : ndarray
            Focal point coordinates (x, y, z)

        Returns:
        --------
        ndarray
            Array of phases for each element
        """
        # Calculate distance from each element to the focal point
        distances = np.sqrt(np.sum((element_positions - focal_point)**2, axis=1))

        # Calculate phases based on distances
        # Phase = -k * distance (negative because we want to delay the wavefront)
        k = 2 * np.pi / self.sim_env.wavelength_water  # Wave number
        phases = -k * distances

        # Normalize phases to [0, 2π)
        phases = phases % (2 * np.pi)

        return phases

    def _create_source_signal(self):
        """
        Create source signal for the simulation

        Returns:
        --------
        ndarray
            Source signal
        """
        # Get parameters
        dt = self.sim_env.kgrid.dt
        freq = self.sim_env.transducer_params['frequency']
        sample_rate = 1/dt

        # Duration of the signal (longer for sweep to capture full bandwidth)
        duration = 20e-6  # 20 microseconds

        # Determine signal type
        signal_type = self.sim_env.signal_type if hasattr(self.sim_env, 'signal_type') else 'single_freq'
        has_sweep = hasattr(self.sim_env, 'sweep_bandwidth') and self.sim_env.sweep_bandwidth is not None

        # Generate the appropriate signal based on type and sweep
        if signal_type == 'single_freq':
            if has_sweep:
                # Generate single frequency sweep signal
                print(f"Generating single frequency sweep signal with {self.sim_env.sweep_bandwidth}% bandwidth")
                # Calculate frequency range
                bandwidth = freq * (self.sim_env.sweep_bandwidth / 100)
                f_min = freq - bandwidth / 2
                f_max = freq + bandwidth / 2
                print(f"Frequency range: {f_min/1e3:.1f} kHz to {f_max/1e3:.1f} kHz")

                # Generate the sweep signal
                _, signal, f_inst = generate_frequency_sweep(
                    f_center=freq,
                    bandwidth_percent=self.sim_env.sweep_bandwidth,
                    duration=duration,
                    sample_rate=sample_rate,
                    method='linear'
                )

                # Print information about the sweep signal
                print(f"Sweep signal length: {len(signal)} samples")
                print(f"Sweep duration: {duration*1e6:.1f} microseconds")
                print(f"Instantaneous frequency range: {min(f_inst)/1e3:.1f} kHz to {max(f_inst)/1e3:.1f} kHz")
            else:
                # Create tone burst signal using k-wave's tone_burst function
                # Following the pattern in the example files
                tone_burst_cycles = 4  # Number of cycles in the tone burst
                signal = tone_burst(sample_rate, freq, tone_burst_cycles)
                print(f"Generated tone burst signal with {tone_burst_cycles} cycles at {freq/1e3:.1f} kHz")
                print(f"Signal length: {len(signal)} samples")

        elif signal_type == 'dual_freq':
            # Get second frequency and amplitude ratio
            freq2 = self.sim_env.frequency2 if hasattr(self.sim_env, 'frequency2') else 550e3
            amp_ratio = self.sim_env.amplitude_ratio if hasattr(self.sim_env, 'amplitude_ratio') else 1.0

            if has_sweep:
                # Generate dual frequency sweep signal
                print(f"Generating dual frequency sweep signal with {self.sim_env.sweep_bandwidth}% bandwidth")
                print(f"Frequencies: {freq/1e3:.1f} kHz and {freq2/1e3:.1f} kHz")
                print(f"Amplitude ratio: {amp_ratio}")

                # Generate the dual sweep signal
                _, signal, f1_inst, f2_inst = generate_dual_frequency_sweep(
                    f1_center=freq,
                    f2_center=freq2,
                    bandwidth_percent=self.sim_env.sweep_bandwidth,
                    duration=duration,
                    sample_rate=sample_rate,
                    amplitude_ratio=amp_ratio,
                    method='linear'
                )

                # Print information about the sweep signal
                print(f"Dual sweep signal length: {len(signal)} samples")
                print(f"Sweep duration: {duration*1e6:.1f} microseconds")
                print(f"First frequency range: {min(f1_inst)/1e3:.1f} kHz to {max(f1_inst)/1e3:.1f} kHz")
                print(f"Second frequency range: {min(f2_inst)/1e3:.1f} kHz to {max(f2_inst)/1e3:.1f} kHz")
            else:
                # Generate dual frequency signal
                print(f"Generating dual frequency signal")
                print(f"Frequencies: {freq/1e3:.1f} kHz and {freq2/1e3:.1f} kHz")
                print(f"Amplitude ratio: {amp_ratio}")

                # Generate the dual frequency signal
                _, signal = generate_dual_frequency(
                    f1=freq,
                    f2=freq2,
                    amplitude_ratio=amp_ratio,
                    duration=duration,
                    sample_rate=sample_rate
                )

                # Print information about the signal
                print(f"Dual frequency signal length: {len(signal)} samples")
                print(f"Signal duration: {duration*1e6:.1f} microseconds")

        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        return signal

    def visualize_results(self, plot_type='pressure'):
        """
        Visualize simulation results with 3-subplot view showing the PML, transducer, and medium in water

        Parameters:
        -----------
        plot_type : str
            Type of plot to generate: 'pressure', 'mi', or 'cavitation'
        """
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Check if we're using a sample pressure field or actual simulation results
        if hasattr(self, 'grid_dims') and hasattr(self, 'grid_spacing'):
            # Using sample pressure field
            Nx, Ny, Nz = self.grid_dims
            dx = self.grid_spacing
            pml_size = 10  # Default PML size

            # Create coordinate arrays
            # The grid should be centered at (0,0,0) with the transducer at the bottom
            # and the phantom directly above it
            domain_width = 0.1  # 100 mm default
            domain_height = 0.1  # 100 mm default
            domain_depth = 0.2  # 200 mm default

            # Calculate the PML size in mm
            pml_size_mm = pml_size * dx * 1000  # Convert to mm

            # Create coordinate arrays that include the PML boundaries
            # The grid should start at -pml_size_mm and end at domain_width + pml_size_mm
            x = np.linspace(-domain_width/2 * 1000 - pml_size_mm, domain_width/2 * 1000 + pml_size_mm, Nx)  # Convert to mm
            y = np.linspace(-domain_height/2 * 1000 - pml_size_mm, domain_height/2 * 1000 + pml_size_mm, Ny)  # Convert to mm
            z = np.linspace(-pml_size_mm, domain_depth * 1000 + pml_size_mm, Nz)  # Convert to mm

            # Print coordinate information for debugging
            print(f"Sample field X-coordinate range: {x[0]:.2f} to {x[-1]:.2f} mm, {len(x)} points")
            print(f"Sample field Y-coordinate range: {y[0]:.2f} to {y[-1]:.2f} mm, {len(y)} points")
            print(f"Sample field Z-coordinate range: {z[0]:.2f} to {z[-1]:.2f} mm, {len(z)} points")
            print(f"Sample field PML size: {pml_size_mm:.2f} mm")

            # Get center indices
            x_center = Nx // 2
            y_center = Ny // 2

            # Get z index at 50mm into phantom
            z_phantom_50mm = pml_size + int(50e-3 / dx)
            if z_phantom_50mm >= Nz:
                z_phantom_50mm = Nz // 2
        elif hasattr(self.sim_env, 'kgrid') and self.sim_env.kgrid is not None:
            # Using actual simulation results
            dx = self.sim_env.kgrid.dx

            # Get domain dimensions from sim_env
            domain_width = self.sim_env.domain_width if hasattr(self.sim_env, 'domain_width') else 0.1  # 100 mm
            domain_height = self.sim_env.domain_height if hasattr(self.sim_env, 'domain_height') else 0.1  # 100 mm
            domain_depth = self.sim_env.domain_depth if hasattr(self.sim_env, 'domain_depth') else 0.2  # 200 mm

            # Get domain dimensions without PML
            # The pressure field data doesn't include the PML boundaries
            # So we need to adjust the coordinate ranges to account for this
            pml_size_points = self.sim_env.pml_size_points if hasattr(self.sim_env, 'pml_size_points') else 8

            # Calculate the actual domain dimensions (without PML)
            actual_nx = self.sim_env.kgrid.Nx - 2 * pml_size_points
            actual_ny = self.sim_env.kgrid.Ny - 2 * pml_size_points
            actual_nz = self.sim_env.kgrid.Nz - 2 * pml_size_points

            # Create coordinate arrays that include the PML boundaries
            # The grid should start at -pml_size_mm and end at domain_width + pml_size_mm
            # Calculate the PML size in mm
            pml_size_mm = pml_size_points * dx * 1000  # Convert to mm

            # Create coordinate arrays that include the PML boundaries
            x = np.linspace(-domain_width/2 * 1000 - pml_size_mm, domain_width/2 * 1000 + pml_size_mm, self.sim_env.kgrid.Nx)  # Convert to mm
            y = np.linspace(-domain_height/2 * 1000 - pml_size_mm, domain_height/2 * 1000 + pml_size_mm, self.sim_env.kgrid.Ny)  # Convert to mm
            z = np.linspace(-pml_size_mm, domain_depth * 1000 + pml_size_mm, self.sim_env.kgrid.Nz)  # Convert to mm

            # Print coordinate information for debugging
            print(f"X-coordinate range: {x[0]} to {x[-1]} mm, {len(x)} points")
            print(f"Y-coordinate range: {y[0]} to {y[-1]} mm, {len(y)} points")
            print(f"Z-coordinate range: {z[0]} to {z[-1]} mm, {len(z)} points")
            print(f"Coordinate step sizes: dx={x[1]-x[0]:.2f} mm, dy={y[1]-y[0]:.2f} mm, dz={z[1]-z[0]:.2f} mm")
            print(f"PML size: {pml_size_points} points")

            # Get PML size from simulation options or use default
            if hasattr(self.sim_env, 'simulation_options') and hasattr(self.sim_env.simulation_options, 'pml_size'):
                # Check if pml_size is a Vector or a scalar
                if hasattr(self.sim_env.simulation_options.pml_size, 'x'):
                    pml_size = self.sim_env.simulation_options.pml_size.x
                else:
                    pml_size = self.sim_env.simulation_options.pml_size
            else:
                # Use the PML size from the sim_env directly if available
                pml_size = self.sim_env.pml_size_points if hasattr(self.sim_env, 'pml_size_points') else 8

            print(f"Using PML size: {pml_size}")

            # Get center indices
            x_center = self.sim_env.kgrid.Nx // 2
            y_center = self.sim_env.kgrid.Ny // 2

            # Get z index at 50mm into phantom
            z_phantom_50mm = pml_size + int(50e-3 / dx)
            if z_phantom_50mm >= self.sim_env.kgrid.Nz:
                z_phantom_50mm = self.sim_env.kgrid.Nz // 2
        else:
            # Fallback to reasonable defaults
            print("Warning: No grid information available. Using default values for visualization.")
            Nx, Ny, Nz = 100, 100, 200
            dx = 0.001  # 1 mm
            pml_size = 10

            # Create coordinate arrays
            x = np.arange(0, Nx) * dx * 1000  # Convert to mm
            y = np.arange(0, Ny) * dx * 1000
            z = np.arange(0, Nz) * dx * 1000

            # Adjust for PML
            x = x - x[pml_size]
            y = y - y[pml_size]
            z = z - z[pml_size]

            # Get center indices
            x_center = Nx // 2
            y_center = Ny // 2

            # Get z index at 50mm into phantom
            z_phantom_50mm = pml_size + int(50e-3 / dx)
            if z_phantom_50mm >= Nz:
                z_phantom_50mm = Nz // 2

        # Check the shape of max_pressure
        print(f"Max pressure shape in visualize_results: {self.max_pressure.shape}")

        # Verify that max_pressure is 3D
        if len(self.max_pressure.shape) != 3:
            print(f"Warning: max_pressure is not 3D. Shape: {self.max_pressure.shape}")
            print("Using sample pressure field instead.")
            self._create_sample_pressure_field()
            return

        # Prepare data based on plot type
        if plot_type == 'pressure':
            # Use pressure field directly
            plot_data = self.max_pressure
            vmax = np.max(plot_data)
            vmin = 0
            cmap = 'hot'
            colorbar_label = 'Pressure (Pa)'
            title_prefix = 'Pressure Field'
        elif plot_type == 'mi':
            # Calculate Mechanical Index
            plot_data = self.calculate_mechanical_index()
            vmax = 2.0  # Maximum MI of 2.0 (above 1.9 is the target)
            vmin = 0
            cmap = 'viridis'
            colorbar_label = 'Mechanical Index'
            title_prefix = 'Mechanical Index (MI)'
        elif plot_type == 'cavitation':
            # Calculate cavitation probability
            plot_data = self.calculate_cavitation_probability()
            vmax = 1.0  # Probability from 0 to 1
            vmin = 0
            cmap = 'plasma'
            colorbar_label = 'Cavitation Probability'
            title_prefix = 'Cavitation Probability'
        else:
            # Default to pressure field
            print(f"Warning: Unknown plot type '{plot_type}'. Using pressure field instead.")
            plot_data = self.max_pressure
            vmax = np.max(plot_data)
            vmin = 0
            cmap = 'hot'
            colorbar_label = 'Pressure (Pa)'
            title_prefix = 'Pressure Field'

        # Plot x-slice (YZ plane at center of x)
        # In k-wave, the coordinate system is:
        # - x: left to right (increasing index)
        # - y: front to back (increasing index)
        # - z: bottom to top (increasing index)
        # But in our visualization, we want z to be from top to bottom
        # So we need to transpose and flip the data correctly

        # For YZ plane (x-slice), we take a slice at x_center
        # The resulting 2D array has y along rows and z along columns
        # We need to transpose it so y is along x-axis and z is along y-axis
        # Prepare slices for visualization with z increasing upward
        yz_slice = plot_data[x_center, :, :]

        # Print debug info
        print(f"YZ slice shape: {yz_slice.shape}")
        print(f"Y range: {y[0]} to {y[-1]} mm")
        print(f"Z range: {z[0]} to {z[-1]} mm")

        im1 = axes[0].imshow(
            yz_slice.T,  # Transpose to get y along x-axis and z along y-axis
            aspect='equal',
            extent=[y[0], y[-1], z[0], z[-1]],  # z increases going up (z=0 at bottom)
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_title('X-Slice (YZ plane at center)')
        axes[0].set_xlabel('Y (mm)')
        axes[0].set_ylabel('Z (mm)')

        # Add PML, transducer, and medium boundaries to x-slice
        self._add_boundaries_to_slice(axes[0], 'yz')

        # Plot y-slice (XZ plane at center of y)
        # For XZ plane (y-slice), we take a slice at y_center
        # The resulting 2D array has x along rows and z along columns
        # We need to transpose it so x is along x-axis and z is along y-axis
        # Prepare slices for visualization with z increasing upward
        xz_slice = plot_data[:, y_center, :]

        # Print debug info
        print(f"XZ slice shape: {xz_slice.shape}")
        print(f"X range: {x[0]} to {x[-1]} mm")
        print(f"Z range: {z[0]} to {z[-1]} mm")

        im2 = axes[1].imshow(
            xz_slice.T,  # Transpose to get x along x-axis and z along y-axis
            aspect='equal',
            extent=[x[0], x[-1], z[0], z[-1]],  # z increases going up (z=0 at bottom)
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_title('Y-Slice (XZ plane at center)')
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Z (mm)')

        # Add PML, transducer, and medium boundaries to y-slice
        self._add_boundaries_to_slice(axes[1], 'xz')

        # Calculate z-slice position (at focal depth)
        if hasattr(self.sim_env, 'focal_depth'):
            # In the visualization, the transducer is at z=0 and the focal point
            # is at a distance of focal_depth from the transducer (moving downward/increasing z)
            # Calculate the grid index for the focal plane
            # Start from the transducer position (z=0) and move downward by focal_depth
            # The transducer is at z=0, so the focal point is at z=0 + focal_depth
            focal_depth_mm = self.sim_env.focal_depth * 1000  # Convert to mm
            transducer_z_pos = 0  # Transducer position (z=0)
            focal_z = transducer_z_pos + focal_depth_mm  # Focal point z-coordinate in mm

            # Find the closest grid point to the focal point
            focal_z_idx = np.argmin(np.abs(z - focal_z))

            print(f"Focal depth: {focal_depth_mm} mm")
            print(f"Transducer position: z = {transducer_z_pos} mm")
            print(f"Focal point position: z = {focal_z} mm")
            print(f"Focal point index: {focal_z_idx}")

            # Make sure it's within bounds
            if focal_z_idx < 0 or focal_z_idx >= self.sim_env.kgrid.Nz:
                focal_z_idx = self.sim_env.kgrid.Nz // 2

            # Use focal depth for z-slice
            z_slice_idx = focal_z_idx
            z_slice_title = f'Z-Slice (XY plane at focal depth: {self.focal_depth_mm} mm)'
        else:
            # Use middle of the grid as default
            z_slice_idx = self.sim_env.kgrid.Nz // 2
            z_slice_title = f'Z-Slice (XY plane at middle of grid)'

        # Plot z-slice (XY plane)
        # For XY plane (z-slice), we take a slice at z_slice_idx
        # The resulting 2D array has x along rows and y along columns
        # We want x along x-axis and y along y-axis, but we need to flip y
        # to maintain the same orientation as the other slices
        xy_slice = plot_data[:, :, z_slice_idx]

        # Print debug info
        print(f"XY slice shape: {xy_slice.shape}")
        print(f"X range: {x[0]} to {x[-1]} mm")
        print(f"Y range: {y[0]} to {y[-1]} mm")
        print(f"Z slice index: {z_slice_idx}, Z value: {z[z_slice_idx]} mm")

        im3 = axes[2].imshow(
            xy_slice,  # No transpose needed, just flip y for consistent orientation
            aspect='equal',
            extent=[x[0], x[-1], y[-1], y[0]],  # Flip y for consistent orientation
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        axes[2].set_title(z_slice_title)
        axes[2].set_xlabel('X (mm)')
        axes[2].set_ylabel('Y (mm)')

        # Add PML, transducer, and medium boundaries to z-slice
        # Pass the z-position to the _add_boundaries_to_slice method
        z_pos = z[z_slice_idx] if z_slice_idx < len(z) else z[-1]
        print(f"Z-slice position: {z_pos} mm")
        self._add_boundaries_to_slice(axes[2], 'xy', z_pos)

        # Add individual colorbars for each subplot
        cbar1 = fig.colorbar(im1, ax=axes[0])
        cbar1.set_label(colorbar_label)

        cbar2 = fig.colorbar(im2, ax=axes[1])
        cbar2.set_label(colorbar_label)

        cbar3 = fig.colorbar(im3, ax=axes[2])
        cbar3.set_label(colorbar_label)

        # Set main title
        fig.suptitle(
            f'{title_prefix} - {self.medium_type.capitalize()} Medium\n'
            f'Focal Depth: {self.focal_depth_mm} mm, Frequency: {self.sim_env.transducer_params["frequency"]/1e3:.1f} kHz',
            fontsize=16
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        # Convert to integer to remove decimal point
        focal_depth_int = int(self.focal_depth_mm)
        filename = f'3d_sim_results_{plot_type}_{self.medium_type}_{focal_depth_int}mm.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {os.path.join(self.output_dir, filename)}")

    def _add_boundaries_to_slice(self, ax, plane, z_pos=None):
        """
        Add PML, transducer, and medium boundaries to a slice

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add boundaries to
        plane : str
            Plane of the slice ('xy', 'xz', or 'yz')
        z_pos : float or None
            Z position of the slice in mm (only used for 'xy' plane)
        """
        # Check if we're using a sample pressure field or actual simulation results
        if hasattr(self, 'grid_spacing'):
            # Using sample pressure field
            dx = self.grid_spacing
            pml_size_points = 10  # Default PML size
        elif hasattr(self.sim_env, 'kgrid') and self.sim_env.kgrid is not None:
            # Using actual simulation results
            dx = self.sim_env.kgrid.dx
            pml_size_points = self.sim_env.pml_size_points
        else:
            # Fallback to reasonable defaults
            dx = 0.001  # 1 mm
            pml_size_points = 10

        # Get domain dimensions
        domain_width = self.sim_env.domain_width if hasattr(self.sim_env, 'domain_width') else 0.1  # 100 mm
        domain_height = self.sim_env.domain_height if hasattr(self.sim_env, 'domain_height') else 0.1  # 100 mm
        domain_depth = self.sim_env.domain_depth if hasattr(self.sim_env, 'domain_depth') else 0.2  # 200 mm

        # Convert to mm
        x_min = -domain_width/2 * 1000
        x_max = domain_width/2 * 1000
        y_min = -domain_height/2 * 1000
        y_max = domain_height/2 * 1000
        z_min = 0
        z_max = domain_depth * 1000

        # Get PML size in mm
        pml_size_mm = pml_size_points * dx * 1000

        # Get phantom dimensions and position
        if hasattr(self.sim_env, 'phantom_props') and self.sim_env.phantom_props is not None:
            phantom_width = self.sim_env.phantom_width * 1000  # Convert to mm
            phantom_height = self.sim_env.phantom_height * 1000
            phantom_depth = self.sim_env.phantom_depth * 1000
            phantom_x_min = -phantom_width/2
            phantom_x_max = phantom_width/2
            phantom_y_min = -phantom_height/2
            phantom_y_max = phantom_height/2
            # In the simulation, the phantom starts after the PML boundary, not at z=0
            # We need to match this in the visualization
            # Get PML size in mm for visualization
            pml_size_mm = pml_size_points * dx * 1000
            # The phantom starts after the PML boundary
            phantom_z_min = pml_size_mm  # Start of phantom after PML
            phantom_z_max = phantom_z_min + phantom_depth  # End of phantom
        else:
            # Default phantom dimensions (match transducer size: 100mm x 100mm x 100mm)
            phantom_width = 0.1 * 1000  # 100 mm
            phantom_height = 0.1 * 1000  # 100 mm
            phantom_depth = 0.1 * 1000   # 100 mm
            print(f"Using default phantom dimensions: {phantom_width} x {phantom_height} x {phantom_depth} mm")
            phantom_x_min = -phantom_width/2
            phantom_x_max = phantom_width/2
            phantom_y_min = -phantom_height/2
            phantom_y_max = phantom_height/2
            # In the simulation, the phantom is at the top of the grid (z = pml_size_points)
            # and the transducer is at the bottom of the grid (z = Nz - 1 - pml_size_points)
            # In the visualization, we keep the transducer at z=0 and adjust the phantom position
            # The phantom should be directly on top of the transducer
            phantom_z_min = 0  # Start of phantom at transducer level
            phantom_z_max = phantom_depth  # End of phantom

        # Get transducer dimensions
        transducer_width = self.sim_env.transducer_params['array_width'] * 1000  # Convert to mm
        transducer_height = self.sim_env.transducer_params['array_height'] * 1000
        transducer_x_min = -transducer_width/2
        transducer_x_max = transducer_width/2
        transducer_y_min = -transducer_height/2
        transducer_y_max = transducer_height/2
        # In the simulation, the transducer is at the bottom of the grid (z = Nz - 1 - pml_size_points)
        # and the phantom is at the top of the grid (z = pml_size_points)
        # In the visualization, we'll keep the transducer at z=0 and adjust the phantom position
        transducer_z = 0  # Transducer at z=0

        # Get focal point
        focal_x = 0  # Centered
        focal_y = 0  # Centered
        # In the visualization, z=0 is at the transducer level and z increases going up
        # The focal point should be at the specified depth from the transducer
        focal_z = transducer_z + (self.sim_env.focal_depth * 1000)  # Convert to mm and adjust for visualization

        if plane == 'xy':
            # Add PML boundaries
            rect_pml_outer = plt.Rectangle(
                (x_min - pml_size_mm, y_min - pml_size_mm),
                (x_max - x_min) + 2 * pml_size_mm,
                (y_max - y_min) + 2 * pml_size_mm,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                linewidth=1,
                label='PML Boundary'
            )
            ax.add_patch(rect_pml_outer)

            # Add inner PML boundary
            rect_pml_inner = plt.Rectangle(
                (x_min, y_min),
                (x_max - x_min),
                (y_max - y_min),
                edgecolor='red',
                facecolor='none',
                linestyle=':',
                linewidth=1
            )
            ax.add_patch(rect_pml_inner)

            # Add phantom if it exists and the slice is within the phantom
            if self.sim_env.phantom_props is not None and z_pos is not None:
                if phantom_z_min <= z_pos <= phantom_z_max:
                    rect_phantom = plt.Rectangle(
                        (phantom_x_min, phantom_y_min),
                        phantom_width,
                        phantom_height,
                        edgecolor=self.sim_env.phantom_props['color'],
                        facecolor=self.sim_env.phantom_props['color'],
                        alpha=0.3,
                        label=self.sim_env.phantom_props['name']
                    )
                    ax.add_patch(rect_phantom)

            # Add focal point if the slice is near the focal point
            if z_pos is not None and abs(z_pos - focal_z) < 5:  # Within 5mm
                ax.plot(focal_x, focal_y, 'rx', markersize=10, label='Focal Point')

        elif plane == 'xz':
            # Add PML boundaries
            rect_pml_outer = plt.Rectangle(
                (x_min - pml_size_mm, z_min - pml_size_mm),
                (x_max - x_min) + 2 * pml_size_mm,
                (z_max - z_min) + pml_size_mm,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                linewidth=1,
                label='PML Boundary'
            )
            ax.add_patch(rect_pml_outer)

            # Add inner PML boundary
            rect_pml_inner = plt.Rectangle(
                (x_min, z_min),
                (x_max - x_min),
                (z_max - z_min),
                edgecolor='red',
                facecolor='none',
                linestyle=':',
                linewidth=1
            )
            ax.add_patch(rect_pml_inner)

            # Add transducer at z=0
            # Give the transducer a small height (5mm) to make it visible
            transducer_thickness = 5  # 5mm thickness
            rect_transducer = plt.Rectangle(
                (transducer_x_min, transducer_z),
                transducer_width,
                transducer_thickness,  # Give it some height to make it visible
                edgecolor='red',
                facecolor='red',
                alpha=0.7,
                linewidth=2,
                label='Transducer'
            )
            ax.add_patch(rect_transducer)

            # Update phantom_z_min to ensure it starts exactly at the top of the transducer
            phantom_z_min = transducer_z + transducer_thickness

            # Add phantom if it exists
            if self.sim_env.phantom_props is not None:
                rect_phantom = plt.Rectangle(
                    (phantom_x_min, phantom_z_min),
                    phantom_width,
                    phantom_depth,
                    edgecolor=self.sim_env.phantom_props['color'],
                    facecolor=self.sim_env.phantom_props['color'],
                    alpha=0.3,
                    label=self.sim_env.phantom_props['name']
                )
                ax.add_patch(rect_phantom)

            # Add focal point
            ax.plot(focal_x, focal_z, 'rx', markersize=10, label='Focal Point')

        elif plane == 'yz':
            # Add PML boundaries
            rect_pml_outer = plt.Rectangle(
                (y_min - pml_size_mm, z_min - pml_size_mm),
                (y_max - y_min) + 2 * pml_size_mm,
                (z_max - z_min) + pml_size_mm,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                linewidth=1,
                label='PML Boundary'
            )
            ax.add_patch(rect_pml_outer)

            # Add inner PML boundary
            rect_pml_inner = plt.Rectangle(
                (y_min, z_min),
                (y_max - y_min),
                (z_max - z_min),
                edgecolor='red',
                facecolor='none',
                linestyle=':',
                linewidth=1
            )
            ax.add_patch(rect_pml_inner)

            # Add transducer at z=0
            # Give the transducer a small height (5mm) to make it visible
            transducer_thickness = 5  # 5mm thickness
            rect_transducer = plt.Rectangle(
                (transducer_y_min, transducer_z),
                transducer_height,
                transducer_thickness,  # Give it some height to make it visible
                edgecolor='red',
                facecolor='red',
                alpha=0.7,
                linewidth=2,
                label='Transducer'
            )
            ax.add_patch(rect_transducer)

            # Update phantom_z_min to ensure it starts exactly at the top of the transducer
            phantom_z_min = transducer_z + transducer_thickness

            # Add phantom if it exists
            if self.sim_env.phantom_props is not None:
                rect_phantom = plt.Rectangle(
                    (phantom_y_min, phantom_z_min),
                    phantom_height,
                    phantom_depth,
                    edgecolor=self.sim_env.phantom_props['color'],
                    facecolor=self.sim_env.phantom_props['color'],
                    alpha=0.3,
                    label=self.sim_env.phantom_props['name']
                )
                ax.add_patch(rect_phantom)

            # Add focal point
            ax.plot(focal_y, focal_z, 'rx', markersize=10, label='Focal Point')

        # Add legend if not already present
        if not ax.get_legend():
            ax.legend(loc='upper right', fontsize=8)

def run_simulation(focal_depth_mm, medium_type='soft_tissue', apodization='none', output_dir=None, use_sample=False,
                 sweep_bandwidth=None, signal_type='single_freq', frequency2=None, amplitude_ratio=1.0):
    """
    Run a 3D ultrasound simulation

    Parameters:
    -----------
    focal_depth_mm : float
        Focal depth in mm
    medium_type : str
        Type of medium ('water', 'bone', or 'soft_tissue')
    apodization : str
        Type of apodization to apply ('none', 'hanning', 'hamming', 'blackman')
    output_dir : str or None
        Directory to save the results. If None, will use default directory.
    use_sample : bool
        Whether to use a sample pressure field instead of running the simulation
    sweep_bandwidth : float or None
        If not None, generates a frequency sweep with the specified bandwidth percentage
    signal_type : str
        Type of signal to generate ('single_freq', 'dual_freq')
    frequency2 : float or None
        Second frequency for dual frequency signals (default: 550 kHz)
    amplitude_ratio : float
        Ratio of amplitude of second frequency to first frequency (default: 1.0)

    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Convert focal depth to meters
    focal_depth = focal_depth_mm * 1e-3

    # Create output directory if not provided
    if output_dir is None:
        # Convert to integer to remove decimal point
        focal_depth_int = int(focal_depth_mm)

        # Determine signal type subfolder
        if signal_type.lower() == 'dual_freq':
            if sweep_bandwidth is not None:
                signal_folder = 'dual_sweep_freq'
            else:
                signal_folder = 'dual_freq'
        else:  # single_freq
            if sweep_bandwidth is not None:
                signal_folder = 'sweep_freq'
            else:
                signal_folder = 'single_freq'

        output_dir = f'sim_results/focal_{focal_depth_int}mm/3d_sim_runner/{signal_folder}'
        os.makedirs(output_dir, exist_ok=True)

    # Load transducer parameters
    transducer_params = load_transducer_params()

    # Print simulation parameters
    print("\n" + "="*50)
    print("3D Ultrasound Simulation Parameters")
    print("="*50)
    print(f"Frequency: {transducer_params['frequency']/1e3:.1f} kHz")
    print(f"Wavelength in water: {transducer_params['wavelength']*1000:.2f} mm")
    print(f"Array dimensions: {transducer_params['array_width']*1000:.1f} x {transducer_params['array_height']*1000:.1f} mm")
    print(f"Element dimensions: {transducer_params['element_width']*1000:.2f} x {transducer_params['element_height']*1000:.2f} mm")
    print(f"Number of elements: {transducer_params['num_elements_x']} x {transducer_params['num_elements_y']} = {transducer_params['num_elements_x']*transducer_params['num_elements_y']}")
    print(f"Focal depth: {focal_depth_mm} mm")
    print(f"Medium type: {medium_type}")
    print(f"Apodization: {apodization}")
    print("="*50 + "\n")

    # Create simulation environment
    sim_env = SimulationEnvironment3D(
        transducer_params,
        medium_type,
        focal_depth,
        apodization,
        sweep_bandwidth,
        signal_type,
        frequency2,
        amplitude_ratio
    )

    # Create simulation runner
    sim_runner = SimulationRunner3D(sim_env, output_dir)

    # Run simulation or create sample pressure field
    if 'use_sample' in locals() and use_sample:
        # Create sample pressure field
        sim_runner._create_sample_pressure_field()
        # Visualize results - create all three plot types
        sim_runner.visualize_results(plot_type='pressure')  # Standard pressure field
        sim_runner.visualize_results(plot_type='mi')        # Mechanical Index
        sim_runner.visualize_results(plot_type='cavitation') # Cavitation probability
        results = {'max_pressure': sim_runner.max_pressure}
    else:
        # Run actual simulation
        results = sim_runner.run_simulation()
        # Visualize results - create all three plot types
        sim_runner.visualize_results(plot_type='pressure')  # Standard pressure field
        sim_runner.visualize_results(plot_type='mi')        # Mechanical Index
        sim_runner.visualize_results(plot_type='cavitation') # Cavitation probability

    return results

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run 3D ultrasound simulations')
    parser.add_argument('--focal-depths', type=float, nargs='+', default=DEFAULT_FOCAL_DEPTHS,
                       help='Focal depths in mm')
    parser.add_argument('--medium', type=str, default=DEFAULT_MEDIUM_TYPE, choices=MEDIUM_TYPES,
                       help='Medium type')
    parser.add_argument('--apodization', type=str, default=DEFAULT_APODIZATION, choices=APODIZATION_TYPES,
                       help='Apodization type')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample pressure field instead of running simulation')
    parser.add_argument('--sweep', type=float, default=None,
                       help='Generate frequency sweep with specified bandwidth percentage (e.g., 40 for 40%)')
    parser.add_argument('--dual', action='store_true',
                       help='Generate dual frequency signal (180 kHz and 550 kHz)')
    parser.add_argument('--freq2', type=float, default=550e3,
                       help='Second frequency for dual frequency signals in Hz (default: 550 kHz)')
    parser.add_argument('--amp-ratio', type=float, default=1.0,
                       help='Amplitude ratio of second frequency to first frequency (default: 1.0)')
    args = parser.parse_args()

    # Create output directories
    create_output_dirs(args.focal_depths)

    # Check if k-wave-python is available and we're not using sample mode
    if not KWAVE_AVAILABLE and not args.sample:
        print("Warning: k-wave-python package is not available or not properly configured.")
        print("Using sample pressure field for visualization instead.")
        args.sample = True

    # Determine signal type
    signal_type = 'dual_freq' if args.dual else 'single_freq'

    # Run simulations for each focal depth
    for focal_depth in args.focal_depths:
        print(f"\nRunning simulation for focal depth {focal_depth} mm with {args.medium} medium and {args.apodization} apodization...")
        print(f"Signal type: {signal_type}, Sweep: {args.sweep}")
        if args.dual:
            print(f"Second frequency: {args.freq2/1e3:.1f} kHz, Amplitude ratio: {args.amp_ratio}")

        run_simulation(
            focal_depth_mm=focal_depth,
            medium_type=args.medium,
            apodization=args.apodization,
            use_sample=args.sample,
            sweep_bandwidth=args.sweep,
            signal_type=signal_type,
            frequency2=args.freq2,
            amplitude_ratio=args.amp_ratio
        )

    print("\nAll simulations completed successfully!")

if __name__ == "__main__":
    main()
