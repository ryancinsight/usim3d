"""
3D Simulation Environment for 2D Transducer Array

This script creates and visualizes a 3D simulation environment for the 2D transducer array
defined in transducer_2d_array.py. It sets up a 3D medium (water, bone, or soft tissue)
sitting directly on top of the transducer, with proper PML boundaries.

The medium dimensions are 100mm x 100mm x 100mm, with the x and y edges aligned with
the transducer surface.

Features:
1. 3D visualization of the simulation environment
2. Configurable medium properties (water, bone, soft tissue)
3. Proper PML boundary implementation
4. Visualization of the focal point and beam path
5. Organized output structure in sim_results folders
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import sys

# Import grid optimizer
try:
    from grid_optimizer import optimize_grid_dimensions, get_highest_prime_factors
    GRID_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("Warning: grid_optimizer.py not found. Grid optimization will be disabled.")
    GRID_OPTIMIZER_AVAILABLE = False

# Try to import from k-wave-python package
try:
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.options.simulation_options import SimulationOptions
    KWAVE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: k-wave-python package not found: {e}")
    print("Simulation environment will be visualized without k-wave functionality.")
    KWAVE_AVAILABLE = False

# Import parameters from transducer_2d_array.py
# We'll use importlib to avoid modifying the original file
import importlib.util
import importlib.machinery

def load_transducer_params():
    """Load parameters from transducer_2d_array.py without executing the whole script"""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("transducer_2d_array", "transducer_2d_array.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract the parameters we need
        params = {
            'frequency': module.frequency,
            'sound_speed_water': module.sound_speed_water,
            'wavelength': module.wavelength,
            'array_width': module.actual_array_width,
            'array_height': module.actual_array_height,
            'element_width': module.element_width,
            'element_height': module.element_height,
            'kerf': module.kerf,
            'num_elements_x': module.num_elements_x,
            'num_elements_y': module.num_elements_y,
            'element_positions': module.element_positions
        }
        return params
    except Exception as e:
        print(f"Error loading transducer parameters: {e}")
        # Default parameters if loading fails
        return {
            'frequency': 180e3,  # 180 kHz
            'sound_speed_water': 1482.0,  # m/s
            'wavelength': 1482.0 / 180e3,  # m
            'array_width': 100e-3,  # 100 mm in meters
            'array_height': 100e-3,  # 100 mm in meters
            'element_width': (1482.0 / 180e3) / 2,  # λ/2
            'element_height': (1482.0 / 180e3) / 2,  # λ/2
            'kerf': (1482.0 / 180e3) / 20,  # λ/20
            'num_elements_x': 22,
            'num_elements_y': 22,
            'element_positions': None  # Will be generated if None
        }

# Define medium properties
class MediumProperties:
    """Class to store medium properties for different materials"""

    @staticmethod
    def get_water():
        """Get properties for water"""
        return {
            'name': 'Water',
            'sound_speed': 1482.0,  # m/s
            'density': 1000.0,      # kg/m^3
            'alpha_coeff': 0.002,   # dB/(MHz^y cm)
            'alpha_power': 2.0,     # y in alpha_coeff
            'BonA': 6.0,            # Parameter of nonlinearity
            'color': 'lightblue',
            'alpha': 0.3            # Transparency for visualization
        }

    @staticmethod
    def get_bone():
        """Get properties for bone"""
        return {
            'name': 'Bone',
            'sound_speed': 3476.0,  # m/s
            'density': 1908.0,      # kg/m^3
            'alpha_coeff': 5.0,     # dB/(MHz^y cm)
            'alpha_power': 1.1,     # y in alpha_coeff
            'BonA': 6.0,            # Parameter of nonlinearity
            'color': 'beige',
            'alpha': 0.7            # Transparency for visualization
        }

    @staticmethod
    def get_soft_tissue():
        """Get properties for soft tissue"""
        return {
            'name': 'Soft Tissue',
            'sound_speed': 1540.0,  # m/s
            'density': 1045.0,      # kg/m^3
            'alpha_coeff': 0.5,     # dB/(MHz^y cm)
            'alpha_power': 1.1,     # y in alpha_coeff
            'BonA': 7.0,            # Parameter of nonlinearity
            'color': 'pink',
            'alpha': 0.5            # Transparency for visualization
        }

class ElementDirectivity:
    """Class to calculate and visualize element directivity patterns"""

    def __init__(self, element_width, element_height, wavelength, num_points=100):
        """
        Initialize the element directivity calculator

        Parameters:
        -----------
        element_width : float
            Width of the element in meters
        element_height : float
            Height of the element in meters
        wavelength : float
            Wavelength in the medium in meters
        num_points : int
            Number of points to calculate for directivity pattern
        """
        self.element_width = element_width
        self.element_height = element_height
        self.wavelength = wavelength
        self.num_points = num_points

        # Calculate normalized element dimensions
        self.width_lambda = self.element_width / self.wavelength
        self.height_lambda = self.element_height / self.wavelength

        # Calculate directivity patterns
        self._calculate_directivity()

    def _calculate_directivity(self):
        """
        Calculate directivity patterns in the x-z and y-z planes

        The directivity function for a rectangular element is:
        D(θ) = sinc(k*a*sin(θ)/2)
        where k = 2π/λ, a is the element dimension, and θ is the angle from normal
        """
        # Calculate angles from -90 to 90 degrees
        self.angles_deg = np.linspace(-90, 90, self.num_points)
        self.angles_rad = np.radians(self.angles_deg)

        # Calculate directivity in x-z plane (affected by element width)
        arg_x = np.pi * self.width_lambda * np.sin(self.angles_rad)
        self.directivity_x = np.sinc(arg_x / np.pi)  # numpy's sinc includes the pi division

        # Calculate directivity in y-z plane (affected by element height)
        arg_y = np.pi * self.height_lambda * np.sin(self.angles_rad)
        self.directivity_y = np.sinc(arg_y / np.pi)

        # Calculate 2D directivity (combined effect)
        self.directivity_2d = np.zeros((self.num_points, self.num_points))
        for i, theta_x in enumerate(self.angles_rad):
            for j, theta_y in enumerate(self.angles_rad):
                # Convert to 3D direction
                sin_theta_x = np.sin(theta_x)
                sin_theta_y = np.sin(theta_y)

                # Ensure the direction is normalized
                norm = np.sqrt(sin_theta_x**2 + sin_theta_y**2 +
                              (1 - sin_theta_x**2 - sin_theta_y**2))

                if norm > 1:  # Invalid direction (outside unit sphere)
                    self.directivity_2d[i, j] = 0
                else:
                    # Calculate directivity
                    arg_x = np.pi * self.width_lambda * sin_theta_x
                    arg_y = np.pi * self.height_lambda * sin_theta_y
                    self.directivity_2d[i, j] = np.sinc(arg_x / np.pi) * np.sinc(arg_y / np.pi)

    def get_directivity(self, theta_x, theta_y):
        """
        Get directivity value for specific angles

        Parameters:
        -----------
        theta_x : float or array
            Angle in x-z plane in radians
        theta_y : float or array
            Angle in y-z plane in radians

        Returns:
        --------
        directivity : float or array
            Directivity value(s) between 0 and 1
        """
        # Calculate directivity
        arg_x = np.pi * self.width_lambda * np.sin(theta_x)
        arg_y = np.pi * self.height_lambda * np.sin(theta_y)
        return np.sinc(arg_x / np.pi) * np.sinc(arg_y / np.pi)

    def plot_directivity(self, ax=None, plane='both'):
        """
        Plot directivity patterns

        Parameters:
        -----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        plane : str
            Which plane to plot: 'x', 'y', or 'both'
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})

        if plane in ['x', 'both']:
            # Plot x-z plane directivity
            ax.plot(self.angles_rad, self.directivity_x**2, 'b-', linewidth=2,
                   label=f'X-Z Plane (Width = {self.width_lambda:.2f}λ)')

        if plane in ['y', 'both']:
            # Plot y-z plane directivity
            ax.plot(self.angles_rad, self.directivity_y**2, 'r--', linewidth=2,
                   label=f'Y-Z Plane (Height = {self.height_lambda:.2f}λ)')

        # Set plot properties
        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)  # clockwise
        ax.set_rlim(0, 1.1)
        ax.set_title('Element Directivity Pattern (Power)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True)

        return ax

    def plot_directivity_2d(self, ax=None):
        """
        Plot 2D directivity pattern

        Parameters:
        -----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Plot 2D directivity as a heatmap
        im = ax.imshow(self.directivity_2d**2, origin='lower', extent=[-90, 90, -90, 90],
                      cmap='hot', vmin=0, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Directivity (Power)')

        # Set plot properties
        ax.set_xlabel('Angle θx (degrees)', fontsize=10)
        ax.set_ylabel('Angle θy (degrees)', fontsize=10)
        ax.set_title('2D Element Directivity Pattern (Power)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        return ax


class SimulationEnvironment3D:
    """Class to set up and visualize a 3D simulation environment for the 2D transducer array"""

    def __init__(self, transducer_params, medium_type='water', focal_depth=None, apodization='none',
                 sweep_bandwidth=None, signal_type='single_freq', frequency2=None, amplitude_ratio=1.0):
        """
        Initialize the simulation environment

        Parameters:
        -----------
        transducer_params : dict
            Dictionary containing transducer parameters
        medium_type : str
            Type of medium ('water', 'bone', or 'soft_tissue')
        focal_depth : float or None
            Focal depth in meters. If None, will use 2x the medium depth.
        apodization : str
            Type of apodization to apply ('none', 'hanning', 'hamming', 'blackman')
        sweep_bandwidth : float or None
            If not None, generates a frequency sweep with the specified bandwidth percentage
        signal_type : str
            Type of signal to generate ('single_freq', 'dual_freq')
        frequency2 : float or None
            Second frequency for dual frequency signals (default: 550 kHz)
        amplitude_ratio : float
            Ratio of amplitude of second frequency to first frequency (default: 1.0)
        """
        self.transducer_params = transducer_params
        self.medium_type = medium_type.lower()
        self.apodization = apodization.lower()
        self.sweep_bandwidth = sweep_bandwidth
        self.signal_type = signal_type.lower()
        self.frequency2 = frequency2 if frequency2 is not None else 550e3  # Default to 550 kHz
        self.amplitude_ratio = amplitude_ratio

        # Set water properties (always needed as the base medium)
        self.water_props = MediumProperties.get_water()

        # Set phantom properties based on type
        if self.medium_type == 'bone':
            self.phantom_props = MediumProperties.get_bone()
        elif self.medium_type == 'soft_tissue':
            self.phantom_props = MediumProperties.get_soft_tissue()
        else:
            # If water is selected, there's no separate phantom
            self.phantom_props = None

        # Set dimensions for the entire simulation domain
        # Reducing domain size to speed up simulation
        # Make it just large enough to contain the transducer and beam
        self.domain_width = max(100e-3, transducer_params['array_width'] * 1.2)  # Reduced from 150mm to 100mm
        self.domain_height = max(100e-3, transducer_params['array_height'] * 1.2)  # Reduced from 150mm to 100mm

        # Set focal depth
        if focal_depth is None:
            self.focal_depth = 100e-3  # Default 100 mm
        else:
            self.focal_depth = focal_depth

        # Set domain depth to be just enough to contain the focal point
        # Reducing domain depth to speed up simulation
        self.domain_depth = max(120e-3, self.focal_depth * 1.5)  # Reduced from 200mm to 120mm and from 2.2x to 1.5x focal depth

        # Set phantom dimensions (placed on top of transducer)
        # Using 100mm x 100mm x 100mm phantom dimensions to match transducer size
        self.phantom_width = 100e-3   # 100mm width
        self.phantom_height = 100e-3  # 100mm height
        self.phantom_depth = 100e-3   # 100mm depth
        self.phantom_position = [0, 0, 0]  # Centered on top of transducer

        # Calculate wavelength in water (for grid resolution)
        self.wavelength_water = self.water_props['sound_speed'] / transducer_params['frequency']

        # Create element directivity calculator
        self.element_directivity = ElementDirectivity(
            element_width=transducer_params['element_width'],
            element_height=transducer_params['element_height'],
            wavelength=self.wavelength_water
        )

        # Calculate apodization weights
        self._calculate_apodization()

        # PML parameters
        # Reducing PML size to speed up simulation
        self.pml_size_points = 8  # Reduced from 10 to 8 grid points for PML
        self.pml_alpha = 2.0  # PML absorption coefficient

        # Grid parameters
        # Reducing points per wavelength to speed up simulation (less accurate but faster)
        # Standard is 6-10 points per wavelength, but we can use 3-4 for faster simulations
        self.points_per_wavelength = 3  # Reduced from 6 to 3 to speed up simulation

        # Initialize grid
        self._setup_grid()

    def _calculate_apodization(self):
        """
        Calculate apodization weights for the transducer elements

        This method creates apodization weights based on the selected window function.
        The weights are normalized to maintain the same total acoustic power.
        """
        # Get number of elements in each dimension
        num_elements_x = self.transducer_params['num_elements_x']
        num_elements_y = self.transducer_params['num_elements_y']

        # Create coordinate arrays for elements
        x = np.linspace(-1, 1, num_elements_x)
        y = np.linspace(-1, 1, num_elements_y)
        X, Y = np.meshgrid(x, y)

        # Calculate radial distance from center (normalized)
        R = np.sqrt(X**2 + Y**2)

        # Apply selected apodization window
        if self.apodization == 'hanning':
            # Hanning window: 0.5 * (1 - cos(2πr))
            weights = 0.5 * (1 - np.cos(np.pi * (1 - R)))
            weights[R > 1] = 0  # Zero outside unit circle
        elif self.apodization == 'hamming':
            # Hamming window: 0.54 - 0.46 * cos(2πr)
            weights = 0.54 - 0.46 * np.cos(np.pi * (1 - R))
            weights[R > 1] = 0  # Zero outside unit circle
        elif self.apodization == 'blackman':
            # Blackman window: 0.42 - 0.5 * cos(2πr) + 0.08 * cos(4πr)
            weights = 0.42 - 0.5 * np.cos(np.pi * (1 - R)) + 0.08 * np.cos(2 * np.pi * (1 - R))
            weights[R > 1] = 0  # Zero outside unit circle
        else:  # 'none' or any other value
            # Uniform weighting (no apodization)
            weights = np.ones((num_elements_y, num_elements_x))

        # Normalize weights to maintain total power
        if np.sum(weights) > 0:
            weights = weights * (num_elements_x * num_elements_y) / np.sum(weights)

        # Store the weights
        self.apodization_weights = weights

        # Calculate effective aperture reduction due to apodization
        # This is the ratio of the sum of squares of weights to the square of the sum
        # It represents how much the aperture is effectively reduced
        if np.sum(weights) > 0:
            self.aperture_efficiency = np.sum(weights**2) / (np.sum(weights)**2) * num_elements_x * num_elements_y
        else:
            self.aperture_efficiency = 1.0

    def _setup_grid(self):
        """Set up the computational grid for the simulation"""
        if not KWAVE_AVAILABLE:
            print("k-wave-python not available, skipping grid setup")
            return

        # Calculate grid spacing based on wavelength in water
        dx = self.wavelength_water / self.points_per_wavelength

        # Calculate physical PML size in meters (typically 2 wavelengths)
        pml_size_meters = 2 * self.wavelength_water

        # Calculate grid dimensions including PML
        Nx = int(np.ceil(self.domain_width / dx)) + 2 * self.pml_size_points
        Ny = int(np.ceil(self.domain_height / dx)) + 2 * self.pml_size_points
        Nz = int(np.ceil(self.domain_depth / dx)) + 2 * self.pml_size_points

        # Optimize grid dimensions for FFT efficiency if grid optimizer is available
        if GRID_OPTIMIZER_AVAILABLE:
            print(f"Original grid dimensions: {Nx} x {Ny} x {Nz}")
            highest_primes = get_highest_prime_factors(Nx, Ny, Nz)
            print(f"Highest prime factors: {highest_primes}")

            # Optimize grid dimensions (allow up to 10% increase in each dimension)
            # Pass the PML size to account for k-wave's internal PML handling
            Nx, Ny, Nz = optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=10, pml_size=self.pml_size_points)

            print(f"Optimized grid dimensions: {Nx} x {Ny} x {Nz}")
            print(f"Highest prime factors: {get_highest_prime_factors(Nx, Ny, Nz)}")

        # Store PML size in meters for visualization
        self.pml_size_meters = pml_size_meters

        # Create k-wave grid
        self.kgrid = kWaveGrid(Vector([Nx, Ny, Nz]), Vector([dx, dx, dx]))

        # Set time step based on CFL condition
        # Using a CFL value appropriate for the medium type
        # For bone, we need a smaller CFL for stability due to higher sound speed
        if self.phantom_props and self.phantom_props['name'].lower() == 'bone':
            # Use a smaller CFL for bone to ensure stability
            cfl = 0.2  # Smaller CFL for bone (more stable but slower)
            print(f"Using smaller CFL number ({cfl}) for bone simulation to ensure stability")
        else:
            # Standard CFL is 0.3, but we can use up to 0.5 for faster simulations with soft tissue
            cfl = 0.5  # Increased from 0.3 to speed up simulation for soft tissue

        # Use the maximum sound speed for CFL calculation to ensure stability
        max_sound_speed = max(self.water_props['sound_speed'],
                             self.phantom_props['sound_speed'] if self.phantom_props else 0)
        dt = cfl * dx / max_sound_speed

        # Calculate how many time steps needed to propagate to focal point and back
        # Reducing the time factor from 2.5 to 1.5 to reduce simulation time
        # This is enough for the wave to reach the focal point and a bit beyond
        t_end = 1.5 * self.domain_depth / self.water_props['sound_speed']  # Reduced from 2.5 to 1.5
        Nt = int(np.ceil(t_end / dt))

        # Set time step
        self.kgrid.setTime(Nt, dt)

        # Create heterogeneous medium with water as base and phantom material
        # First, create arrays for each property
        sound_speed = np.ones((self.kgrid.Nx, self.kgrid.Ny, self.kgrid.Nz)) * self.water_props['sound_speed']
        density = np.ones((self.kgrid.Nx, self.kgrid.Ny, self.kgrid.Nz)) * self.water_props['density']
        alpha_coeff = np.ones((self.kgrid.Nx, self.kgrid.Ny, self.kgrid.Nz)) * self.water_props['alpha_coeff']
        # alpha_power must be scalar according to k-wave
        alpha_power = self.water_props['alpha_power']
        BonA = np.ones((self.kgrid.Nx, self.kgrid.Ny, self.kgrid.Nz)) * self.water_props['BonA']

        # If we have a phantom (not water), create a heterogeneous medium
        if self.phantom_props is not None:
            print(f"Creating heterogeneous medium with {self.phantom_props['name']} phantom")

            # Calculate phantom dimensions in grid points
            phantom_width_points = int(self.phantom_width / dx)
            phantom_height_points = int(self.phantom_height / dx)
            phantom_depth_points = int(self.phantom_depth / dx)

            # Calculate phantom position in grid points (centered in x and y)
            phantom_x_start = self.kgrid.Nx // 2 - phantom_width_points // 2
            phantom_y_start = self.kgrid.Ny // 2 - phantom_height_points // 2
            # Position the phantom directly on top of the transducer
            # The transducer is at z = Nz - 1 - pml_size_points
            # We'll position the phantom to start at the transducer position - phantom_depth_points
            phantom_z_start = self.kgrid.Nz - 1 - self.pml_size_points - phantom_depth_points

            # Make sure phantom fits within grid
            phantom_width_points = min(phantom_width_points, self.kgrid.Nx - 2 * self.pml_size_points)
            phantom_height_points = min(phantom_height_points, self.kgrid.Ny - 2 * self.pml_size_points)
            phantom_depth_points = min(phantom_depth_points, self.kgrid.Nz - 2 * self.pml_size_points)

            # Set phantom properties in the grid
            phantom_x_end = phantom_x_start + phantom_width_points
            phantom_y_end = phantom_y_start + phantom_height_points
            phantom_z_end = phantom_z_start + phantom_depth_points

            # For bone, create a gradual transition layer to reduce sharp impedance mismatch
            # This helps with numerical stability
            if self.phantom_props['name'].lower() == 'bone':
                # Create a transition layer of 3 grid points
                transition_width = 3
                print(f"Creating gradual transition layer of {transition_width} grid points for bone phantom")

                # Create transition masks for each side of the phantom
                # We'll create a linear gradient from water to bone properties
                for i in range(transition_width):
                    # Calculate transition factor (0 at water, 1 at bone)
                    factor = (i + 1) / (transition_width + 1)

                    # Calculate transition properties
                    trans_sound_speed = self.water_props['sound_speed'] + factor * (self.phantom_props['sound_speed'] - self.water_props['sound_speed'])
                    trans_density = self.water_props['density'] + factor * (self.phantom_props['density'] - self.water_props['density'])
                    trans_alpha_coeff = self.water_props['alpha_coeff'] + factor * (self.phantom_props['alpha_coeff'] - self.water_props['alpha_coeff'])

                    # Apply transition properties to the bottom layer (z direction)
                    if phantom_z_start + i < phantom_z_end:
                        sound_speed[phantom_x_start:phantom_x_end,
                                  phantom_y_start:phantom_y_end,
                                  phantom_z_start + i] = trans_sound_speed

                        density[phantom_x_start:phantom_x_end,
                              phantom_y_start:phantom_y_end,
                              phantom_z_start + i] = trans_density

                        alpha_coeff[phantom_x_start:phantom_x_end,
                                  phantom_y_start:phantom_y_end,
                                  phantom_z_start + i] = trans_alpha_coeff

                # Apply core phantom properties (after transition layer)
                sound_speed[phantom_x_start:phantom_x_end,
                          phantom_y_start:phantom_y_end,
                          phantom_z_start + transition_width:phantom_z_end] = self.phantom_props['sound_speed']

                density[phantom_x_start:phantom_x_end,
                      phantom_y_start:phantom_y_end,
                      phantom_z_start + transition_width:phantom_z_end] = self.phantom_props['density']

                alpha_coeff[phantom_x_start:phantom_x_end,
                          phantom_y_start:phantom_y_end,
                          phantom_z_start + transition_width:phantom_z_end] = self.phantom_props['alpha_coeff']
            else:
                # For soft tissue, apply properties directly without transition
                sound_speed[phantom_x_start:phantom_x_end,
                          phantom_y_start:phantom_y_end,
                          phantom_z_start:phantom_z_end] = self.phantom_props['sound_speed']

                density[phantom_x_start:phantom_x_end,
                      phantom_y_start:phantom_y_end,
                      phantom_z_start:phantom_z_end] = self.phantom_props['density']

                alpha_coeff[phantom_x_start:phantom_x_end,
                          phantom_y_start:phantom_y_end,
                          phantom_z_start:phantom_z_end] = self.phantom_props['alpha_coeff']

            # alpha_power must be scalar according to k-wave
            # We'll use a weighted average of water and phantom alpha_power
            # This is a compromise since k-wave doesn't support heterogeneous alpha_power
            phantom_volume = phantom_width_points * phantom_height_points * phantom_depth_points
            total_volume = self.kgrid.Nx * self.kgrid.Ny * self.kgrid.Nz
            phantom_fraction = phantom_volume / total_volume
            alpha_power = (1 - phantom_fraction) * self.water_props['alpha_power'] + phantom_fraction * self.phantom_props['alpha_power']
            print(f"Using weighted average alpha_power: {alpha_power:.3f} (water: {self.water_props['alpha_power']}, phantom: {self.phantom_props['alpha_power']})")

            BonA[phantom_x_start:phantom_x_end,
                phantom_y_start:phantom_y_end,
                phantom_z_start:phantom_z_end] = self.phantom_props['BonA']

            print(f"Phantom dimensions: {phantom_width_points} x {phantom_height_points} x {phantom_depth_points} grid points")
            print(f"Phantom position: ({phantom_x_start}, {phantom_y_start}, {phantom_z_start})")

            # Store phantom grid position for visualization
            self.phantom_grid_position = {
                'x_start': phantom_x_start,
                'y_start': phantom_y_start,
                'z_start': phantom_z_start,
                'x_end': phantom_x_end,
                'y_end': phantom_y_end,
                'z_end': phantom_z_end
            }

        # Create the medium with the heterogeneous properties
        self.medium = kWaveMedium(
            sound_speed=sound_speed,
            density=density,
            alpha_coeff=alpha_coeff,
            alpha_power=alpha_power,
            BonA=BonA
        )

        # Create simulation options with PML
        self.simulation_options = SimulationOptions(
            pml_size=Vector([self.pml_size_points, self.pml_size_points, self.pml_size_points]),
            pml_alpha=self.pml_alpha,
            save_to_disk=False,
            data_cast='single'
        )

    def visualize_directivity(self, output_dir=None):
        """
        Visualize the element directivity patterns

        Parameters:
        -----------
        output_dir : str or None
            Directory to save the visualization. If None, will show the plot.
        """
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 8))

        # Plot polar directivity pattern
        ax1 = plt.subplot(1, 2, 1, projection='polar')
        self.element_directivity.plot_directivity(ax=ax1)

        # Plot 2D directivity pattern
        ax2 = plt.subplot(1, 2, 2)
        self.element_directivity.plot_directivity_2d(ax=ax2)

        # Add element dimensions information
        element_width_mm = self.transducer_params['element_width'] * 1000
        element_height_mm = self.transducer_params['element_height'] * 1000
        wavelength_mm = self.wavelength_water * 1000

        info_text = (
            f"Element Width: {element_width_mm:.2f} mm ({self.element_directivity.width_lambda:.2f}λ)\n"
            f"Element Height: {element_height_mm:.2f} mm ({self.element_directivity.height_lambda:.2f}λ)\n"
            f"Wavelength: {wavelength_mm:.2f} mm\n"
            f"Frequency: {self.transducer_params['frequency']/1e3:.1f} kHz"
        )

        fig.suptitle("Element Directivity Patterns", fontsize=14, weight='bold')
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'element_directivity.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_apodization(self, output_dir=None):
        """
        Visualize the apodization weights

        Parameters:
        -----------
        output_dir : str or None
            Directory to save the visualization. If None, will show the plot.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot apodization weights as a heatmap
        im = ax.imshow(self.apodization_weights, cmap='viridis',
                      interpolation='nearest', origin='lower')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude Weight', fontsize=12)

        # Set labels and title
        ax.set_xlabel('Element X Index', fontsize=12)
        ax.set_ylabel('Element Y Index', fontsize=12)
        ax.set_title(f'Apodization Weights ({self.apodization.capitalize()})', fontsize=14)

        # Add text with apodization information
        info_text = (
            f"Apodization: {self.apodization.capitalize()}\n"
            f"Aperture Efficiency: {self.aperture_efficiency:.2f}\n"
            f"Effective Aperture Reduction: {1/np.sqrt(self.aperture_efficiency):.2f}x"
        )

        # Add text box with apodization info
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='bottom')

        plt.tight_layout()

        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'apodization_{self.apodization}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_environment(self, output_dir=None, filename='3d_simulation_environment.png'):
        """
        Visualize the 3D simulation environment

        Parameters:
        -----------
        output_dir : str or None
            Directory to save the visualization. If None, will show the plot.
        filename : str
            Filename to save the visualization.
        """
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Set aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])

        # Plot medium as a transparent box
        self._plot_medium(ax)

        # Plot transducer elements
        self._plot_transducer(ax)

        # Plot PML boundaries
        self._plot_pml_boundaries(ax)

        # Plot focal point and beam
        self._plot_focal_point_and_beam(ax)

        # Set labels and title
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        # Set title based on medium type
        if self.phantom_props is not None:
            ax.set_title(f'3D Simulation Environment with {self.phantom_props["name"]} Phantom\nFocal Depth: {self.focal_depth*1000:.1f} mm', fontsize=14)
        else:
            ax.set_title(f'3D Simulation Environment with Water Medium\nFocal Depth: {self.focal_depth*1000:.1f} mm', fontsize=14)

        # Set axis limits to include PML
        if not KWAVE_AVAILABLE:
            # If k-wave is not available, use estimated PML size
            pml_size_mm = self.wavelength_water * 2 * 1000  # 2 wavelengths in mm
        else:
            # Use the stored PML size
            pml_size_mm = self.pml_size_meters * 1000  # Convert to mm

        ax.set_xlim([-self.domain_width/2*1000 - pml_size_mm, self.domain_width/2*1000 + pml_size_mm])
        ax.set_ylim([-self.domain_height/2*1000 - pml_size_mm, self.domain_height/2*1000 + pml_size_mm])
        ax.set_zlim([-pml_size_mm, self.domain_depth*1000 + pml_size_mm])

        # Convert axis ticks to mm
        ax.set_xticks(np.linspace(-self.domain_width/2*1000, self.domain_width/2*1000, 5))
        ax.set_yticks(np.linspace(-self.domain_height/2*1000, self.domain_height/2*1000, 5))
        ax.set_zticks(np.linspace(0, self.domain_depth*1000, 5))

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        # Add text with simulation parameters
        if self.phantom_props is not None:
            # If we have a phantom, show its properties
            param_text = (
                f"Water Domain: {self.domain_width*1000:.1f} × {self.domain_height*1000:.1f} × {self.domain_depth*1000:.1f} mm\n"
                f"Phantom: {self.phantom_props['name']}\n"
                f"Phantom Sound Speed: {self.phantom_props['sound_speed']:.1f} m/s\n"
                f"Phantom Density: {self.phantom_props['density']:.1f} kg/m³\n"
                f"Frequency: {self.transducer_params['frequency']/1e3:.1f} kHz\n"
                f"Wavelength in Water: {self.wavelength_water*1000:.2f} mm\n"
                f"Focal Depth: {self.focal_depth*1000:.1f} mm\n"
                f"PML Size: {self.pml_size_points} grid points"
            )
        else:
            # If we only have water, show water properties
            param_text = (
                f"Water Domain: {self.domain_width*1000:.1f} × {self.domain_height*1000:.1f} × {self.domain_depth*1000:.1f} mm\n"
                f"Water Sound Speed: {self.water_props['sound_speed']:.1f} m/s\n"
                f"Water Density: {self.water_props['density']:.1f} kg/m³\n"
                f"Frequency: {self.transducer_params['frequency']/1e3:.1f} kHz\n"
                f"Wavelength in Water: {self.wavelength_water*1000:.2f} mm\n"
                f"Focal Depth: {self.focal_depth*1000:.1f} mm\n"
                f"PML Size: {self.pml_size_points} grid points"
            )

        # Position text in the top left corner
        ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Adjust view angle for better visualization
        ax.view_init(elev=20, azim=-35)

        # Tight layout
        plt.tight_layout()

        # Save or show the plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_medium(self, ax):
        """Plot the water domain and phantom (if any) as transparent boxes"""
        # Define the vertices of the water domain box
        x_domain = np.array([-self.domain_width/2, self.domain_width/2]) * 1000  # Convert to mm
        y_domain = np.array([-self.domain_height/2, self.domain_height/2]) * 1000
        z_domain = np.array([0, self.domain_depth]) * 1000

        # Create a meshgrid for the water domain box
        xx_domain, yy_domain = np.meshgrid(x_domain, y_domain)

        # Plot the water domain as a transparent box
        # Bottom face (z=0)
        ax.plot_surface(xx_domain, yy_domain, np.zeros_like(xx_domain),
                       color=self.water_props['color'], alpha=0.15, shade=True, label='Water Domain')

        # Top face (z=domain_depth)
        ax.plot_surface(xx_domain, yy_domain, np.ones_like(xx_domain) * z_domain[1],
                       color=self.water_props['color'], alpha=0.15, shade=True)

        # Side faces (just edges for clarity)
        for i in range(2):
            for j in range(2):
                ax.plot3D([x_domain[i], x_domain[i]], [y_domain[j], y_domain[j]],
                         [z_domain[0], z_domain[1]], color='lightblue', alpha=0.3, linewidth=1)

        for i in range(2):
            ax.plot3D([x_domain[0], x_domain[1]], [y_domain[i], y_domain[i]],
                     [z_domain[0], z_domain[0]], color='lightblue', alpha=0.3, linewidth=1)
            ax.plot3D([x_domain[0], x_domain[1]], [y_domain[i], y_domain[i]],
                     [z_domain[1], z_domain[1]], color='lightblue', alpha=0.3, linewidth=1)
            ax.plot3D([x_domain[i], x_domain[i]], [y_domain[0], y_domain[1]],
                     [z_domain[0], z_domain[0]], color='lightblue', alpha=0.3, linewidth=1)
            ax.plot3D([x_domain[i], x_domain[i]], [y_domain[0], y_domain[1]],
                     [z_domain[1], z_domain[1]], color='lightblue', alpha=0.3, linewidth=1)

        # If we have a phantom (not water), plot it
        if self.phantom_props is not None:
            # Define the vertices of the phantom box
            x_phantom = np.array([-self.phantom_width/2, self.phantom_width/2]) * 1000  # Convert to mm
            y_phantom = np.array([-self.phantom_height/2, self.phantom_height/2]) * 1000
            z_phantom = np.array([0, self.phantom_depth]) * 1000

            # Create a meshgrid for the phantom box
            xx_phantom, yy_phantom = np.meshgrid(x_phantom, y_phantom)

            # Plot the phantom as a transparent box
            # Bottom face (z=0)
            ax.plot_surface(xx_phantom, yy_phantom, np.zeros_like(xx_phantom),
                           color=self.phantom_props['color'], alpha=self.phantom_props['alpha'],
                           shade=True, label=self.phantom_props['name'])

            # Top face (z=phantom_depth)
            ax.plot_surface(xx_phantom, yy_phantom, np.ones_like(xx_phantom) * z_phantom[1],
                           color=self.phantom_props['color'], alpha=self.phantom_props['alpha'],
                           shade=True)

            # Side faces
            for i in range(2):
                for j in range(2):
                    ax.plot3D([x_phantom[i], x_phantom[i]], [y_phantom[j], y_phantom[j]],
                             [z_phantom[0], z_phantom[1]], color='gray', alpha=0.5, linewidth=1)

            for i in range(2):
                ax.plot3D([x_phantom[0], x_phantom[1]], [y_phantom[i], y_phantom[i]],
                         [z_phantom[0], z_phantom[0]], color='gray', alpha=0.5, linewidth=1)
                ax.plot3D([x_phantom[0], x_phantom[1]], [y_phantom[i], y_phantom[i]],
                         [z_phantom[1], z_phantom[1]], color='gray', alpha=0.5, linewidth=1)
                ax.plot3D([x_phantom[i], x_phantom[i]], [y_phantom[0], y_phantom[1]],
                         [z_phantom[0], z_phantom[0]], color='gray', alpha=0.5, linewidth=1)
                ax.plot3D([x_phantom[i], x_phantom[i]], [y_phantom[0], y_phantom[1]],
                         [z_phantom[1], z_phantom[1]], color='gray', alpha=0.5, linewidth=1)

    def _plot_transducer(self, ax):
        """Plot the transducer elements"""
        # Get transducer parameters
        element_width = self.transducer_params['element_width'] * 1000  # Convert to mm
        element_height = self.transducer_params['element_height'] * 1000

        # If element positions are provided, use them
        if self.transducer_params['element_positions'] is not None:
            element_positions = self.transducer_params['element_positions'] * 1000  # Convert to mm
        else:
            # Otherwise, generate element positions based on array dimensions
            num_elements_x = self.transducer_params['num_elements_x']
            num_elements_y = self.transducer_params['num_elements_y']
            kerf = self.transducer_params['kerf'] * 1000  # Convert to mm

            # Calculate element positions
            x_positions = np.linspace(-(num_elements_x-1)/2 * (element_width + kerf),
                                     (num_elements_x-1)/2 * (element_width + kerf),
                                     num_elements_x)
            y_positions = np.linspace(-(num_elements_y-1)/2 * (element_height + kerf),
                                     (num_elements_y-1)/2 * (element_height + kerf),
                                     num_elements_y)

            # Create meshgrid of positions
            x_pos, y_pos = np.meshgrid(x_positions, y_positions)
            element_positions = np.column_stack((x_pos.flatten(), y_pos.flatten()))

        # Plot each element
        for pos in element_positions:
            # Define the vertices of the element (bottom face at z=0)
            x = np.array([pos[0] - element_width/2, pos[0] + element_width/2])
            y = np.array([pos[1] - element_height/2, pos[1] + element_height/2])

            # Create a meshgrid for the element
            xx, yy = np.meshgrid(x, y)

            # Plot the element
            ax.plot_surface(xx, yy, np.zeros_like(xx), color='darkblue', alpha=0.7)

        # Add a label for the transducer array
        ax.text3D(0, 0, -5, 'Transducer Array', color='darkblue', fontsize=10,
                 ha='center', va='center')

    def _plot_pml_boundaries(self, ax):
        """Plot the PML boundaries"""
        if not KWAVE_AVAILABLE:
            # If k-wave is not available, use estimated PML size
            pml_size_mm = self.wavelength_water * 2 * 1000  # 2 wavelengths in mm
        else:
            # Use the stored PML size
            pml_size_mm = self.pml_size_meters * 1000  # Convert to mm

        # Define the vertices of the PML boundaries
        x_inner = np.array([-self.domain_width/2, self.domain_width/2]) * 1000
        y_inner = np.array([-self.domain_height/2, self.domain_height/2]) * 1000
        z_inner = np.array([0, self.domain_depth]) * 1000

        x_outer = np.array([x_inner[0] - pml_size_mm, x_inner[1] + pml_size_mm])
        y_outer = np.array([y_inner[0] - pml_size_mm, y_inner[1] + pml_size_mm])
        z_outer = np.array([z_inner[0] - pml_size_mm, z_inner[1] + pml_size_mm])

        # Plot PML boundaries as transparent surfaces with wireframe for clarity
        # Bottom PML
        xx, yy = np.meshgrid([x_inner[0], x_inner[1]], [y_inner[0], y_inner[1]])
        ax.plot_surface(xx, yy, np.ones_like(xx) * z_outer[0], color='red', alpha=0.05)

        # Top PML
        ax.plot_surface(xx, yy, np.ones_like(xx) * z_outer[1], color='red', alpha=0.05)

        # Side PMLs (x-direction)
        for i in range(2):
            yy, zz = np.meshgrid([y_inner[0], y_inner[1]], [z_inner[0], z_inner[1]])
            if i == 0:
                # Left side
                ax.plot_surface(np.ones_like(yy) * x_outer[0], yy, zz, color='red', alpha=0.05)
            else:
                # Right side
                ax.plot_surface(np.ones_like(yy) * x_outer[1], yy, zz, color='red', alpha=0.05)

        # Side PMLs (y-direction)
        for i in range(2):
            xx, zz = np.meshgrid([x_inner[0], x_inner[1]], [z_inner[0], z_inner[1]])
            if i == 0:
                # Front side
                ax.plot_surface(xx, np.ones_like(xx) * y_outer[0], zz, color='red', alpha=0.05)
            else:
                # Back side
                ax.plot_surface(xx, np.ones_like(xx) * y_outer[1], zz, color='red', alpha=0.05)

        # Add wireframe edges to make PML boundaries more visible
        # Outer box edges
        for i in range(2):
            for j in range(2):
                ax.plot3D([x_outer[i], x_outer[i]], [y_outer[j], y_outer[j]],
                         [z_outer[0], z_outer[1]], color='red', alpha=0.3, linewidth=1)
                ax.plot3D([x_outer[i], x_outer[i]], [y_outer[0], y_outer[1]],
                         [z_outer[j], z_outer[j]], color='red', alpha=0.3, linewidth=1)
                ax.plot3D([x_outer[0], x_outer[1]], [y_outer[i], y_outer[i]],
                         [z_outer[j], z_outer[j]], color='red', alpha=0.3, linewidth=1)

        # Add a label for the PML boundaries
        ax.text3D(x_outer[1] + 5, 0, z_inner[1]/2, 'PML Boundaries', color='red', fontsize=10,
                 ha='left', va='center')

        # Add a label for the PML size
        ax.text3D(x_inner[1], y_inner[1], z_inner[1] + pml_size_mm/2,
                 f'PML: {pml_size_mm:.1f} mm', color='red', fontsize=8,
                 ha='right', va='center')

    def _plot_focal_point_and_beam(self, ax):
        """Plot the focal point and beam path"""
        # Define the focal point
        focal_point = np.array([0, 0, self.focal_depth * 1000])  # Convert to mm

        # Plot the focal point
        ax.scatter(focal_point[0], focal_point[1], focal_point[2], color='red', s=100,
                  label='Focal Point')

        # Calculate the beam profile at different depths for a rectangular array
        z_points = np.linspace(0, self.domain_depth * 1000, 200)

        # Get array dimensions
        array_width_mm = self.transducer_params['array_width'] * 1000  # mm
        array_height_mm = self.transducer_params['array_height'] * 1000  # mm

        # Calculate beam widths in x and y directions at each z position
        beam_width_x = np.zeros_like(z_points)
        beam_width_y = np.zeros_like(z_points)

        # Calculate directivity-modified beam profile
        for i, z in enumerate(z_points):
            if z < focal_point[2]:
                # Special case for z=0: use exact transducer dimensions without directivity
                if z == 0 or np.isclose(z, 0):
                    # At z=0, beam width should exactly match the transducer dimensions without any modification
                    beam_width_x[i] = array_width_mm/2
                    beam_width_y[i] = array_height_mm/2
                else:
                    # For non-zero z, calculate base width with linear convergence
                    base_width_x = array_width_mm/2 * (1 - z/focal_point[2])
                    base_width_y = array_height_mm/2 * (1 - z/focal_point[2])

                    # Apply directivity for non-zero z
                    # Calculate angle from transducer center to edge of beam at this depth
                    angle_x = np.arctan2(base_width_x, z)
                    angle_y = np.arctan2(base_width_y, z)

                    # Apply directivity factor (wider beam due to directivity)
                    # The directivity narrows the effective aperture
                    directivity_factor_x = 1.0 / max(0.1, self.element_directivity.get_directivity(angle_x, 0))
                    directivity_factor_y = 1.0 / max(0.1, self.element_directivity.get_directivity(0, angle_y))

                    # Apply directivity to beam width (wider beam)
                    beam_width_x[i] = base_width_x * directivity_factor_x
                    beam_width_y[i] = base_width_y * directivity_factor_y
            else:
                # Diverging beam after focal point
                # For a phased array with directivity, the effective aperture is smaller
                # which increases the divergence angle

                # Calculate effective aperture reduction due to directivity and apodization
                # This is a simplified model - in reality, it's more complex
                directivity_factor = 0.8  # Directivity reduces effective aperture

                # Apply apodization effect if not 'none'
                if self.apodization != 'none':
                    # Use the aperture efficiency to calculate effective aperture reduction
                    apodization_factor = 1.0 / np.sqrt(self.aperture_efficiency)
                else:
                    apodization_factor = 1.0

                # Calculate effective aperture dimensions
                effective_width = array_width_mm * directivity_factor / apodization_factor
                effective_height = array_height_mm * directivity_factor / apodization_factor

                # Calculate divergence angles (increased due to smaller effective aperture)
                divergence_angle_x = 1.2 * self.wavelength_water * 1000 / effective_width  # in radians
                divergence_angle_y = 1.2 * self.wavelength_water * 1000 / effective_height  # in radians

                # Calculate beam width based on divergence angle
                beam_width_x[i] = np.tan(divergence_angle_x) * (z - focal_point[2])
                beam_width_y[i] = np.tan(divergence_angle_y) * (z - focal_point[2])

        # Ensure minimum beam width
        min_width_x = array_width_mm * 0.02
        min_width_y = array_height_mm * 0.02
        beam_width_x = np.maximum(beam_width_x, min_width_x)
        beam_width_y = np.maximum(beam_width_y, min_width_y)

        # Plot beam outline as a series of rectangles
        for i, z in enumerate(z_points):
            if i % 20 == 0:  # Plot every 20th point to reduce clutter
                # Special case for z=0 (transducer surface)
                if z == 0 or np.isclose(z, 0):
                    # Use actual transducer dimensions at z=0
                    x_rect = np.array([-array_width_mm/2, array_width_mm/2, array_width_mm/2, -array_width_mm/2, -array_width_mm/2])
                    y_rect = np.array([-array_height_mm/2, -array_height_mm/2, array_height_mm/2, array_height_mm/2, -array_height_mm/2])
                else:
                    # Create rectangle corners for other points
                    x_rect = np.array([-beam_width_x[i], beam_width_x[i], beam_width_x[i], -beam_width_x[i], -beam_width_x[i]])
                    y_rect = np.array([-beam_width_y[i], -beam_width_y[i], beam_width_y[i], beam_width_y[i], -beam_width_y[i]])

                z_rect = np.ones_like(x_rect) * z

                # Plot the rectangle
                ax.plot(x_rect, y_rect, z_rect, color='blue', alpha=0.2)

        # Add beam profile visualization at key points
        key_z_points = [0, focal_point[2]/2, focal_point[2], focal_point[2]*1.5]
        key_z_points = [z for z in key_z_points if z <= self.domain_depth * 1000]

        # Get transducer dimensions for the starting point
        array_width_mm = self.transducer_params['array_width'] * 1000  # mm
        array_height_mm = self.transducer_params['array_height'] * 1000  # mm

        for z in key_z_points:
            # Special case for z=0 (transducer surface)
            if z == 0 or np.isclose(z, 0):
                # Use actual transducer dimensions at z=0
                x_rect = np.array([-array_width_mm/2, array_width_mm/2, array_width_mm/2, -array_width_mm/2, -array_width_mm/2])
                y_rect = np.array([-array_height_mm/2, -array_height_mm/2, array_height_mm/2, array_height_mm/2, -array_height_mm/2])
            else:
                # Find the closest z-point index for non-zero z
                z_idx = np.argmin(np.abs(z_points - z))

                # Create rectangle for beam profile at other points
                x_rect = np.array([-beam_width_x[z_idx], beam_width_x[z_idx], beam_width_x[z_idx], -beam_width_x[z_idx], -beam_width_x[z_idx]])
                y_rect = np.array([-beam_width_y[z_idx], -beam_width_y[z_idx], beam_width_y[z_idx], beam_width_y[z_idx], -beam_width_y[z_idx]])

            z_rect = np.ones_like(x_rect) * z

            # Plot the rectangle with higher opacity
            ax.plot(x_rect, y_rect, z_rect, color='blue', alpha=0.5, linewidth=2)

            # Fill the rectangle with a transparent surface
            verts = [list(zip(x_rect, y_rect, z_rect))]
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(verts, alpha=0.1, facecolor='blue', edgecolor='blue')
            ax.add_collection3d(poly)

        # Plot beam centerline
        ax.plot([0, 0], [0, 0], [0, self.domain_depth * 1000], 'b--', alpha=0.5,
               label='Beam Axis')

        # Calculate near field distance for matrix array
        # For a rectangular aperture, we calculate separate near field distances for each dimension
        # and use the longer one as the overall near field distance
        array_width_mm = self.transducer_params['array_width'] * 1000  # mm
        array_height_mm = self.transducer_params['array_height'] * 1000  # mm

        # Near field distance for width dimension = L²/(4λ) where L is aperture width
        near_field_width = (array_width_mm**2) / (4 * self.wavelength_water * 1000)

        # Near field distance for height dimension = L²/(4λ) where L is aperture height
        near_field_height = (array_height_mm**2) / (4 * self.wavelength_water * 1000)

        # Use the longer distance as the overall near field distance
        near_field = max(near_field_width, near_field_height)

        # Plot near field boundary as a rectangular plane
        if near_field < self.domain_depth * 1000:
            # Create a rectangular plane at the near field distance
            x_plane = np.array([-array_width_mm/2, array_width_mm/2, array_width_mm/2, -array_width_mm/2, -array_width_mm/2])
            y_plane = np.array([-array_height_mm/2, -array_height_mm/2, array_height_mm/2, array_height_mm/2, -array_height_mm/2])
            z_plane = np.ones_like(x_plane) * near_field

            # Plot the rectangular boundary
            ax.plot(x_plane, y_plane, z_plane, 'g-', linewidth=2, alpha=0.7,
                   label=f'Near Field Boundary: {near_field:.1f} mm')

            # Fill the plane with a transparent surface
            verts = [list(zip(x_plane, y_plane, z_plane))]
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(verts, alpha=0.2, facecolor='green', edgecolor='green')
            ax.add_collection3d(poly)

            # Add vertical lines from the corners of the transducer to the near field boundary
            for i in range(4):
                ax.plot([x_plane[i], x_plane[i]], [y_plane[i], y_plane[i]], [0, near_field], 'g--', alpha=0.3)

            # Add text label for near field
            ax.text3D(0, 0, near_field + 10, f'Near Field Boundary: {near_field:.1f} mm',
                     color='green', fontsize=10, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Add text with near field calculation details
            detail_text = (f"Width Near Field: {near_field_width:.1f} mm\n"
                          f"Height Near Field: {near_field_height:.1f} mm")
            ax.text3D(array_width_mm/2 + 10, 0, near_field, detail_text,
                     color='green', fontsize=8, ha='left', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Add text label for focal depth
        ax.text3D(0, -array_height_mm/4, focal_point[2], f'Focal Depth\n{focal_point[2]:.1f} mm',
                 color='red', fontsize=8, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Calculate effective aperture size (average of width and height)
        effective_aperture = (array_width_mm + array_height_mm) / 2

        # Calculate F-number (focal depth / aperture diameter)
        f_number = (self.focal_depth * 1000) / effective_aperture

        # Add F-number annotation
        ax.text3D(array_width_mm/4, array_height_mm/4, focal_point[2],
                 f'F-number: {f_number:.1f}', color='blue', fontsize=8,
                 ha='left', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

def create_3d_sim_env(focal_depth_mm, medium_type='water', apodization='none'):
    """
    Create and visualize a 3D simulation environment for the given focal depth

    Parameters:
    -----------
    focal_depth_mm : float
        Focal depth in mm
    medium_type : str
        Type of medium ('water', 'bone', or 'soft_tissue')
    apodization : str
        Type of apodization to apply ('none', 'hanning', 'hamming', 'blackman')
    """
    # Load transducer parameters
    transducer_params = load_transducer_params()

    # Convert focal depth to meters
    focal_depth = focal_depth_mm * 1e-3

    # Create output directory
    output_dir = f'sim_results/focal_{focal_depth_mm:.0f}mm/3d_sim_env'
    os.makedirs(output_dir, exist_ok=True)

    # Create simulation environment
    sim_env = SimulationEnvironment3D(transducer_params, medium_type, focal_depth, apodization)

    # Create filename with medium type and apodization
    if apodization.lower() != 'none':
        filename = f'3d_simulation_environment_{medium_type.lower()}_{apodization.lower()}.png'
    else:
        filename = f'3d_simulation_environment_{medium_type.lower()}.png'

    # Visualize environment
    sim_env.visualize_environment(output_dir, filename)

    # Visualize element directivity
    sim_env.visualize_directivity(output_dir)

    # Visualize apodization if not 'none'
    if apodization.lower() != 'none':
        sim_env.visualize_apodization(output_dir)
        print(f"3D simulation environment with {medium_type} medium and {apodization} apodization created and saved to {output_dir}/{filename}")
        print(f"Element directivity patterns saved to {output_dir}/element_directivity.png")
        print(f"Apodization weights saved to {output_dir}/apodization_{apodization}.png")
    else:
        print(f"3D simulation environment with {medium_type} medium created and saved to {output_dir}/{filename}")
        print(f"Element directivity patterns saved to {output_dir}/element_directivity.png")

    return sim_env

def main():
    """Main function to create 3D simulation environments for all focal depths"""
    # Define focal depths
    focal_depths = [50, 100, 200, 400, 800, 1600]  # in mm

    # Define medium types
    medium_types = ['water', 'bone', 'soft_tissue']

    # Define apodization types
    apodization_types = ['none', 'hanning', 'hamming', 'blackman']

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Create 3D simulation environment for 2D transducer array')
    parser.add_argument('--focal-depth', type=float, default=None, help='Focal depth in mm')
    parser.add_argument('--medium', type=str, default='water', choices=medium_types, help='Medium type')
    parser.add_argument('--apodization', type=str, default='none', choices=apodization_types,
                       help='Apodization type to apply to the array elements')
    args = parser.parse_args()

    if args.focal_depth is not None:
        # Create environment for specified focal depth
        create_3d_sim_env(args.focal_depth, args.medium, args.apodization)
    else:
        # Create environments for all focal depths
        for focal_depth in focal_depths:
            create_3d_sim_env(focal_depth, args.medium, args.apodization)

if __name__ == "__main__":
    main()
