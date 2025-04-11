<<<<<<< HEAD
"""
2D Transducer Array Simulation and Visualization

This script creates and visualizes a 2D ultrasound transducer array based on
specified frequency and dimensions. It calculates appropriate element sizes
based on wavelength and generates phase configurations for focusing at
different depths.

Key features:
1. Transducer configuration based on 180 kHz frequency in water
2. 100mm x 100mm array dimensions
3. Element sizing based on wavelength
4. Signal generation and visualization
5. Phase configurations for focusing at multiple depths
6. Comprehensive visualization of array properties
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import matplotlib.patches as patches

# Import grid optimizer
try:
    from grid_optimizer import optimize_grid_dimensions, get_highest_prime_factors
    GRID_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("Warning: grid_optimizer.py not found. Grid optimization will be disabled.")
    GRID_OPTIMIZER_AVAILABLE = False

# Create directories if they don't exist
os.makedirs('sim_results', exist_ok=True)
focal_depths = [50, 100, 200, 400, 800, 1600]  # in mm
for depth in focal_depths:
    os.makedirs(f'sim_results/focal_{depth}mm', exist_ok=True)

# Demonstrate grid optimization for FFT operations
if GRID_OPTIMIZER_AVAILABLE:
    print("\nDemonstrating grid optimization for FFT operations:")
    # Example grid sizes that might be used in simulations
    example_grids = [
        (128, 128, 128),  # Powers of 2 (optimal)
        (100, 100, 100),  # Round numbers (not optimal)
        (130, 130, 166),  # From the warning message
        (131, 131, 167)   # Prime numbers (worst case)
    ]

    # Smaller PML size for faster simulation
    pml_size = 8  # Reduced from 10 to 8 to match other scripts

    for grid in example_grids:
        Nx, Ny, Nz = grid
        highest_primes = get_highest_prime_factors(Nx, Ny, Nz)
        print(f"\nOriginal grid: {Nx} x {Ny} x {Nz}")
        print(f"Highest prime factors: {highest_primes}")

        # Calculate grid with PML added (as k-wave would do internally)
        Nx_with_pml = Nx + 2 * pml_size
        Ny_with_pml = Ny + 2 * pml_size
        Nz_with_pml = Nz + 2 * pml_size
        highest_primes_with_pml = get_highest_prime_factors(Nx_with_pml, Ny_with_pml, Nz_with_pml)
        print(f"Grid with PML: {Nx_with_pml} x {Ny_with_pml} x {Nz_with_pml}")
        print(f"Highest prime factors with PML: {highest_primes_with_pml}")

        # Standard optimization (without considering PML)
        opt_Nx, opt_Ny, opt_Nz = optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=10)
        opt_highest_primes = get_highest_prime_factors(opt_Nx, opt_Ny, opt_Nz)

        print(f"Standard optimized grid: {opt_Nx} x {opt_Ny} x {opt_Nz}")
        print(f"Standard optimized highest prime factors: {opt_highest_primes}")

        # Calculate what happens when k-wave adds PML to our optimized grid
        opt_Nx_with_pml = opt_Nx + 2 * pml_size
        opt_Ny_with_pml = opt_Ny + 2 * pml_size
        opt_Nz_with_pml = opt_Nz + 2 * pml_size
        opt_highest_primes_with_pml = get_highest_prime_factors(opt_Nx_with_pml, opt_Ny_with_pml, opt_Nz_with_pml)
        print(f"Standard optimized grid with PML: {opt_Nx_with_pml} x {opt_Ny_with_pml} x {opt_Nz_with_pml}")
        print(f"Standard optimized highest prime factors with PML: {opt_highest_primes_with_pml}")

        # PML-aware optimization
        pml_opt_Nx, pml_opt_Ny, pml_opt_Nz = optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=10, pml_size=pml_size)
        pml_opt_highest_primes = get_highest_prime_factors(pml_opt_Nx, pml_opt_Ny, pml_opt_Nz)

        print(f"PML-aware optimized grid: {pml_opt_Nx} x {pml_opt_Ny} x {pml_opt_Nz}")
        print(f"PML-aware optimized highest prime factors: {pml_opt_highest_primes}")

        # Calculate what happens when k-wave adds PML to our PML-aware optimized grid
        pml_opt_Nx_with_pml = pml_opt_Nx + 2 * pml_size
        pml_opt_Ny_with_pml = pml_opt_Ny + 2 * pml_size
        pml_opt_Nz_with_pml = pml_opt_Nz + 2 * pml_size
        pml_opt_highest_primes_with_pml = get_highest_prime_factors(pml_opt_Nx_with_pml, pml_opt_Ny_with_pml, pml_opt_Nz_with_pml)
        print(f"PML-aware optimized grid with PML: {pml_opt_Nx_with_pml} x {pml_opt_Ny_with_pml} x {pml_opt_Nz_with_pml}")
        print(f"PML-aware optimized highest prime factors with PML: {pml_opt_highest_primes_with_pml}")
        print(f"Size increase: {(pml_opt_Nx/Nx - 1)*100:.1f}% x {(pml_opt_Ny/Ny - 1)*100:.1f}% x {(pml_opt_Nz/Nz - 1)*100:.1f}%")

# Physical parameters
frequency = 180e3  # 180 kHz
sound_speed_water = 1482.0  # m/s
wavelength = sound_speed_water / frequency  # m
print(f"Wavelength in water at {frequency/1e3} kHz: {wavelength*1000:.2f} mm")

# Transducer dimensions
array_width = 100e-3  # 100 mm in meters
array_height = 100e-3  # 100 mm in meters

# Element sizing based on wavelength
element_width = wavelength / 2  # Typical element width is λ/2
element_height = wavelength / 2  # Square elements
kerf = wavelength / 20  # Small gap between elements

# Calculate number of elements that fit in the array
num_elements_x = int(np.floor(array_width / (element_width + kerf)))
num_elements_y = int(np.floor(array_height / (element_height + kerf)))

# Adjust array dimensions to fit exact number of elements
actual_array_width = num_elements_x * (element_width + kerf) - kerf
actual_array_height = num_elements_y * (element_height + kerf) - kerf

print(f"Number of elements: {num_elements_x} x {num_elements_y} = {num_elements_x * num_elements_y} elements")
print(f"Element width: {element_width*1000:.2f} mm")
print(f"Element height: {element_height*1000:.2f} mm")
print(f"Kerf (spacing between elements): {kerf*1000:.2f} mm")
print(f"Actual array dimensions: {actual_array_width*1000:.2f} mm x {actual_array_height*1000:.2f} mm")

# Create element positions (center coordinates)
element_positions_x = np.linspace(-actual_array_width/2 + element_width/2,
                                 actual_array_width/2 - element_width/2,
                                 num_elements_x)
element_positions_y = np.linspace(-actual_array_height/2 + element_height/2,
                                 actual_array_height/2 - element_height/2,
                                 num_elements_y)

# Create meshgrid of element positions
element_pos_x, element_pos_y = np.meshgrid(element_positions_x, element_positions_y)
element_positions = np.column_stack((element_pos_x.flatten(), element_pos_y.flatten()))

# Function to calculate phase delays for focusing at a specific depth
def calculate_focus_phases(element_positions, focal_point, frequency, sound_speed):
    """
    Calculate phase delays for focusing at a specific point.

    Parameters:
    -----------
    element_positions : array
        Array of (x, y) positions for each element
    focal_point : array
        (x, y, z) coordinates of the focal point
    frequency : float
        Operating frequency in Hz
    sound_speed : float
        Speed of sound in the medium in m/s

    Returns:
    --------
    phases : array
        Phase delays for each element in radians
    """
    # Calculate distances from each element to the focal point
    distances = np.sqrt((element_positions[:, 0] - focal_point[0])**2 +
                      (element_positions[:, 1] - focal_point[1])**2 +
                      focal_point[2]**2)

    # Find the maximum distance
    max_distance = np.max(distances)

    # Calculate time delays (negative because we want to delay elements that are closer)
    time_delays = (max_distance - distances) / sound_speed

    # Convert time delays to phase delays
    phase_delays = 2 * np.pi * frequency * time_delays

    # Normalize phases to [0, 2π)
    phase_delays = phase_delays % (2 * np.pi)

    return phase_delays

# Generate signal
def generate_tone_burst(frequency, num_cycles, sample_rate, envelope='hanning'):
    """
    Generate a tone burst signal.

    Parameters:
    -----------
    frequency : float
        Signal frequency in Hz
    num_cycles : int
        Number of cycles in the burst
    sample_rate : float
        Sampling rate in Hz
    envelope : str
        Type of envelope ('hanning', 'hamming', 'rectangular')

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values
    """
    duration = num_cycles / frequency
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    signal = np.sin(2 * np.pi * frequency * t)

    # Apply envelope
    if envelope == 'hanning':
        window = np.hanning(num_samples)
    elif envelope == 'hamming':
        window = np.hamming(num_samples)
    elif envelope == 'rectangular':
        window = np.ones(num_samples)
    else:
        window = np.ones(num_samples)

    signal = signal * window

    return t, signal

# Generate continuous wave signal
def generate_continuous_wave(frequency, duration, sample_rate):
    """
    Generate a continuous wave signal.

    Parameters:
    -----------
    frequency : float
        Signal frequency in Hz
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    signal = np.sin(2 * np.pi * frequency * t)

    return t, signal

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
def generate_dual_frequency(f1, f2, amplitude_ratio=1.0, duration=None, sample_rate=None, phase1=0, phase2=0):
    """
    Generate a signal with two frequency components.

    Parameters:
    -----------
    f1 : float
        First frequency component in Hz
    f2 : float
        Second frequency component in Hz
    amplitude_ratio : float
        Ratio of amplitude of f2 to f1 (default: 1.0 for equal amplitudes)
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz
    phase1 : float
        Phase offset for first frequency in radians
    phase2 : float
        Phase offset for second frequency in radians

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

# Plot transducer configuration
def plot_transducer_config(element_positions, element_width, element_height, kerf, array_width, array_height, filename):
    """
    Plot the transducer configuration with a side-by-side view of the full array and a zoomed-in element.

    Parameters:
    -----------
    element_positions : array
        Array of (x, y) positions for each element
    element_width : float
        Width of each element in meters
    element_height : float
        Height of each element in meters
    kerf : float
        Spacing between elements in meters
    array_width : float
        Width of the array in meters
    array_height : float
        Height of the array in meters
    filename : str
        Filename to save the plot
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # Left subplot: Full array
    # Plot each element
    for pos in element_positions:
        rect = patches.Rectangle(
            (pos[0] - element_width/2, pos[1] - element_height/2),
            element_width, element_height,
            linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax1.add_patch(rect)

    # Set axis limits with some padding
    padding = max(element_width, element_height) * 2
    ax1.set_xlim(-array_width/2 - padding/2, array_width/2 + padding/2)
    ax1.set_ylim(-array_height/2 - padding/2, array_height/2 + padding/2)

    # Add labels and title
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Y Position (mm)', fontsize=12)
    ax1.set_title('2D Transducer Array Configuration', fontsize=14)

    # Convert axis ticks to mm
    ax1.set_xticks(np.linspace(-array_width/2, array_width/2, 7))
    ax1.set_yticks(np.linspace(-array_height/2, array_height/2, 7))
    ax1.set_xticklabels([f'{x*1000:.1f}' for x in ax1.get_xticks()])
    ax1.set_yticklabels([f'{y*1000:.1f}' for y in ax1.get_yticks()])

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add a scale bar
    scale_bar_length = 10e-3  # 10 mm
    ax1.plot([array_width/2 - scale_bar_length - padding/4, array_width/2 - padding/4],
            [-array_height/2 + padding/4, -array_height/2 + padding/4], 'k-', linewidth=3)
    ax1.text(array_width/2 - scale_bar_length/2 - padding/4, -array_height/2 + padding/3,
            f'{scale_bar_length*1000:.0f} mm', ha='center', va='bottom')

    # Add a box to show the zoomed area (center of the array)
    zoom_width = 3 * (element_width + kerf)
    zoom_height = 3 * (element_height + kerf)
    zoom_rect = patches.Rectangle(
        (-zoom_width/2, -zoom_height/2),
        zoom_width, zoom_height,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax1.add_patch(zoom_rect)

    # Add an arrow connecting the zoom box to the zoomed view
    ax1.annotate('', xy=(array_width/2 + padding/4, 0), xytext=(zoom_width/2, 0),
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5))

    # Right subplot: Zoomed-in view of elements
    # Calculate the center positions of a few elements (3x3 grid)
    center_x = 0
    center_y = 0
    element_grid_size = 3
    element_positions_zoomed = []
    for i in range(-element_grid_size//2, element_grid_size//2 + 1):
        for j in range(-element_grid_size//2, element_grid_size//2 + 1):
            x = center_x + i * (element_width + kerf)
            y = center_y + j * (element_height + kerf)
            element_positions_zoomed.append((x, y))

    # Plot each element in the zoomed view
    for pos in element_positions_zoomed:
        rect = patches.Rectangle(
            (pos[0] - element_width/2, pos[1] - element_height/2),
            element_width, element_height,
            linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax2.add_patch(rect)

    # Set axis limits for zoomed view
    ax2.set_xlim(-zoom_width/2 - element_width/2, zoom_width/2 + element_width/2)
    ax2.set_ylim(-zoom_height/2 - element_height/2, zoom_height/2 + element_height/2)

    # Add labels and title
    ax2.set_xlabel('X Position (mm)', fontsize=12)
    ax2.set_ylabel('Y Position (mm)', fontsize=12)
    ax2.set_title('Zoomed View of Transducer Elements', fontsize=14)

    # Convert axis ticks to mm
    ax2.set_xticks(np.linspace(-zoom_width/2, zoom_width/2, 5))
    ax2.set_yticks(np.linspace(-zoom_height/2, zoom_height/2, 5))
    ax2.set_xticklabels([f'{x*1000:.1f}' for x in ax2.get_xticks()])
    ax2.set_yticklabels([f'{y*1000:.1f}' for y in ax2.get_yticks()])

    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add dimension lines and labels for element width, height, and kerf
    # Element width dimension line
    center_element_pos = element_positions_zoomed[len(element_positions_zoomed)//2]  # Center element
    x_start = center_element_pos[0] - element_width/2
    x_end = center_element_pos[0] + element_width/2
    y_pos = center_element_pos[1] - element_height/2 - element_height/4

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text((x_start + x_end)/2, y_pos - element_height/8,
            f'Element Width\n{element_width*1000:.2f} mm',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Element height dimension line
    x_pos = center_element_pos[0] + element_width/2 + element_width/4
    y_start = center_element_pos[1] - element_height/2
    y_end = center_element_pos[1] + element_height/2

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_pos, y_start), xytext=(x_pos, y_end),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text(x_pos + element_width/8, (y_start + y_end)/2,
            f'Element Height\n{element_height*1000:.2f} mm',
            ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Kerf dimension line (horizontal)
    right_element_pos = element_positions_zoomed[len(element_positions_zoomed)//2 + 1]  # Element to the right of center
    x_start = center_element_pos[0] + element_width/2
    x_end = right_element_pos[0] - element_width/2
    y_pos = center_element_pos[1]

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax2.text((x_start + x_end)/2, y_pos + element_height/8,
            f'Kerf\n{kerf*1000:.2f} mm',
            ha='center', va='bottom', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add pitch dimension (element width + kerf)
    pitch = element_width + kerf
    x_start = center_element_pos[0] - element_width/2
    x_end = right_element_pos[0] - element_width/2
    y_pos = center_element_pos[1] + element_height/2 + element_height/4

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax2.text((x_start + x_end)/2, y_pos + element_height/8,
            f'Pitch\n{pitch*1000:.2f} mm',
            ha='center', va='bottom', fontsize=10, color='blue',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add annotations with key parameters
    plt.figtext(0.5, 0.01,
                f'Array dimensions: {array_width*1000:.1f} mm × {array_height*1000:.1f} mm\n'
                f'Element dimensions: {element_width*1000:.2f} mm × {element_height*1000:.2f} mm\n'
                f'Kerf (spacing): {kerf*1000:.2f} mm, Pitch: {pitch*1000:.2f} mm\n'
                f'Number of elements: {len(element_positions)} ({int(np.sqrt(len(element_positions)))} × {int(np.sqrt(len(element_positions)))})\n'
                f'Frequency: {frequency/1e3:.1f} kHz, Wavelength: {wavelength*1000:.2f} mm\n'
                f'Element width/wavelength ratio: {element_width/(wavelength):.2f}',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot transducer with phase information
def plot_transducer_phases(element_positions, element_width, element_height, phases, focal_depth, filename):
    """
    Plot the transducer configuration with phase information and beam visualization.

    Parameters:
    -----------
    element_positions : array
        Array of (x, y) positions for each element
    element_width : float
        Width of each element in meters
    element_height : float
        Height of each element in meters
    phases : array
        Phase values for each element in radians
    focal_depth : float
        Focal depth in meters
    filename : str
        Filename to save the plot
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.2]})

    # Left subplot: Transducer array with phase information
    # Create a colormap for phases
    norm = Normalize(vmin=0, vmax=2*np.pi)
    cmap = cm.hsv

    # Plot each element with color based on phase
    for pos, phase in zip(element_positions, phases):
        rect = patches.Rectangle(
            (pos[0] - element_width/2, pos[1] - element_height/2),
            element_width, element_height,
            linewidth=1, edgecolor='black', facecolor=cmap(norm(phase)), alpha=0.7
        )
        ax1.add_patch(rect)

    # Set axis limits with some padding
    padding = max(element_width, element_height) * 2
    ax1.set_xlim(-actual_array_width/2 - padding/2, actual_array_width/2 + padding/2)
    ax1.set_ylim(-actual_array_height/2 - padding/2, actual_array_height/2 + padding/2)

    # Add labels and title
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Y Position (mm)', fontsize=12)
    ax1.set_title(f'2D Transducer Array Phase Configuration\nFocal Depth: {focal_depth*1000:.0f} mm', fontsize=14)

    # Convert axis ticks to mm
    ax1.set_xticks(np.linspace(-actual_array_width/2, actual_array_width/2, 7))
    ax1.set_yticks(np.linspace(-actual_array_height/2, actual_array_height/2, 7))
    ax1.set_xticklabels([f'{x*1000:.1f}' for x in ax1.get_xticks()])
    ax1.set_yticklabels([f'{y*1000:.1f}' for y in ax1.get_yticks()])

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, ticks=np.linspace(0, 2*np.pi, 5))
    cbar.set_label('Phase (radians)', fontsize=12)
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Right subplot: Beam visualization with focal point
    # Create a simplified beam visualization
    # Define the beam path
    z_max = focal_depth * 1.5  # Extend beyond focal point
    z = np.linspace(0, z_max, 1000)

    # Calculate beam width at each z position (simplified model)
    # Using a simple model where the beam converges to the focal point and then diverges
    array_width_half = actual_array_width/2
    array_height_half = actual_array_height/2

    # Calculate beam width at each z position
    beam_width = np.zeros_like(z)

    for i, z_val in enumerate(z):
        if z_val == 0:
            # At z=0, beam width should exactly match the transducer width
            beam_width[i] = array_width_half
        elif z_val < focal_depth:
            # Converging beam - linear convergence to focal point
            beam_width[i] = array_width_half * (1 - z_val/focal_depth)
        else:
            # Diverging beam (slower divergence than convergence)
            beam_width[i] = array_width_half * 0.5 * (z_val/focal_depth - 1)

    # Ensure minimum beam width
    beam_width = np.maximum(beam_width, array_width_half * 0.05)  # Minimum beam width

    # Plot the beam outline
    ax2.fill_betweenx(z * 1000, -beam_width * 1000, beam_width * 1000,
                     color='lightblue', alpha=0.3, label='Beam')
    ax2.plot(beam_width * 1000, z * 1000, 'b-', linewidth=1.5)
    ax2.plot(-beam_width * 1000, z * 1000, 'b-', linewidth=1.5)

    # Plot the transducer array at z=0
    ax2.plot([-actual_array_width/2 * 1000, actual_array_width/2 * 1000], [0, 0], 'k-', linewidth=3)

    # Plot the focal point
    ax2.plot(0, focal_depth * 1000, 'ro', markersize=8, label='Focal Point')

    # Add beam centerline
    ax2.plot([0, 0], [0, z_max * 1000], 'k--', linewidth=1, alpha=0.7, label='Beam Axis')

    # Calculate near field distance (Fresnel zone)
    # For a rectangular aperture, use the average of width and height
    array_radius = np.sqrt((array_width_half**2 + array_height_half**2) / 2)
    near_field = (array_radius**2) / wavelength

    # Add near field marker
    if near_field < z_max:
        ax2.axhline(y=near_field * 1000, color='g', linestyle='--', linewidth=1.5,
                   label=f'Near Field Boundary: {near_field*1000:.1f} mm')

    # Add labels and title
    ax2.set_xlabel('Lateral Position (mm)', fontsize=12)
    ax2.set_ylabel('Axial Distance (mm)', fontsize=12)
    ax2.set_title('Ultrasound Beam Visualization', fontsize=14)

    # Set axis limits
    ax2.set_xlim(-actual_array_width * 1000, actual_array_width * 1000)
    ax2.set_ylim(-5, z_max * 1000 * 1.05)

    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax2.legend(loc='upper right', fontsize=10)

    # Add annotations with key parameters
    plt.figtext(0.5, 0.01,
                f'Array dimensions: {actual_array_width*1000:.1f} mm × {actual_array_height*1000:.1f} mm\n'
                f'Element dimensions: {element_width*1000:.2f} mm × {element_height*1000:.2f} mm\n'
                f'Frequency: {frequency/1e3:.1f} kHz, Wavelength: {wavelength*1000:.2f} mm\n'
                f'Focal point: (0, 0, {focal_depth*1000:.0f} mm), F-number: {focal_depth/array_radius:.1f}\n'
                f'Near field distance: {near_field*1000:.1f} mm',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot signal in time and frequency domain
def plot_signal(t, signal, frequency, filename_time, filename_freq, filename_spectrogram=None):
    """
    Plot the signal in time and frequency domain and optionally a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    frequency : float
        Signal frequency in Hz
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str, optional
        Filename to save the spectrogram plot
    """
    # Create a figure with 2 or 3 subplots depending on whether spectrogram is requested
    if filename_spectrogram:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Time domain
        ax2 = fig.add_subplot(gs[0, 1])  # Frequency domain
        ax3 = fig.add_subplot(gs[1, :])  # Spectrogram (full width)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time domain plot (ax1)
    # Plot signal
    ax1.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    # Calculate and plot instantaneous phase
    phase = np.unwrap(np.angle(analytic_signal))
    phase_norm = (phase - phase.min()) / (phase.max() - phase.min()) * 2 - 1  # Normalize to [-1, 1]

    # Add labels and title
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'Signal Waveform at {frequency/1e3:.1f} kHz', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # Add cycles count and period markers
    period = 1.0 / frequency  # Period in seconds
    num_cycles_shown = min(5, int(t[-1] * frequency))  # Show markers for up to 5 cycles

    for i in range(num_cycles_shown):
        cycle_start = i * period
        if cycle_start < t[-1]:
            ax1.axvline(x=cycle_start * 1e6, color='gray', linestyle=':', alpha=0.5)
            if i > 0:  # Skip labeling the first line at t=0
                ax1.text(cycle_start * 1e6, ax1.get_ylim()[0] * 0.9, f'{i}T',
                        ha='center', va='bottom', fontsize=8, alpha=0.7)

    # Add wavelength annotation
    ax1.annotate(f'Period (T) = {period*1e6:.2f} μs',
                xy=(1.5*period*1e6, 0),
                xytext=(1.5*period*1e6, ax1.get_ylim()[1]*0.8),
                arrowprops=dict(arrowstyle='->', color='gray'),
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Frequency domain plot (ax2)
    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the signal frequency
    positive_freq_idx = np.where((xf >= 0) & (xf <= 3*frequency))
    ax2.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    # Add labels and title
    ax2.set_xlabel('Frequency (kHz)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Signal Frequency Spectrum', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add vertical line at the fundamental frequency
    ax2.axvline(x=frequency/1e3, color='r', linestyle='--', alpha=0.7)
    ax2.text(frequency/1e3, ax2.get_ylim()[1]*0.9, f'{frequency/1e3:.1f} kHz',
            ha='center', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add bandwidth markers if applicable (for tone burst)
    if np.std(envelope) > 0.01:  # Check if it's not a continuous wave
        # Estimate bandwidth from spectrum
        magnitude = 2.0/n * np.abs(yf[positive_freq_idx])
        max_mag = np.max(magnitude)
        half_power_level = max_mag / 2  # -3dB point

        # Find frequencies at half power
        above_half_power = magnitude > half_power_level
        if np.sum(above_half_power) > 1:  # If we have points above half power
            f_indices = np.where(above_half_power)[0]
            f_low = xf[positive_freq_idx][f_indices[0]] / 1e3
            f_high = xf[positive_freq_idx][f_indices[-1]] / 1e3
            bandwidth = f_high - f_low

            # Add markers for bandwidth
            ax2.axvline(x=f_low, color='g', linestyle=':', alpha=0.7)
            ax2.axvline(x=f_high, color='g', linestyle=':', alpha=0.7)
            ax2.axhline(y=half_power_level, color='g', linestyle=':', alpha=0.7)

            # Add bandwidth annotation
            ax2.annotate(f'Bandwidth (-3dB): {bandwidth:.1f} kHz',
                        xy=((f_low + f_high)/2, half_power_level),
                        xytext=((f_low + f_high)/2, half_power_level*2),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        ha='center', va='bottom', fontsize=10, color='green',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add spectrogram if requested
    if filename_spectrogram:
        # Calculate spectrogram using scipy.signal
        from scipy import signal as sig
        nperseg = min(256, len(t) // 10)  # Number of points per segment
        f_sample = 1.0/(t[1]-t[0])  # Sample frequency

        # Calculate spectrogram
        f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

        # Convert to dB scale, avoiding log of zero
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Plot spectrogram
        im = ax3.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

        # Add labels and title
        ax3.set_xlabel('Time (μs)', fontsize=12)
        ax3.set_ylabel('Frequency (kHz)', fontsize=12)
        ax3.set_title(f'Spectrogram of {frequency/1e3:.1f} kHz Signal', fontsize=14)

        # Set y-axis limits to focus on the frequency range of interest
        ax3.set_ylim([0, frequency/1e3 * 3])  # Show up to 3x the fundamental frequency

        # Add horizontal line at the fundamental frequency
        ax3.axhline(y=frequency/1e3, color='r', linestyle='--', alpha=0.7)

        # Add text label
        ax3.text(ax3.get_xlim()[1]*0.95, frequency/1e3, f'{frequency/1e3:.1f} kHz',
                ha='right', va='bottom', color='r', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add annotations with key parameters
    plt.figtext(0.5, 0.01,
                f'Frequency: {frequency/1e3:.1f} kHz\n'
                f'Wavelength in water: {wavelength*1000:.2f} mm\n'
                f'Period: {period*1e6:.2f} μs\n'
                f'Signal duration: {t[-1]*1e6:.1f} μs\n'
                f'Cycles: {t[-1]*frequency:.1f}',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time if not filename_spectrogram else filename_spectrogram, dpi=300, bbox_inches='tight')

    # If we're creating a combined plot with spectrogram, also save individual plots
    if filename_spectrogram:
        plt.close()

        # Save individual time domain plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')
        ax.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Signal Waveform at {frequency/1e3:.1f} kHz', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(filename_time, dpi=300, bbox_inches='tight')
        plt.close()

        # Save individual frequency domain plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)
        ax.set_xlabel('Frequency (kHz)', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title('Signal Frequency Spectrum', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axvline(x=frequency/1e3, color='r', linestyle='--', alpha=0.7)
        ax.text(frequency/1e3, ax.get_ylim()[1]*0.9, f'{frequency/1e3:.1f} kHz',
                ha='center', va='top', color='r', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.close()

# Plot dual frequency signal
def plot_dual_frequency_signal(t, signal, f1, f2, filename_time, filename_freq, filename_spectrogram=None):
    """
    Plot the dual frequency signal in time and frequency domains, plus optionally a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    f1 : float
        First frequency component in Hz
    f2 : float
        Second frequency component in Hz
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str, optional
        Filename to save the spectrogram plot
    """
    # Time domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot signal
    ax.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Dual Frequency Signal ({f1/1e3:.1f} kHz + {f2/1e3:.1f} kHz)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Add annotations
    plt.figtext(0.5, 0.01,
                f'Frequencies: {f1/1e3:.1f} kHz + {f2/1e3:.1f} kHz\n'
                f'Wavelengths: {(sound_speed_water/f1)*1000:.2f} mm + {(sound_speed_water/f2)*1000:.2f} mm\n'
                f'Duration: {t[-1]*1e6:.1f} μs',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time, dpi=300, bbox_inches='tight')
    plt.close()

    # Frequency domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the max frequency
    max_freq = max(f1, f2)
    positive_freq_idx = np.where((xf >= 0) & (xf <= 3*max_freq))
    ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Dual Frequency Spectrum', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at the two frequencies
    ax.axvline(x=f1/1e3, color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=f2/1e3, color='g', linestyle='--', alpha=0.7)

    # Add text labels
    ax.text(f1/1e3, ax.get_ylim()[1]*0.9, f'{f1/1e3:.1f} kHz',
            ha='center', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f2/1e3, ax.get_ylim()[1]*0.9, f'{f2/1e3:.1f} kHz',
            ha='center', va='top', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate spectrogram if filename is provided
    if filename_spectrogram:
        # Spectrogram plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate spectrogram using scipy.signal
        from scipy import signal as sig
        nperseg = min(256, len(t) // 10)  # Number of points per segment
        f_sample = 1.0/(t[1]-t[0])  # Sample frequency

        # Calculate spectrogram
        f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

        # Convert to dB scale, avoiding log of zero
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Plot spectrogram
        im = ax.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

        # Add labels and title
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Frequency (kHz)', fontsize=12)
        ax.set_title(f'Spectrogram of Dual Frequency Signal', fontsize=14)

        # Set y-axis limits to focus on the frequency range of interest
        ax.set_ylim([0, max(f1, f2)/1e3 * 3])  # Show up to 3x the max frequency

        # Add horizontal lines at the two frequencies
        ax.axhline(y=f1/1e3, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=f2/1e3, color='g', linestyle='--', alpha=0.7)

        # Add text labels
        ax.text(ax.get_xlim()[1]*0.95, f1/1e3, f'{f1/1e3:.1f} kHz',
                ha='right', va='bottom', color='r', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(ax.get_xlim()[1]*0.95, f2/1e3, f'{f2/1e3:.1f} kHz',
                ha='right', va='bottom', color='g', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename_spectrogram, dpi=300, bbox_inches='tight')
        plt.close()

# Plot dual frequency sweep signal
def plot_dual_frequency_sweep(t, signal, f1_inst, f2_inst, f1_center, f2_center, bandwidth_percent,
                           filename_time, filename_freq, filename_spectrogram):
    """
    Plot the dual frequency sweep signal in time and frequency domains, plus a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    f1_inst : array
        Instantaneous frequency of first component at each time point
    f2_inst : array
        Instantaneous frequency of second component at each time point
    f1_center : float
        Center frequency of first component in Hz
    f2_center : float
        Center frequency of second component in Hz
    bandwidth_percent : float
        Bandwidth as percentage of center frequency
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str
        Filename to save the spectrogram plot
    """
    # Time domain plot with instantaneous frequencies
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot signal
    ax1.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    # Add second y-axis for frequencies
    ax2 = ax1.twinx()
    ax2.plot(t * 1e6, f1_inst / 1e3, 'g-', linewidth=1.5, label=f'Freq 1 ({f1_center/1e3:.1f} kHz)')
    ax2.plot(t * 1e6, f2_inst / 1e3, 'm-', linewidth=1.5, label=f'Freq 2 ({f2_center/1e3:.1f} kHz)')
    ax2.set_ylabel('Frequency (kHz)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add title and legend
    plt.title(f'Dual Frequency Sweep Signal', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add annotations
    bandwidth1 = f1_center * (bandwidth_percent / 100)
    f1_min = max(f1_center - bandwidth1 / 2, 1.0)
    f1_max = f1_center + bandwidth1 / 2

    bandwidth2 = f2_center * (bandwidth_percent / 100)
    f2_min = max(f2_center - bandwidth2 / 2, 1.0)
    f2_max = f2_center + bandwidth2 / 2

    plt.figtext(0.5, 0.01,
                f'Frequency 1: {f1_center/1e3:.1f} kHz (Range: {f1_min/1e3:.1f} - {f1_max/1e3:.1f} kHz)\n'
                f'Frequency 2: {f2_center/1e3:.1f} kHz (Range: {f2_min/1e3:.1f} - {f2_max/1e3:.1f} kHz)\n'
                f'Bandwidth: {bandwidth_percent}%\n'
                f'Duration: {t[-1]*1e6:.1f} μs',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time, dpi=300, bbox_inches='tight')
    plt.close()

    # Frequency domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the max frequency
    max_freq = max(f2_max, f1_max)
    positive_freq_idx = np.where((xf >= 0) & (xf <= 3*max_freq))
    ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Dual Frequency Sweep Spectrum', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at min, center, and max frequencies for both components
    ax.axvline(x=f1_min/1e3, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=f1_center/1e3, color='r', linestyle='-', alpha=0.7)
    ax.axvline(x=f1_max/1e3, color='r', linestyle='--', alpha=0.5)

    ax.axvline(x=f2_min/1e3, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=f2_center/1e3, color='g', linestyle='-', alpha=0.7)
    ax.axvline(x=f2_max/1e3, color='g', linestyle='--', alpha=0.5)

    # Add text labels
    ax.text(f1_center/1e3, ax.get_ylim()[1]*0.9, f'{f1_center/1e3:.1f} kHz',
            ha='center', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f2_center/1e3, ax.get_ylim()[1]*0.9, f'{f2_center/1e3:.1f} kHz',
            ha='center', va='top', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
    plt.close()

    # Spectrogram plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate spectrogram using scipy.signal
    from scipy import signal as sig
    nperseg = min(256, len(t) // 10)  # Number of points per segment
    f_sample = 1.0/(t[1]-t[0])  # Sample frequency

    # Calculate spectrogram
    f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

    # Convert to dB scale, avoiding log of zero
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plot spectrogram
    im = ax.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

    # Add labels and title
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    ax.set_title('Spectrogram of Dual Frequency Sweep Signal', fontsize=14)

    # Set y-axis limits to focus on the frequency range of interest
    ax.set_ylim([0, max(f1_max, f2_max)/1e3 * 1.5])

    # Add horizontal lines at min, center, and max frequencies for both components
    ax.axhline(y=f1_min/1e3, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=f1_center/1e3, color='r', linestyle='-', alpha=0.7)
    ax.axhline(y=f1_max/1e3, color='r', linestyle='--', alpha=0.5)

    ax.axhline(y=f2_min/1e3, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=f2_center/1e3, color='g', linestyle='-', alpha=0.7)
    ax.axhline(y=f2_max/1e3, color='g', linestyle='--', alpha=0.5)

    # Add text labels
    ax.text(ax.get_xlim()[1]*0.95, f1_center/1e3, f'{f1_center/1e3:.1f} kHz',
            ha='right', va='center', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(ax.get_xlim()[1]*0.95, f2_center/1e3, f'{f2_center/1e3:.1f} kHz',
            ha='right', va='center', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_spectrogram, dpi=300, bbox_inches='tight')
    plt.close()

# Plot frequency sweep signal
def plot_sweep_signal(t, signal, f_inst, f_center, bandwidth_percent, filename_time, filename_freq, filename_spectrogram):
    """
    Plot the frequency sweep signal in time and frequency domains, plus a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    f_inst : array
        Instantaneous frequency at each time point
    f_center : float
        Center frequency in Hz
    bandwidth_percent : float
        Bandwidth as percentage of center frequency
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str
        Filename to save the spectrogram plot
    """
    # Time domain plot with instantaneous frequency
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot signal
    ax1.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    # Add second y-axis for frequency
    ax2 = ax1.twinx()
    ax2.plot(t * 1e6, f_inst / 1e3, 'g-', linewidth=1.5, label='Instantaneous Frequency')
    ax2.set_ylabel('Frequency (kHz)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add title and legend
    plt.title(f'Frequency Sweep Signal (Center: {f_center/1e3:.1f} kHz, Bandwidth: {bandwidth_percent}%)', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add annotations
    bandwidth = f_center * (bandwidth_percent / 100)
    f_min = max(f_center - bandwidth / 2, 1.0)
    f_max = f_center + bandwidth / 2

    plt.figtext(0.5, 0.01,
                f'Center Frequency: {f_center/1e3:.1f} kHz\n'
                f'Bandwidth: {bandwidth/1e3:.1f} kHz ({bandwidth_percent}%)\n'
                f'Frequency Range: {f_min/1e3:.1f} - {f_max/1e3:.1f} kHz\n'
                f'Duration: {t[-1]*1e6:.1f} μs',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time, dpi=300, bbox_inches='tight')
    plt.close()

    # Frequency domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the max frequency
    max_plot_freq = 3 * f_max
    positive_freq_idx = np.where((xf >= 0) & (xf <= max_plot_freq))
    ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Frequency Sweep Spectrum', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at min, center, and max frequencies
    ax.axvline(x=f_min/1e3, color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=f_center/1e3, color='g', linestyle='--', alpha=0.7)
    ax.axvline(x=f_max/1e3, color='r', linestyle='--', alpha=0.7)

    # Add text labels
    ax.text(f_min/1e3, ax.get_ylim()[1]*0.9, f'{f_min/1e3:.1f} kHz',
            ha='right', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f_center/1e3, ax.get_ylim()[1]*0.9, f'{f_center/1e3:.1f} kHz',
            ha='center', va='top', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f_max/1e3, ax.get_ylim()[1]*0.9, f'{f_max/1e3:.1f} kHz',
            ha='left', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
    plt.close()

    # Spectrogram plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate spectrogram using scipy.signal
    from scipy import signal as sig
    nperseg = min(256, len(t) // 10)  # Number of points per segment
    f_sample = 1.0/(t[1]-t[0])  # Sample frequency

    # Calculate spectrogram
    f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

    # Convert to dB scale, avoiding log of zero
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plot spectrogram
    im = ax.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

    # Add labels and title
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    ax.set_title('Spectrogram of Frequency Sweep Signal', fontsize=14)

    # Set y-axis limits to focus on the frequency range of interest
    ax.set_ylim([0, f_max/1e3 * 1.5])

    # Add horizontal lines at min, center, and max frequencies
    ax.axhline(y=f_min/1e3, color='r', linestyle='--', alpha=0.7)
    ax.axhline(y=f_center/1e3, color='g', linestyle='--', alpha=0.7)
    ax.axhline(y=f_max/1e3, color='r', linestyle='--', alpha=0.7)

    # Add text labels
    ax.text(ax.get_xlim()[1]*0.95, f_min/1e3, f'{f_min/1e3:.1f} kHz',
            ha='right', va='bottom', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(ax.get_xlim()[1]*0.95, f_center/1e3, f'{f_center/1e3:.1f} kHz',
            ha='right', va='center', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(ax.get_xlim()[1]*0.95, f_max/1e3, f'{f_max/1e3:.1f} kHz',
            ha='right', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_spectrogram, dpi=300, bbox_inches='tight')
    plt.close()

# Create directories for signal results with consistent structure
os.makedirs('sim_results/single_freq', exist_ok=True)
os.makedirs('sim_results/single_freq/sweep_freq', exist_ok=True)
os.makedirs('sim_results/dual_freq', exist_ok=True)
os.makedirs('sim_results/dual_freq/sweep_freq', exist_ok=True)

# Signal parameters
sample_rate = 100 * frequency  # 100 samples per period
num_cycles = 10
signal_duration = 100 / frequency  # Same duration for all signals
bandwidth_percent = 40  # Default 40% bandwidth

# Second frequency for dual frequency signals
frequency2 = 550e3  # 550 kHz

# Generate signals
# Single frequency signals
t_burst, signal_burst = generate_tone_burst(frequency, num_cycles, sample_rate)
t_continuous, signal_continuous = generate_continuous_wave(frequency, signal_duration, sample_rate)

# Single frequency sweep
t_sweep, signal_sweep, f_inst = generate_frequency_sweep(frequency, bandwidth_percent, signal_duration, sample_rate)

# Dual frequency signals
t_dual, signal_dual = generate_dual_frequency(frequency, frequency2, 1.0, signal_duration, sample_rate)

# Dual frequency sweep
t_dual_sweep, signal_dual_sweep, f1_inst, f2_inst = generate_dual_frequency_sweep(
    frequency, frequency2, bandwidth_percent, signal_duration, sample_rate)

# Plot transducer configuration
plot_transducer_config(
    element_positions,
    element_width,
    element_height,
    kerf,
    actual_array_width,
    actual_array_height,
    'sim_results/transducer_configuration.png'
)

# Plot single frequency signals
plot_signal(
    t_burst,
    signal_burst,
    frequency,
    'sim_results/single_freq/signal_time_domain.png',
    'sim_results/single_freq/signal_frequency_domain.png',
    'sim_results/single_freq/signal_spectrogram.png'
)

# Plot single frequency sweep signals
plot_sweep_signal(
    t_sweep,
    signal_sweep,
    f_inst,
    frequency,
    bandwidth_percent,
    'sim_results/single_freq/sweep_freq/signal_time_domain.png',
    'sim_results/single_freq/sweep_freq/signal_frequency_domain.png',
    'sim_results/single_freq/sweep_freq/signal_spectrogram.png'
)

# Plot dual frequency signals
plot_dual_frequency_signal(
    t_dual,
    signal_dual,
    frequency,
    frequency2,
    'sim_results/dual_freq/signal_time_domain.png',
    'sim_results/dual_freq/signal_frequency_domain.png',
    'sim_results/dual_freq/signal_spectrogram.png'
)

# Plot dual frequency sweep signals
plot_dual_frequency_sweep(
    t_dual_sweep,
    signal_dual_sweep,
    f1_inst,
    f2_inst,
    frequency,
    frequency2,
    bandwidth_percent,
    'sim_results/dual_freq/sweep_freq/signal_time_domain.png',
    'sim_results/dual_freq/sweep_freq/signal_frequency_domain.png',
    'sim_results/dual_freq/sweep_freq/signal_spectrogram.png'
)

# Calculate and plot phase configurations for different focal depths
for depth_mm in focal_depths:
    depth = depth_mm * 1e-3  # Convert mm to m
    focal_point = np.array([0, 0, depth])
    phases = calculate_focus_phases(element_positions, focal_point, frequency, sound_speed_water)

    # Reshape phases to match the grid
    phases_grid = phases.reshape(num_elements_y, num_elements_x)

    # Plot transducer with phase information
    plot_transducer_phases(
        element_positions,
        element_width,
        element_height,
        phases,
        depth,
        f'sim_results/focal_{depth_mm}mm/phase_configuration.png'
    )

    # Create a 2D plot of the phase distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(phases_grid, cmap='hsv', origin='lower',
                  extent=[-actual_array_width/2*1000, actual_array_width/2*1000,
                          -actual_array_height/2*1000, actual_array_height/2*1000],
                  vmin=0, vmax=2*np.pi)

    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title(f'Phase Distribution for Focal Depth {depth_mm} mm', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ticks=np.linspace(0, 2*np.pi, 5))
    cbar.set_label('Phase (radians)', fontsize=12)
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()
    plt.savefig(f'sim_results/focal_{depth_mm}mm/phase_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

print("All simulations and visualizations completed successfully.")
=======
"""
2D Transducer Array Simulation and Visualization

This script creates and visualizes a 2D ultrasound transducer array based on
specified frequency and dimensions. It calculates appropriate element sizes
based on wavelength and generates phase configurations for focusing at
different depths.

Key features:
1. Transducer configuration based on 180 kHz frequency in water
2. 100mm x 100mm array dimensions
3. Element sizing based on wavelength
4. Signal generation and visualization
5. Phase configurations for focusing at multiple depths
6. Comprehensive visualization of array properties
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import matplotlib.patches as patches

# Import grid optimizer
try:
    from grid_optimizer import optimize_grid_dimensions, get_highest_prime_factors
    GRID_OPTIMIZER_AVAILABLE = True
except ImportError:
    print("Warning: grid_optimizer.py not found. Grid optimization will be disabled.")
    GRID_OPTIMIZER_AVAILABLE = False

# Create directories if they don't exist
os.makedirs('sim_results', exist_ok=True)
focal_depths = [50, 100, 200, 400, 800, 1600]  # in mm
for depth in focal_depths:
    os.makedirs(f'sim_results/focal_{depth}mm', exist_ok=True)

# Demonstrate grid optimization for FFT operations
if GRID_OPTIMIZER_AVAILABLE:
    print("\nDemonstrating grid optimization for FFT operations:")
    # Example grid sizes that might be used in simulations
    example_grids = [
        (128, 128, 128),  # Powers of 2 (optimal)
        (100, 100, 100),  # Round numbers (not optimal)
        (130, 130, 166),  # From the warning message
        (131, 131, 167)   # Prime numbers (worst case)
    ]

    # Smaller PML size for faster simulation
    pml_size = 8  # Reduced from 10 to 8 to match other scripts

    for grid in example_grids:
        Nx, Ny, Nz = grid
        highest_primes = get_highest_prime_factors(Nx, Ny, Nz)
        print(f"\nOriginal grid: {Nx} x {Ny} x {Nz}")
        print(f"Highest prime factors: {highest_primes}")

        # Calculate grid with PML added (as k-wave would do internally)
        Nx_with_pml = Nx + 2 * pml_size
        Ny_with_pml = Ny + 2 * pml_size
        Nz_with_pml = Nz + 2 * pml_size
        highest_primes_with_pml = get_highest_prime_factors(Nx_with_pml, Ny_with_pml, Nz_with_pml)
        print(f"Grid with PML: {Nx_with_pml} x {Ny_with_pml} x {Nz_with_pml}")
        print(f"Highest prime factors with PML: {highest_primes_with_pml}")

        # Standard optimization (without considering PML)
        opt_Nx, opt_Ny, opt_Nz = optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=10)
        opt_highest_primes = get_highest_prime_factors(opt_Nx, opt_Ny, opt_Nz)

        print(f"Standard optimized grid: {opt_Nx} x {opt_Ny} x {opt_Nz}")
        print(f"Standard optimized highest prime factors: {opt_highest_primes}")

        # Calculate what happens when k-wave adds PML to our optimized grid
        opt_Nx_with_pml = opt_Nx + 2 * pml_size
        opt_Ny_with_pml = opt_Ny + 2 * pml_size
        opt_Nz_with_pml = opt_Nz + 2 * pml_size
        opt_highest_primes_with_pml = get_highest_prime_factors(opt_Nx_with_pml, opt_Ny_with_pml, opt_Nz_with_pml)
        print(f"Standard optimized grid with PML: {opt_Nx_with_pml} x {opt_Ny_with_pml} x {opt_Nz_with_pml}")
        print(f"Standard optimized highest prime factors with PML: {opt_highest_primes_with_pml}")

        # PML-aware optimization
        pml_opt_Nx, pml_opt_Ny, pml_opt_Nz = optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=10, pml_size=pml_size)
        pml_opt_highest_primes = get_highest_prime_factors(pml_opt_Nx, pml_opt_Ny, pml_opt_Nz)

        print(f"PML-aware optimized grid: {pml_opt_Nx} x {pml_opt_Ny} x {pml_opt_Nz}")
        print(f"PML-aware optimized highest prime factors: {pml_opt_highest_primes}")

        # Calculate what happens when k-wave adds PML to our PML-aware optimized grid
        pml_opt_Nx_with_pml = pml_opt_Nx + 2 * pml_size
        pml_opt_Ny_with_pml = pml_opt_Ny + 2 * pml_size
        pml_opt_Nz_with_pml = pml_opt_Nz + 2 * pml_size
        pml_opt_highest_primes_with_pml = get_highest_prime_factors(pml_opt_Nx_with_pml, pml_opt_Ny_with_pml, pml_opt_Nz_with_pml)
        print(f"PML-aware optimized grid with PML: {pml_opt_Nx_with_pml} x {pml_opt_Ny_with_pml} x {pml_opt_Nz_with_pml}")
        print(f"PML-aware optimized highest prime factors with PML: {pml_opt_highest_primes_with_pml}")
        print(f"Size increase: {(pml_opt_Nx/Nx - 1)*100:.1f}% x {(pml_opt_Ny/Ny - 1)*100:.1f}% x {(pml_opt_Nz/Nz - 1)*100:.1f}%")

# Physical parameters
frequency = 180e3  # 180 kHz
sound_speed_water = 1482.0  # m/s
wavelength = sound_speed_water / frequency  # m
print(f"Wavelength in water at {frequency/1e3} kHz: {wavelength*1000:.2f} mm")

# Transducer dimensions
array_width = 100e-3  # 100 mm in meters
array_height = 100e-3  # 100 mm in meters

# Element sizing based on wavelength
element_width = wavelength / 2  # Typical element width is λ/2
element_height = wavelength / 2  # Square elements
kerf = wavelength / 20  # Small gap between elements

# Calculate number of elements that fit in the array
num_elements_x = int(np.floor(array_width / (element_width + kerf)))
num_elements_y = int(np.floor(array_height / (element_height + kerf)))

# Adjust array dimensions to fit exact number of elements
actual_array_width = num_elements_x * (element_width + kerf) - kerf
actual_array_height = num_elements_y * (element_height + kerf) - kerf

print(f"Number of elements: {num_elements_x} x {num_elements_y} = {num_elements_x * num_elements_y} elements")
print(f"Element width: {element_width*1000:.2f} mm")
print(f"Element height: {element_height*1000:.2f} mm")
print(f"Kerf (spacing between elements): {kerf*1000:.2f} mm")
print(f"Actual array dimensions: {actual_array_width*1000:.2f} mm x {actual_array_height*1000:.2f} mm")

# Create element positions (center coordinates)
element_positions_x = np.linspace(-actual_array_width/2 + element_width/2,
                                 actual_array_width/2 - element_width/2,
                                 num_elements_x)
element_positions_y = np.linspace(-actual_array_height/2 + element_height/2,
                                 actual_array_height/2 - element_height/2,
                                 num_elements_y)

# Create meshgrid of element positions
element_pos_x, element_pos_y = np.meshgrid(element_positions_x, element_positions_y)
element_positions = np.column_stack((element_pos_x.flatten(), element_pos_y.flatten()))

# Function to calculate phase delays for focusing at a specific depth
def calculate_focus_phases(element_positions, focal_point, frequency, sound_speed):
    """
    Calculate phase delays for focusing at a specific point.

    Parameters:
    -----------
    element_positions : array
        Array of (x, y) positions for each element
    focal_point : array
        (x, y, z) coordinates of the focal point
    frequency : float
        Operating frequency in Hz
    sound_speed : float
        Speed of sound in the medium in m/s

    Returns:
    --------
    phases : array
        Phase delays for each element in radians
    """
    # Calculate distances from each element to the focal point
    distances = np.sqrt((element_positions[:, 0] - focal_point[0])**2 +
                      (element_positions[:, 1] - focal_point[1])**2 +
                      focal_point[2]**2)

    # Find the maximum distance
    max_distance = np.max(distances)

    # Calculate time delays (negative because we want to delay elements that are closer)
    time_delays = (max_distance - distances) / sound_speed

    # Convert time delays to phase delays
    phase_delays = 2 * np.pi * frequency * time_delays

    # Normalize phases to [0, 2π)
    phase_delays = phase_delays % (2 * np.pi)

    return phase_delays

# Generate signal
def generate_tone_burst(frequency, num_cycles, sample_rate, envelope='hanning'):
    """
    Generate a tone burst signal.

    Parameters:
    -----------
    frequency : float
        Signal frequency in Hz
    num_cycles : int
        Number of cycles in the burst
    sample_rate : float
        Sampling rate in Hz
    envelope : str
        Type of envelope ('hanning', 'hamming', 'rectangular')

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values
    """
    duration = num_cycles / frequency
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    signal = np.sin(2 * np.pi * frequency * t)

    # Apply envelope
    if envelope == 'hanning':
        window = np.hanning(num_samples)
    elif envelope == 'hamming':
        window = np.hamming(num_samples)
    elif envelope == 'rectangular':
        window = np.ones(num_samples)
    else:
        window = np.ones(num_samples)

    signal = signal * window

    return t, signal

# Generate continuous wave signal
def generate_continuous_wave(frequency, duration, sample_rate):
    """
    Generate a continuous wave signal.

    Parameters:
    -----------
    frequency : float
        Signal frequency in Hz
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz

    Returns:
    --------
    t : array
        Time array
    signal : array
        Signal values
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    signal = np.sin(2 * np.pi * frequency * t)

    return t, signal

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
def generate_dual_frequency(f1, f2, amplitude_ratio=1.0, duration=None, sample_rate=None, phase1=0, phase2=0):
    """
    Generate a signal with two frequency components.

    Parameters:
    -----------
    f1 : float
        First frequency component in Hz
    f2 : float
        Second frequency component in Hz
    amplitude_ratio : float
        Ratio of amplitude of f2 to f1 (default: 1.0 for equal amplitudes)
    duration : float
        Signal duration in seconds
    sample_rate : float
        Sampling rate in Hz
    phase1 : float
        Phase offset for first frequency in radians
    phase2 : float
        Phase offset for second frequency in radians

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

# Plot transducer configuration
def plot_transducer_config(element_positions, element_width, element_height, kerf, array_width, array_height, filename):
    """
    Plot the transducer configuration with a side-by-side view of the full array and a zoomed-in element.

    Parameters:
    -----------
    element_positions : array
        Array of (x, y) positions for each element
    element_width : float
        Width of each element in meters
    element_height : float
        Height of each element in meters
    kerf : float
        Spacing between elements in meters
    array_width : float
        Width of the array in meters
    array_height : float
        Height of the array in meters
    filename : str
        Filename to save the plot
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # Left subplot: Full array
    # Plot each element
    for pos in element_positions:
        rect = patches.Rectangle(
            (pos[0] - element_width/2, pos[1] - element_height/2),
            element_width, element_height,
            linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax1.add_patch(rect)

    # Set axis limits with some padding
    padding = max(element_width, element_height) * 2
    ax1.set_xlim(-array_width/2 - padding/2, array_width/2 + padding/2)
    ax1.set_ylim(-array_height/2 - padding/2, array_height/2 + padding/2)

    # Add labels and title
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Y Position (mm)', fontsize=12)
    ax1.set_title('2D Transducer Array Configuration', fontsize=14)

    # Convert axis ticks to mm
    ax1.set_xticks(np.linspace(-array_width/2, array_width/2, 7))
    ax1.set_yticks(np.linspace(-array_height/2, array_height/2, 7))
    ax1.set_xticklabels([f'{x*1000:.1f}' for x in ax1.get_xticks()])
    ax1.set_yticklabels([f'{y*1000:.1f}' for y in ax1.get_yticks()])

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add a scale bar
    scale_bar_length = 10e-3  # 10 mm
    ax1.plot([array_width/2 - scale_bar_length - padding/4, array_width/2 - padding/4],
            [-array_height/2 + padding/4, -array_height/2 + padding/4], 'k-', linewidth=3)
    ax1.text(array_width/2 - scale_bar_length/2 - padding/4, -array_height/2 + padding/3,
            f'{scale_bar_length*1000:.0f} mm', ha='center', va='bottom')

    # Add a box to show the zoomed area (center of the array)
    zoom_width = 3 * (element_width + kerf)
    zoom_height = 3 * (element_height + kerf)
    zoom_rect = patches.Rectangle(
        (-zoom_width/2, -zoom_height/2),
        zoom_width, zoom_height,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax1.add_patch(zoom_rect)

    # Add an arrow connecting the zoom box to the zoomed view
    ax1.annotate('', xy=(array_width/2 + padding/4, 0), xytext=(zoom_width/2, 0),
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5))

    # Right subplot: Zoomed-in view of elements
    # Calculate the center positions of a few elements (3x3 grid)
    center_x = 0
    center_y = 0
    element_grid_size = 3
    element_positions_zoomed = []
    for i in range(-element_grid_size//2, element_grid_size//2 + 1):
        for j in range(-element_grid_size//2, element_grid_size//2 + 1):
            x = center_x + i * (element_width + kerf)
            y = center_y + j * (element_height + kerf)
            element_positions_zoomed.append((x, y))

    # Plot each element in the zoomed view
    for pos in element_positions_zoomed:
        rect = patches.Rectangle(
            (pos[0] - element_width/2, pos[1] - element_height/2),
            element_width, element_height,
            linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7
        )
        ax2.add_patch(rect)

    # Set axis limits for zoomed view
    ax2.set_xlim(-zoom_width/2 - element_width/2, zoom_width/2 + element_width/2)
    ax2.set_ylim(-zoom_height/2 - element_height/2, zoom_height/2 + element_height/2)

    # Add labels and title
    ax2.set_xlabel('X Position (mm)', fontsize=12)
    ax2.set_ylabel('Y Position (mm)', fontsize=12)
    ax2.set_title('Zoomed View of Transducer Elements', fontsize=14)

    # Convert axis ticks to mm
    ax2.set_xticks(np.linspace(-zoom_width/2, zoom_width/2, 5))
    ax2.set_yticks(np.linspace(-zoom_height/2, zoom_height/2, 5))
    ax2.set_xticklabels([f'{x*1000:.1f}' for x in ax2.get_xticks()])
    ax2.set_yticklabels([f'{y*1000:.1f}' for y in ax2.get_yticks()])

    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add dimension lines and labels for element width, height, and kerf
    # Element width dimension line
    center_element_pos = element_positions_zoomed[len(element_positions_zoomed)//2]  # Center element
    x_start = center_element_pos[0] - element_width/2
    x_end = center_element_pos[0] + element_width/2
    y_pos = center_element_pos[1] - element_height/2 - element_height/4

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text((x_start + x_end)/2, y_pos - element_height/8,
            f'Element Width\n{element_width*1000:.2f} mm',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Element height dimension line
    x_pos = center_element_pos[0] + element_width/2 + element_width/4
    y_start = center_element_pos[1] - element_height/2
    y_end = center_element_pos[1] + element_height/2

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_pos, y_start), xytext=(x_pos, y_end),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text(x_pos + element_width/8, (y_start + y_end)/2,
            f'Element Height\n{element_height*1000:.2f} mm',
            ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Kerf dimension line (horizontal)
    right_element_pos = element_positions_zoomed[len(element_positions_zoomed)//2 + 1]  # Element to the right of center
    x_start = center_element_pos[0] + element_width/2
    x_end = right_element_pos[0] - element_width/2
    y_pos = center_element_pos[1]

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax2.text((x_start + x_end)/2, y_pos + element_height/8,
            f'Kerf\n{kerf*1000:.2f} mm',
            ha='center', va='bottom', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add pitch dimension (element width + kerf)
    pitch = element_width + kerf
    x_start = center_element_pos[0] - element_width/2
    x_end = right_element_pos[0] - element_width/2
    y_pos = center_element_pos[1] + element_height/2 + element_height/4

    # Draw dimension line with arrows
    ax2.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax2.text((x_start + x_end)/2, y_pos + element_height/8,
            f'Pitch\n{pitch*1000:.2f} mm',
            ha='center', va='bottom', fontsize=10, color='blue',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add annotations with key parameters
    plt.figtext(0.5, 0.01,
                f'Array dimensions: {array_width*1000:.1f} mm × {array_height*1000:.1f} mm\n'
                f'Element dimensions: {element_width*1000:.2f} mm × {element_height*1000:.2f} mm\n'
                f'Kerf (spacing): {kerf*1000:.2f} mm, Pitch: {pitch*1000:.2f} mm\n'
                f'Number of elements: {len(element_positions)} ({int(np.sqrt(len(element_positions)))} × {int(np.sqrt(len(element_positions)))})\n'
                f'Frequency: {frequency/1e3:.1f} kHz, Wavelength: {wavelength*1000:.2f} mm\n'
                f'Element width/wavelength ratio: {element_width/(wavelength):.2f}',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot transducer with phase information
def plot_transducer_phases(element_positions, element_width, element_height, phases, focal_depth, filename):
    """
    Plot the transducer configuration with phase information and beam visualization.

    Parameters:
    -----------
    element_positions : array
        Array of (x, y) positions for each element
    element_width : float
        Width of each element in meters
    element_height : float
        Height of each element in meters
    phases : array
        Phase values for each element in radians
    focal_depth : float
        Focal depth in meters
    filename : str
        Filename to save the plot
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.2]})

    # Left subplot: Transducer array with phase information
    # Create a colormap for phases
    norm = Normalize(vmin=0, vmax=2*np.pi)
    cmap = cm.hsv

    # Plot each element with color based on phase
    for pos, phase in zip(element_positions, phases):
        rect = patches.Rectangle(
            (pos[0] - element_width/2, pos[1] - element_height/2),
            element_width, element_height,
            linewidth=1, edgecolor='black', facecolor=cmap(norm(phase)), alpha=0.7
        )
        ax1.add_patch(rect)

    # Set axis limits with some padding
    padding = max(element_width, element_height) * 2
    ax1.set_xlim(-actual_array_width/2 - padding/2, actual_array_width/2 + padding/2)
    ax1.set_ylim(-actual_array_height/2 - padding/2, actual_array_height/2 + padding/2)

    # Add labels and title
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Y Position (mm)', fontsize=12)
    ax1.set_title(f'2D Transducer Array Phase Configuration\nFocal Depth: {focal_depth*1000:.0f} mm', fontsize=14)

    # Convert axis ticks to mm
    ax1.set_xticks(np.linspace(-actual_array_width/2, actual_array_width/2, 7))
    ax1.set_yticks(np.linspace(-actual_array_height/2, actual_array_height/2, 7))
    ax1.set_xticklabels([f'{x*1000:.1f}' for x in ax1.get_xticks()])
    ax1.set_yticklabels([f'{y*1000:.1f}' for y in ax1.get_yticks()])

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, ticks=np.linspace(0, 2*np.pi, 5))
    cbar.set_label('Phase (radians)', fontsize=12)
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Right subplot: Beam visualization with focal point
    # Create a simplified beam visualization
    # Define the beam path
    z_max = focal_depth * 1.5  # Extend beyond focal point
    z = np.linspace(0, z_max, 1000)

    # Calculate beam width at each z position (simplified model)
    # Using a simple model where the beam converges to the focal point and then diverges
    array_width_half = actual_array_width/2
    array_height_half = actual_array_height/2

    # Calculate beam width at each z position
    beam_width = np.zeros_like(z)

    for i, z_val in enumerate(z):
        if z_val == 0:
            # At z=0, beam width should exactly match the transducer width
            beam_width[i] = array_width_half
        elif z_val < focal_depth:
            # Converging beam - linear convergence to focal point
            beam_width[i] = array_width_half * (1 - z_val/focal_depth)
        else:
            # Diverging beam (slower divergence than convergence)
            beam_width[i] = array_width_half * 0.5 * (z_val/focal_depth - 1)

    # Ensure minimum beam width
    beam_width = np.maximum(beam_width, array_width_half * 0.05)  # Minimum beam width

    # Plot the beam outline
    ax2.fill_betweenx(z * 1000, -beam_width * 1000, beam_width * 1000,
                     color='lightblue', alpha=0.3, label='Beam')
    ax2.plot(beam_width * 1000, z * 1000, 'b-', linewidth=1.5)
    ax2.plot(-beam_width * 1000, z * 1000, 'b-', linewidth=1.5)

    # Plot the transducer array at z=0
    ax2.plot([-actual_array_width/2 * 1000, actual_array_width/2 * 1000], [0, 0], 'k-', linewidth=3)

    # Plot the focal point
    ax2.plot(0, focal_depth * 1000, 'ro', markersize=8, label='Focal Point')

    # Add beam centerline
    ax2.plot([0, 0], [0, z_max * 1000], 'k--', linewidth=1, alpha=0.7, label='Beam Axis')

    # Calculate near field distance (Fresnel zone)
    # For a rectangular aperture, use the average of width and height
    array_radius = np.sqrt((array_width_half**2 + array_height_half**2) / 2)
    near_field = (array_radius**2) / wavelength

    # Add near field marker
    if near_field < z_max:
        ax2.axhline(y=near_field * 1000, color='g', linestyle='--', linewidth=1.5,
                   label=f'Near Field Boundary: {near_field*1000:.1f} mm')

    # Add labels and title
    ax2.set_xlabel('Lateral Position (mm)', fontsize=12)
    ax2.set_ylabel('Axial Distance (mm)', fontsize=12)
    ax2.set_title('Ultrasound Beam Visualization', fontsize=14)

    # Set axis limits
    ax2.set_xlim(-actual_array_width * 1000, actual_array_width * 1000)
    ax2.set_ylim(-5, z_max * 1000 * 1.05)

    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax2.legend(loc='upper right', fontsize=10)

    # Add annotations with key parameters
    plt.figtext(0.5, 0.01,
                f'Array dimensions: {actual_array_width*1000:.1f} mm × {actual_array_height*1000:.1f} mm\n'
                f'Element dimensions: {element_width*1000:.2f} mm × {element_height*1000:.2f} mm\n'
                f'Frequency: {frequency/1e3:.1f} kHz, Wavelength: {wavelength*1000:.2f} mm\n'
                f'Focal point: (0, 0, {focal_depth*1000:.0f} mm), F-number: {focal_depth/array_radius:.1f}\n'
                f'Near field distance: {near_field*1000:.1f} mm',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot signal in time and frequency domain
def plot_signal(t, signal, frequency, filename_time, filename_freq, filename_spectrogram=None):
    """
    Plot the signal in time and frequency domain and optionally a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    frequency : float
        Signal frequency in Hz
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str, optional
        Filename to save the spectrogram plot
    """
    # Create a figure with 2 or 3 subplots depending on whether spectrogram is requested
    if filename_spectrogram:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Time domain
        ax2 = fig.add_subplot(gs[0, 1])  # Frequency domain
        ax3 = fig.add_subplot(gs[1, :])  # Spectrogram (full width)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time domain plot (ax1)
    # Plot signal
    ax1.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    # Calculate and plot instantaneous phase
    phase = np.unwrap(np.angle(analytic_signal))
    phase_norm = (phase - phase.min()) / (phase.max() - phase.min()) * 2 - 1  # Normalize to [-1, 1]

    # Add labels and title
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'Signal Waveform at {frequency/1e3:.1f} kHz', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # Add cycles count and period markers
    period = 1.0 / frequency  # Period in seconds
    num_cycles_shown = min(5, int(t[-1] * frequency))  # Show markers for up to 5 cycles

    for i in range(num_cycles_shown):
        cycle_start = i * period
        if cycle_start < t[-1]:
            ax1.axvline(x=cycle_start * 1e6, color='gray', linestyle=':', alpha=0.5)
            if i > 0:  # Skip labeling the first line at t=0
                ax1.text(cycle_start * 1e6, ax1.get_ylim()[0] * 0.9, f'{i}T',
                        ha='center', va='bottom', fontsize=8, alpha=0.7)

    # Add wavelength annotation
    ax1.annotate(f'Period (T) = {period*1e6:.2f} μs',
                xy=(1.5*period*1e6, 0),
                xytext=(1.5*period*1e6, ax1.get_ylim()[1]*0.8),
                arrowprops=dict(arrowstyle='->', color='gray'),
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Frequency domain plot (ax2)
    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the signal frequency
    positive_freq_idx = np.where((xf >= 0) & (xf <= 3*frequency))
    ax2.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    # Add labels and title
    ax2.set_xlabel('Frequency (kHz)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Signal Frequency Spectrum', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add vertical line at the fundamental frequency
    ax2.axvline(x=frequency/1e3, color='r', linestyle='--', alpha=0.7)
    ax2.text(frequency/1e3, ax2.get_ylim()[1]*0.9, f'{frequency/1e3:.1f} kHz',
            ha='center', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add bandwidth markers if applicable (for tone burst)
    if np.std(envelope) > 0.01:  # Check if it's not a continuous wave
        # Estimate bandwidth from spectrum
        magnitude = 2.0/n * np.abs(yf[positive_freq_idx])
        max_mag = np.max(magnitude)
        half_power_level = max_mag / 2  # -3dB point

        # Find frequencies at half power
        above_half_power = magnitude > half_power_level
        if np.sum(above_half_power) > 1:  # If we have points above half power
            f_indices = np.where(above_half_power)[0]
            f_low = xf[positive_freq_idx][f_indices[0]] / 1e3
            f_high = xf[positive_freq_idx][f_indices[-1]] / 1e3
            bandwidth = f_high - f_low

            # Add markers for bandwidth
            ax2.axvline(x=f_low, color='g', linestyle=':', alpha=0.7)
            ax2.axvline(x=f_high, color='g', linestyle=':', alpha=0.7)
            ax2.axhline(y=half_power_level, color='g', linestyle=':', alpha=0.7)

            # Add bandwidth annotation
            ax2.annotate(f'Bandwidth (-3dB): {bandwidth:.1f} kHz',
                        xy=((f_low + f_high)/2, half_power_level),
                        xytext=((f_low + f_high)/2, half_power_level*2),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        ha='center', va='bottom', fontsize=10, color='green',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add spectrogram if requested
    if filename_spectrogram:
        # Calculate spectrogram using scipy.signal
        from scipy import signal as sig
        nperseg = min(256, len(t) // 10)  # Number of points per segment
        f_sample = 1.0/(t[1]-t[0])  # Sample frequency

        # Calculate spectrogram
        f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

        # Convert to dB scale, avoiding log of zero
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Plot spectrogram
        im = ax3.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

        # Add labels and title
        ax3.set_xlabel('Time (μs)', fontsize=12)
        ax3.set_ylabel('Frequency (kHz)', fontsize=12)
        ax3.set_title(f'Spectrogram of {frequency/1e3:.1f} kHz Signal', fontsize=14)

        # Set y-axis limits to focus on the frequency range of interest
        ax3.set_ylim([0, frequency/1e3 * 3])  # Show up to 3x the fundamental frequency

        # Add horizontal line at the fundamental frequency
        ax3.axhline(y=frequency/1e3, color='r', linestyle='--', alpha=0.7)

        # Add text label
        ax3.text(ax3.get_xlim()[1]*0.95, frequency/1e3, f'{frequency/1e3:.1f} kHz',
                ha='right', va='bottom', color='r', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add annotations with key parameters
    plt.figtext(0.5, 0.01,
                f'Frequency: {frequency/1e3:.1f} kHz\n'
                f'Wavelength in water: {wavelength*1000:.2f} mm\n'
                f'Period: {period*1e6:.2f} μs\n'
                f'Signal duration: {t[-1]*1e6:.1f} μs\n'
                f'Cycles: {t[-1]*frequency:.1f}',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time if not filename_spectrogram else filename_spectrogram, dpi=300, bbox_inches='tight')

    # If we're creating a combined plot with spectrogram, also save individual plots
    if filename_spectrogram:
        plt.close()

        # Save individual time domain plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')
        ax.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Signal Waveform at {frequency/1e3:.1f} kHz', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(filename_time, dpi=300, bbox_inches='tight')
        plt.close()

        # Save individual frequency domain plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)
        ax.set_xlabel('Frequency (kHz)', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_title('Signal Frequency Spectrum', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axvline(x=frequency/1e3, color='r', linestyle='--', alpha=0.7)
        ax.text(frequency/1e3, ax.get_ylim()[1]*0.9, f'{frequency/1e3:.1f} kHz',
                ha='center', va='top', color='r', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.close()

# Plot dual frequency signal
def plot_dual_frequency_signal(t, signal, f1, f2, filename_time, filename_freq, filename_spectrogram=None):
    """
    Plot the dual frequency signal in time and frequency domains, plus optionally a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    f1 : float
        First frequency component in Hz
    f2 : float
        Second frequency component in Hz
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str, optional
        Filename to save the spectrogram plot
    """
    # Time domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot signal
    ax.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Dual Frequency Signal ({f1/1e3:.1f} kHz + {f2/1e3:.1f} kHz)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Add annotations
    plt.figtext(0.5, 0.01,
                f'Frequencies: {f1/1e3:.1f} kHz + {f2/1e3:.1f} kHz\n'
                f'Wavelengths: {(sound_speed_water/f1)*1000:.2f} mm + {(sound_speed_water/f2)*1000:.2f} mm\n'
                f'Duration: {t[-1]*1e6:.1f} μs',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time, dpi=300, bbox_inches='tight')
    plt.close()

    # Frequency domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the max frequency
    max_freq = max(f1, f2)
    positive_freq_idx = np.where((xf >= 0) & (xf <= 3*max_freq))
    ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Dual Frequency Spectrum', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at the two frequencies
    ax.axvline(x=f1/1e3, color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=f2/1e3, color='g', linestyle='--', alpha=0.7)

    # Add text labels
    ax.text(f1/1e3, ax.get_ylim()[1]*0.9, f'{f1/1e3:.1f} kHz',
            ha='center', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f2/1e3, ax.get_ylim()[1]*0.9, f'{f2/1e3:.1f} kHz',
            ha='center', va='top', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate spectrogram if filename is provided
    if filename_spectrogram:
        # Spectrogram plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate spectrogram using scipy.signal
        from scipy import signal as sig
        nperseg = min(256, len(t) // 10)  # Number of points per segment
        f_sample = 1.0/(t[1]-t[0])  # Sample frequency

        # Calculate spectrogram
        f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

        # Convert to dB scale, avoiding log of zero
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Plot spectrogram
        im = ax.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

        # Add labels and title
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Frequency (kHz)', fontsize=12)
        ax.set_title(f'Spectrogram of Dual Frequency Signal', fontsize=14)

        # Set y-axis limits to focus on the frequency range of interest
        ax.set_ylim([0, max(f1, f2)/1e3 * 3])  # Show up to 3x the max frequency

        # Add horizontal lines at the two frequencies
        ax.axhline(y=f1/1e3, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=f2/1e3, color='g', linestyle='--', alpha=0.7)

        # Add text labels
        ax.text(ax.get_xlim()[1]*0.95, f1/1e3, f'{f1/1e3:.1f} kHz',
                ha='right', va='bottom', color='r', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(ax.get_xlim()[1]*0.95, f2/1e3, f'{f2/1e3:.1f} kHz',
                ha='right', va='bottom', color='g', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename_spectrogram, dpi=300, bbox_inches='tight')
        plt.close()

# Plot dual frequency sweep signal
def plot_dual_frequency_sweep(t, signal, f1_inst, f2_inst, f1_center, f2_center, bandwidth_percent,
                           filename_time, filename_freq, filename_spectrogram):
    """
    Plot the dual frequency sweep signal in time and frequency domains, plus a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    f1_inst : array
        Instantaneous frequency of first component at each time point
    f2_inst : array
        Instantaneous frequency of second component at each time point
    f1_center : float
        Center frequency of first component in Hz
    f2_center : float
        Center frequency of second component in Hz
    bandwidth_percent : float
        Bandwidth as percentage of center frequency
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str
        Filename to save the spectrogram plot
    """
    # Time domain plot with instantaneous frequencies
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot signal
    ax1.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    # Add second y-axis for frequencies
    ax2 = ax1.twinx()
    ax2.plot(t * 1e6, f1_inst / 1e3, 'g-', linewidth=1.5, label=f'Freq 1 ({f1_center/1e3:.1f} kHz)')
    ax2.plot(t * 1e6, f2_inst / 1e3, 'm-', linewidth=1.5, label=f'Freq 2 ({f2_center/1e3:.1f} kHz)')
    ax2.set_ylabel('Frequency (kHz)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add title and legend
    plt.title(f'Dual Frequency Sweep Signal', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add annotations
    bandwidth1 = f1_center * (bandwidth_percent / 100)
    f1_min = max(f1_center - bandwidth1 / 2, 1.0)
    f1_max = f1_center + bandwidth1 / 2

    bandwidth2 = f2_center * (bandwidth_percent / 100)
    f2_min = max(f2_center - bandwidth2 / 2, 1.0)
    f2_max = f2_center + bandwidth2 / 2

    plt.figtext(0.5, 0.01,
                f'Frequency 1: {f1_center/1e3:.1f} kHz (Range: {f1_min/1e3:.1f} - {f1_max/1e3:.1f} kHz)\n'
                f'Frequency 2: {f2_center/1e3:.1f} kHz (Range: {f2_min/1e3:.1f} - {f2_max/1e3:.1f} kHz)\n'
                f'Bandwidth: {bandwidth_percent}%\n'
                f'Duration: {t[-1]*1e6:.1f} μs',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time, dpi=300, bbox_inches='tight')
    plt.close()

    # Frequency domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the max frequency
    max_freq = max(f2_max, f1_max)
    positive_freq_idx = np.where((xf >= 0) & (xf <= 3*max_freq))
    ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Dual Frequency Sweep Spectrum', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at min, center, and max frequencies for both components
    ax.axvline(x=f1_min/1e3, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=f1_center/1e3, color='r', linestyle='-', alpha=0.7)
    ax.axvline(x=f1_max/1e3, color='r', linestyle='--', alpha=0.5)

    ax.axvline(x=f2_min/1e3, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=f2_center/1e3, color='g', linestyle='-', alpha=0.7)
    ax.axvline(x=f2_max/1e3, color='g', linestyle='--', alpha=0.5)

    # Add text labels
    ax.text(f1_center/1e3, ax.get_ylim()[1]*0.9, f'{f1_center/1e3:.1f} kHz',
            ha='center', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f2_center/1e3, ax.get_ylim()[1]*0.9, f'{f2_center/1e3:.1f} kHz',
            ha='center', va='top', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
    plt.close()

    # Spectrogram plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate spectrogram using scipy.signal
    from scipy import signal as sig
    nperseg = min(256, len(t) // 10)  # Number of points per segment
    f_sample = 1.0/(t[1]-t[0])  # Sample frequency

    # Calculate spectrogram
    f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

    # Convert to dB scale, avoiding log of zero
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plot spectrogram
    im = ax.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

    # Add labels and title
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    ax.set_title('Spectrogram of Dual Frequency Sweep Signal', fontsize=14)

    # Set y-axis limits to focus on the frequency range of interest
    ax.set_ylim([0, max(f1_max, f2_max)/1e3 * 1.5])

    # Add horizontal lines at min, center, and max frequencies for both components
    ax.axhline(y=f1_min/1e3, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=f1_center/1e3, color='r', linestyle='-', alpha=0.7)
    ax.axhline(y=f1_max/1e3, color='r', linestyle='--', alpha=0.5)

    ax.axhline(y=f2_min/1e3, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=f2_center/1e3, color='g', linestyle='-', alpha=0.7)
    ax.axhline(y=f2_max/1e3, color='g', linestyle='--', alpha=0.5)

    # Add text labels
    ax.text(ax.get_xlim()[1]*0.95, f1_center/1e3, f'{f1_center/1e3:.1f} kHz',
            ha='right', va='center', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(ax.get_xlim()[1]*0.95, f2_center/1e3, f'{f2_center/1e3:.1f} kHz',
            ha='right', va='center', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_spectrogram, dpi=300, bbox_inches='tight')
    plt.close()

# Plot frequency sweep signal
def plot_sweep_signal(t, signal, f_inst, f_center, bandwidth_percent, filename_time, filename_freq, filename_spectrogram):
    """
    Plot the frequency sweep signal in time and frequency domains, plus a spectrogram.

    Parameters:
    -----------
    t : array
        Time array
    signal : array
        Signal values
    f_inst : array
        Instantaneous frequency at each time point
    f_center : float
        Center frequency in Hz
    bandwidth_percent : float
        Bandwidth as percentage of center frequency
    filename_time : str
        Filename to save the time domain plot
    filename_freq : str
        Filename to save the frequency domain plot
    filename_spectrogram : str
        Filename to save the spectrogram plot
    """
    # Time domain plot with instantaneous frequency
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot signal
    ax1.plot(t * 1e6, signal, 'b-', linewidth=2, label='Signal')
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Calculate and plot envelope
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t * 1e6, envelope, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')

    # Add second y-axis for frequency
    ax2 = ax1.twinx()
    ax2.plot(t * 1e6, f_inst / 1e3, 'g-', linewidth=1.5, label='Instantaneous Frequency')
    ax2.set_ylabel('Frequency (kHz)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add title and legend
    plt.title(f'Frequency Sweep Signal (Center: {f_center/1e3:.1f} kHz, Bandwidth: {bandwidth_percent}%)', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add annotations
    bandwidth = f_center * (bandwidth_percent / 100)
    f_min = max(f_center - bandwidth / 2, 1.0)
    f_max = f_center + bandwidth / 2

    plt.figtext(0.5, 0.01,
                f'Center Frequency: {f_center/1e3:.1f} kHz\n'
                f'Bandwidth: {bandwidth/1e3:.1f} kHz ({bandwidth_percent}%)\n'
                f'Frequency Range: {f_min/1e3:.1f} - {f_max/1e3:.1f} kHz\n'
                f'Duration: {t[-1]*1e6:.1f} μs',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(filename_time, dpi=300, bbox_inches='tight')
    plt.close()

    # Frequency domain plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute FFT
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, t[1] - t[0])

    # Plot only positive frequencies up to 3x the max frequency
    max_plot_freq = 3 * f_max
    positive_freq_idx = np.where((xf >= 0) & (xf <= max_plot_freq))
    ax.plot(xf[positive_freq_idx] / 1e3, 2.0/n * np.abs(yf[positive_freq_idx]), 'b-', linewidth=2)

    ax.set_xlabel('Frequency (kHz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Frequency Sweep Spectrum', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines at min, center, and max frequencies
    ax.axvline(x=f_min/1e3, color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=f_center/1e3, color='g', linestyle='--', alpha=0.7)
    ax.axvline(x=f_max/1e3, color='r', linestyle='--', alpha=0.7)

    # Add text labels
    ax.text(f_min/1e3, ax.get_ylim()[1]*0.9, f'{f_min/1e3:.1f} kHz',
            ha='right', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f_center/1e3, ax.get_ylim()[1]*0.9, f'{f_center/1e3:.1f} kHz',
            ha='center', va='top', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(f_max/1e3, ax.get_ylim()[1]*0.9, f'{f_max/1e3:.1f} kHz',
            ha='left', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_freq, dpi=300, bbox_inches='tight')
    plt.close()

    # Spectrogram plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate spectrogram using scipy.signal
    from scipy import signal as sig
    nperseg = min(256, len(t) // 10)  # Number of points per segment
    f_sample = 1.0/(t[1]-t[0])  # Sample frequency

    # Calculate spectrogram
    f, t_spec, Sxx = sig.spectrogram(signal, fs=f_sample, nperseg=nperseg, noverlap=nperseg//2)

    # Convert to dB scale, avoiding log of zero
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plot spectrogram
    im = ax.pcolormesh(t_spec * 1e6, f / 1e3, Sxx_db, shading='auto', cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)

    # Add labels and title
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    ax.set_title('Spectrogram of Frequency Sweep Signal', fontsize=14)

    # Set y-axis limits to focus on the frequency range of interest
    ax.set_ylim([0, f_max/1e3 * 1.5])

    # Add horizontal lines at min, center, and max frequencies
    ax.axhline(y=f_min/1e3, color='r', linestyle='--', alpha=0.7)
    ax.axhline(y=f_center/1e3, color='g', linestyle='--', alpha=0.7)
    ax.axhline(y=f_max/1e3, color='r', linestyle='--', alpha=0.7)

    # Add text labels
    ax.text(ax.get_xlim()[1]*0.95, f_min/1e3, f'{f_min/1e3:.1f} kHz',
            ha='right', va='bottom', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(ax.get_xlim()[1]*0.95, f_center/1e3, f'{f_center/1e3:.1f} kHz',
            ha='right', va='center', color='g', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(ax.get_xlim()[1]*0.95, f_max/1e3, f'{f_max/1e3:.1f} kHz',
            ha='right', va='top', color='r', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename_spectrogram, dpi=300, bbox_inches='tight')
    plt.close()

# Create directories for signal results with consistent structure
os.makedirs('sim_results/single_freq', exist_ok=True)
os.makedirs('sim_results/single_freq/sweep_freq', exist_ok=True)
os.makedirs('sim_results/dual_freq', exist_ok=True)
os.makedirs('sim_results/dual_freq/sweep_freq', exist_ok=True)

# Signal parameters
sample_rate = 100 * frequency  # 100 samples per period
num_cycles = 10
signal_duration = 100 / frequency  # Same duration for all signals
bandwidth_percent = 40  # Default 40% bandwidth

# Second frequency for dual frequency signals
frequency2 = 550e3  # 550 kHz

# Generate signals
# Single frequency signals
t_burst, signal_burst = generate_tone_burst(frequency, num_cycles, sample_rate)
t_continuous, signal_continuous = generate_continuous_wave(frequency, signal_duration, sample_rate)

# Single frequency sweep
t_sweep, signal_sweep, f_inst = generate_frequency_sweep(frequency, bandwidth_percent, signal_duration, sample_rate)

# Dual frequency signals
t_dual, signal_dual = generate_dual_frequency(frequency, frequency2, 1.0, signal_duration, sample_rate)

# Dual frequency sweep
t_dual_sweep, signal_dual_sweep, f1_inst, f2_inst = generate_dual_frequency_sweep(
    frequency, frequency2, bandwidth_percent, signal_duration, sample_rate)

# Plot transducer configuration
plot_transducer_config(
    element_positions,
    element_width,
    element_height,
    kerf,
    actual_array_width,
    actual_array_height,
    'sim_results/transducer_configuration.png'
)

# Plot single frequency signals
plot_signal(
    t_burst,
    signal_burst,
    frequency,
    'sim_results/single_freq/signal_time_domain.png',
    'sim_results/single_freq/signal_frequency_domain.png',
    'sim_results/single_freq/signal_spectrogram.png'
)

# Plot single frequency sweep signals
plot_sweep_signal(
    t_sweep,
    signal_sweep,
    f_inst,
    frequency,
    bandwidth_percent,
    'sim_results/single_freq/sweep_freq/signal_time_domain.png',
    'sim_results/single_freq/sweep_freq/signal_frequency_domain.png',
    'sim_results/single_freq/sweep_freq/signal_spectrogram.png'
)

# Plot dual frequency signals
plot_dual_frequency_signal(
    t_dual,
    signal_dual,
    frequency,
    frequency2,
    'sim_results/dual_freq/signal_time_domain.png',
    'sim_results/dual_freq/signal_frequency_domain.png',
    'sim_results/dual_freq/signal_spectrogram.png'
)

# Plot dual frequency sweep signals
plot_dual_frequency_sweep(
    t_dual_sweep,
    signal_dual_sweep,
    f1_inst,
    f2_inst,
    frequency,
    frequency2,
    bandwidth_percent,
    'sim_results/dual_freq/sweep_freq/signal_time_domain.png',
    'sim_results/dual_freq/sweep_freq/signal_frequency_domain.png',
    'sim_results/dual_freq/sweep_freq/signal_spectrogram.png'
)

# Calculate and plot phase configurations for different focal depths
for depth_mm in focal_depths:
    depth = depth_mm * 1e-3  # Convert mm to m
    focal_point = np.array([0, 0, depth])
    phases = calculate_focus_phases(element_positions, focal_point, frequency, sound_speed_water)

    # Reshape phases to match the grid
    phases_grid = phases.reshape(num_elements_y, num_elements_x)

    # Plot transducer with phase information
    plot_transducer_phases(
        element_positions,
        element_width,
        element_height,
        phases,
        depth,
        f'sim_results/focal_{depth_mm}mm/phase_configuration.png'
    )

    # Create a 2D plot of the phase distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(phases_grid, cmap='hsv', origin='lower',
                  extent=[-actual_array_width/2*1000, actual_array_width/2*1000,
                          -actual_array_height/2*1000, actual_array_height/2*1000],
                  vmin=0, vmax=2*np.pi)

    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title(f'Phase Distribution for Focal Depth {depth_mm} mm', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ticks=np.linspace(0, 2*np.pi, 5))
    cbar.set_label('Phase (radians)', fontsize=12)
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout()
    plt.savefig(f'sim_results/focal_{depth_mm}mm/phase_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

print("All simulations and visualizations completed successfully.")
>>>>>>> 793ef48ece9336b7d7632f02e7da7317c59632a2
