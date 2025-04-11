<<<<<<< HEAD
"""
Ultrasound Signal Generator

This module provides functions for generating various ultrasound signals,
including single frequency, frequency sweeps, dual frequency, and dual
frequency sweeps.
"""

import numpy as np

class SignalGenerator:
    """Class for generating various ultrasound signals"""
    
    def __init__(self, sample_rate=20e6):
        """
        Initialize the signal generator
        
        Parameters:
        -----------
        sample_rate : float
            Sampling rate in Hz (default: 20 MHz)
        """
        self.sample_rate = sample_rate
    
    def tone_burst(self, frequency, num_cycles, envelope='hanning'):
        """
        Generate a tone burst signal
        
        Parameters:
        -----------
        frequency : float
            Signal frequency in Hz
        num_cycles : int
            Number of cycles in the burst
        envelope : str
            Type of envelope ('hanning', 'hamming', 'rectangular')
            
        Returns:
        --------
        tuple
            (t, signal) - time array and signal array
        """
        # Calculate signal duration
        duration = num_cycles / frequency
        
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate signal
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        if envelope == 'hanning':
            window = np.hanning(num_samples)
            signal = signal * window
        elif envelope == 'hamming':
            window = np.hamming(num_samples)
            signal = signal * window
        # For rectangular, no window is applied
        
        return t, signal
    
    def continuous_wave(self, frequency, duration, envelope='rectangular'):
        """
        Generate a continuous wave signal
        
        Parameters:
        -----------
        frequency : float
            Signal frequency in Hz
        duration : float
            Signal duration in seconds
        envelope : str
            Type of envelope ('hanning', 'hamming', 'rectangular')
            
        Returns:
        --------
        tuple
            (t, signal) - time array and signal array
        """
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate signal
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        if envelope == 'hanning':
            window = np.hanning(num_samples)
            signal = signal * window
        elif envelope == 'hamming':
            window = np.hamming(num_samples)
            signal = signal * window
        # For rectangular, no window is applied
        
        return t, signal
    
    def frequency_sweep(self, center_frequency, bandwidth_percent, duration, phase=0):
        """
        Generate a frequency sweep signal
        
        Parameters:
        -----------
        center_frequency : float
            Center frequency in Hz
        bandwidth_percent : float
            Bandwidth as a percentage of center frequency
        duration : float
            Signal duration in seconds
        phase : float
            Initial phase in radians
            
        Returns:
        --------
        tuple
            (t, signal, f_inst) - time array, signal array, and instantaneous frequency array
        """
        # Calculate frequency range
        bandwidth = center_frequency * bandwidth_percent / 100
        f_min = center_frequency - bandwidth / 2
        f_max = center_frequency + bandwidth / 2
        
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Calculate instantaneous frequency (linear sweep)
        f_inst = f_min + (f_max - f_min) * t / duration
        
        # Calculate phase by integrating frequency
        dt = 1 / self.sample_rate
        phase_t = phase + 2 * np.pi * np.cumsum(f_inst) * dt
        
        # Generate signal
        signal = np.sin(phase_t)
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(num_samples)
        signal = signal * window
        
        return t, signal, f_inst
    
    def dual_frequency(self, frequency1, frequency2, amplitude_ratio=1.0, duration=0.1, phase1=0, phase2=0):
        """
        Generate a dual frequency signal
        
        Parameters:
        -----------
        frequency1 : float
            First frequency in Hz
        frequency2 : float
            Second frequency in Hz
        amplitude_ratio : float
            Amplitude ratio of second frequency to first frequency
        duration : float
            Signal duration in seconds
        phase1 : float
            Initial phase of first frequency in radians
        phase2 : float
            Initial phase of second frequency in radians
            
        Returns:
        --------
        tuple
            (t, signal) - time array and signal array
        """
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate individual components
        signal1 = np.sin(2 * np.pi * frequency1 * t + phase1)
        signal2 = amplitude_ratio * np.sin(2 * np.pi * frequency2 * t + phase2)
        
        # Combine signals
        signal = (signal1 + signal2) / (1 + amplitude_ratio)  # Normalize to keep amplitude in [-1, 1]
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(num_samples)
        signal = signal * window
        
        return t, signal
    
    def dual_frequency_sweep(self, frequency1, frequency2, bandwidth_percent, duration, amplitude_ratio=1.0, phase1=0, phase2=0):
        """
        Generate a dual frequency sweep signal
        
        Parameters:
        -----------
        frequency1 : float
            First center frequency in Hz
        frequency2 : float
            Second center frequency in Hz
        bandwidth_percent : float
            Bandwidth as a percentage of center frequency
        duration : float
            Signal duration in seconds
        amplitude_ratio : float
            Amplitude ratio of second frequency to first frequency
        phase1 : float
            Initial phase of first frequency in radians
        phase2 : float
            Initial phase of second frequency in radians
            
        Returns:
        --------
        tuple
            (t, signal, f1_inst, f2_inst) - time array, signal array, and instantaneous frequency arrays
        """
        # Calculate frequency ranges
        bandwidth1 = frequency1 * bandwidth_percent / 100
        f1_min = frequency1 - bandwidth1 / 2
        f1_max = frequency1 + bandwidth1 / 2
        
        bandwidth2 = frequency2 * bandwidth_percent / 100
        f2_min = frequency2 - bandwidth2 / 2
        f2_max = frequency2 + bandwidth2 / 2
        
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Calculate instantaneous frequencies (linear sweeps)
        f1_inst = f1_min + (f1_max - f1_min) * t / duration
        f2_inst = f2_min + (f2_max - f2_min) * t / duration
        
        # Calculate phases by integrating frequencies
        dt = 1 / self.sample_rate
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
    
    def generate_signal(self, signal_type, **kwargs):
        """
        Generate a signal based on the specified type
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to generate ('single_freq', 'sweep_freq', 'dual_freq', 'dual_sweep_freq')
        **kwargs : dict
            Additional parameters for the specific signal type
            
        Returns:
        --------
        tuple
            Signal data (format depends on signal type)
        """
        if signal_type == 'single_freq':
            # Required parameters: frequency, duration
            frequency = kwargs.get('frequency', 180e3)
            duration = kwargs.get('duration', 0.1)
            return self.continuous_wave(frequency, duration)
        
        elif signal_type == 'sweep_freq':
            # Required parameters: frequency, bandwidth_percent, duration
            frequency = kwargs.get('frequency', 180e3)
            bandwidth_percent = kwargs.get('bandwidth_percent', 40)
            duration = kwargs.get('duration', 0.1)
            return self.frequency_sweep(frequency, bandwidth_percent, duration)
        
        elif signal_type == 'dual_freq':
            # Required parameters: frequency1, frequency2, duration
            frequency1 = kwargs.get('frequency1', 180e3)
            frequency2 = kwargs.get('frequency2', 550e3)
            amplitude_ratio = kwargs.get('amplitude_ratio', 1.0)
            duration = kwargs.get('duration', 0.1)
            return self.dual_frequency(frequency1, frequency2, amplitude_ratio, duration)
        
        elif signal_type == 'dual_sweep_freq':
            # Required parameters: frequency1, frequency2, bandwidth_percent, duration
            frequency1 = kwargs.get('frequency1', 180e3)
            frequency2 = kwargs.get('frequency2', 550e3)
            bandwidth_percent = kwargs.get('bandwidth_percent', 40)
            amplitude_ratio = kwargs.get('amplitude_ratio', 1.0)
            duration = kwargs.get('duration', 0.1)
            return self.dual_frequency_sweep(frequency1, frequency2, bandwidth_percent, duration, amplitude_ratio)
        
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create signal generator
    sg = SignalGenerator()
    
    # Generate signals
    t1, signal1 = sg.generate_signal('single_freq', frequency=180e3, duration=0.1)
    t2, signal2, f_inst = sg.generate_signal('sweep_freq', frequency=180e3, bandwidth_percent=40, duration=0.1)
    t3, signal3 = sg.generate_signal('dual_freq', frequency1=180e3, frequency2=550e3, duration=0.1)
    t4, signal4, f1_inst, f2_inst = sg.generate_signal('dual_sweep_freq', frequency1=180e3, frequency2=550e3, bandwidth_percent=40, duration=0.1)
    
    # Plot signals
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(t1 * 1000, signal1)
    plt.title('Single Frequency (180 kHz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 2)
    plt.plot(t2 * 1000, signal2)
    plt.title('Frequency Sweep (180 kHz ± 20%)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 3)
    plt.plot(t3 * 1000, signal3)
    plt.title('Dual Frequency (180 kHz + 550 kHz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 4)
    plt.plot(t4 * 1000, signal4)
    plt.title('Dual Frequency Sweep (180 kHz ± 20% + 550 kHz ± 20%)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('signal_examples.png', dpi=300)
    plt.show()
=======
"""
Ultrasound Signal Generator

This module provides functions for generating various ultrasound signals,
including single frequency, frequency sweeps, dual frequency, and dual
frequency sweeps.
"""

import numpy as np

class SignalGenerator:
    """Class for generating various ultrasound signals"""
    
    def __init__(self, sample_rate=20e6):
        """
        Initialize the signal generator
        
        Parameters:
        -----------
        sample_rate : float
            Sampling rate in Hz (default: 20 MHz)
        """
        self.sample_rate = sample_rate
    
    def tone_burst(self, frequency, num_cycles, envelope='hanning'):
        """
        Generate a tone burst signal
        
        Parameters:
        -----------
        frequency : float
            Signal frequency in Hz
        num_cycles : int
            Number of cycles in the burst
        envelope : str
            Type of envelope ('hanning', 'hamming', 'rectangular')
            
        Returns:
        --------
        tuple
            (t, signal) - time array and signal array
        """
        # Calculate signal duration
        duration = num_cycles / frequency
        
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate signal
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        if envelope == 'hanning':
            window = np.hanning(num_samples)
            signal = signal * window
        elif envelope == 'hamming':
            window = np.hamming(num_samples)
            signal = signal * window
        # For rectangular, no window is applied
        
        return t, signal
    
    def continuous_wave(self, frequency, duration, envelope='rectangular'):
        """
        Generate a continuous wave signal
        
        Parameters:
        -----------
        frequency : float
            Signal frequency in Hz
        duration : float
            Signal duration in seconds
        envelope : str
            Type of envelope ('hanning', 'hamming', 'rectangular')
            
        Returns:
        --------
        tuple
            (t, signal) - time array and signal array
        """
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate signal
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        if envelope == 'hanning':
            window = np.hanning(num_samples)
            signal = signal * window
        elif envelope == 'hamming':
            window = np.hamming(num_samples)
            signal = signal * window
        # For rectangular, no window is applied
        
        return t, signal
    
    def frequency_sweep(self, center_frequency, bandwidth_percent, duration, phase=0):
        """
        Generate a frequency sweep signal
        
        Parameters:
        -----------
        center_frequency : float
            Center frequency in Hz
        bandwidth_percent : float
            Bandwidth as a percentage of center frequency
        duration : float
            Signal duration in seconds
        phase : float
            Initial phase in radians
            
        Returns:
        --------
        tuple
            (t, signal, f_inst) - time array, signal array, and instantaneous frequency array
        """
        # Calculate frequency range
        bandwidth = center_frequency * bandwidth_percent / 100
        f_min = center_frequency - bandwidth / 2
        f_max = center_frequency + bandwidth / 2
        
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Calculate instantaneous frequency (linear sweep)
        f_inst = f_min + (f_max - f_min) * t / duration
        
        # Calculate phase by integrating frequency
        dt = 1 / self.sample_rate
        phase_t = phase + 2 * np.pi * np.cumsum(f_inst) * dt
        
        # Generate signal
        signal = np.sin(phase_t)
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(num_samples)
        signal = signal * window
        
        return t, signal, f_inst
    
    def dual_frequency(self, frequency1, frequency2, amplitude_ratio=1.0, duration=0.1, phase1=0, phase2=0):
        """
        Generate a dual frequency signal
        
        Parameters:
        -----------
        frequency1 : float
            First frequency in Hz
        frequency2 : float
            Second frequency in Hz
        amplitude_ratio : float
            Amplitude ratio of second frequency to first frequency
        duration : float
            Signal duration in seconds
        phase1 : float
            Initial phase of first frequency in radians
        phase2 : float
            Initial phase of second frequency in radians
            
        Returns:
        --------
        tuple
            (t, signal) - time array and signal array
        """
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate individual components
        signal1 = np.sin(2 * np.pi * frequency1 * t + phase1)
        signal2 = amplitude_ratio * np.sin(2 * np.pi * frequency2 * t + phase2)
        
        # Combine signals
        signal = (signal1 + signal2) / (1 + amplitude_ratio)  # Normalize to keep amplitude in [-1, 1]
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(num_samples)
        signal = signal * window
        
        return t, signal
    
    def dual_frequency_sweep(self, frequency1, frequency2, bandwidth_percent, duration, amplitude_ratio=1.0, phase1=0, phase2=0):
        """
        Generate a dual frequency sweep signal
        
        Parameters:
        -----------
        frequency1 : float
            First center frequency in Hz
        frequency2 : float
            Second center frequency in Hz
        bandwidth_percent : float
            Bandwidth as a percentage of center frequency
        duration : float
            Signal duration in seconds
        amplitude_ratio : float
            Amplitude ratio of second frequency to first frequency
        phase1 : float
            Initial phase of first frequency in radians
        phase2 : float
            Initial phase of second frequency in radians
            
        Returns:
        --------
        tuple
            (t, signal, f1_inst, f2_inst) - time array, signal array, and instantaneous frequency arrays
        """
        # Calculate frequency ranges
        bandwidth1 = frequency1 * bandwidth_percent / 100
        f1_min = frequency1 - bandwidth1 / 2
        f1_max = frequency1 + bandwidth1 / 2
        
        bandwidth2 = frequency2 * bandwidth_percent / 100
        f2_min = frequency2 - bandwidth2 / 2
        f2_max = frequency2 + bandwidth2 / 2
        
        # Generate time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Calculate instantaneous frequencies (linear sweeps)
        f1_inst = f1_min + (f1_max - f1_min) * t / duration
        f2_inst = f2_min + (f2_max - f2_min) * t / duration
        
        # Calculate phases by integrating frequencies
        dt = 1 / self.sample_rate
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
    
    def generate_signal(self, signal_type, **kwargs):
        """
        Generate a signal based on the specified type
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to generate ('single_freq', 'sweep_freq', 'dual_freq', 'dual_sweep_freq')
        **kwargs : dict
            Additional parameters for the specific signal type
            
        Returns:
        --------
        tuple
            Signal data (format depends on signal type)
        """
        if signal_type == 'single_freq':
            # Required parameters: frequency, duration
            frequency = kwargs.get('frequency', 180e3)
            duration = kwargs.get('duration', 0.1)
            return self.continuous_wave(frequency, duration)
        
        elif signal_type == 'sweep_freq':
            # Required parameters: frequency, bandwidth_percent, duration
            frequency = kwargs.get('frequency', 180e3)
            bandwidth_percent = kwargs.get('bandwidth_percent', 40)
            duration = kwargs.get('duration', 0.1)
            return self.frequency_sweep(frequency, bandwidth_percent, duration)
        
        elif signal_type == 'dual_freq':
            # Required parameters: frequency1, frequency2, duration
            frequency1 = kwargs.get('frequency1', 180e3)
            frequency2 = kwargs.get('frequency2', 550e3)
            amplitude_ratio = kwargs.get('amplitude_ratio', 1.0)
            duration = kwargs.get('duration', 0.1)
            return self.dual_frequency(frequency1, frequency2, amplitude_ratio, duration)
        
        elif signal_type == 'dual_sweep_freq':
            # Required parameters: frequency1, frequency2, bandwidth_percent, duration
            frequency1 = kwargs.get('frequency1', 180e3)
            frequency2 = kwargs.get('frequency2', 550e3)
            bandwidth_percent = kwargs.get('bandwidth_percent', 40)
            amplitude_ratio = kwargs.get('amplitude_ratio', 1.0)
            duration = kwargs.get('duration', 0.1)
            return self.dual_frequency_sweep(frequency1, frequency2, bandwidth_percent, duration, amplitude_ratio)
        
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create signal generator
    sg = SignalGenerator()
    
    # Generate signals
    t1, signal1 = sg.generate_signal('single_freq', frequency=180e3, duration=0.1)
    t2, signal2, f_inst = sg.generate_signal('sweep_freq', frequency=180e3, bandwidth_percent=40, duration=0.1)
    t3, signal3 = sg.generate_signal('dual_freq', frequency1=180e3, frequency2=550e3, duration=0.1)
    t4, signal4, f1_inst, f2_inst = sg.generate_signal('dual_sweep_freq', frequency1=180e3, frequency2=550e3, bandwidth_percent=40, duration=0.1)
    
    # Plot signals
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(t1 * 1000, signal1)
    plt.title('Single Frequency (180 kHz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 2)
    plt.plot(t2 * 1000, signal2)
    plt.title('Frequency Sweep (180 kHz ± 20%)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 3)
    plt.plot(t3 * 1000, signal3)
    plt.title('Dual Frequency (180 kHz + 550 kHz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.subplot(4, 1, 4)
    plt.plot(t4 * 1000, signal4)
    plt.title('Dual Frequency Sweep (180 kHz ± 20% + 550 kHz ± 20%)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('signal_examples.png', dpi=300)
    plt.show()
>>>>>>> 793ef48ece9336b7d7632f02e7da7317c59632a2
