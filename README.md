# Ultrasound Simulation Toolkit

A comprehensive toolkit for simulating and analyzing ultrasound fields, with a focus on dual-frequency applications and cavitation probability.

## Features

- 3D ultrasound simulation using k-wave-python
- Support for various signal types:
  - Single frequency
  - Frequency sweep
  - Dual frequency
  - Dual frequency sweep
- Mechanical Index (MI) calculation
- Cavitation probability estimation
- Comparison tools for different signal types
- Visualization of pressure fields, MI, and cavitation probability
- Support for different medium types (water, bone, soft tissue)

## Requirements

- Python 3.7+
- k-wave-python
- NumPy
- Matplotlib
- h5py

## Structure

The toolkit consists of several modules:

- `transducer_2d_array.py`: Creates and visualizes a 2D ultrasound transducer array
- `transducer_3d_sim_env.py`: Sets up a 3D simulation environment
- `transducer_3d_sim_runner.py`: Runs 3D ultrasound simulations
- `signal_generator.py`: Generates various ultrasound signals
- `ultrasound_utils.py`: Common utilities for data loading and analysis
- `compare_mechanical_index.py`: Compares Mechanical Index across different signal types
- `compare_cavitation.py`: Compares cavitation probability across different signal types
- `grid_optimizer.py`: Optimizes grid sizes for k-wave simulations

## Usage

### Running a simulation

```bash
python transducer_3d_sim_runner.py --focal-depth 50 --medium water
```

### Comparing different signal types

```bash
python compare_mechanical_index.py --focal-depth 50 --medium water --roi-start 20 --roi-end 80 --exclude-interface
python compare_cavitation.py --focal-depth 50 --medium water --roi-start 20 --roi-end 80 --exclude-interface
```

## License

MIT
