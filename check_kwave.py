import sys

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import kwave
    print(f"k-wave version: {kwave.__version__}")
    print(f"Available modules: {dir(kwave)}")
except ImportError as e:
    print(f"Error importing kwave: {e}")

try:
    from kwave.kgrid import kWaveGrid
    print("kWaveGrid imported successfully")
except ImportError as e:
    print(f"Error importing kWaveGrid: {e}")

try:
    from kwave.kmedium import kWaveMedium
    print("kWaveMedium imported successfully")
except ImportError as e:
    print(f"Error importing kWaveMedium: {e}")

try:
    from kwave.ksensor import kSensor
    print("kSensor imported successfully")
except ImportError as e:
    print(f"Error importing kSensor: {e}")

try:
    from kwave.ksource import kSource
    print("kSource imported successfully")
except ImportError as e:
    print(f"Error importing kSource: {e}")

try:
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
    print("kspaceFirstOrder2DC imported successfully")
except ImportError as e:
    print(f"Error importing kspaceFirstOrder2DC: {e}")

try:
    from kwave.utils.kwave_array import kWaveArray
    print("kWaveArray imported successfully")
except ImportError as e:
    print(f"Error importing kWaveArray: {e}")
