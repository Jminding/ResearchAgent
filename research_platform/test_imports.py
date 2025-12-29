#!/usr/bin/env python3
import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy not available: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas not available: {e}")

try:
    from scipy import stats
    print(f"SciPy available")
except ImportError as e:
    print(f"SciPy not available: {e}")

print("\nAll imports successful!")
