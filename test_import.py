
import sys
import os

_THIS_DIR = os.getcwd()
_KRONOS_DIR = os.path.join(_THIS_DIR, "kronos_model")

# Current approach in kronos_predictor.py
sys.path.insert(0, _KRONOS_DIR)

print(f"sys.path[0]: {sys.path[0]}")

try:
    from kronos_model import Kronos
    print("Import from kronos_model Success")
except ImportError as e:
    print(f"Import from kronos_model Failed: {e}")

try:
    import kronos
    print("Import kronos Success")
except ImportError as e:
    print(f"Import kronos Failed: {e}")
