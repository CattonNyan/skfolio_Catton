"""Environment verification script for skfolio_Catton."""

import sys


def verify():
    print("Python version:", sys.version)
    modules = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("cvxpy", "CVXPY"),
        ("clarabel", "Clarabel Solver"),
        ("sklearn", "Scikit-Learn"),
        ("plotly", "Plotly"),
        ("streamlit", "Streamlit"),
        ("ccxt", "CCXT Crypto API"),
        ("skfolio", "skfolio"),
    ]

    all_passed = True
    for module_name, display_name in modules:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"[OK] {display_name:<20}: {ver}")
        except ImportError as err:
            print(f"[FAILED] {display_name:<20}: {err}")
            all_passed = False

    if all_passed:
        print("\nAll required core dependencies are correctly installed and verified!")
    else:
        print("\nSome dependencies failed to load. Please run setup.ps1 to install requirements.")


if __name__ == "__main__":
    verify()
