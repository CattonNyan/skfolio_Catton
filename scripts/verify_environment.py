"""Environment verification script for skfolio_Catton."""

import sys
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


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

    print("\n--- Korean Crypto Quant Tools Verification ---")
    korean_tools = [
        ("scripts.fetch_upbit_crypto", "Upbit Price Fetcher"),
        ("scripts.crypto_krw_fee_calculator", "KRW Fee Drag Simulator"),
        ("scripts.crypto_kimchi_regime", "Kimchi Regime Allocator"),
        ("scripts.crypto_travel_rule_advisor", "Travel Rule Advisor"),
        ("scripts.crypto_kimchi_premium", "Kimchi Premium Analyzer"),
        ("scripts.crypto_tax_calculator", "Crypto Tax Calculator"),
    ]
    for module_name, display_name in korean_tools:
        try:
            __import__(module_name)
            print(f"[OK] {display_name:<25}: loaded")
        except Exception as err:
            print(f"[FAILED] {display_name:<25}: {err}")
            all_passed = False

    if all_passed:
        print("\nAll core dependencies and Korean quant tools are correctly verified!")
    else:
        print("\nSome modules failed to load. Please verify scripts and virtual environment.")


if __name__ == "__main__":
    verify()
