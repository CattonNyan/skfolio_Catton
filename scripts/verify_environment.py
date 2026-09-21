"""Environment verification script for skfolio_Catton."""

import sys
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def verify(verbose: bool = True) -> bool:
    if verbose:
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
            if verbose:
                print(f"[OK] {display_name:<20}: {ver}")
        except ImportError as err:
            if verbose:
                print(f"[FAILED] {display_name:<20}: {err}")
            all_passed = False

    if verbose:
        print("\n--- Korean Crypto Quant Tools Verification ---")
    korean_tools = [
        ("scripts.fetch_upbit_crypto", "Upbit Price Fetcher"),
        ("scripts.crypto_krw_fee_calculator", "KRW Fee Drag Simulator"),
        ("scripts.crypto_kimchi_regime", "Kimchi Regime Allocator"),
        ("scripts.crypto_travel_rule_advisor", "Travel Rule Advisor"),
        ("scripts.crypto_kimchi_premium", "Kimchi Premium Analyzer"),
        ("scripts.crypto_tax_calculator", "Crypto Tax Calculator"),
        ("scripts.crypto_factor_analyzer", "Multi-Factor Screener"),
        ("scripts.crypto_black_litterman", "Black-Litterman Engine"),
        ("scripts.crypto_liquidity_filter", "Liquidity Risk Filter"),
        ("scripts.crypto_kelly_sizer", "Kelly Position Sizer"),
        ("scripts.fetch_bithumb_crypto", "Bithumb Price Fetcher"),
        ("scripts.crypto_triangular_arbitrage", "Triangular Arbitrage Scanner"),
        ("scripts.crypto_kimchi_hedging", "Kimchi Hedging Simulator"),
        ("scripts.crypto_synthetic_data", "Jump-Diffusion Path Generator"),
        ("scripts.freqtrade_funding_arbitrage", "Funding Arbitrage Configurator"),
        ("scripts.fetch_coinbase_crypto", "Coinbase Price Fetcher"),
        ("scripts.crypto_vol_target_allocator", "Vol-Targeting Allocator"),
        ("scripts.crypto_tail_dependence", "Tail Dependence Analyzer"),
        ("scripts.crypto_drawdown_metrics", "Drawdown & Ulcer Index Engine"),
        ("scripts.http_retry_helper", "HTTP Retry Helper"),
    ]
    for module_name, display_name in korean_tools:
        try:
            __import__(module_name)
            if verbose:
                print(f"[OK] {display_name:<25}: loaded")
        except Exception as err:
            if verbose:
                print(f"[FAILED] {display_name:<25}: {err}")
            all_passed = False

    if verbose:
        if all_passed:
            print("\nAll core dependencies and Korean quant tools are correctly verified!")
        else:
            print("\nSome modules failed to load. Please verify scripts and virtual environment.")

    return all_passed


if __name__ == "__main__":
    success = verify(verbose=True)
    sys.exit(0 if success else 1)
