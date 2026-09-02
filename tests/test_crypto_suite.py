import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from tests.test_crypto_optimizer import CryptoOptimizerTests
from tests.test_hrp_clustering import HrpClusteringTests
from tests.test_live_fetcher import LiveFetcherTests
from tests.test_dashboard import DashboardTests
from tests.test_rebalancing import RebalancingTests
from tests.test_html_report import HtmlReportTests
from tests.test_risk_calculator import RiskCalculatorTests
from tests.test_freqtrade_integration import FreqtradeAllocationTests
from tests.test_stake_allocator import StakeAllocatorTests
from tests.test_black_litterman import BlackLittermanTests
from tests.test_stress_tester import StressTesterTests
from tests.test_kimchi_premium import KimchiPremiumTests
from tests.test_strategy_optimizer import StrategyOptimizerTests
from tests.test_monte_carlo import MonteCarloTests
from tests.test_macro_regime import MacroRegimeTests
from tests.test_factor_analyzer import FactorAnalyzerTests
from tests.test_enhanced_strategy import EnhancedStrategyTests
from tests.test_correlation_breakdown import CorrelationBreakdownTests
from tests.test_tax_calculator import TaxCalculatorTests


def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTests(loader.loadTestsFromTestCase(CryptoOptimizerTests))
    s.addTests(loader.loadTestsFromTestCase(HrpClusteringTests))
    s.addTests(loader.loadTestsFromTestCase(LiveFetcherTests))
    s.addTests(loader.loadTestsFromTestCase(DashboardTests))
    s.addTests(loader.loadTestsFromTestCase(RebalancingTests))
    s.addTests(loader.loadTestsFromTestCase(HtmlReportTests))
    s.addTests(loader.loadTestsFromTestCase(RiskCalculatorTests))
    s.addTests(loader.loadTestsFromTestCase(FreqtradeAllocationTests))
    s.addTests(loader.loadTestsFromTestCase(StakeAllocatorTests))
    s.addTests(loader.loadTestsFromTestCase(BlackLittermanTests))
    s.addTests(loader.loadTestsFromTestCase(StressTesterTests))
    s.addTests(loader.loadTestsFromTestCase(KimchiPremiumTests))
    s.addTests(loader.loadTestsFromTestCase(StrategyOptimizerTests))
    s.addTests(loader.loadTestsFromTestCase(MonteCarloTests))
    s.addTests(loader.loadTestsFromTestCase(MacroRegimeTests))
    s.addTests(loader.loadTestsFromTestCase(FactorAnalyzerTests))
    s.addTests(loader.loadTestsFromTestCase(EnhancedStrategyTests))
    s.addTests(loader.loadTestsFromTestCase(CorrelationBreakdownTests))
    s.addTests(loader.loadTestsFromTestCase(TaxCalculatorTests))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
