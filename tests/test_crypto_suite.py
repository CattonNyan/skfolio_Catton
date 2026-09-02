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
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
