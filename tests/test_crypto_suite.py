import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from tests.test_crypto_optimizer import CryptoOptimizerTests
from tests.test_hrp_clustering import HrpClusteringTests
from tests.test_live_fetcher import LiveFetcherTests


def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTests(loader.loadTestsFromTestCase(CryptoOptimizerTests))
    s.addTests(loader.loadTestsFromTestCase(HrpClusteringTests))
    s.addTests(loader.loadTestsFromTestCase(LiveFetcherTests))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
