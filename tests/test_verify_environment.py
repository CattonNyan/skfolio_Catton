import io
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_environment import verify


class VerifyEnvironmentTests(unittest.TestCase):
    def test_verify_environment_passes(self):
        passed = verify(verbose=False)
        self.assertTrue(passed)

    def test_verify_environment_verbose_output(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            passed = verify(verbose=True)
        out = buf.getvalue()
        self.assertTrue(passed)
        self.assertIn("Python version:", out)
        self.assertIn("NumPy", out)
        self.assertIn("All core dependencies and Korean quant tools are correctly verified!", out)

    def test_verify_handles_import_failure(self):
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "ccxt":
                raise ImportError("Mocked ccxt missing")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            passed = verify(verbose=False)
            self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
