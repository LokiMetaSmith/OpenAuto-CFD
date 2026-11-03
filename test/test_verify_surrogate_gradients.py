import unittest
import os
import sys

# Ensure we can import the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test')))
from verify_surrogate_gradients import verify_surrogate_drift

class TestVerifySurrogateDrift(unittest.TestCase):
    def test_verify_surrogate_drift(self):
        verify_surrogate_drift()

if __name__ == '__main__':
    unittest.main()
