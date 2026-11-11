import unittest
import os
import sys

# Ensure we can import the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test')))
from verify_surrogate_gradients import verify_surrogate_drift

class TestVerifySurrogateDrift(unittest.TestCase):
    def test_verify_surrogate_drift(self):
        verify_surrogate_drift()

    def test_model_fusion_drift_logic(self):
        import numpy as np
        from unittest.mock import patch, MagicMock
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
        from model_fusion_multiphysics import MultiPhysicsModelFusionOptimizer

        opt = MultiPhysicsModelFusionOptimizer(
            physics_driver=MagicMock(),
            parameter_defs={"test": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}},
            domain="cfd",
            verbose=True
        )
        opt.surrogate = MagicMock()
        opt.evaluate_surrogate = MagicMock(return_value=({"delta_p": 110.0, "separation_efficiency": 85.0}, 0.5))
        opt.driver.case_dir = "/tmp"

        # Populate history to trigger drift check
        for i in range(5):
            opt.history.append({
                "params": {"test": 0.5},
                "actual_metrics": {"delta_p": 100.0, "separation_efficiency": 90.0},
                "pred_metrics": {"delta_p": 110.0, "separation_efficiency": 85.0}  # ~11% error
            })

        with patch("builtins.print") as mock_print:
            # We must pass the mock parameters since selecting a candidate tries to run surrogate gradients
            opt.step(candidate_params={"test": 0.5}, mock_run=True)

            # Verify the drift warning was printed
            drift_call = any("Warning: Surrogate drift" in call.args[0] for call in mock_print.call_args_list if call.args)
            self.assertTrue(drift_call, "Surrogate drift warning should have been printed")
            # And verify fit was called
            self.assertTrue(opt.surrogate.fit.called, "Surrogate fit should have been called upon large drift")

if __name__ == '__main__':
    unittest.main()
