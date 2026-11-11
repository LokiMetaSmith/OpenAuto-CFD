import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
import main

class TestMainFallback(unittest.TestCase):
    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.DataStore')
    @patch('main.LLMAgent')
    @patch('main.CadEngineFactory.get_driver')
    @patch('main.PhysicsEngineFactory.get_driver')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="geometry: {}\noptimization: {}")
    def test_dummy_parameter_injection_no_params_file(self, mock_open, mock_phys, mock_cad, mock_agent, mock_ds, mock_args):
        mock_args_obj = MagicMock()
        mock_args_obj.config_file = "test.yaml"
        mock_args_obj.params_file = None
        mock_args_obj.iterations = "1"
        mock_args_obj.harness = False
        mock_args_obj.dry_run = True
        mock_args_obj.skip_cfd = True
        mock_args_obj.reuse_mesh = False
        mock_args_obj.verbose = False
        mock_args_obj.parallel_workers = 0
        mock_args_obj.batch_size = 1
        mock_args_obj.use_ml_optimizer = False
        mock_args_obj.use_model_fusion = False
        mock_args_obj.schema = False
        mock_args_obj.epsilon = 5.0
        mock_args.return_value = mock_args_obj

        # We want to catch the initial_params right after they are created.
        with patch('main.run_simulation', return_value=({}, [], "", "", "")) as mock_run:
            with patch('sys.argv', ['main.py', 'test.yaml', '--iterations', '1']):
                try:
                    main.main()
                except SystemExit:
                    pass

            # Assert run_simulation was called with _dummy=True
            self.assertTrue(mock_run.called)
            params_arg = mock_run.call_args[0][2] # params is the 3rd positional argument
            self.assertIn("_dummy", params_arg)
            self.assertTrue(params_arg["_dummy"])

    @patch('main.argparse.ArgumentParser.parse_args')
    @patch('main.DataStore')
    @patch('main.LLMAgent')
    @patch('main.CadEngineFactory.get_driver')
    @patch('main.PhysicsEngineFactory.get_driver')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="geometry: {}\noptimization: {}")
    def test_with_params_file_source_only(self, mock_open, mock_phys, mock_cad, mock_agent, mock_ds, mock_args):
        mock_args_obj = MagicMock()
        mock_args_obj.config_file = "test.yaml"
        mock_args_obj.params_file = "my_params.scad"
        mock_args_obj.iterations = "1"
        mock_args_obj.harness = False
        mock_args_obj.dry_run = True
        mock_args_obj.skip_cfd = True
        mock_args_obj.reuse_mesh = False
        mock_args_obj.verbose = False
        mock_args_obj.parallel_workers = 0
        mock_args_obj.batch_size = 1
        mock_args_obj.use_ml_optimizer = False
        mock_args_obj.use_model_fusion = False
        mock_args_obj.schema = False
        mock_args_obj.epsilon = 5.0
        mock_args.return_value = mock_args_obj

        with patch('main.run_simulation', return_value=({}, [], "", "", "")) as mock_run:
            with patch('sys.argv', ['main.py', 'test.yaml', '--iterations', '1', '--params-file', 'my_params.scad']):
                try:
                    main.main()
                except SystemExit:
                    pass

            self.assertTrue(mock_run.called)
            params_arg = mock_run.call_args[0][2]
            self.assertEqual(params_arg["_source"], "my_params.scad")

if __name__ == "__main__":
    unittest.main()
