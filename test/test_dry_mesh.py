import sys
import os
import shutil
import unittest
import tempfile

# Ensure we can import from optimizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
from optimizer.auto_tuner import DryMeshObjective

class TestDryMeshValidation(unittest.TestCase):
    def test_dry_mesh_eval(self):
        """Test the DryMeshObjective parses checkmesh output correctly."""
        objective = DryMeshObjective()
        # Test the parsing logic with a mock log file
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "log.checkMesh")
            with open(log_path, 'w') as f:
                f.write('''
cells:      150000
Max non-orthogonality = 65.4
Max skewness = 3.2
Failed 1 or more mesh checks
''')

            non_ortho, skewness, failed, cells = objective.parse_checkmesh_output(temp_dir)
            self.assertEqual(non_ortho, 65.4)
            self.assertEqual(skewness, 3.2)
            self.assertEqual(failed, 1)
            self.assertEqual(cells, 150000)

if __name__ == '__main__':
    unittest.main()
