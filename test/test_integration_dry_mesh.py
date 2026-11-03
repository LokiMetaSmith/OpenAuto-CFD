import sys
import os
import shutil
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
from optimizer.auto_tuner import DryMeshObjective

class TestIntegrationDryMesh(unittest.TestCase):
    def test_dry_mesh_classification(self):
        objective = DryMeshObjective()

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "log.checkMesh")

            # Good mesh test
            with open(log_path, 'w') as f:
                f.write('''
cells:      150000
Max non-orthogonality = 45.4
Max skewness = 2.2
Mesh OK.
''')
            non_ortho, skewness, failed, cells = objective.parse_checkmesh_output(temp_dir)
            self.assertEqual(non_ortho, 45.4)
            self.assertEqual(skewness, 2.2)
            self.assertEqual(failed, 0)

            # Bad mesh test
            with open(log_path, 'w') as f:
                f.write('''
cells:      150000
Max non-orthogonality = 85.4
Max skewness = 9.2
Failed 3 or more mesh checks
''')
            non_ortho, skewness, failed, cells = objective.parse_checkmesh_output(temp_dir)
            self.assertEqual(non_ortho, 85.4)
            self.assertEqual(skewness, 9.2)
            self.assertEqual(failed, 3)

if __name__ == '__main__':
    unittest.main()
