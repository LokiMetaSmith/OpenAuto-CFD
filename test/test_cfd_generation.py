import unittest
import os
import sys
import tempfile
import trimesh
from unittest.mock import patch
import subprocess

# Ensure we can import from optimizer and its dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))
from optimizer.build123d_driver import Build123dDriver

class TestCFDGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure we are in the repo root
        cls.original_cwd = os.getcwd()
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        os.chdir(cls.repo_root)

        cls.corkscrew_driver = Build123dDriver("corkscrew.scad")
        cls.manifold_driver = Build123dDriver("cyclone_filter_manifold.scad", fluid_volume_module="fluid_volume")


    @classmethod
    def tearDownClass(cls):
        # Restore original working directory
        os.chdir(cls.original_cwd)

    def verify_stl(self, stl_path):
        """Verifies that the STL file exists and is a valid, watertight mesh."""
        self.assertTrue(os.path.exists(stl_path), f"STL file not found: {stl_path}")

        # Load the mesh using trimesh
        try:
            mesh = trimesh.load(stl_path, file_type='stl')
        except Exception as e:
            self.fail(f"Failed to load STL: {e}")

        self.assertIsNotNone(mesh, "Failed to load mesh")

        # Handle Scene objects (if multiple bodies)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                self.fail("Loaded mesh is an empty scene")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        self.assertGreater(mesh.volume, 0, "Mesh volume is not positive")
        self.assertTrue(mesh.is_watertight, "Mesh is not watertight")

    def test_inlet_cap_repro_failure(self):
        """Test generation of Inlet Cap STL using exact parameters from user report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "inlet.stl")
            params = {
                "part_to_generate": "inlet_cap",
                "num_bins": 1,
                "number_of_complete_revolutions": 5,
                "helix_path_radius_mm": 6.0,
                "helix_profile_radius_mm": 4.0,
                "helix_void_profile_radius_mm": 2.5,
                "helix_profile_scale_ratio": 1.0,
                "tube_od_mm": 32,
                "insert_length_mm": 50,
                "GENERATE_CFD_VOLUME": True,
                "slit_axial_length_mm": 2.5,
                "slit_chamfer_height": 0.7,
                "GENERATE_SLICE": False,
                "CUT_FOR_VISIBILITY": False,
                "SHOW_TUBE": False,
                "high_res_fn": 100
            }

            print("Generating Inlet Cap with repro parameters...")
            success = self.corkscrew_driver.generate_stl(params, output_path)
            self.assertTrue(success, "Inlet Cap generation failed")
            self.verify_stl(output_path)

    def test_outlet_cap_repro_failure(self):
        """Test generation of Outlet Cap STL using exact parameters from user report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "outlet.stl")
            params = {
                "part_to_generate": "outlet_cap",
                "num_bins": 1,
                "number_of_complete_revolutions": 5,
                "helix_path_radius_mm": 6.0,
                "helix_profile_radius_mm": 4.0,
                "helix_void_profile_radius_mm": 2.5,
                "helix_profile_scale_ratio": 1.0,
                "tube_od_mm": 32,
                "insert_length_mm": 50,
                "GENERATE_CFD_VOLUME": True,
                "slit_axial_length_mm": 2.5,
                "slit_chamfer_height": 0.7,
                "GENERATE_SLICE": False,
                "CUT_FOR_VISIBILITY": False,
                "SHOW_TUBE": False,
                "high_res_fn": 100
            }

            print("Generating Outlet Cap with repro parameters...")
            success = self.corkscrew_driver.generate_stl(params, output_path)
            self.assertTrue(success, "Outlet Cap generation failed")
            self.verify_stl(output_path)

    def test_fluid_volume_watertightness(self):
        """Test generation of Fluid Volume STL (main CFD body) using exact parameters from user report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "corkscrew_fluid.stl")
            params = {
                "part_to_generate": "modular_filter_assembly",
                "num_bins": 1,
                "number_of_complete_revolutions": 2,
                "helix_path_radius_mm": 8.0,
                "helix_profile_radius_mm": 6.0,
                "helix_void_profile_radius_mm": 4.0,
                "helix_profile_scale_ratio": 1.0,
                "tube_od_mm": 32,
                "insert_length_mm": 50,
                "GENERATE_CFD_VOLUME": True,
                "slit_axial_length_mm": 2.0,
                "slit_chamfer_height": 0.5,
                "GENERATE_SLICE": False,
                "CUT_FOR_VISIBILITY": False,
                "SHOW_TUBE": False,
                "high_res_fn": 100
            }

            print("Generating Fluid Volume with repro parameters...")
            success = self.corkscrew_driver.generate_stl(params, output_path)
            # generate_stl returns an anchor coordinate list for CFD volumes, or True for normal parts
            self.assertTrue(success if isinstance(success, bool) else (success is not None), "Fluid Volume generation failed")
            self.verify_stl(output_path)

if __name__ == '__main__':
    unittest.main()
