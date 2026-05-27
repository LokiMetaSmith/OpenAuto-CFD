import os
import shutil
import tempfile
import sys
from physics_driver import PhysicsDriver
from utils import run_command_with_spinner, ProcessAbortedError

class OpenEMSDriver(PhysicsDriver):
    """
    Driver for executing electromagnetic simulations using openEMS.
    """

    def __init__(self, case_dir, config=None, template_dir=None, container_engine="auto", num_processors=1, verbose=False, debug=False):
        super().__init__(case_dir, config=config, container_engine=container_engine, verbose=verbose, debug=debug)
        self.num_processors = num_processors
        self.template_dir = os.path.abspath(template_dir) if template_dir else os.path.abspath(case_dir)

        # Use a RAM disk for temporary execution if possible
        if os.path.exists("/dev/shm") and sys.platform.startswith('linux'):
            self.ram_disk_base = tempfile.mkdtemp(dir="/dev/shm", prefix="em_run_")
        else:
            self.ram_disk_base = tempfile.mkdtemp(prefix="em_run_")

        self.case_dir = os.path.join(self.ram_disk_base, os.path.basename(case_dir))
        self.log_file = os.path.join(self.case_dir, "run_openems.log")

        # Determine execution environment (future robust logic here)
        self.has_tools = True # Assume true for now or add detection logic

    def prepare_case(self, **kwargs):
        """
        Sets up the openEMS execution directory.
        """
        if self.case_dir != self.template_dir:
            if os.path.exists(self.case_dir):
                shutil.rmtree(self.case_dir)
            if os.path.exists(self.template_dir):
                shutil.copytree(self.template_dir, self.case_dir)
            else:
                os.makedirs(self.case_dir, exist_ok=True)

        # Future: Generate python/octave scripts or CSXCAD geometry definitions here based on config.
        print(f"Prepared openEMS case at {self.case_dir}")

    def run_meshing(self, log_file=None, **kwargs):
        """
        openEMS typically uses FDTD rectilinear grids handled within the solver script (CSXCAD).
        We may not need a discrete meshing step like OpenFOAM's blockMesh/snappyHexMesh,
        so this can either be a no-op or trigger a pre-processing validation script.
        """
        print("Skipping discrete meshing step (FDTD grid handled in solver).")
        return True

    def run_solver(self, log_file=None, **kwargs):
        """
        Executes the openEMS solver via a python or octave script.
        """
        print("Running openEMS solver...")
        # Future: self.run_command(["python3", "run_em_simulation.py"])
        return True

    def get_metrics(self, log_file=None):
        """
        Parses S-parameters, gain, etc. from openEMS output formats (CSV, HDF5, or log files).
        """
        print("Extracting EM metrics...")
        metrics = {
            "S11": -15.0, # Mock value for skeleton
            "gain": 2.1   # Mock value for skeleton
        }
        return metrics

    def cleanup_ram_disk(self):
        """
        Cleans up the temporary RAM disk used by this driver.
        """
        if self.ram_disk_base and os.path.exists(self.ram_disk_base):
            try:
                shutil.rmtree(self.ram_disk_base)
                if self.verbose:
                    print(f"Cleaned up EM RAM disk: {self.ram_disk_base}")
            except Exception as e:
                print(f"Warning: Failed to clean up EM RAM disk {self.ram_disk_base}: {e}")
