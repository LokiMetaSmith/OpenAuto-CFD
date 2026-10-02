import optuna
import os
import subprocess
import re
from foam_driver import FoamDriver
import shutil

class DryMeshObjective:
    def __init__(self, base_project_name="corkscrewFilter"):
        self.base_project_name = base_project_name
        self.work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cases", base_project_name))

    def parse_checkmesh_output(self, case_dir: str):
        max_non_ortho = 100.0  # Penalty default
        max_skewness = 100.0
        failed_checks = 0
        total_cells = 1000000 # Penalty default

        log_path = os.path.join(case_dir, "log.checkMesh")
        if not os.path.exists(log_path):
            return max_non_ortho, max_skewness, 10, total_cells

        with open(log_path, 'r') as f:
            output = f.read()

        for line in output.splitlines():
            if "Failed 1 or more mesh checks" in line:
                failed_checks += 1
            if "cells:" in line:
                match = re.search(r"cells:\s+(\d+)", line)
                if match:
                    total_cells = int(match.group(1))
            if "Max non-orthogonality" in line:
                match = re.search(r"Max non-orthogonality = ([\d\.]+)", line)
                if match:
                    max_non_ortho = float(match.group(1))
            elif "Max skewness" in line:
                match = re.search(r"Max skewness = ([\d\.]+)", line)
                if match:
                    max_skewness = float(match.group(1))

        return max_non_ortho, max_skewness, failed_checks, total_cells

    def __call__(self, trial):
        # Suggest parameters to tune
        target_cell_size_divisor = trial.suggest_float("target_cell_size_divisor", 4.0, 10.0)
        n_surface_layers = trial.suggest_int("nSurfaceLayers", 1, 5)
        max_non_ortho = trial.suggest_float("maxNonOrtho", 60.0, 80.0)
        max_boundary_skewness = trial.suggest_float("maxBoundarySkewness", 10.0, 30.0)
        included_angle = trial.suggest_int("includedAngle", 100, 150)

        trial_name = f"{self.base_project_name}_trial_{trial.number}"

        # Init driver with specific tuning parameters
        driver = FoamDriver(
            case_dir=os.path.join(os.path.dirname(__file__), "..", "cases", trial_name),
            template_kwargs={
                "target_cell_size_divisor": target_cell_size_divisor,
                "nSurfaceLayers": n_surface_layers,
                "maxNonOrtho": max_non_ortho,
                "maxBoundarySkewness": max_boundary_skewness,
                "includedAngle": included_angle
            }
        )

        try:
            driver.initialize_case()
            driver._generate_mesh()
            driver.run_command(["checkMesh"], log_file=os.path.join(driver.case_dir, "log.checkMesh"), description="Checking Mesh Quality")

            non_ortho, skewness, failed, cells = self.parse_checkmesh_output(driver.case_dir)
            score = non_ortho + (skewness * 2.0) - (n_surface_layers * 0.5) + (failed * 1000) + (cells / 10000.0)
            return score

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 1000000.0  # Heavy penalty for failure
        finally:
            if hasattr(driver, 'case_dir') and os.path.exists(driver.case_dir):
                shutil.rmtree(driver.case_dir, ignore_errors=True)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    objective = DryMeshObjective()
    study.optimize(objective, n_trials=5)
    print("Best parameters:", study.best_params)
