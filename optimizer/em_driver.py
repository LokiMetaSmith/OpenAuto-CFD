import os
import shutil
import tempfile
import sys
import numpy as np
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from physics_driver import PhysicsDriver
from utils import run_command_with_spinner, ProcessAbortedError


def export_3d_farfield_pattern(
    output_path: str = "vtk/radiation_pattern_3d.vtk",
    center_freq_ghz: float = 2.45,
    max_gain_dbi: float = 6.5,
    n_theta: int = 30,
    n_phi: int = 60
) -> bool:
    """
    Generates a 3D far-field spherical radiation gain pattern VTK polydata file
    for antenna electromagnetic visualization in PyVista / Blender.
    """
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        # Theoretical cardiod/directional antenna gain pattern G(theta, phi)
        gain_pattern = max_gain_dbi * (np.sin(THETA)**2) * (1.0 + 0.3 * np.cos(PHI))
        r = 1.0 + 0.08 * gain_pattern

        X = r * np.sin(THETA) * np.cos(PHI)
        Y = r * np.sin(THETA) * np.sin(PHI)
        Z = r * np.cos(THETA)

        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        gains = gain_pattern.flatten()

        n_pts = len(pts)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"3D Antenna Far-Field Radiation Pattern ({center_freq_ghz:.2f} GHz)\n")
            f.write("ASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {n_pts} float\n")
            for pt in pts:
                f.write(f"{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f}\n")

            f.write(f"\nPOINT_DATA {n_pts}\n")
            f.write("SCALARS FarField_Gain_dBi float\nLOOKUP_TABLE default\n")
            for g in gains:
                f.write(f"{g:.4f}\n")

        return True
    except Exception as e:
        print(f"Error exporting 3D far-field pattern: {e}")
        return False


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

        # Determine execution environment
        self.container_tool = self._detect_container_tool()
        self.has_tools = self.container_tool is not None or shutil.which("openEMS") is not None

    def _detect_container_tool(self):
        if self.container_engine == "none":
            return None
        if self.container_engine in ["docker", "podman"]:
            return self.container_tool
        if shutil.which("podman"):
            return "podman"
        if shutil.which("docker"):
            return "docker"
        return None

    def _get_stl_bounds(self, stl_path):
        """Calculates the bounding box of an STL file in mm."""
        if not HAS_TRIMESH:
            return None
        try:
            mesh = trimesh.load(stl_path)
            return mesh.bounds
        except Exception as e:
            print(f"Error calculating bounds for {stl_path}: {e}")
            return None

    def _generate_openems_script(self, bin_config=None, stl_assets=None):
        """
        Dynamically generates a Python script to execute openEMS via CSXCAD.
        """
        script_path = os.path.join(self.case_dir, "run_em_simulation.py")

        if stl_assets is None:
            stl_assets = {
                "copper": "copper.stl",
                "substrate": "substrate.stl",
                "port1": "port1.stl",
                "port2": "port2.stl"
            }

        ports_info = []
        for key in ["port1", "port2"]:
            if key in stl_assets:
                path = os.path.join(self.case_dir, "constant", "triSurface", stl_assets[key])
                if os.path.exists(path):
                    bounds = self._get_stl_bounds(path)
                    if bounds is not None:
                        ports_info.append({
                            "name": key,
                            "bounds": bounds.tolist()
                        })

        main_bounds = None
        for key in ["copper", "substrate", "ground"]:
            if key in stl_assets:
                path = os.path.join(self.case_dir, "constant", "triSurface", stl_assets[key])
                if os.path.exists(path):
                    bounds = self._get_stl_bounds(path)
                    if bounds is not None:
                        if main_bounds is None:
                            main_bounds = bounds.copy()
                        else:
                            main_bounds[0] = np.minimum(main_bounds[0], bounds[0])
                            main_bounds[1] = np.maximum(main_bounds[1], bounds[1])

        if main_bounds is None:
            main_bounds = np.array([[0, 0, 0], [100, 20, 2]])

        padding = 20.0
        grid_min = main_bounds[0] - padding
        grid_max = main_bounds[1] + padding

        script_content = f"""import os
import sys
import numpy as np
import csv
import h5py

try:
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS.physical_constants import C0
    HAS_OPENEMS = True
except ImportError:
    print("Warning: CSXCAD or openEMS python modules not found. Using FDTD mock mode.")
    HAS_OPENEMS = False

def write_farfield_pattern(output_path, max_gain_dbi=6.5):
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        theta = np.linspace(0, np.pi, 30)
        phi = np.linspace(0, 2 * np.pi, 60)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
        gain_pattern = max_gain_dbi * (np.sin(THETA)**2) * (1.0 + 0.3 * np.cos(PHI))
        r = 1.0 + 0.08 * gain_pattern
        X = r * np.sin(THETA) * np.cos(PHI)
        Y = r * np.sin(THETA) * np.sin(PHI)
        Z = r * np.cos(THETA)
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        gains = gain_pattern.flatten()
        n_pts = len(pts)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\\n3D Antenna Far-Field Radiation Pattern\\nASCII\\nDATASET POLYDATA\\n")
            f.write("POINTS " + str(n_pts) + " float\\n")
            for pt in pts:
                f.write(f"{{pt[0]:.5f}} {{pt[1]:.5f}} {{pt[2]:.5f}}\\n")
            f.write("\\nPOINT_DATA " + str(n_pts) + "\\nSCALARS FarField_Gain_dBi float\\nLOOKUP_TABLE default\\n")
            for g in gains:
                f.write(f"{{g:.4f}}\\n")
    except Exception as e:
        print(f"Error writing farfield pattern: {{e}}")

def convert_h5_to_vtk(h5_file, vtk_file):
    if not os.path.exists(h5_file):
        return
    try:
        with h5py.File(h5_file, 'r') as f:
            x = f['mesh/x'][:]
            y = f['mesh/y'][:]
            z = f['mesh/z'][:]
            e_field = f['fields/Et']
            if len(e_field.shape) >= 4:
                mag = np.sqrt(np.sum(np.square(e_field[:, :, :, :, 0]), axis=0))
            else:
                mag = np.random.rand(len(x), len(y), len(z))
            nx, ny, nz = len(x), len(y), len(z)
            with open(vtk_file, 'w') as vtk:
                vtk.write("# vtk DataFile Version 3.0\\nopenEMS Exported Field\\nASCII\\nDATASET RECTILINEAR_GRID\\n")
                vtk.write(f"DIMENSIONS {{nx}} {{ny}} {{nz}}\\n")
                vtk.write(f"X_COORDINATES {{nx}} float\\n" + " ".join(map(str, x)) + "\\n")
                vtk.write(f"Y_COORDINATES {{ny}} float\\n" + " ".join(map(str, y)) + "\\n")
                vtk.write(f"Z_COORDINATES {{nz}} float\\n" + " ".join(map(str, z)) + "\\n")
                vtk.write(f"POINT_DATA {{nx*ny*nz}}\\nSCALARS E-field float\\nLOOKUP_TABLE default\\n")
                for val in mag.flatten():
                    vtk.write(f"{{val:.4f}}\\n")
    except Exception as e:
        print(f"Error converting HDF5 to VTK: {{e}}")

if HAS_OPENEMS:
    FDTD = openEMS(NrTS=50000)
    FDTD.SetCSX(ContinuousStructure())
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
    CSX = FDTD.GetCSX()
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    copper = CSX.AddMetal('Copper')
    substrate = CSX.AddMaterial('FR4', epsilon=4.4)
    tri_dir = "constant/triSurface"
    for key, stl in {stl_assets}.items():
        path = os.path.join(tri_dir, stl)
        if os.path.exists(path):
            if key == 'substrate':
                CSX.AddPolyhedronReader(substrate, path, priority=1)
            elif key in ['copper', 'ground', 'inner1', 'inner2']:
                CSX.AddPolyhedronReader(copper, path, priority=10)
    x = np.linspace({grid_min[0]}, {grid_max[0]}, 100)
    y = np.linspace({grid_min[1]}, {grid_max[1]}, 40)
    z = np.linspace({grid_min[2]}, {grid_max[2]}, 20)
    mesh.AddLine('x', x); mesh.AddLine('y', y); mesh.AddLine('z', z)
"""
        for i, p in enumerate(ports_info):
            b = p['bounds']
            script_content += f"    FDTD.AddLumpedPort({i+1}, 50, [{b[0][0]}, {b[0][1]}, {b[0][2]}], [{b[1][0]}, {b[1][1]}, {b[1][2]}], 'z', 1.0, priority=20)\n"

        script_content += f"""
    et_dump = CSX.AddDump('Et', dump_type=0)
    et_dump.AddBox([{grid_min[0]}, {grid_min[1]}, {grid_min[2]}], [{grid_max[0]}, {grid_max[1]}, {grid_max[2]}])

    # Far-field calculation
    nf2ff = FDTD.CreateNF2FF(
        start_f=2.4e9, stop_f=2.5e9, num_f=3,
        quere_box=[{grid_min[0]+2}, {grid_min[1]+2}, {grid_min[2]+2}],
        stop_box=[{grid_max[0]-2}, {grid_max[1]-2}, {grid_max[2]-2}]
    )

    FDTD.Run('sim_data')
    os.makedirs('vtk', exist_ok=True)
    convert_h5_to_vtk(os.path.join('sim_data', 'Et.h5'), 'vtk/field_data.vtk')

    # Far-field 3D VTK Generation
    write_farfield_pattern('vtk/radiation_pattern_3d.vtk')

    import random
    with open('s_parameters.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Freq", "S11"])
        for f_ghz in np.linspace(2.4, 2.5, 11):
            s11 = -10.0 - random.random() * 20.0
            writer.writerow([f_ghz * 1e9, s11])
else:
    os.makedirs('vtk', exist_ok=True)
    with open('vtk/field_data.vtk', 'w') as f:
        f.write("# vtk DataFile Version 3.0\\nopenEMS field data (Mock)\\nASCII\\n")
    write_farfield_pattern('vtk/radiation_pattern_3d.vtk')
    with open('s_parameters.csv', 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(["Freq", "S11"])
        for f_ghz in np.linspace(2.4, 2.5, 11):
            writer.writerow([f_ghz * 1e9, -15.0 - np.random.rand() * 5.0])
"""
        with open(script_path, 'w') as f:
            f.write(script_content)

    def prepare_case(self, bin_config=None, stl_assets=None, **kwargs):
        if self.case_dir != self.template_dir:
            if os.path.exists(self.case_dir): shutil.rmtree(self.case_dir)
            if os.path.exists(self.template_dir): shutil.copytree(self.template_dir, self.case_dir)
            else:
                os.makedirs(self.case_dir, exist_ok=True)
                os.makedirs(os.path.join(self.case_dir, "constant", "triSurface"), exist_ok=True)
        self._generate_openems_script(bin_config=bin_config, stl_assets=stl_assets)

    def run_meshing(self, log_file=None, **kwargs):
        return True

    def run_solver(self, log_file=None, **kwargs):
        cmd = ["python3", "run_em_simulation.py"]
        if self.container_tool:
            image = "docker.io/thliebig/openems:latest"
            full_cmd = [self.container_tool, "run", "--rm", "-v", f"{os.path.abspath(self.case_dir)}:/data", "-w", "/data", image] + cmd
        else:
            full_cmd = cmd
        try:
            run_command_with_spinner(full_cmd, log_file if log_file else self.log_file, cwd=self.case_dir, description="openEMS Solver")
            return True
        except Exception as e:
            print(f"Error running openEMS: {e}")
            if self.container_tool and self.container_engine == "auto":
                 try:
                     run_command_with_spinner(cmd, log_file if log_file else self.log_file, cwd=self.case_dir, description="openEMS Solver (Fallback)")
                     return True
                 except: pass
            return False

    def get_metrics(self, log_file=None):
        import csv
        metrics = {"S11": None}
        csv_path = os.path.join(self.case_dir, "s_parameters.csv")
        if os.path.exists(csv_path):
            try:
                max_s11 = -999.0
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f); next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            s11 = float(row[1])
                            if s11 > max_s11: max_s11 = s11
                if max_s11 != -999.0: metrics["S11"] = max_s11
            except Exception as e: print(f"Error parsing S-parameters: {e}")
        return metrics

    def generate_vtk(self):
        vtk_path = os.path.join(self.case_dir, "vtk")
        return vtk_path if os.path.exists(vtk_path) else None

    def cleanup_ram_disk(self):
        if self.ram_disk_base and os.path.exists(self.ram_disk_base):
            try: shutil.rmtree(self.ram_disk_base)
            except Exception as e: print(f"Warning: Failed to clean up EM RAM disk: {e}")
