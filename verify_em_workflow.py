import os
import sys
import shutil

# Add optimizer to path
sys.path.append(os.path.abspath("optimizer"))

from em_driver import OpenEMSDriver, export_3d_farfield_pattern
from scad_driver import ScadDriver
import simulation_runner

def test_em_workflow():
    print("--- Testing EM Workflow ---")

    case_dir = "test_em_case"
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)

    # Initialize Drivers
    scad = ScadDriver("trace_optimizer.scad")
    # Initialize OpenEMSDriver with container_engine="none" for local mock testing
    em = OpenEMSDriver(case_dir, container_engine="none", verbose=True)

    params = {
        "trace_offset": 0.5,
        "substrate_thickness": 1.6,
        "part_to_generate": "all"
    }

    # Manually run the steps to verify the driver logic
    print("Step 1: Preparing Case...")
    # Setup the directory structure expected by simulation_runner
    tri_dir = os.path.join(em.case_dir, "constant", "triSurface")
    os.makedirs(tri_dir, exist_ok=True)

    # Simulate STL generation (place dummy files)
    for p in ["copper", "substrate", "port1", "port2"]:
        with open(os.path.join(tri_dir, f"{p}.stl"), "w") as f:
            f.write("solid dummy\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid dummy")

    em.prepare_case(stl_assets={
        "copper": "copper.stl",
        "substrate": "substrate.stl",
        "port1": "port1.stl",
        "port2": "port2.stl"
    })

    print("Step 2: Running Solver (Mock)...")
    success = em.run_solver()
    print(f"Solver Success: {success}")

    print("Step 3: Extracting Metrics...")
    metrics = em.get_metrics()
    print(f"Metrics: {metrics}")

    print("Step 4: Checking VTK Export & 3D Radiation Pattern...")
    script_path = os.path.join(em.case_dir, "run_em_simulation.py")
    if os.path.exists(script_path):
        print("run_em_simulation.py exists.")
    else:
        print("run_em_simulation.py MISSING")

    vtk_path = em.generate_vtk()
    if vtk_path:
        print(f"VTK Directory: {vtk_path}")
        files = os.listdir(vtk_path)
        print(f"Files in VTK dir: {files}")
        if "field_data.vtk" in files and "radiation_pattern_3d.vtk" in files:
            print("VTK Export verified: field_data.vtk & radiation_pattern_3d.vtk found.")
        else:
            print("VTK Export FAILED: field_data or radiation_pattern_3d missing.")
    else:
        print("VTK Export FAILED: generate_vtk returned None")

    # Direct 3D farfield pattern test
    rad_ok = export_3d_farfield_pattern(os.path.join(case_dir, "test_pattern.vtk"), center_freq_ghz=2.45)
    assert rad_ok, "Far-field 3D pattern export failed"
    print("3D Far-Field Radiation Pattern VTK Export verified.")

    # Cleanup
    em.cleanup_ram_disk()
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    test_em_workflow()
