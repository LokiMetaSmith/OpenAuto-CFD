import os
import time
import math
import numpy as np
import shutil
from utils import Timer, get_container_memory_gb
from parameter_validator import validate_parameters
from validator import Validator

def run_simulation(scad_driver, physics_driver, params, output_stl_name="corkscrew_fluid.stl", dry_run=False, skip_cfd=False, dry_mesh=False, iteration=0, reuse_mesh=False, output_prefix=None, verbose=False, params_file=None, turbulence="laminar", debug=False):
    """
    Executes the full simulation pipeline:
    1. Generate Fluid Geometry (STL)
    2. Update OpenFOAM BlockMesh (if CFD)
    3. Run Physics Simulation
    4. Extract Metrics
    5. Generate Visualization Images (Solid Model)

    Args:
        scad_driver (ScadDriver): Initialized ScadDriver instance.
        physics_driver (PhysicsDriver): Initialized PhysicsDriver instance (e.g. FoamDriver or OpenEMSDriver).
        params (dict): Dictionary of parameters for the run.
        output_stl_name (str): Filename for the fluid STL (inside case/constant/triSurface).
        dry_run (bool): If True, skips actual processing and returns mock data.
        skip_cfd (bool): If True, generates geometry but skips CFD simulation.
        iteration (int): The current iteration number (used for logging).
        reuse_mesh (bool): If True, skips geometry generation and meshing, using existing mesh.
        output_prefix (str): Prefix for output files (e.g. "exports/run_123"). If None, generates timestamped default.
        params_file (str): Path to a SCAD parameter file to use as config.

    Returns:
        tuple: (metrics, image_paths, solid_stl_path, fluid_stl_path, vtk_zip_path)
            metrics (dict): Simulation results (delta_p, residuals, etc).
            image_paths (list): List of paths to generated PNG visualizations.
            solid_stl_path (str): Path to the generated solid visualization STL.
            fluid_stl_path (str): Path to the archived fluid STL (negative volume).
            vtk_zip_path (str): Path to the zipped VTK directory (or None).
    """

    # Setup Log Directory
    log_dir = os.path.join("logs", f"iteration_{iteration}")
    os.makedirs(log_dir, exist_ok=True)

    # Log files
    geom_log = os.path.join(log_dir, "geometry.log")
    mesh_log = os.path.join(log_dir, "meshing.log")
    solver_log = os.path.join(log_dir, "solver.log")
    vis_log = os.path.join(log_dir, "visualization.log")

    # 0. Validate Parameters
    # If params_file is provided, we trust the file content (or cannot validate it easily from Python)
    if not params_file:
        is_valid, error_msg = validate_parameters(params)
        if not is_valid:
            print(f"Parameter Validation Failed: {error_msg}")
            return {"error": "invalid_parameters", "details": error_msg}, [], None, None, None
    else:
        print("Skipping parameter validation due to external params file.")

    # 0.5. Prepare Case Directory and Configs
    if not dry_run and not skip_cfd:
        # Early check for environment
        if getattr(physics_driver, 'has_tools', False) == False:
            print("Physics simulation tools not found. Skipping simulation.")
            return {"error": "environment_missing_tools", "details": "Required physics simulation tools not found"}, [], None, None, None

        # Override turbulence if defined in config
        cfd_settings = physics_driver.config.get('cfd_settings', {})
        turbulence = "laminar"
        if cfd_settings and 'turbulence_model' in cfd_settings:
            turbulence = cfd_settings['turbulence_model']

        # Prepare Bin Configuration for Meshing/Tracking
        bin_config = {
            "num_bins": int(params.get("num_bins", 1)),
            "insert_length_mm": float(params.get("insert_length_mm", 50.0)),

            # --- NEW: Dynamic Physics & Dust Parameters ---
            "tube_od_mm": float(params.get("tube_od_mm", 32.0)),
            "fluid_velocity": float(params.get("fluid_velocity", 5.0)),
            "dust_density": float(params.get("dust_density", 3100)), # Default: Moon dust
            "dust_sizes_um": params.get("dust_sizes_um", [5, 10, 20, 50, 100])
        }

        # Initialize base case directory and templates
        physics_driver.prepare_case(keep_mesh=reuse_mesh, turbulence=turbulence, bin_config=bin_config)

    # 1. Generate Geometry (Fluid Volume + CFD Assets)
    tri_surface_dir = os.path.join(physics_driver.case_dir, "constant", "triSurface")
    os.makedirs(tri_surface_dir, exist_ok=True)

    # Use specified output name for fluid, others are fixed
    fluid_stl_path = os.path.join(tri_surface_dir, output_stl_name)

    SCALE_FACTOR = 0.001 # mm to meters
    cfd_assets = {} # Dictionary of absolute paths

    if not dry_run:
        if not reuse_mesh:
            with Timer("Geometry Generation"):
                # Use generate_cfd_assets to create all required STLs
                # Note: generate_cfd_assets returns paths to 'corkscrew_fluid.stl', 'inlet.stl', etc.
                # We should rename the fluid one if output_stl_name is different, but for now we trust generate_cfd_assets defaults or move it.
                assets = scad_driver.generate_cfd_assets(params, tri_surface_dir, log_file=geom_log, params_file=params_file)

            if assets:
                cfd_assets = assets
                mesh_anchor = cfd_assets.get("mesh_anchor")
                if "mesh_anchor" in cfd_assets:
                    del cfd_assets["mesh_anchor"]
                # If the generated fluid name doesn't match requested output_stl_name, rename/copy?
                # generate_cfd_assets produces 'corkscrew_fluid.stl' fixed name.
                # If output_stl_name is different, we handle it.
                generated_fluid = cfd_assets["fluid"]
                if os.path.basename(generated_fluid) != output_stl_name:
                    shutil.move(generated_fluid, fluid_stl_path)
                    cfd_assets["fluid"] = fluid_stl_path

                # 1.1 Validation (in mm, before scaling)
                print("Validating Geometry...")
                validator = Validator(verbose=verbose)
                boundaries_config = physics_driver.config.get("physics", {}).get("boundaries", {})
                val_res = validator.validate_assembly(
                    cfd_assets["fluid"],
                    cfd_assets["inlet"],
                    cfd_assets["outlet"],
                    cfd_assets["wall"],
                    tolerance=1.0, # 1mm tolerance
                    boundaries_config=boundaries_config
                )

                # Print any warnings from validation
                for msg in val_res.get('messages', []):
                    if msg.startswith("Warning:"):
                        print(msg)

                if not val_res["valid"]:
                    print(f"Geometry Validation Failed: {val_res['messages']}")
                    return {"error": "geometry_validation_failed", "details": val_res['messages']}, [], None, None, None

                print("Geometry Validation Passed.")

                # 1.2 Scaling (to meters)
                print(f"Scaling STLs to meters...")
                for key, path in cfd_assets.items():
                    if not scad_driver.scale_mesh(path, SCALE_FACTOR):
                        print(f"Failed to scale mesh: {key}")
                        return {"error": "mesh_scaling_failed"}, [], None, None, None
            else:
                print(f"Geometry generation failed. Check {geom_log} for details.")
                return {"error": "geometry_generation_failed"}, [], None, None, None
        else:
            print("[Reuse Mesh] Skipping geometry generation.")
            # Populate cfd_assets assuming files exist
            cfd_assets = {
                "fluid": fluid_stl_path,
                "inlet": os.path.join(tri_surface_dir, "inlet.stl"),
                "outlet": os.path.join(tri_surface_dir, "outlet.stl"),
                "wall": os.path.join(tri_surface_dir, "wall.stl")
            }
    else:
        print(f"[Dry Run] Generated STL at {fluid_stl_path}")
        if not os.path.exists(fluid_stl_path):
            with open(fluid_stl_path, 'w') as f: f.write("solid dryrun\nendsolid dryrun")

    # 2. Prepare Case (BlockMesh update)
    stl_path = fluid_stl_path # For bounds check

    if not dry_run and not skip_cfd:
        if not reuse_mesh:
            # STL is now in METERS (scaled above or from previous run).
            bounds = scad_driver.get_bounds(stl_path)
            if bounds is None or bounds[0] is None:
                print("ERROR: Failed to get bounds. STL geometry is likely invalid or empty.")
                return {"error": "invalid_geometry_bounds"}, [], None, None, None
            else:
                REFINEMENT_LEVEL = 1 # Match level set in snappyHexMeshDict

                # Bounds are already in meters
                bounds_arr = bounds

                # Calculate dynamic target cell size based on smallest feature
                # Default scaled
                target_cell_size = float(params.get("target_cell_size", 1.5)) * SCALE_FACTOR

                # Allow override from params if provided, bypassing automatic void_r logic if explicit
                if "target_cell_size" not in params:
                    void_r = params.get("helix_void_profile_radius_mm")
                    if "void_r" in locals() and void_r:
                        try:
                            # Ensure resolution is sufficient for small channels (at least ~2.5 cells radius)
                            # We use 0.3 * radius to be safe (diameter / 6), clamped between 0.2mm and 0.8mm
                            # This ensures small inlet patches are captured by snappyHexMesh
                            calculated_size_mm = float(void_r) * 0.3
                            target_cell_size_mm = max(0.2, min(0.8, calculated_size_mm))

                            # Adjust target size by refinement factor because blockMesh is coarser
                            # The final surface resolution will be target_cell_size / (2^REFINEMENT_LEVEL)
                            # So we multiply here to set blockMesh size.
                            target_cell_size = target_cell_size_mm * SCALE_FACTOR * (2 ** REFINEMENT_LEVEL)
                        except (ValueError, TypeError):
                            pass

                # Estimate cell count to prevent OOM
                # Dynamically fetch block_margin from config to support various geometries
                config_margin = physics_driver.config.get("geometry", {}).get("block_margin", [1.2, 1.2, 1.2])
                try:
                    BLOCK_MARGIN = np.array(config_margin)
                except Exception:
                    BLOCK_MARGIN = np.array([1.2, 1.2, 1.2])

                # Ensure bounds are numpy arrays for subtraction
                # bounds_arr is already scaled to meters
                size = bounds_arr[1] - bounds_arr[0]

                block_size = size * BLOCK_MARGIN
                block_volume = np.prod(block_size)

                estimated_cells = block_volume / (target_cell_size ** 3)

                # Dynamic Memory Limit
                try:
                    # Estimate available RAM in GB
                    ram_gb = get_container_memory_gb(getattr(physics_driver, 'container_tool', None))
                    print(f"Detected available memory: {ram_gb:.2f} GB")

                    if ram_gb < 7.5 and getattr(physics_driver, 'container_tool', None) == "podman":
                        print(f"Tip: Your container memory is low ({ram_gb:.1f}GB < 8GB). You can increase it by running:")
                        print("     python optimizer/setup_machine.py --memory 16384")

                except Exception as e:
                    print(f"Warning: Failed to detect memory ({e}). Using default.")
                    ram_gb = 4.0 # Conservative default

                # Heuristic: 1.5 to 2 gigabytes of RAM per 1 million cells. Let's use 1.75 GB / 1M cells.
                # So MAX_CELLS = (ram_gb / 1.75) * 1_000_000
                calculated_limit = int((ram_gb / 1.75) * 1_000_000)

                # Clamp limits
                MIN_LIMIT = 100_000
                MAX_LIMIT = 10_000_000 # Cap reasonable enough for 16GB+ systems

                MAX_CELLS = max(MIN_LIMIT, min(calculated_limit, MAX_LIMIT))

                # Check for minimum resolution requirement (void_r)
                if "void_r" in locals() and void_r:
                    # Require at least 4 cells across the channel diameter (radius / 2)
                    # This ensures features aren't completely lost
                    min_res_cell_size = (float(void_r) * SCALE_FACTOR) / 2.0
                    min_required_cells = block_volume / (min_res_cell_size ** 3)

                    if min_required_cells > MAX_CELLS:
                        print(f"WARNING: Geometry requires ~{min_required_cells:.0f} cells for accuracy, but memory limit is {MAX_CELLS}.")

                mesh_scaled_for_memory = False
                if estimated_cells > MAX_CELLS:
                    print(f"ERROR: Estimated cell count {estimated_cells:.0f} exceeds {MAX_CELLS} limit (RAM: {ram_gb:.1f}GB).")
                    print("Simulation fundamentally compromised. Rejecting geometrically intractable configuration.")
                    # Return a structured penalty score directly to the LLM agent
                    error_details = f"Estimated cell count {estimated_cells:.0f} exceeds hardware capacity of {MAX_CELLS} cells (based on {ram_gb:.1f} GB RAM)."
                    metrics = {
                        "error": "computationally_intractable",
                        "penalty": 1e9,
                        "details": error_details
                    }
                    return metrics, [], None, None, None

                print(f"Updating blockMesh with target_cell_size={target_cell_size:.3f}m")
                # Pass scaled bounds to physics_driver (if it supports blockMesh)
                if hasattr(physics_driver, 'update_blockMesh'):
                    physics_driver.update_blockMesh(bounds_arr, margin=BLOCK_MARGIN, target_cell_size=target_cell_size)

                # 1. Try to use MESH_ANCHOR from OpenSCAD script
                custom_location = None
                mesh_anchor_scaled = None
                if 'mesh_anchor' in locals() and mesh_anchor:
                    mesh_anchor_scaled = [mesh_anchor[0] * SCALE_FACTOR, mesh_anchor[1] * SCALE_FACTOR, mesh_anchor[2] * SCALE_FACTOR]
                    print(f"Verifying MESH_ANCHOR from OpenSCAD: {mesh_anchor_scaled}")

                # 2. Ray trace/Grid search to verify or find a new internal point
                # Pass mesh_anchor_scaled as the given point so it's checked first
                custom_location = scad_driver.get_internal_point(stl_path, given_point=mesh_anchor_scaled)

                if custom_location:
                    print(f"Using verified internal point: {custom_location}")
                else:
                    print("ERROR: Could not find ANY point strictly inside the mesh.")
                    print("This usually means the geometry is completely non-manifold, zero-thickness, or inverted.")
                    print("Rejecting geometry to prevent snappyHexMesh segfault (Exit 139).")

                    error_details = "Geometry generated a completely invalid volume (no internal point found). This leads to snappyHexMesh segfaults."
                    metrics = {
                        "error": "geometry_invalid_volume",
                        "penalty": 1e9,
                        "details": error_details
                    }
                    return metrics, [], None, None, None

                # Update location
                if hasattr(physics_driver, 'update_snappyHexMesh_location'):
                    physics_driver.update_snappyHexMesh_location(bounds_arr, custom_location=custom_location)
        else:
            print("[Reuse Mesh] Skipping BlockMesh update.")
    elif skip_cfd:
        print("[Skip CFD] Skipping BlockMesh update.")
    else:
         print("[Dry Run] Updated blockMeshDict")

    # 3. Run Simulation
    metrics = {}

    # Store the calculated cell size for reproducibility in the logs
    if not reuse_mesh and 'target_cell_size' in locals():
        metrics['target_cell_size_m'] = target_cell_size

    vtk_zip_path = None

    if not dry_run and not skip_cfd:
        if not reuse_mesh:
            with Timer("Meshing"):
                # Pass bin config AND assets
                # We pass just filenames as they are in triSurface
                asset_filenames = {k: os.path.basename(v) for k, v in cfd_assets.items()}

                # Extract add_layers parameter, default True
                add_layers = params.get("add_layers", True)

                success = physics_driver.run_meshing(log_file=mesh_log, bin_config=bin_config, stl_assets=asset_filenames, add_layers=add_layers)
        else:
            print("[Reuse Mesh] Skipping meshing pipeline.")
            success = True

        if success:
            if dry_mesh:
                print("[Dry Mesh] Evaluating mesh quality and skipping CFD solver...")
                if hasattr(physics_driver, '_run_checkMesh') and hasattr(physics_driver, '_classify_mesh'):
                    mesh_metrics = physics_driver._run_checkMesh()
                    mesh_class = physics_driver._classify_mesh(mesh_metrics)
                else:
                    mesh_metrics = {}
                    mesh_class = "unknown"
                metrics.update({
                    "mesh_quality_class": mesh_class,
                    "max_non_orthogonality": mesh_metrics.get("max_non_orthogonality", 0.0),
                    "max_skewness": mesh_metrics.get("max_skewness", 0.0),
                    "failed_checks": mesh_metrics.get("failed_checks", 0),
                    "dry_mesh_completed": True
                })
            else:
                with Timer("Solver"):
                    _scaled = mesh_scaled_for_memory if 'mesh_scaled_for_memory' in locals() else False
                    # User directive: keep turbulence model active on coarse mesh instead of trapping into laminar FPE

                    success = physics_driver.run_solver(log_file=solver_log, mesh_scaled_for_memory=_scaled)

                if success:
                    # Fetch physics metrics and preserve any existing custom metrics (like cell size)
                    physics_metrics = physics_driver.get_metrics(log_file=solver_log)
                    metrics.update(physics_metrics)

                    # CFD-specific particle tracking check
                    if hasattr(physics_driver, 'run_particle_tracking'):
                        delta_p = metrics.get('delta_p')
                        residuals = metrics.get('residuals')
                        if delta_p is None or (residuals is not None and residuals > 1e-3):
                            print(f"Skipping particle tracking: flow field appears invalid or unconverged (delta_p={delta_p}, residuals={residuals}).")
                        else:
                            with Timer("Particle Tracking"):
                                physics_driver.run_particle_tracking(log_file=solver_log, bin_config=bin_config, turbulence=turbulence, mesh_scaled_for_memory=_scaled)

                            particle_metrics = physics_driver.get_metrics(log_file=solver_log)
                            metrics.update(particle_metrics)

                    # Generate VTK if supported
                    if hasattr(physics_driver, 'generate_vtk'):
                        vtk_dir = physics_driver.generate_vtk()
                        if vtk_dir:
                            timestamp = int(time.time())
                            # Archive to exports/
                            zip_name = os.path.join("exports", f"run_{timestamp}_vtk")
                            print(f"Zipping VTK output to {zip_name}.zip...")
                            shutil.make_archive(zip_name, 'zip', vtk_dir)
                            vtk_zip_path = zip_name + ".zip"
                else:
                    print(f"Solver failed. Check {solver_log}")
                    metrics = {"error": "solver_failed"}
        else:
            print(f"Meshing failed. Check {mesh_log}")
            metrics = {"error": "meshing_failed"}
    elif skip_cfd:
        print("[Skip CFD] Skipping CFD simulation.")
        metrics = {"skipped": True, "note": "CFD simulation skipped by user request"}
    else:
        print("[Dry Run] Ran OpenFOAM simulation")
        # Mock metrics for dry run
        import random
        metrics = {
            "delta_p": 100 + random.randint(0, 50),
            "residuals": 1e-5
        }

    # 4. Generate Visualization (Solid Model for LLM/Human Review) and Archive Fluid STL
    png_paths = []

    # Determine output paths
    if output_prefix:
        vis_base = f"{output_prefix}_solid"
        fluid_stl_dest = f"{output_prefix}_fluid.stl"
    else:
        timestamp = int(time.time())
        vis_base = os.path.join("exports", f"run_{timestamp}_solid")
        fluid_stl_dest = os.path.join("exports", f"run_{timestamp}_fluid.stl")

    os.makedirs(os.path.dirname(vis_base), exist_ok=True)

    solid_stl_path = f"{vis_base}.stl"
    fluid_stl_final_path = None

    if not dry_run:
        # Use lower resolution for vis to speed up
        vis_params = params.copy()
        vis_params["high_res_fn"] = 20 # Low res enough for shape check

        with Timer("Visualization"):
            png_paths = scad_driver.generate_visualization(vis_params, vis_base, log_file=vis_log, params_file=params_file)

        # Copy Fluid STL to output location
        if os.path.exists(stl_path):
            try:
                shutil.copy(stl_path, fluid_stl_dest)
                fluid_stl_final_path = fluid_stl_dest
                print(f"Archived fluid STL to {fluid_stl_dest}")
            except Exception as e:
                print(f"Warning: Failed to copy fluid STL: {e}")
        else:
            print(f"Warning: Fluid STL not found at {stl_path}, cannot archive.")

        # Handle debug logic: Save successful case directories out of the volatile RAM disk,
        # or clear the case directory to save space if not successful and debug is off.
        if "error" not in metrics or debug:
            if hasattr(physics_driver, 'template_dir') and physics_driver.case_dir != os.path.abspath(physics_driver.template_dir):
                # Run was successful (or we are in debug mode), so we must preserve the data to persistent disk.
                # Copy the volatile case_dir (likely in /dev/shm) to a persistent exports location
                persistent_case_dest = f"{output_prefix}_case"
                try:
                    shutil.copytree(physics_driver.case_dir, persistent_case_dest, dirs_exist_ok=True)
                    print(f"Archived successful case to {persistent_case_dest}")
                except Exception as e:
                    print(f"Warning: Failed to archive successful case: {e}")
        elif "error" in metrics and not debug:
            if hasattr(physics_driver, 'template_dir') and physics_driver.case_dir != os.path.abspath(physics_driver.template_dir):
                # We can safely clear the volatile case dir contents since it failed and we don't need them
                try:
                    for filename in os.listdir(physics_driver.case_dir):
                        file_path = os.path.join(physics_driver.case_dir, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                except Exception as e:
                    if verbose: print(f"Warning: Failed to clean up failed case dir: {e}")

    else:
        print(f"[Dry Run] Generated Visualization at {vis_base}.png")
        # Create dummy path for dry run consistency
        png_paths = []
        fluid_stl_final_path = fluid_stl_dest # Pretend we copied it

    return metrics, png_paths, solid_stl_path, fluid_stl_final_path, vtk_zip_path
