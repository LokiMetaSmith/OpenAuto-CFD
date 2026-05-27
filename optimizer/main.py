import os
import json
import argparse
import time
import subprocess
import shutil
import hashlib
import sys
import uuid
import yaml
from dotenv import load_dotenv
from scad_driver import ScadDriver
from physics_factory import PhysicsEngineFactory
from llm_agent import LLMAgent
from data_store import DataStore
from simulation_runner import run_simulation
from job_manager import JobManager
from constraints import CONSTRAINTS
try:
    from scoring import calculate_score
except ImportError:
    from optimizer.scoring import calculate_score

def get_params_hash(params):
    """Generates a stable hash for a parameter set to detect duplicates."""
    normalized = {}
    for k, v in params.items():
        if isinstance(v, float):
            normalized[k] = round(v, 4)
        else:
            normalized[k] = v
    s = json.dumps(normalized, sort_keys=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generative AI Optimizer for Config-Driven Workflows")
    parser.add_argument("config_file", type=str, help="Path to the problem definition YAML file")
    parser.add_argument("--iterations", type=str, default="5", help="Number of iterations (int), or 'inf'/'infinite'/-1 for infinite loop")
    parser.add_argument("--case-dir", type=str, default="corkscrewFilter", help="Path to OpenFOAM case directory")
    parser.add_argument("--output-stl", type=str, default="corkscrew_fluid.stl", help="Output STL filename (for OpenFOAM usage)")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual OpenFOAM execution (mocks everything)")
    parser.add_argument("--skip-cfd", action="store_true", help="Generate geometry but skip CFD simulation")
    parser.add_argument("--dry-mesh", action="store_true", help="Run geometry generation and meshing, evaluate mesh quality, but skip CFD simulation")
    parser.add_argument("--reuse-mesh", action="store_true", help="Reuse existing mesh (skips geometry generation and meshing)")
    parser.add_argument("--container-engine", type=str, default="auto", choices=["auto", "podman", "docker"], help="Force specific container engine")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to use for parallel execution (default: 1)")
    parser.add_argument("--no-llm", action="store_true", help="Explicitly disable LLM and use random/fallback strategy (also suppresses prompts in startup script)")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of parameter sets to generate per LLM call")
    parser.add_argument("--no-cleanup", action="store_true", help="Disable cleanup of artifacts (STLs, images) for non-top runs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output (e.g. error logs)")
    parser.add_argument("--debug", action="store_true", help="Run CFD inside a RAM disk to save disk wear and preserve all logs")
    parser.add_argument("--parallel-workers", type=int, default=0, help="Number of parallel worker processes to spawn (0 = sequential)")
    parser.add_argument("--params-file", type=str, help="Path to a SCAD parameter file to use as the base configuration (overrides defaults)")
    parser.add_argument("--turbulence", type=str, default="laminar", help="Turbulence model to use (default: laminar, can be kOmegaSST, RNGkEpsilon, etc.)")
    args = parser.parse_args()

    # Parse iterations argument
    infinite_mode = False
    try:
        if args.iterations.lower() in ["inf", "infinite", "-1"]:
            infinite_mode = True
            max_iterations = float("inf")
        else:
            max_iterations = int(args.iterations)
            if max_iterations < 0:
                infinite_mode = True
                max_iterations = float("inf")
    except ValueError:
        print(f"Error: Invalid iterations value '{args.iterations}'. Using default 5.")
        max_iterations = 5

    # Load YAML config
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file {args.config_file}: {e}")
        sys.exit(1)

    scad_file = config.get('geometry', {}).get('scad_file', 'corkscrew.scad')
    fluid_volume_module = config.get('geometry', {}).get('fluid_volume_module', 'modular_filter_assembly')

    # Initialize components
    scad = ScadDriver(scad_file, fluid_volume_module=fluid_volume_module)

    # Instantiate physics driver using the factory
    physics_driver = PhysicsEngineFactory.get_driver(
        args.case_dir,
        config=config,
        container_engine=args.container_engine,
        num_processors=args.cpus,
        verbose=args.verbose,
        debug=args.debug
    )

    # Handle --no-llm logic
    if args.no_llm and "GEMINI_API_KEY" in os.environ:
        print("Explicitly disabling LLM due to --no-llm flag.")
        del os.environ["GEMINI_API_KEY"]

    agent = LLMAgent()
    store = DataStore()
    manager = JobManager(store)

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    # Get git commit
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        if status:
            git_commit += " (dirty)"
    except Exception as e:
        print(f"Warning: Failed to retrieve git commit info: {e}")
        git_commit = "unknown"

    # Initial parameters
    if args.params_file:
        print(f"Using parameters file: {args.params_file}. Clearing initial_params to allow file to take precedence.")
        # We must provide a non-empty dictionary for DataStore validation, but we don't want to override file values.
        # Adding a metadata key like "_source" is safe as OpenSCAD likely ignores it or it just sets a variable.
        initial_params = {"_source": args.params_file}
    else:
        # Extract default parameters from YAML config
        initial_params = {}
        for param_name, param_def in config.get('geometry', {}).get('parameters', {}).items():
            if 'default' in param_def:
                initial_params[param_name] = param_def['default']
            elif param_def.get('constant', False) and 'value' in param_def:
                initial_params[param_name] = param_def['value']

    # Fallback to avoid empty initial parameters if no parameters are defined but the optimization loop expects them
    if not initial_params and not args.params_file:
        initial_params = {"_dummy": True}

    # Extract constraints for the LLM
    constraints_str = config.get('optimization', {}).get('constraints', '')
    objective_func = config.get('optimization', {}).get('objective_function', 'separation_efficiency')
    optimization_target = config.get('optimization', {}).get('target', 'maximize')
    optimization_desc = config.get('optimization', {}).get('description', '')

    # Load History & Populate Visited Set
    print("Loading history from data store...")
    full_history = store.load_history()
    agent.history = full_history

    visited_params = set()
    for run in full_history:
        if "parameters" in run:
            visited_params.add(get_params_hash(run["parameters"]))

    print(f"Loaded {len(full_history)} past runs. Found {len(visited_params)} unique parameter sets.")
    print(f"Starting optimization loop... (Target: {max_iterations} iterations, Parallel Workers: {args.parallel_workers})")

    parameter_queue = []

    # Initial seed
    if get_params_hash(initial_params) not in visited_params:
        parameter_queue.append(initial_params)

    last_run_images = []
    i = 0

    while True:
        if not infinite_mode and i >= max_iterations:
            break

        print(f"\n=== Iteration {i+1} ===")

        # Refill Queue if empty
        if not parameter_queue:
            print(f"Parameter queue empty. Requesting {args.batch_size} new sets from LLM...")

            campaign_params = agent.suggest_campaign(
                history=full_history,
                constraints=constraints_str,
                objective=objective_func,
                target=optimization_target,
                description=optimization_desc,
                parameters_def=config.get('geometry', {}).get('parameters', {}),
                count=args.batch_size,
                image_paths=last_run_images
            )

            if campaign_params and campaign_params[0].get("stop_optimization") is True:
                print("\n>>> OPTIMIZATION COMPLETE: LLM signaled stop. <<<")
                break

            if campaign_params:
                print(f"LLM returned {len(campaign_params)} parameter sets.")
                # Filter duplicates immediately
                added = 0
                for p in campaign_params:
                     if get_params_hash(p) not in visited_params:
                         parameter_queue.append(p)
                         added += 1
                     else:
                         print("Skipping duplicate suggested by LLM.")
                if added == 0:
                    print("All suggestions were duplicates.")
            else:
                print("LLM failed/fallback. Generating random.")
                base_params = full_history[-1]["parameters"] if full_history else initial_params
                random_params = agent._generate_random_parameters(base_params, config.get('geometry', {}).get('parameters', {}))
                if get_params_hash(random_params) not in visited_params:
                    parameter_queue.append(random_params)

        if not parameter_queue:
            print("Queue empty after refill. Retrying next loop (or breaking if strictly limited).")
            # Force random if stuck?
            # For now, break to avoid infinite loop of nothing.
            break

        # === EXECUTION PHASE ===

        if args.parallel_workers > 0:
            # Parallel Execution Strategy
            print(f"Parallel Mode: Processing {len(parameter_queue)} items with {args.parallel_workers} workers.")

            current_batch_ids = []

            # 1. Submit all queue items to JobManager
            while parameter_queue:
                params = parameter_queue.pop(0)
                phash = get_params_hash(params)
                visited_params.add(phash)

                job_id = manager.create_job(params)
                current_batch_ids.append(job_id)
                print(f"Submitted Job {job_id}")

            # 2. Spawn Workers
            workers = []
            print(f"Spawning {args.parallel_workers} workers...")
            for w in range(args.parallel_workers):
                cmd = [sys.executable, "optimizer/worker.py", "--id", f"worker-{w}", "--loop", "--local", args.config_file]
                if args.dry_run: cmd.append("--dry-run")
                if args.case_dir: cmd.extend(["--case-dir", args.case_dir])
                # We assume workers pick up from the same DB file location (default behavior)

                p = subprocess.Popen(cmd)
                workers.append(p)

            # 3. Wait for Batch
            try:
                while True:
                    time.sleep(5)
                    states = manager._get_all_latest_states()

                    completed_count = 0
                    failed_count = 0
                    running_count = 0

                    for jid in current_batch_ids:
                        state = states.get(jid, {})
                        status = state.get("status", "unknown")
                        if status == "completed": completed_count += 1
                        elif status == "failed": failed_count += 1
                        elif status == "running": running_count += 1

                    print(f"Batch Progress: {completed_count} done, {failed_count} failed, {running_count} running...", end='\r')

                    if completed_count + failed_count == len(current_batch_ids):
                        print("\nBatch Complete.")
                        break
            except KeyboardInterrupt:
                print("\nInterrupted. Killing workers...")
                for p in workers: p.terminate()
                raise
            finally:
                # Terminate workers
                for p in workers:
                    if p.poll() is None:
                        p.terminate()

            # 4. Collect Results
            full_history = store.load_history()
            agent.history = full_history

            # Update last_run_images from one of the successful jobs
            for run in full_history:
                if run.get("id") in current_batch_ids and run.get("status") == "completed":
                    if run.get("images"):
                        last_run_images = run["images"]

            i += len(current_batch_ids)

        else:
            current_params = parameter_queue.pop(0)

            # Deduplication Check
            param_hash = get_params_hash(current_params)
            if param_hash in visited_params:
                print(f"Skipping duplicate parameters (Hash: {param_hash}).")
                # If queue is empty, we'll hit refill next loop. If not, we just take next.
                continue

            print(f"Testing parameters: {current_params}")
            visited_params.add(param_hash)

            # Generate Run ID and Output Prefix
            # Use timestamp to ensure uniqueness
            run_timestamp = time.time()
            run_id_hash = hashlib.md5(f"{i}_{run_timestamp}".encode()).hexdigest()
            run_id_short = run_id_hash[:8]
            output_prefix = os.path.join("exports", f"run_{run_id_short}")

            # Run Simulation via Runner
            # output_stl is strictly 'corkscrew_fluid.stl' for OpenFOAM compatibility
            metrics, png_paths, solid_stl_path, fluid_stl_path, vtk_zip_path = run_simulation(
                scad,
                physics_driver,
                current_params,
                output_stl_name=args.output_stl,
                dry_run=args.dry_run,
                skip_cfd=args.skip_cfd,
                dry_mesh=args.dry_mesh,
                iteration=i,
                reuse_mesh=args.reuse_mesh,
                output_prefix=output_prefix,
                verbose=args.verbose,
                params_file=args.params_file,
                turbulence=args.turbulence,
                debug=args.debug
            )

            print(f"Result metrics: {metrics}")

            # Update images for next LLM call
            if png_paths:
                last_run_images = png_paths

            # Check for critical failure
            if "error" in metrics:
                if metrics["error"] == "environment_missing_tools":
                    print("\nCRITICAL ERROR: OpenFOAM tools not found.")
                    print("Switching to geometry-only mode (--skip-cfd) for remaining iterations.")
                    args.skip_cfd = True
                elif metrics["error"] == "geometry_generation_failed":
                    print("Geometry generation failed.")

            # Save Results
            run_data = {
                "id": run_id_hash, # Unique ID
                "status": "completed",
                "git_commit": git_commit,
                "agent_id": "optimizer-script",
                "iteration": i,
                "timestamp": run_timestamp,
                "parameters": current_params.copy(),
                "params_file": args.params_file,
                "metrics": metrics,
                "images": png_paths,
                "solid_stl_path": solid_stl_path,
                "fluid_stl_path": fluid_stl_path,
                "artifact_stl_path": fluid_stl_path, # Backward compatibility / Alias
                "artifact_vtk_path": vtk_zip_path
            }
            store.append_result(run_data)

            # Reload history to include this run
            full_history.append(run_data)
            agent.history = full_history

            # Cleanup Artifacts (Keep Top 10)
            # We do this every run to save space
            if not args.no_cleanup:
                # Use custom top runs method to support configs
                try:
                    from scoring import calculate_score
                    sorted_runs = sorted(full_history, key=lambda r: calculate_score(r.get("metrics", {}), config), reverse=True)
                    top_runs = sorted_runs[:10]
                    store.clean_artifacts(top_runs)
                except Exception as e:
                    print(f"Error cleaning artifacts: {e}")
                    # Fallback
                    top_runs = store.get_top_runs(10)
                    store.clean_artifacts(top_runs)

            i += 1

    # Cleanup the driver RAM disk when loop is finished
    physics_driver.cleanup_ram_disk()

    print("\nOptimization loop finished.")

if __name__ == "__main__":
    main()
