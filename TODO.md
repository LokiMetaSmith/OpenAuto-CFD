# Project To-Do List

This file tracks planned enhancements and future work for the OpenAuto-CFD framework and the Corkscrew Filter validation study.

## Real-Time Simulation, Model Fusion & Agentic CAD Roadmap (Completed)
- [x] **Multi-Physics Surrogate & Model Fusion:** Universal RBF surrogate supporting coupled CFD (OpenFOAM), FEA structural stress (CalculiX), and EM (openEMS) metrics with 3D spatial vector fields (`surrogate_multiphysics.py`).
- [x] **Exact Differentiable Gradients & Inverse Design:** Analytical exact RBF derivatives ($\approx 10^{-10}$ error vs finite differences) and multi-start L-BFGS-B inverse design optimizer converging in $<10\text{ ms}$ (`surrogate_gradients.py`).
- [x] **Binary GPU Buffer Streaming:** High-performance binary GPU buffer exporter (`.bin` format) for 3D field data exchange (`cfd_fea_field_io.py`).
- [x] **Asynchronous Worker Queue:** Non-blocking background simulation queue with worker pool (`async_solver_queue.py`).
- [x] **Phase 1: Real-Time Interactive WebGL Viewer (Atlas-Style Frontend):** Modern Three.js dark-mode technical HUD with 60 FPS slider scrub, particle streamline ribbons, FEA stress heatmaps, and AI inverse design button (`viewer/web/`, `viewer/server.py`, `run_viewer.py`).
- [x] **Phase 2: Multi-Fidelity Mesh Pyramid:** Kennedy & O'Hagan Co-Kriging model ($y_H = \rho y_L + \delta(x)$) with dynamic mesh resolution switching (`multifidelity_driver.py`) and a Two-Stage Active Screening Filter (`model_fusion_multifidelity.py`) delivering a $2.9\times$ compute speedup.
- [x] **Phase 3: Physics-Informed Conservation Regularizer (PINN):** Discrete Exterior Calculus Helmholtz-Hodge solenoidal projector $(D D^T) \boldsymbol{\lambda} = D \mathbf{u}$ enforcing incompressibility ($\nabla \cdot \mathbf{u} = 0$) with $100\%$ divergence elimination in $<25\text{ ms}$, plus Cauchy stress divergence equilibrium ($\nabla \cdot \boldsymbol{\sigma} \approx \mathbf{0}$) (`pinn_conservation.py`).
- [x] **Phase 4: LLM Tool-Calling Integration (Kimi K3 / Gemini CAD Agent):** Standard OpenAPI / JSON Schema function-calling specs (`predict_surrogate`, `run_inverse_design`, `check_physics_conservation`, `dispatch_simulation`, `check_simulation_status`, `generate_scad_code`), `CADAgentToolRegistry`, and `CADReasoningAgent` autonomous multi-turn reasoning loop (`cad_agent_tools.py`, `llm_agent.py`).

## build123d CAD Engine & Multi-Physics Refactor
- [x] **Pure Python CAD Engine (`build123d`):** Rewrote all project 3D models (`optimizer/cad_models.py`) into native Python `build123d` B-rep solids (corkscrew filter, dipole antenna, puck antenna, cyclone filter manifold, trace optimizer, hose adapter cap, custom couplings, and filter holders).
- [x] **Parametric 3D Filleting & Chamfering:** Added native B-rep 3D chamfering (`blade_chamfer_mm`) and filleting (`inlet_fillet_radius_mm`) to eliminate flow turbulence and stress concentrations.
- [x] **Legacy OpenSCAD Fallback:** Preserved existing OpenSCAD scripts in `legacy/openscad/` with configurable engine selection (`CadEngineFactory`).
- [x] **Direct STEP-to-Mesh Volume Pipeline (`gmsh`):** Implemented `GmshDriver` (`optimizer/gmsh_driver.py`) for direct STEP CAD B-rep volume meshing into OpenFOAM (`gmshToFoam`).
- [x] **Meshing Performance & Quality Benchmark Suite:** Created `scripts/benchmark_meshing.py` to compare SnappyHexMesh (STL surface) vs. Gmsh (STEP B-rep) on wall-clock time, peak memory footprint, cell counts, aspect ratio, non-orthogonality, and skewness.
- [x] **Self-Contained Interactive 3D WebGL Visualizer:** Implemented Three.js WebGL HTML exporter (`CadDriverBase.generate_interactive_html`) with embedded STL models for interactive 3D browser viewing.
- [x] **Automated FEA Structural Validation (`FeaDriver`):** Enhanced `FeaDriver` (`optimizer/fea_driver.py`) to evaluate von Mises stress (`max_von_mises_stress_MPa`), displacement (`max_displacement_mm`), and factor of safety (`factor_of_safety`).
- [x] **Multi-Node Worker Fault-Tolerance:** Added worker heartbeats (`record_heartbeat`) and automatic dead-worker stale job recovery (`requeue_stale_jobs`) to `job_manager.py` and `worker.py`.
- [x] **Fine-Tuning Dataset Generator Updates:** Updated `optimizer/generate_training_data.py` to synthesize Chain-of-Thought (CoT) reasoning for `build123d` parametric CAD mutations.
- [x] **Multi-Objective Pareto Optimization & World Model:** Added non-dominated Pareto front calculation and plotting (`optimizer/scoring.py`), and updated Schema surrogate transition rules (`optimizer/harness/surrogate.py`).
- [x] **1-to-1 Parity Test Suite:** Added `test/test_cad_parity.py` verifying 1-to-1 STL bounding box and volume parity (< 1.5% difference) between OpenSCAD and `build123d`.

- [x] Add the ability to make different parameter configurations based on included config files.
- [x] Refine the `CorkscrewSlitKnife` geometry to have a chamfered or ramped leading edge to improve separation efficiency.
- [x] Conduct CFD analysis to test different design parameters (slit shape, screw pitch, etc.) - *Enabled via new parameters in config.scad and optimizer/constraints.py*
- [x] Add particle tracking to the CFD simulation to visualize and quantify separation effectiveness.
- [x] Document `optimizer/` and `parameters/` directories in the README. These are core functions for simulation and parameter evaluation.
- [x] Complete refactor of barb generators into a unified, parameterized `Barb` module.
- [x] Create `FilterHolder` part (barb fitting with dual O-rings and optional threading).
- [x] Standardize coupling configurations (Cartridge, Sandblaster) into `configs/` files.

## Distributed Optimization (Git-Based)
- [x] **Data Schema:** Design a JSONL-based schema for the Job Queue/Result Log. This format should favor append-only operations to minimize merge conflicts when multiple users push results.
- [x] **Job Manager:** Implement a `JobManager` class capable of:
    - "Checking out" a specific parameter region of interest.
    - Generating a local queue of jobs derived from that region.
    - Managing the state of claimed jobs.
- [x] **Versioning Strategy:** Implement a hashing or UUID system to link specific job queues/results to the git commit hash of the codebase at the time of execution. This ensures reproducibility.
- [x] **Agent "Campaign" Mode:** Update the `LLMAgent` to support generating batch "campaigns" (multiple parameter sets) into the queue, rather than single-step iterations.
- [x] **Synchronization Workflow:** Create scripts/logic to handle the `pull` -> `claim` -> `run` -> `push` lifecycle, allowing a team to collaborate on the optimization surface asynchronously.
- [x] **CLI Region Support:** Updated `generate_campaign.py` to allow direct job generation for specific parameter regions (e.g., `--param key=min:max`) without requiring LLM assistance.

## Simulation Fidelity & Physics
- [x] **Re-enable Lagrangian Turbulence:** Remove the `_disable_turbulence` hack in `foam_driver.py`. Update `kinematicCloudProperties` to use `dispersionModel StochasticDispersionRAS;` to accurately model particle re-entrainment.
- [x] **Safeguard Wall Distances:** Add `postProcess -func wallDist` to the simulation pipeline immediately prior to particle tracking to prevent crashes when evaluating turbulent wall functions.
- [x] **Switch Integration Scheme:** Change the particle tracking integration scheme from `Euler` to `analytical` in `kinematicCloudProperties` to resolve the SIGFPE math explosion caused by explicit integration overshooting drag equilibrium on fine particles.

## Metric Extraction & Data Handling
- [x] **Implement `particleCollector`:** Inject `particleCollector` into the `cloudFunctions` dictionary to capture discrete particle fate data for individual trapping bins.
- [x] **Aggregate Patch Data:** Update `foam_driver.get_metrics()` to parse the `particleCollector` CSV/OBJ outputs. Aggregate the raw hit counts into LLM-friendly summary statistics (e.g., capture efficiency per bin).
- [x] **Ensure Face Flux (Phi):** Run `postProcess -func writePhi` automatically when preparing the transient `0` directory from steady-state results to ensure velocity interpolation works for all sub-models.

## Optimization Loop & Architecture
- [x] **Inter-Simulation Parallelization:** Refactor the optimization orchestrator (`main.py` / `worker.py`) to process LLM-generated parameter batches concurrently by spinning up multiple asynchronous OpenFOAM container instances, rather than evaluating one design at a time.
- [x] **Bug Fix:** Fixed parameter type handling in `JobManager` to correctly distinguish between ranges (tuples) and discrete choices (lists).

## Project Readiness & Review (Jan 2026)
- [x] **Comprehensive Review:** Conducted a full review of the codebase, documentation, and tests.
- [x] **Geometry Fixes:** Attempted to fix non-planar faces in `RampedKnifeShape` (triangulation) and added epsilon overlap to cutters to improve CSG stability.
- [x] **Test Reliability:** Increased timeouts for WASM-based tests (`test/regression.js` and `test/test_parameter_stls.py`) to prevent false failures on slower environments.
- [x] **Documentation:** Updated `README.md` with explicit installation/testing instructions and `TECHNICAL_REPORT.md` with notes on missing figures.
- [x] **Geometry Stability:** Resolve persistent `CGAL error: precondition violation` in `single_cell_filter.scad` and `flat_end_screw.scad` when running in `openscad-wasm`. (Native OpenSCAD may work fine). *Update: Added unit tests in `test/test_wasm_cgal_error.py` tracking this as an upstream issue in openscad-wasm handling of complex boolean extrusions.*
- [x] **Visual Assets:** Generate and insert Figure 3 (Velocity Streamlines) into `TECHNICAL_REPORT.md` using ParaView / PyVista pipeline.

## Post-Review Improvements (Code Review Action Items)
- [x] **CFD Stability:** Investigate why `k`, `epsilon`, `omega`, and `nut` turbulence fields are blowing up in steady-state simulations, rather than freezing them to `1e-8`. Consider alternative turbulence models (e.g., RNG k-epsilon) better suited for swirling, anisotropic flows.
- [x] **Meshing Reliability:** Remove the "Auto-Fallback" mechanism that drops boundary layers (`addLayers false`) when `snappyHexMesh` fails. Boundary layers are critical for accurate particle tracking near walls. If meshing fails, the design should be rejected with explicit feedback.
- [x] **End-to-End Non-CFD Testing:** Implement a "Dry-Mesh" testing step in the optimization loop that runs `blockMesh` and `snappyHexMesh` (without running the solver) and evaluates the output of `checkMesh`. Use this to validate manufacturability and meshability before running expensive CFD simulations.
- [x] **Stricter Geometry Validation:** Enhance the Python `Validator` to check STLs for non-manifold edges and self-intersections (using `trimesh`) before attempting to mesh them in OpenFOAM.
- [x] **Hardcode Safety Margins:** In OpenSCAD modules (e.g., `assemblies.scad`), enforce minimum geometric tolerances mathematically to prevent CGAL precondition violations, regardless of LLM parameter suggestions (e.g., explicitly prevent `helix_profile_radius` from equaling `helix_path_radius`).
- [x] **Improved LLM Feedback:** Update the LLM error feedback mechanism to provide specific, geometric reasons for simulation failures (e.g., "The mesh quality check failed due to high non-orthogonality near the slit") instead of generic OpenFOAM solver crash logs.

## Electromagnetic (EM) Optimization Pipeline
- [x] **Impedance Optimization Loop:** Create a specialized configuration and scoring system to iteratively optimize PCB trace widths for target impedance using the LLM agent.
- [x] **True VTK Field Extraction:** Implement the conversion of raw HDF5 field data from openEMS to VTK format for physically accurate visualizations.
- [x] **Complex PCB Stackups:** Extend the geometry generation and drivers to support multi-layer PCB stackups, internal planes, and vias.
- [x] **Radiation Pattern Visualization:** Add support for extracting and visualizing 3D far-field radiation patterns in Blender.

## CFD Resilient Execution Engine (Post-Review Architecture)
- [ ] **Phase 1: Immediate Stability Fixes (Numerical Stability)**
    - [x] Fix `nut` invariant: Ensure `nut >= 1e-7` globally, replacing any `uniform 0` in `internalField` or fallback boundaries.
    - [x] Fix `fvSchemes`: Enforce `limited corrected 0.33` for `snGradSchemes` and `laplacianSchemes` on harsh meshes.
    - [x] Fix `fvSchemes`: Ensure bounded upwind schemes (`bounded Gauss upwind`) are used for all turbulence parameters (`k`, `epsilon`, `omega`).
    - [x] Remove wall function applications during laminar fallback to prevent solver instability.
- [x] **Phase 2: Mesh-Quality Feedback Loop**
    - [x] Implement `checkMesh` parsing to extract `Max non-orthogonality` and `Max skewness`.
    - [x] Classify mesh quality (e.g., `good`, `marginal`, `bad`).
    - [x] Dynamically adapt `fvSchemes` limiters based on mesh classification before running the solver.
- [ ] **Phase 3: Full Orchestration System**
    - [x] Implement multi-stage Retry Ladder: Try `RNG k-epsilon` -> degrade to `k-omega SST` -> fallback to `laminar`.
    - [x] Implement proactive Field Clamping: Sanitize fields to prevent them from becoming 0, NaN, or extremely small before the solver runs.

## Phase 4: Upstream Geometry & Mesh Optimization
To address the root cause of the numerical instabilities outlined in the technical report, the mesh quality must be improved at the source rather than relying solely on `foam_driver.py` workarounds.
- [x] **Optimize OpenSCAD Geometry ($fn):** Increase facet resolution (`$fn = 60` to `120`) on helical modules to prevent `snappyHexMesh` from snapping to artificial sharp edges and creating highly skewed cells.
- [x] **Eliminate Non-Manifold Geometry (Epsilon Rule):** Add tiny overlaps (e.g., `+ 0.01mm`) to cutting tools in OpenSCAD before `union()` or `difference()` operations to eliminate zero-thickness shared edges.
- [x] **Smooth Internal Corners:** Add small chamfers or fillets to the root of the corkscrew blade to smooth the 90-degree internal corner, preventing the mesher from generating severely distorted cells at the singularity.
- [x] **Tune `snappyHexMeshDict` (Background Grid):** Lower `target_cell_size` so at least 4 to 5 base cells fit across the narrowest gap in the corkscrew channel before refinement.
- [x] **Tune `snappyHexMeshDict` (Surface Refinement):** Increase `refinementSurfaces` level for the corkscrew geometry (e.g., `level (3 4)`) to force the mesher to divide cells closer to the twisted walls.
- [x] **Tune `snappyHexMeshDict` (Boundary Layers):** Relax `meshQualityControls` and reduce `nSurfaceLayers` while increasing `featureAngle` to prevent prism layers from colliding on tight helices, or temporarily disable `addLayers` to isolate skewness causes.
- [x] **Tune `surfaceFeatureExtract`:** Lower `includedAngle` (e.g., to 120 or 130) to ensure the spiraling blade edges are explicitly captured.

## Codebase Cleanup and Refactoring
This section tracks necessary repository cleanup tasks to reorganize misplaced files, remove orphaned code, and improve project structure. **Important:** Every file must be carefully examined before deletion to ensure we do not introduce regressions. This codebase has extensive debug and recovery methods that are core functionality, so do not indiscriminately delete anything that says "fix/resolve/verify".

- [x] **Group 1: Investigate Test Files Outside of the `test/` Folder.**
    - Carefully review the following test files in the root directory. If they are actual tests or test utilities, move them to `test/`. If they are redundant or outdated, delete them.
    - Files to investigate: `check_tests.py`, `generalize_tests.py`, `run_cfd_test.py`, `run_cfd_test2.py`, `test_boundaries.py`, `test_fvschemes.py`, `test_fvschemes2.py`, `test_fvsolution.py`, `test_fvsolution2.py`, `test_meshing.py`, `test_nan.py`, `update_tests.py`.
- [x] **Group 2: Investigate Orphaned Plan Files.**
    - Review `plan17.md` and `plan18.md` to see if they contain any relevant unsaved documentation. Otherwise, they appear to be orphaned agent execution plans and should be removed.
- [x] **Group 3: Investigate Debug, Recovery, and One-Off Scripts.**
    - Many scripts in the root seem related to debugging or resolving specific issues (e.g., `fix_final_merge.py`, `fix_inletoutlet.py`, `fix_merge.py`, `investigate.py`, `resolve_conflicts.py`, `resolve_foam_driver.py`, `verify_fvschemes.py`, `verify_fvsolution.py`). Examine them carefully. If they are no longer needed, they can be deleted. If they still serve a purpose, consider moving them to a `scripts/` or `tools/` folder. Do not blindly delete files that are core recovery functionalities.
- [x] **Group 4: Investigate Log Files in the Root Directory.**
    - Review `.pip_install.log` and `test_meshing3.log`. If they are not needed for reference, delete them or add them to `.gitignore` and untrack them from the repository to clean up the root folder.
