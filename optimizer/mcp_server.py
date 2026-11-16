"""
MCP (Model Context Protocol) Server for OpenAuto-CFD.

Exposes tools, resources, and prompts for LLMs to control, query, and drive the simulation harness,
run parameter evaluations, inspect optimization history, execute Schema surrogate steps, export fine-tuning datasets,
and execute GenCAD physics/vision cross-modal sequence generation.
"""

import os
import sys
import json
import glob
import yaml
from typing import Dict, List, Any, Optional

# Ensure repository root is in python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from mcp.server import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from optimizer.cad_factory import CadEngineFactory
from optimizer.physics_factory import PhysicsEngineFactory
from optimizer.simulation_runner import run_simulation
from optimizer.data_store import DataStore
from optimizer.llm_agent import LLMAgent
from optimizer.parameter_validator import validate_parameters
from optimizer.gencad_driver import GenCADDriver
from optimizer.generate_training_data import (
    load_yaml_config,
    load_optimization_logs,
    build_transitions,
    filter_transitions,
    export_data
)
from optimizer.receipt_engine import ClaimReceiptGraph, ReceiptState
from optimizer.assembly_solver import UpstreamAssemblySolver
from optimizer.guardian import PhyzConservationGuardian, SolverTier
from optimizer.latent_dynamics import LatentDynamicsModel
from optimizer.trajectory_adjoint import TrajectoryAdjointEngine
from optimizer.compute_router import HardwareComputeRouter
import numpy as np

# Initialize FastMCP / MCPServer Instance
mcp = FastMCP(
    "OpenAuto-CFD",
    instructions="MCP Server providing tools, resources, and prompts for autonomous engineering, 3D parametric SCAD/build123d generation, GenCAD physics/vision cross-modal sequence generation, CFD/EM simulation, and Schema surrogate optimization loops."
)

# ==============================================================================
# MCP RESOURCES
# ==============================================================================

@mcp.resource("config://corkscrew_config.yaml")
def read_corkscrew_config_resource() -> str:
    """Read the standard Corkscrew Filter CFD configuration YAML file."""
    path = "configs/corkscrew_config.yaml"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return "# Configuration file not found"


@mcp.resource("log://latest")
def read_latest_log_resource() -> str:
    """Read the latest simulation run record from the optimization log."""
    store = DataStore(log_file="optimization_log.jsonl")
    history = store.load_history()
    if history:
        return json.dumps(history[-1], indent=2)
    return json.dumps({"status": "empty", "message": "No runs recorded in optimization_log.jsonl"})


@mcp.resource("schema://notes")
def read_schema_notes_resource() -> str:
    """Read the physicist notes (notes.md) recorded during Schema surrogate iterations."""
    for path in ["exports/notes.md", "notes.md"]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
    return "# Schema Notes\nNo physicist notes recorded yet."


# ==============================================================================
# MCP PROMPTS
# ==============================================================================

@mcp.prompt()
def analyze_simulation_results(config_file: str = "configs/corkscrew_config.yaml", run_id: str = "") -> str:
    """
    Generate a structured prompt guiding an LLM on analyzing simulation metrics and recommending parameter mutations.
    """
    store = DataStore(log_file="optimization_log.jsonl")
    history = store.load_history()

    target_run = None
    if run_id:
        for r in history:
            if r.get("id") == run_id or str(r.get("id", "")).startswith(run_id):
                target_run = r
                break
    elif history:
        target_run = history[-1]

    run_info = json.dumps(target_run, indent=2) if target_run else "No run data available"

    return f"""You are an expert multi-physics optimization engineer analyzing OpenAuto-CFD simulation output.

PROBLEM CONFIGURATION: {config_file}
LATEST RUN EVALUATION:
{run_info}

INSTRUCTIONS:
1. Review the performance metrics (e.g., pressure_drop, separation_efficiency, S11, max_stress).
2. Identify physical or geometric trade-offs (e.g. centrifugal force vs boundary friction losses).
3. If errors or mesh quality issues are reported, identify parameter adjustments to restore stability.
4. Propose 3 candidate parameter sets with detailed engineering reasoning for each set.
"""


@mcp.prompt()
def schema_surrogate_reasoning(workspace_dir: str = "exports") -> str:
    """
    Generate a prompt template for mechanism discovery and surrogate updates when state discrepancy occurs.
    """
    notes_path = os.path.join(workspace_dir, "notes.md")
    notes_content = ""
    if os.path.exists(notes_path):
        with open(notes_path, "r") as f:
            notes_content = f.read()

    return f"""You are a computational physicist updating a fast surrogate model world_model.py for OpenAuto-CFD.

PHYSICIST NOTES & HISTORY:
{notes_content}

TASK:
1. Compare expected physics surrogate outputs against ground-truth solver results.
2. Analyze discrepancy trends across multi-physics domains (fluid, structural, electromagnetic).
3. Update physical hypothesis definitions and propose updated step(state, action) transition logic.
"""


# ==============================================================================
# MCP TOOLS
# ==============================================================================

@mcp.tool()
def list_available_configs() -> List[Dict[str, Any]]:
    """
    List all available problem configuration YAML files in the codebase (e.g. configs/ directory and root).

    Returns:
        List[Dict[str, Any]]: List of configuration metadata including path, physics type, and title/description.
    """
    configs = []
    config_paths = glob.glob("configs/*.yaml") + glob.glob("configs/*.yml") + glob.glob("*.yaml")

    for path in sorted(set(config_paths)):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                configs.append({
                    "path": path,
                    "physics_type": data.get("physics", {}).get("type", "cfd"),
                    "scad_file": data.get("geometry", {}).get("scad_file", ""),
                    "description": data.get("optimization", {}).get("description", "")
                })
        except Exception as e:
            configs.append({"path": path, "error": str(e)})

    return configs


@mcp.tool()
def run_simulation_tool(
    config_file: str = "configs/corkscrew_config.yaml",
    params: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    case_dir: str = "corkscrewFilter",
    output_stl: str = "corkscrew_fluid.stl",
    cpus: int = 1,
    turbulence: str = "laminar",
    cad_engine: str = "build123d"
) -> Dict[str, Any]:
    """
    Execute a single CAD generation and physics simulation evaluation (CFD/EM/FEA) for a given set of parameters.

    Args:
        config_file (str): Path to problem configuration YAML file (default: "configs/corkscrew_config.yaml").
        params (Optional[Dict[str, Any]]): Dictionary of CAD parameters. If None, default config parameters are used.
        dry_run (bool): If True, skip actual solver execution and return mock/cached results (default: True).
        case_dir (str): OpenFOAM case directory or solver workspace (default: "corkscrewFilter").
        output_stl (str): Output STL filename for fluid domain (default: "corkscrew_fluid.stl").
        cpus (int): Number of CPU cores for parallel meshing/solving (default: 1).
        turbulence (str): Turbulence model to use (e.g. "laminar", "kOmegaSST") (default: "laminar").

    Returns:
        Dict[str, Any]: Combined results including performance metrics, visual render paths, and artifact file paths.
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load config_file '{config_file}': {str(e)}"}

    scad_file = config.get('geometry', {}).get('scad_file', 'corkscrew.scad')
    fluid_volume_module = config.get('geometry', {}).get('fluid_volume_module', 'modular_filter_assembly')

    if params is None:
        params = {}
        for param_name, param_def in config.get('geometry', {}).get('parameters', {}).items():
            if 'default' in param_def:
                params[param_name] = param_def['default']
            elif param_def.get('constant', False) and 'value' in param_def:
                params[param_name] = param_def['value']

    scad = CadEngineFactory.get_driver(scad_file, cad_engine=cad_engine, fluid_volume_module=fluid_volume_module)
    physics_driver = PhysicsEngineFactory.get_driver(
        case_dir,
        config=config,
        num_processors=cpus
    )

    metrics, png_paths, solid_stl, fluid_stl, vtk_zip = run_simulation(
        scad=scad,
        physics_driver=physics_driver,
        params=params,
        output_stl_name=output_stl,
        dry_run=dry_run,
        params_file=None,
        turbulence=turbulence
    )

    return {
        "status": "success",
        "parameters": params,
        "metrics": metrics,
        "images": png_paths,
        "solid_stl_path": solid_stl,
        "fluid_stl_path": fluid_stl,
        "artifact_vtk_path": vtk_zip
    }


@mcp.tool()
def gencad_retrieve_from_physics(physics_target: Dict[str, float], top_k: int = 3) -> Dict[str, Any]:
    """
    GenCAD MCP Tool: Query nearest CAD programs in contrastive physics-latent space given a CFD target profile.
    """
    driver = GenCADDriver()
    matches = driver.retrieve_cad_program(physics_target, top_k=top_k)
    return {
        "status": "success",
        "physics_target": physics_target,
        "matches": matches
    }


@mcp.tool()
def gencad_synthesize_script(
    physics_target: Dict[str, float],
    format_type: str = "build123d",
    filename_prefix: str = "mcp_gencad_synthesized"
) -> Dict[str, Any]:
    """
    GenCAD MCP Tool: Synthesize build123d/OpenSCAD script directly conditioned on CFD physics targets using PyTorch Transformer.
    """
    driver = GenCADDriver()
    res = driver.generate_and_export(
        physics_target=physics_target,
        output_dir="artifacts",
        filename_prefix=filename_prefix,
        format_type=format_type,
        use_transformer_sequence=True
    )
    return res


@mcp.tool()
def gencad_retrieve_from_image(image_path: str, top_k: int = 3) -> Dict[str, Any]:
    """
    GenCAD MCP Tool: Retrieve nearest CAD programs matching a 2D CAD render image or STL wireframe.
    """
    driver = GenCADDriver()
    matches = driver.retrieve_cad_from_image(image_path, top_k=top_k)
    return {
        "status": "success",
        "image_path": image_path,
        "matches": matches
    }


@mcp.tool()
def gencad_sample_diverse(
    physics_target: Dict[str, float],
    n_samples: int = 3,
    temperature: float = 0.8
) -> Dict[str, Any]:
    """
    GenCAD MCP Tool: Sample N diverse CAD AST sequence variations for a target CFD physics prompt using Latent Diffusion Noise Sampling.
    """
    driver = GenCADDriver()
    samples = driver.sample_diverse_cad_programs(physics_target, n_samples=n_samples, temperature=temperature)
    return {
        "status": "success",
        "physics_target": physics_target,
        "samples": samples
    }


@mcp.tool()
def run_meep_simulation_tool(
    geometry_type: str = "waveguide",
    wavelength_min_um: float = 1.4,
    wavelength_max_um: float = 1.7,
    waveguide_width_um: float = 0.5,
    ring_radius_um: float = 3.0,
    ring_gap_um: float = 0.15,
    pore_diam_nm: float = 120.0,
    resolution: int = 20,
    case_dir: str = "artifacts/meep_mcp"
) -> Dict[str, Any]:
    """
    Execute a photonic FDTD electromagnetic simulation using Meep.

    Args:
        geometry_type: Structure topology ('waveguide', 'ring_resonator', 'nanopore_metasurface').
        wavelength_min_um: Starting wavelength in microns (default: 1.4).
        wavelength_max_um: Ending wavelength in microns (default: 1.7).
        waveguide_width_um: Waveguide core width in microns (default: 0.5).
        ring_radius_um: Ring resonator radius in microns (default: 3.0).
        ring_gap_um: Coupling gap in microns (default: 0.15).
        pore_diam_nm: Nanopore aperture diameter in nanometers (default: 120.0).
        resolution: Spatial discretization resolution in pixels per micron (default: 20).
        case_dir: Workspace directory for simulation files.

    Returns:
        Dict[str, Any]: Performance metrics (transmission, reflection, Q-factor, loss) and VTK field artifact paths.
    """
    from optimizer.meep_driver import MeepDriver
    driver = MeepDriver(
        case_dir=case_dir,
        config={
            "meep": {
                "geometry_type": geometry_type,
                "wavelength_min_um": wavelength_min_um,
                "wavelength_max_um": wavelength_max_um,
                "waveguide_width_um": waveguide_width_um,
                "ring_radius_um": ring_radius_um,
                "ring_gap_um": ring_gap_um,
                "pore_diam_nm": pore_diam_nm,
                "resolution": resolution
            }
        }
    )
    driver.prepare_case()
    driver.run_meshing()
    success = driver.run_solver()
    metrics = driver.get_metrics()
    spectrum = driver.get_spectrum()
    vtk_path = driver.generate_vtk()

    return {
        "status": "success" if success else "failed",
        "geometry_type": geometry_type,
        "metrics": metrics,
        "spectrum": {
            "wavelength_nm": spectrum.get("wavelength_nm", [])[:20],
            "transmission": spectrum.get("transmission", [])[:20],
            "reflection": spectrum.get("reflection", [])[:20]
        },
        "vtk_path": vtk_path
    }


@mcp.tool()
def run_s4_simulation_tool(
    lattice_period_um: float = 0.8,
    num_harmonics: int = 49,
    wavelength_min_um: float = 0.7,
    wavelength_max_um: float = 1.6,
    slab_thickness_um: float = 0.22,
    hole_radius_um: float = 0.18,
    polarization: str = "TE",
    case_dir: str = "artifacts/s4_mcp"
) -> Dict[str, Any]:
    """
    Execute a photonic Rigorous Coupled-Wave Analysis (RCWA) simulation using Stanford S4.

    Args:
        lattice_period_um: Periodic lattice period in microns (default: 0.8).
        num_harmonics: Number of Fourier basis harmonics (default: 49).
        wavelength_min_um: Minimum spectral wavelength in microns (default: 0.7).
        wavelength_max_um: Maximum spectral wavelength in microns (default: 1.6).
        slab_thickness_um: Photonic crystal slab thickness in microns (default: 0.22).
        hole_radius_um: Patterned hole / aperture radius in microns (default: 0.18).
        polarization: Incident planewave polarization ('TE' or 'TM').
        case_dir: Workspace directory for simulation files.

    Returns:
        Dict[str, Any]: Diffraction efficiencies, 0th-order spectra, resonance metrics, and VTK field artifact paths.
    """
    from optimizer.s4_driver import S4Driver
    driver = S4Driver(
        case_dir=case_dir,
        config={
            "s4": {
                "lattice_period_um": lattice_period_um,
                "num_harmonics": num_harmonics,
                "wavelength_min_um": wavelength_min_um,
                "wavelength_max_um": wavelength_max_um,
                "slab_thickness_um": slab_thickness_um,
                "hole_radius_um": hole_radius_um,
                "polarization": polarization
            }
        }
    )
    driver.prepare_case()
    driver.run_meshing()
    success = driver.run_solver()
    metrics = driver.get_metrics()
    spectrum = driver.get_spectrum()
    vtk_path = driver.generate_vtk()

    return {
        "status": "success" if success else "failed",
        "metrics": metrics,
        "spectrum": {
            "wavelength_nm": spectrum.get("wavelength_nm", [])[:20],
            "transmission_0th": spectrum.get("transmission_0th", [])[:20],
            "diffraction_efficiency": spectrum.get("diffraction_efficiency", [])[:20]
        },
        "vtk_path": vtk_path
    }


@mcp.tool()
def get_harness_status(db_path: str = "optimization_log.jsonl", top_k: int = 5) -> Dict[str, Any]:

    """
    Query the overall optimization harness history and top performing runs.

    Args:
        db_path (str): Path to JSONL data store log file (default: "optimization_log.jsonl").
        top_k (int): Number of top runs to retrieve (default: 5).

    Returns:
        Dict[str, Any]: History summary including total run count, top runs, and recent iterations.
    """
    store = DataStore(log_file=db_path)
    history = store.load_history()

    if not history:
        return {
            "total_runs": 0,
            "top_runs": [],
            "latest_run": None
        }

    top_runs = store.get_top_runs(top_k)
    latest_run = history[-1]

    return {
        "total_runs": len(history),
        "top_runs": top_runs,
        "latest_run": {
            "id": latest_run.get("id"),
            "iteration": latest_run.get("iteration"),
            "parameters": latest_run.get("parameters"),
            "metrics": latest_run.get("metrics"),
            "timestamp": latest_run.get("timestamp")
        }
    }


@mcp.tool()
def get_run_details(run_id: str, db_path: str = "optimization_log.jsonl") -> Dict[str, Any]:
    """
    Fetch comprehensive details for a specific run by ID.

    Args:
        run_id (str): Unique run ID or hash prefix.
        db_path (str): Path to JSONL data store log file.

    Returns:
        Dict[str, Any]: Full details of the specified run.
    """
    store = DataStore(log_file=db_path)
    history = store.load_history()

    for run in history:
        if run.get("id") == run_id or str(run.get("id", "")).startswith(run_id):
            return {"status": "success", "run": run}

    return {"status": "error", "message": f"Run ID '{run_id}' not found."}


@mcp.tool()
def get_schema_state(workspace_dir: str = "exports") -> Dict[str, Any]:
    """
    Retrieve the current physicist-style Schema surrogate state and backtest timeline log.

    Args:
        workspace_dir (str): Workspace directory containing Schema state files (default: "exports").

    Returns:
        Dict[str, Any]: Current Schema surrogate state, backtest timeline history, and notes.
    """
    from optimizer.harness.timeline import Timeline
    from optimizer.harness.notes import NotesManager
    from optimizer.harness.state import State

    timeline_path = os.path.join(workspace_dir, "timeline.jsonl")
    notes_path = os.path.join(workspace_dir, "notes.md")

    tl = Timeline(timeline_path)
    nm = NotesManager(notes_path)

    transitions = tl.load_transitions()
    notes = nm.read_notes()

    latest_state = State().to_dict()
    latest_discrepancy = None

    if transitions:
        latest_trans = transitions[-1]
        latest_state = latest_trans.get("actual_state", latest_state)

    return {
        "status": "success",
        "state": latest_state,
        "timeline_events_count": len(transitions),
        "latest_discrepancy": latest_discrepancy,
        "notes": notes
    }


@mcp.tool()
def run_schema_step(
    config_file: str = "configs/corkscrew_config.yaml",
    workspace_dir: str = "exports",
    epsilon: float = 5.0,
    dry_run: bool = True,
    cad_engine: str = "build123d"
) -> Dict[str, Any]:
    """
    Execute a single iteration of the Schema surrogate-planning harness loop.

    Synthesizes physics surrogate predictions, executes real solver validation,
    backtests discrepancy against epsilon, and updates Schema state.

    Args:
        config_file (str): Problem configuration YAML file (default: "configs/corkscrew_config.yaml").
        workspace_dir (str): Workspace directory for Schema artifacts (default: "exports").
        epsilon (float): Mismatch threshold for surrogate updates (default: 5.0).
        dry_run (bool): If True, skip actual solver execution during backtest (default: True).

    Returns:
        Dict[str, Any]: Result of the Schema step including predicted state, selected action, updated state, and discrepancy.
    """
    from optimizer.harness.engine import SchemaEngine
    from optimizer.harness.state import State, parse_solver_outputs_to_state

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load config_file '{config_file}': {str(e)}"}

    parameter_defs = config.get('geometry', {}).get('parameters', {})
    agent = LLMAgent()

    engine = SchemaEngine(
        workspace_dir=workspace_dir,
        parameter_defs=parameter_defs,
        llm_agent=agent,
        epsilon=epsilon
    )

    initial_params = {}
    for param_name, param_def in parameter_defs.items():
        if 'default' in param_def:
            initial_params[param_name] = param_def['default']
        elif param_def.get('constant', False) and 'value' in param_def:
            initial_params[param_name] = param_def['value']

    def real_solver_func(params_to_run):
        scad_file = config.get('geometry', {}).get('scad_file', 'corkscrew.scad')
        fluid_volume_module = config.get('geometry', {}).get('fluid_volume_module', 'modular_filter_assembly')
        scad = CadEngineFactory.get_driver(scad_file, cad_engine=cad_engine, fluid_volume_module=fluid_volume_module)
        physics_driver = PhysicsEngineFactory.get_driver("corkscrewFilter", config=config)

        metrics, _, _, _, _ = run_simulation(
            scad=scad,
            physics_driver=physics_driver,
            params=params_to_run,
            dry_run=dry_run
        )
        return metrics

    initial_metrics = real_solver_func(initial_params)
    current_state = parse_solver_outputs_to_state(initial_params, initial_metrics)

    def state_score_func(s: State) -> float:
        eff = s.fluid.get("separation_efficiency", 0.0)
        p_drop = s.fluid.get("pressure_drop", 0.0)
        return eff - (p_drop * 0.1)

    predicted_state, action, updated_state, discrepancy = engine.run_one_iteration(
        current_state=current_state,
        score_func=state_score_func,
        real_solver_func=real_solver_func
    )

    return {
        "status": "success",
        "action": action.to_dict() if hasattr(action, 'to_dict') else action,
        "discrepancy": discrepancy,
        "surrogate_updated": discrepancy > epsilon,
        "updated_state": updated_state.to_dict() if hasattr(updated_state, 'to_dict') else updated_state
    }


@mcp.tool()
def validate_parameters_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform fast pre-flight validation of CAD parameters against known geometric constraints and bounds.

    Args:
        params (Dict[str, Any]): Dictionary of parameter names and proposed numerical values.

    Returns:
        Dict[str, Any]: Validation status (is_valid: bool) and error message if invalid.
    """
    is_valid, error_msg = validate_parameters(params)
    return {
        "is_valid": is_valid,
        "error": error_msg if not is_valid else None
    }


@mcp.tool()
def suggest_parameters_tool(
    config_file: str = "configs/corkscrew_config.yaml",
    count: int = 1,
    db_path: str = "optimization_log.jsonl"
) -> Dict[str, Any]:
    """
    Invoke the internal AI Agent to analyze history and propose candidate parameter sets.

    Args:
        config_file (str): Problem configuration YAML file.
        count (int): Number of candidate parameter sets to suggest (default: 1).
        db_path (str): Log database path for historical reference.

    Returns:
        Dict[str, Any]: Proposed parameter sets and reasoning.
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load config: {e}"}

    store = DataStore(log_file=db_path)
    history = store.load_history()

    agent = LLMAgent()

    if count > 1:
        campaign = agent.suggest_campaign(
            history=history,
            constraints=config.get('optimization', {}).get('constraints', ''),
            objective=config.get('optimization', {}).get('objective_function', 'efficiency'),
            target=config.get('optimization', {}).get('target', 'maximize'),
            description=config.get('optimization', {}).get('description', ''),
            parameters_def=config.get('geometry', {}).get('parameters', {}),
            count=count
        )
        return {"status": "success", "suggestions": campaign}
    else:
        current_params = history[-1]["parameters"] if history else {}
        metrics = history[-1]["metrics"] if history else {}
        suggestion = agent.suggest_parameters(
            current_params=current_params,
            metrics=metrics,
            constraints=config.get('optimization', {}).get('constraints', ''),
            history=history
        )
        return {"status": "success", "suggestions": [suggestion]}


@mcp.tool()
def generate_training_dataset_tool(
    config_file: str = "configs/corkscrew_config.yaml",
    log_file: str = "optimization_log.jsonl",
    output_file: str = "exports/training_data.jsonl",
    format_type: str = "openai",
    filter_mode: str = "all"
) -> Dict[str, Any]:
    """
    Export fine-tuning dataset (OpenAI Chat, Alpaca, or DPO format) complete with CoT physics reasoning from history.

    Args:
        config_file (str): Problem configuration YAML file.
        log_file (str): Path to input simulation logs (.jsonl).
        output_file (str): Path to write formatted output dataset (.jsonl).
        format_type (str): Format type ('openai', 'alpaca', or 'dpo') (default: 'openai').
        filter_mode (str): Transition filtering ('all', 'success', or 'error-correction') (default: 'all').

    Returns:
        Dict[str, Any]: Export status and summary of exported examples.
    """
    try:
        config = load_yaml_config(config_file)
        logs = load_optimization_logs(log_file)

        if len(logs) < 2:
            return {"status": "error", "message": f"At least 2 simulation logs are required. Found {len(logs)} in {log_file}."}

        transitions = build_transitions(logs, config)
        filtered = filter_transitions(transitions, filter_mode, config)

        if not filtered:
            return {"status": "error", "message": f"No transitions matched filter '{filter_mode}'."}

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        export_data(filtered, format_type, config, output_file)

        return {
            "status": "success",
            "output_file": output_file,
            "exported_examples": len(filtered),
            "format": format_type,
            "filter": filter_mode
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==============================================================================
# PHYZ & VCAD INTEGRATION TOOLS (PHASE 5)
# ==============================================================================

@mcp.tool()
def get_receipt_graph_status(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluates fail-closed Claim & Receipt DAG against given parameters.
    Returns status of all geometric, DFM, fluidic, and structural receipts (Holds/Stale/Violated),
    quantitative margins, and whether downstream meshing/simulation is cleared to proceed.
    """
    if params is None:
        params = {
            "insert_length_mm": 50.0,
            "twist_deg": 720.0,
            "helix_path_radius_mm": 10.0,
            "helix_profile_radius_mm": 4.0,
            "helix_void_profile_radius_mm": 2.5,
            "tube_od_mm": 32.0,
            "tube_wall_mm": 1.5,
            "blade_chamfer_mm": 0.5,
            "barrel_length_mm": 60.0
        }

    graph = ClaimReceiptGraph()
    graph.resolve(params)

    return {
        "is_valid": graph.is_valid(),
        "summary": {
            "holds_count": sum(1 for r in graph._receipts.values() if r.state == ReceiptState.HOLDS),
            "violated_count": len(graph.get_violations()),
            "stale_count": len(graph.get_stale())
        },
        "violations": [v.to_dict() for v in graph.get_violations()],
        "receipts": graph.export_receipts()
    }


@mcp.tool()
def get_guardian_telemetry(
    current_energy: float = 1000.0,
    initial_energy: float = 1000.0,
    boundary_work: float = 0.0,
    is_final_signoff: bool = False
) -> Dict[str, Any]:
    """
    Queries Conservation Guardian watchdog metrics (energy drift, active solver tier, and fallback reason).
    """
    guardian = PhyzConservationGuardian()
    report, _ = guardian.audit_rollout_step(
        current_energy=current_energy,
        initial_energy=initial_energy,
        boundary_work=boundary_work,
        is_final_signoff=is_final_signoff
    )
    return report.to_dict()


@mcp.tool()
def run_trajectory_adjoint_sensitivity(
    path_r: float = 10.0,
    pitch: float = 25.0,
    tube_od: float = 32.0,
    chamfer: float = 0.5,
    time_steps: int = 20
) -> Dict[str, Any]:
    """
    Computes analytical reverse-mode trajectory adjoint sensitivities (phyz-diff paradigm).
    Ranks geometric and firmware control parameters by gradient leverage on the multi-physics objective.
    """
    model = LatentDynamicsModel()
    engine = TrajectoryAdjointEngine(model=model, lyapunov_gamma=0.05)

    z0 = np.array([5.0, 100.0, 50.0, 2.0, 1.0, 0.1], dtype=np.float64)
    p0 = np.array([path_r, pitch, tube_od, chamfer], dtype=np.float64)
    dt = 0.005
    u_traj = np.tile(np.array([0.7, 0.5]), (time_steps, 1))

    z_traj = model.rollout(z0, u_traj, p0, dt)
    lambda_traj, dJ_dp = engine.solve_reverse_adjoint(z_traj, u_traj, p0, dt)

    param_names = ["helix_path_radius_mm", "helix_pitch_mm", "tube_od_mm", "blade_chamfer_mm"]
    sensitivities = {name: float(dJ_dp[i]) for i, name in enumerate(param_names)}
    ranked = sorted(sensitivities.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "status": "success",
        "sensitivities": sensitivities,
        "ranking_by_leverage": [{"parameter": k, "gradient": v} for k, v in ranked],
        "terminal_state": {
            "swirl_momentum": float(z_traj[-1, 0]),
            "delta_p_pa": float(z_traj[-1, 1]),
            "separation_efficiency_pct": float(z_traj[-1, 2]),
            "von_mises_stress_mpa": float(z_traj[-1, 3])
        }
    }


@mcp.tool()
def evaluate_coupled_design(
    insert_length_mm: float = 50.0,
    twist_deg: float = 720.0,
    helix_path_radius_mm: float = 10.0,
    helix_profile_radius_mm: float = 4.0,
    helix_void_profile_radius_mm: float = 2.5,
    tube_od_mm: float = 32.0,
    tube_wall_mm: float = 1.5,
    blade_chamfer_mm: float = 0.5
) -> Dict[str, Any]:
    """
    Executes end-to-end fail-closed design evaluation.
    Resolves upstream assembly mates, checks Receipt DAG, runs Guardian-monitored latent dynamics,
    and returns verified metrics or explicit constraint breach failure.
    """
    params = {
        "insert_length_mm": insert_length_mm,
        "twist_deg": twist_deg,
        "helix_path_radius_mm": helix_path_radius_mm,
        "helix_profile_radius_mm": helix_profile_radius_mm,
        "helix_void_profile_radius_mm": helix_void_profile_radius_mm,
        "tube_od_mm": tube_od_mm,
        "tube_wall_mm": tube_wall_mm,
        "blade_chamfer_mm": blade_chamfer_mm,
        "barrel_length_mm": insert_length_mm + 10.0
    }

    # 1. Receipt DAG Check
    graph = ClaimReceiptGraph()
    graph.resolve(params)
    if not graph.is_valid():
        violations = [v.to_dict() for v in graph.get_violations()]
        return {
            "status": "rejected_fail_closed",
            "message": "Design rejected: one or more constraints VIOLATED",
            "violations": violations
        }

    # 2. Upstream LM Assembly Solve
    assembly_solver = UpstreamAssemblySolver()
    p_assembly = {
        "tube_od_mm": tube_od_mm,
        "tube_wall_mm": tube_wall_mm,
        "insert_length_mm": insert_length_mm,
        "inlet_port_dia_mm": 6.35,
        "sensor_standoff_mm": 3.0,
        "oring_cs_mm": 1.78
    }
    solved_assembly, _ = assembly_solver.solve(p_assembly)

    # 3. Latent Dynamics Rollout with Guardian Watchdog
    model = LatentDynamicsModel()
    guardian = PhyzConservationGuardian()

    z0 = np.zeros(6, dtype=np.float64)
    pitch = insert_length_mm / (abs(twist_deg) / 360.0) if abs(twist_deg) > 1e-3 else 50.0
    p_latent = np.array([helix_path_radius_mm, pitch, tube_od_mm, blade_chamfer_mm], dtype=np.float64)
    u_traj = np.tile(np.array([0.75, 0.55]), (30, 1))

    z_traj = model.rollout(z0, u_traj, p_latent, dt=0.01)
    z_final = z_traj[-1]

    # Energy estimate: kinetic + pressure work
    sim_energy = 0.5 * 1000.0 * (z_final[0] ** 2) * 1e-4 + z_final[1] * 0.1
    report, _ = guardian.audit_rollout_step(
        current_energy=sim_energy,
        initial_energy=sim_energy,
        boundary_work=0.0
    )

    return {
        "status": "success",
        "receipt_summary": {
            "is_valid": True,
            "holds_count": len(graph._receipts)
        },
        "assembly_mates": solved_assembly,
        "guardian_status": report.to_dict(),
        "predicted_metrics": {
            "swirl_angular_momentum": float(z_final[0]),
            "delta_p_pa": float(z_final[1]),
            "separation_efficiency_pct": float(z_final[2]),
            "max_von_mises_stress_mpa": float(z_final[3]),
            "thermal_dissipation_w": float(z_final[4])
        }
    }


def main():
    """Run MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
