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


def main():
    """Run MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
