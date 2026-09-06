# OpenAuto-CFD: Universal Configuration-Driven CFD Optimizer

OpenAuto-CFD is a powerful "Software-Defined Engineering" framework designed for the autonomous generation, simulation, and optimization of 3D-printable fluid dynamics components. It creates a seamless loop integrating parametric Computer-Aided Design (OpenSCAD), Computational Fluid Dynamics (OpenFOAM), and Generative Artificial Intelligence (LLM) to perform automated searches of high-dimensional design spaces.

By leveraging a universal Configuration-Driven Architecture (via YAML Problem Definition files), OpenAuto-CFD is not tied to a single geometry. Instead, it allows users to specify parameter ranges, physics constraints, and optimization goals, enabling an AI "Virtual Engineer" to iteratively design, simulate, and refine models toward an optimal solution.

## Key Features

*   **Universal Configuration-Driven Architecture**: Define any parametric optimization problem via a single YAML file, decoupled from hardcoded logic.
*   **KiCad 10 Action Plugin (OpenAuto EM Live Bridge)**: Official KiCad 10/9/8/7 `pcbnew.ActionPlugin` creating an instant electromagnetic feedback loop (<100 ms). Modifying traces on the PCB canvas and saving (`Ctrl+S`) automatically recalculates controlled impedance ($Z_0, Z_{diff}$), S-parameters ($S_{11}, S_{21}$), and regenerates HyperLynx (`.hyp`) and openEMS FDTD models.
*   **EDA, Chip Design & RF Transmission Line Synthesis**: Wheeler and Hammerstad conformal mapping for microstrip and coupled differential pairs with finite copper thickness, frequency-dependent skin depth attenuation, and automated `.kicad_pcb` layout generation.
*   **Multi-Physics Surrogate & Model Fusion**: Universal RBF surrogate supporting coupled CFD (OpenFOAM), FEA structural stress (CalculiX), and EM (openEMS) metrics with 3D spatial vector fields and binary GPU buffer export.
*   **Real-Time Interactive WebGL Viewer (OpenAuto-CFD Studio HUD)**: Modern Three.js dark-mode HUD with 60 FPS real-time parameter scrub, swirling particle ribbons, FEA stress heatmaps, AI inverse design controls, and KiCad EM sync toast alerts.
*   **Multi-Fidelity Mesh Pyramid**: Kennedy & O'Hagan Co-Kriging model ($y_H = \rho y_L + \delta(x)$) with dynamic mesh resolution switching and a Two-Stage Active Screening Filter delivering a $2.9\times$ compute speedup.
*   **Physics-Informed Conservation Regularizer (PINN)**: Discrete Exterior Calculus Helmholtz-Hodge solenoidal projector $(D D^T) \boldsymbol{\lambda} = D \mathbf{u}$ enforcing incompressibility continuity ($\nabla \cdot \mathbf{u} = 0$) with $100\%$ divergence elimination in $<25\text{ ms}$, plus Cauchy stress divergence equilibrium ($\nabla \cdot \boldsymbol{\sigma} \approx \mathbf{0}$).
*   **LLM Tool-Calling CAD & EDA Reasoning Agents**: Exposes standard OpenAPI / JSON Schema function-calling tools (`predict_surrogate`, `run_inverse_design`, `optimize_trace_impedance`, `evaluate_rf_transmission`, `generate_kicad_pcb`, `generate_scad_code`) for autonomous multi-turn engineering reasoning.
*   **Exact Differentiable Gradients**: Analytic exact derivatives ($\approx 10^{-10}$ error vs finite differences) and multi-start L-BFGS-B inverse design converging in $<10\text{ ms}$.
*   **Model Context Protocol (MCP) Interface**: Exposes a standard MCP server for external LLMs (e.g. Claude Desktop, Cursor) to directly query harness state, list configs, run parametric CAD & physics simulations, and execute Schema surrogate steps.


---

## Case Study: Parametric Corkscrew Filter

To demonstrate the capabilities of OpenAuto-CFD, this repository includes a comprehensive system validation study: the **Parametric Corkscrew Filter**.

This is a 3D-printable inertial filter designed for separating particles from a fluid stream—ideal for challenging environments like mitigating abrasive lunar regolith.

### Theory of Operation

The corkscrew filter combines advanced fluid dynamics principles with the AI-driven engineering of OpenAuto-CFD.
See the [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md) for a detailed explanation of the physics and validation results.

#### Physics of Inertial Separation

The core mechanism of the corkscrew filter is **inertial separation**. As fluid traverses the helical channel, it is subjected to rapid changes in direction, inducing specific forces:

1.  **Centrifugal Force ($$F_c = m \frac{v^2}{r}$$)**: The helical geometry acts as a continuous centrifuge. Heavier particles possess greater inertia and are flung toward the outer wall of the channel.
2.  **Dean Vortices**: In curved pipes, the velocity differential creates secondary flows known as Dean Vortices that sweep the cross-section and transport particles toward trapping zones.
3.  **"Clog-Free" Trapping**: Unlike barrier filters (e.g., HEPA), this design uses "stepped traps" to eject particles *out* of the main flow stream into a quiescent collection bin. The filter maintains constant flow conductance until the bin is physically full.

## Getting Started

### 1. Install Dependencies

*   **Node.js**: Run `npm install` in the root directory to install geometry generation tools (`openscad-wasm`).
*   **Python**:
    It is highly recommended to use a virtual environment. The `start_optimization.sh` script handles this automatically (creating `.venv`). If installing manually:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r optimizer/requirements.txt
    ```

### 2. Generating the 3D Models (OpenSCAD)

The 3D models for the filter are generated using OpenSCAD.
1.  **Open the Main File**: Open the `corkscrew.scad` file in OpenSCAD (or use the CLI tools provided).
2.  **Configure Parameters**: Adjust the parameters in `config.scad` or the `parameters/` directory.
3.  **Render and Export**: Render the model (F6) and export it as an STL file for 3D printing.

#### Key Parameters (`corkscrew.scad`)
*   `filter_height_mm`: The height of the filter.
*   `number_of_complete_revolutions`: The number of turns in the corkscrew channels.
*   `screw_OD_mm`, `screw_ID_mm`: The outer and inner diameters of the corkscrew channels.
*   `num_screws`: The number of parallel corkscrew channels.
*   `num_bins`: The number of collection bins.

### 3. Automated Optimization & Parameters

The `optimizer/` directory contains tools to automate the design-simulation-analysis loop using the OpenAuto-CFD framework.
*   See [optimizer/README.md](./optimizer/README.md) for details on the AI-driven optimization workflow and MCP server usage.
*   See [parameters/README.md](./parameters/README.md) for information on parameter configuration files.

> **Pro Tip:** When running the optimizer `main.py`, you can parallelize the meshing and CFD solver by using the `--cpus X` flag (where X is the number of cores). This makes OpenFOAM execute much faster!

### 4. Running the MCP Server

OpenAuto-CFD includes a Model Context Protocol (MCP) interface allowing AI assistants (Claude Desktop, Cursor, etc.) to drive evaluations and inspect harness state:

```bash
PYTHONPATH=optimizer python3 -m optimizer.mcp_server
```

To configure with Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "openauto-cfd": {
      "command": "python3",
      "args": ["-m", "optimizer.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/OpenAuto-CFD/optimizer"
      }
    }
  }
}
```

### 4. CFD Simulation (Advanced)

This project includes a base case setup for running a CFD simulation using OpenFOAM. For detailed instructions on how to set up and run the simulation, please see the [README.md in the `corkscrewFilter` directory](./corkscrewFilter/README.md).

### 5. Launching the Real-Time Interactive WebGL Viewer (OpenAuto-CFD Studio)

To launch the real-time browser-based OpenAuto-CFD 3D simulation viewer:
```bash
python run_viewer.py
```
This starts the local threaded REST server on port 8080 and opens the WebGL interface. The viewer features:
* **60 FPS Parameter Scrub**: Scrub geometric sliders with instant surrogate response.
* **Particle Streamline Ribbons**: Dynamic swirling flow ribbons colored by velocity magnitude.
* **FEA Stress Heatmaps**: Visualizes von Mises stress distributions across the geometry.
* **AI Inverse Design**: One-click multi-start gradient descent button to find optimal parameters.

### 6. Running the Autonomous CAD Reasoning Agent (Kimi K3 / Gemini / OpenAI)

You can launch an autonomous engineering session where the agent executes multi-turn tool calling (predicting surrogate surfaces, verifying physical conservation, and writing valid `.scad` files):
```bash
PYTHONPATH=optimizer python3 -c "
from cad_agent_tools import CADReasoningAgent
agent = CADReasoningAgent()
res = agent.run_goal('Optimize lunar regolith corkscrew filter for >95% efficiency and low pressure drop')
print(res['summary'])
"
```

### 7. KiCad 10 Action Plugin: OpenAuto EM Live Bridge

The repository includes a native **KiCad 10.0** (and 9/8/7) Action Plugin located in [`kicad_plugin/`](./kicad_plugin/). It connects KiCad PCB Editor (`pcbnew`) directly to the simulation engine, providing a live on-save electromagnetic feedback loop.

#### 1-Command Installation
```powershell
# Windows PowerShell
.\install_kicad_plugin.ps1
```
```bash
# Linux / macOS
./install_kicad_plugin.sh
```
*Or directly via Python:*
```bash
python kicad_plugin/install_plugin.py
```

#### How to Use in KiCad 10:
1. Start the simulation viewer: `python run_viewer.py`
2. Open KiCad 10 PCB Editor (`pcbnew`).
3. Click **Tools** $\rightarrow$ **External Plugins** $\rightarrow$ **Refresh Plugins**.
4. Click the **OpenAuto EM Live Bridge** toolbar button.
5. Edit trace widths or differential pairs and hit **Ctrl+S**: the WebGL HUD instantly updates with live $Z_0$ and $S_{11}$ telemetry (<100 ms).

*See the dedicated [kicad_plugin/README.md](./kicad_plugin/README.md) for architecture, Wheeler conformal equations, and openEMS FDTD script generation.*

---

## Testing & Verification

To run the automated verification suites covering all multi-physics, EDA, distributed swarm, and KiCad plugin modules:

```bash
# Set PYTHONPATH to include optimizer, viewer, and kicad_plugin directories
export PYTHONPATH="optimizer:viewer:kicad_plugin"  # On Windows PowerShell: $env:PYTHONPATH="optimizer;viewer;kicad_plugin"

# 1. KiCad 10 Action Plugin & Real-Time On-Save EM Recalculation Loop
python verify_kicad_plugin.py

# 2. EDA, Chip Design & RF Transmission Lines (Wheeler Conformal Mapping & S-Params)
python verify_eda_chip_design.py

# 3. Live WebGL Viewer Upgrades (HUD Physics Telemetry & AI Agent Chat Drawer)
python verify_viewer_upgrades.py

# 4. Multi-Algorithm Benchmark & Pareto Front Analysis (Random vs L-BFGS-B vs PINN vs CAD Agent)
python verify_benchmark_suite.py

# 5. Distributed Worker Swarm & Git-Based Job Queue (Atomic Claims & Heartbeat)
python verify_distributed_swarm.py

# 6. Physics-Informed Conservation Regularizer (PINN & Discrete Exterior Calculus)
python verify_pinn_conservation.py

# 7. Autonomous CAD Agent Tool-Calling Suite
python verify_cad_agent.py

# 8. Multi-Physics Surrogate Fusion & Binary GPU Buffer IO
python verify_cfd_fea_fusion.py

# 9. Multi-Fidelity Mesh Pyramid & Two-Stage Active Screening
python verify_multifidelity.py
```

## Assembly

For a complete list of materials required and assembly instructions for the Corkscrew Filter validation study, please see the [Bill of Materials (BOM.md)](./BOM.md).

## Future Work

For a list of planned enhancements and roadmap items, please see the [TODO list (TODO.md)](./TODO.md).
