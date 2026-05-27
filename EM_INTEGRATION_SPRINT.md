# EM Integration Sprint Report

## Overview
We are evolving the OpenAuto-CFD repository from a single-physics (CFD) framework into a Multi-Physics Generative Framework. The immediate objective of this sprint is to extend the current configuration-driven architecture to support electromagnetic (EM) modeling using **openEMS** as an alternative physics backend to **OpenFOAM**.

This shift enables ML and LLM-based parametric searches in free space for both constrained and unconstrained antennas, given manufacturing limitations, using high-speed, scriptable 3D EM simulation without graphical interfaces.

## Architectural Guidance

### Maintain Orthogonality
The existing optimization loop (`optimizer/main.py`), simulation pipeline (`optimizer/simulation_runner.py`), and configuration parsing logic must remain as solver-agnostic as possible. Hardcoded CFD logic will be moved out of the core `optimizer/` module wherever feasible, or abstracted to handle varying data structures from different physics engines.

### Factory Pattern Integration
We will implement a **Physics Engine Factory** (`optimizer/physics_factory.py` or similar). The system will examine the `physics.type` field in the YAML configuration and dynamically bind the appropriate driver:
- `physics.type: 'cfd'` -> `OpenFOAM_Driver`
- `physics.type: 'em'` -> `OpenEMSDriver`

### Interface Consistency
The new EM driver will implement the same base interface as the CFD driver (e.g., `run_solver()`, `get_metrics()`, `prepare_case()`). The output performance metrics must be structured identically to our existing `calculate_physics.py` and downstream LLM evaluation scripts (a flat JSON dictionary with values mapping to LLM-readable names like `S11`, `gain`).

### Containerization & Tooling
The `openEMS` toolkit, including `CSXCAD` and its Python interface, will be incorporated into our containerized workflow. `repair_podman.ps1` and related shell scripts will be updated to fetch or build an openEMS-compatible container image, running simulations just as we currently orchestrate OpenFOAM via Docker/Podman.

### Extensibility
The `PhysicsEngine` base interface will be designed such that adding future physics modules (e.g., thermal, structural) is a trivial matter of subclassing and mapping the physics `type` to the new class in the factory.

## Proposed File Modifications

1. **`optimizer/physics_driver.py` (New)**
   - Define a `PhysicsDriver` base class containing the interface methods: `prepare_case`, `run_meshing`, `run_solver`, `get_metrics`, and `cleanup_ram_disk`.

2. **`optimizer/physics_factory.py` (New)**
   - Define `PhysicsEngineFactory`.
   - `get_driver(config, ...)` will return the correct driver based on `config['physics']['type']`.

3. **`optimizer/foam_driver.py` (Refactor)**
   - Subclass `PhysicsDriver`.
   - Ensure it strictly conforms to the expected base class interface.

4. **`optimizer/em_driver.py` (New)**
   - Implement `OpenEMSDriver`, subclassing `PhysicsDriver`.
   - Map `run_solver` to executing openEMS python/octave scripts inside the container.
   - Map `get_metrics` to reading S-parameters and gain from openEMS output (CSV/HDF5).
   - Implement geometry conversion handling (STL/DXF parsing for CSXCAD).

5. **`optimizer/simulation_runner.py` (Refactor)**
   - Replace explicit `FoamDriver` initialization with a factory call.
   - Abstract the steps. For example, EM might skip OpenFOAM's BlockMesh step entirely or map it to a generic bounding box generation for FDTD grid setup.
   - Separate CFD-specific logic (`run_particle_tracking`) into the driver or conditional blocks based on driver capability.

6. **`calculate_physics.py` (Refactor)**
   - Refactor to act as a bridge/factory for standalone metric calculation, querying the correct engine based on `config.yaml`.
   - Ensure the JSON objects output are consistent regardless of engine.

7. **`configs/example_config.yaml` (New)**
   - Define the new schema with `physics.type`.

8. **`repair_podman.ps1` & Tooling (Refactor)**
   - Ensure the deployment of `openEMS` images is supported alongside `OpenFOAM`.

## Architecture Draft: Physics Engine Interface

```python
# optimizer/physics_driver.py
class PhysicsDriver:
    def __init__(self, case_dir, config=None, container_engine="auto", **kwargs):
        self.case_dir = case_dir
        self.config = config

    def prepare_case(self, **kwargs):
        raise NotImplementedError

    def run_meshing(self, **kwargs):
        raise NotImplementedError

    def run_solver(self, log_file=None, **kwargs):
        raise NotImplementedError

    def get_metrics(self, log_file=None):
        raise NotImplementedError

    def cleanup_ram_disk(self):
        pass
```

```python
# optimizer/physics_factory.py
from foam_driver import FoamDriver
from em_driver import OpenEMSDriver

class PhysicsEngineFactory:
    @staticmethod
    def get_solver(case_dir, config, **kwargs):
        physics_type = config.get('physics', {}).get('type', 'cfd')

        if physics_type == 'cfd':
            return FoamDriver(case_dir, config=config, **kwargs)
        elif physics_type == 'em':
            return OpenEMSDriver(case_dir, config=config, **kwargs)
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")
```

## Sprint Todo List

- [ ] Create `example_config.yaml` with the Multi-Physics schema.
- [ ] Define `PhysicsDriver` base class (`optimizer/physics_driver.py`).
- [ ] Define `PhysicsEngineFactory` (`optimizer/physics_factory.py`).
- [ ] Refactor `FoamDriver` to inherit from `PhysicsDriver`.
- [ ] Create `OpenEMSDriver` skeleton (`optimizer/em_driver.py`).
- [ ] Refactor `optimizer/main.py` and `optimizer/simulation_runner.py` to use `PhysicsEngineFactory` instead of hardcoded `FoamDriver`.
- [ ] Refactor `calculate_physics.py` to support engine-agnostic metric calculation.
- [ ] Update `repair_podman.ps1` and container scripts to pull an openEMS image.
- [ ] Implement `OpenEMSDriver` logic (geometry parsing, FDTD setup, container execution).
- [ ] Implement `OpenEMSDriver` metric extraction (S-parameters, gain).
- [ ] Create `run_em_test.py` proof-of-concept for a simple patch/dipole antenna.
- [ ] Verify LLM context window consistency with new EM JSON outputs.
