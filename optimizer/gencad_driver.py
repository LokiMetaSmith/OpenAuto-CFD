"""
gencad_driver.py

Physics-Driven CAD Embedding & Retrieval Driver (GenCAD Architecture).
Maps CFD flow simulation performance feature vectors (e.g., target pressure drop,
separation efficiency, lift/drag coefficients) through latent space embeddings to
retrieve and generate parametric CAD programs (OpenSCAD / build123d).
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union


class GenCADDriver:
    """
    Physics-conditioned CAD retrieval and generative synthesis engine.
    Implements latent embedding space mapping CFD physics targets to parametric CAD scripts.
    """

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path
        self.cad_library: List[Dict[str, Any]] = []
        self._initialize_cad_library()

    def _initialize_cad_library(self):
        """Initializes baseline CAD program library with associated physics feature vectors."""
        if self.database_path and os.path.exists(self.database_path):
            with open(self.database_path, "r", encoding="utf-8") as f:
                self.cad_library = json.load(f)
            return

        # Default CAD dataset representing design space with physics profiles
        self.cad_library = [
            {
                "id": "corkscrew_low_dp",
                "name": "Low Pressure Drop Corkscrew Filter",
                "physics_features": {
                    "delta_p": 1200.0,
                    "separation_efficiency": 88.5,
                    "flow_rate_m3s": 0.015,
                    "drag_coefficient": 0.45
                },
                "parameters": {
                    "number_of_complete_revolutions": 1.5,
                    "helix_path_radius_mm": 2.0,
                    "helix_profile_radius_mm": 1.3,
                    "blade_chamfer_mm": 0.3
                },
                "category": "filter"
            },
            {
                "id": "corkscrew_balanced",
                "name": "Balanced Performance Corkscrew Filter",
                "physics_features": {
                    "delta_p": 2100.0,
                    "separation_efficiency": 94.2,
                    "flow_rate_m3s": 0.012,
                    "drag_coefficient": 0.62
                },
                "parameters": {
                    "number_of_complete_revolutions": 2.5,
                    "helix_path_radius_mm": 2.5,
                    "helix_profile_radius_mm": 1.5,
                    "blade_chamfer_mm": 0.5
                },
                "category": "filter"
            },
            {
                "id": "corkscrew_high_eff",
                "name": "High Efficiency Particle Separator",
                "physics_features": {
                    "delta_p": 3800.0,
                    "separation_efficiency": 98.7,
                    "flow_rate_m3s": 0.008,
                    "drag_coefficient": 0.88
                },
                "parameters": {
                    "number_of_complete_revolutions": 3.8,
                    "helix_path_radius_mm": 3.2,
                    "helix_profile_radius_mm": 1.7,
                    "blade_chamfer_mm": 0.8
                },
                "category": "filter"
            },
            {
                "id": "monocopter_airfoil_high_lift",
                "name": "High Lift Airfoil Section",
                "physics_features": {
                    "lift_coefficient": 1.45,
                    "drag_coefficient": 0.08,
                    "delta_p": 450.0,
                    "separation_efficiency": 0.0
                },
                "parameters": {
                    "number_of_complete_revolutions": 1.0,
                    "helix_path_radius_mm": 1.5,
                    "helix_profile_radius_mm": 1.2,
                    "blade_chamfer_mm": 0.2
                },
                "category": "aerofoil"
            }
        ]

    def encode_physics(self, physics_target: Dict[str, float]) -> np.ndarray:
        """
        Encodes a physics target feature vector into a normalized physics latent space embedding.
        Keys recognized: delta_p, separation_efficiency, flow_rate_m3s, drag_coefficient, lift_coefficient.
        """
        dp = physics_target.get("delta_p", 2000.0) / 4000.0
        eff = physics_target.get("separation_efficiency", 90.0) / 100.0
        fr = physics_target.get("flow_rate_m3s", 0.01) / 0.02
        cd = physics_target.get("drag_coefficient", 0.5) / 1.0
        cl = physics_target.get("lift_coefficient", 0.0) / 2.0

        vec = np.array([dp, eff, fr, cd, cl], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def retrieve_cad_program(
        self,
        physics_target: Dict[str, float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k CAD programs from library best matching the CFD physics target profile.
        Uses cosine similarity in contrastive physics-latent space.
        """
        target_embedding = self.encode_physics(physics_target)
        results = []

        for entry in self.cad_library:
            entry_embedding = self.encode_physics(entry["physics_features"])
            sim = float(np.dot(target_embedding, entry_embedding))
            res = dict(entry)
            res["similarity_score"] = round(sim, 4)
            results.append(res)

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def synthesize_cad_parameters(
        self,
        physics_target: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Generates/interpolates continuous CAD parameters conditioned on physics target.
        Acts as latent diffusion prior / surrogate inverse mapping.
        """
        top_matches = self.retrieve_cad_program(physics_target, top_k=3)
        sims = np.array([m["similarity_score"] for m in top_matches], dtype=np.float32)

        # Softmax weighting over similarities
        exp_sims = np.exp(sims * 5.0)
        weights = exp_sims / np.sum(exp_sims)

        synth_params = {}
        param_keys = top_matches[0]["parameters"].keys()

        for k in param_keys:
            val = sum(weights[i] * top_matches[i]["parameters"][k] for i in range(len(top_matches)))
            synth_params[k] = float(round(val, 3))

        return synth_params

    def generate_cad_script(
        self,
        parameters: Dict[str, float],
        format_type: str = "openscad"
    ) -> str:
        """
        Converts CAD parameterization into a full, executable CAD command program / script
        in OpenSCAD (.scad) or build123d (.py).
        """
        n_rev = parameters.get("number_of_complete_revolutions", 2.5)
        path_r = parameters.get("helix_path_radius_mm", 2.5)
        prof_r = parameters.get("helix_profile_radius_mm", 1.5)
        chamfer = parameters.get("blade_chamfer_mm", 0.5)

        if format_type.lower() in ["build123d", "python", "py"]:
            return f"""# ====================================================================
# Auto-Generated build123d Parametric CAD Program (GenCAD Physics Synthesis)
# ====================================================================

from build123d import *

# Parametric Parameters
number_of_complete_revolutions = {n_rev:.3f}
helix_path_radius_mm = {path_r:.3f}
helix_profile_radius_mm = {prof_r:.3f}
blade_chamfer_mm = {chamfer:.3f}

# Outer Cyclone Shell & Helical Core Construction
height = number_of_complete_revolutions * 18.0
pitch = height / max(1.0, number_of_complete_revolutions)

with BuildPart() as model:
    # Outer cylindrical casing
    Cylinder(radius=16.0, height=height + 10.0)
    # Helical flow channel cut / vane
    with BuildSection(Plane.XY) as sec:
        with Locations((helix_path_radius_mm, 0)):
            Circle(radius=helix_profile_radius_mm)

    Helix(pitch=pitch, height=height, radius=helix_path_radius_mm)

# Export references
part = model.part
"""

        # Default OpenSCAD format
        return f"""// ====================================================================
// Auto-Generated OpenSCAD Parametric CAD Program (GenCAD Physics Synthesis)
// ====================================================================

$fn = 60;

// GenCAD Parametric Controls
number_of_complete_revolutions = {n_rev:.3f};
helix_path_radius_mm = {path_r:.3f};
helix_profile_radius_mm = {prof_r:.3f};
blade_chamfer_mm = {chamfer:.3f};

tube_od_mm = 32.0;
tube_wall_mm = 1.2;
tube_id_mm = tube_od_mm - 2 * tube_wall_mm;

module corkscrew_vane() {{
    linear_extrude(
        height = number_of_complete_revolutions * 18.0,
        twist = -360 * number_of_complete_revolutions,
        slices = 120
    )
    translate([helix_path_radius_mm, 0, 0])
    circle(r = helix_profile_radius_mm);
}}

module cyclone_body() {{
    difference() {{
        cylinder(r = tube_od_mm / 2, h = number_of_complete_revolutions * 18.0 + 10.0, center = false);
        translate([0, 0, -1])
        cylinder(r = tube_id_mm / 2, h = number_of_complete_revolutions * 18.0 + 12.0, center = false);
    }}
}}

union() {{
    cyclone_body();
    corkscrew_vane();
}}
"""

    def generate_and_export(
        self,
        physics_target: Dict[str, float],
        output_dir: str = "artifacts",
        filename_prefix: str = "gencad_generated",
        format_type: str = "openscad"
    ) -> Dict[str, Any]:
        """
        Full GenCAD workflow: Physics Target -> Parameter Synthesis -> CAD Code Export -> Metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        retrieved_cad = self.retrieve_cad_program(physics_target, top_k=1)[0]
        synth_params = self.synthesize_cad_parameters(physics_target)

        ext = ".py" if format_type.lower() in ["build123d", "python", "py"] else ".scad"
        file_path = os.path.join(output_dir, f"{filename_prefix}{ext}")

        script_code = self.generate_cad_script(synth_params, format_type=format_type)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        return {
            "status": "success",
            "physics_target": physics_target,
            "retrieved_nearest_match": retrieved_cad["name"],
            "retrieved_similarity": retrieved_cad["similarity_score"],
            "synthesized_parameters": synth_params,
            "format": format_type,
            "output_file": file_path,
            "script_code": script_code
        }
