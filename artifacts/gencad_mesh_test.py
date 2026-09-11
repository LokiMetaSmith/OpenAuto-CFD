# ====================================================================
# Auto-Generated build123d CAD AST Program (GenCAD Transformer Sequence)
# ====================================================================

from build123d import *

# Parametric Controls from Transformer AST Tokens
number_of_complete_revolutions = 2.500
helix_path_radius_mm = 2.500
helix_profile_radius_mm = 1.500
blade_chamfer_mm = 0.500

height = number_of_complete_revolutions * 18.0
pitch = height / max(1.0, number_of_complete_revolutions)

with BuildPart() as model:
    Cylinder(radius=16.0, height=height + 10.0)
    with BuildSection(Plane.XY) as sec:
        with Locations((helix_path_radius_mm, 0)):
            Circle(radius=helix_profile_radius_mm)

    Helix(pitch=pitch, height=height, radius=helix_path_radius_mm)

part = model.part
