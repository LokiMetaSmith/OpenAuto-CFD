# ====================================================================
# Auto-Generated build123d Parametric CAD Program (GenCAD Physics Synthesis)
# ====================================================================

from build123d import *

# Parametric Parameters
number_of_complete_revolutions = 2.626
helix_path_radius_mm = 2.579
helix_profile_radius_mm = 1.505
blade_chamfer_mm = 0.538

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
