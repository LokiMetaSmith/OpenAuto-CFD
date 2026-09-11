// ====================================================================
// Auto-Generated Parametric Corkscrew Filter
// Optimized via Multi-Physics Surrogate & PINN Conservation Regularizer
// Generated: 2026-09-11 04:12:03
// ====================================================================

$fn = 60;

// Optimized Design Parameters
number_of_complete_revolutions = 3.170;
helix_path_radius_mm = 2.243;
helix_profile_radius_mm = 1.634;
blade_chamfer_mm = 1.000;

// Base Tube Envelope
tube_od_mm = 32.0;
tube_wall_mm = 1.2;
tube_id_mm = tube_od_mm - 2 * tube_wall_mm;

module corkscrew_vane() {
    echo("Generating helical vane with revolutions:", number_of_complete_revolutions);
    linear_extrude(
        height = number_of_complete_revolutions * 18.0,
        twist = -360 * number_of_complete_revolutions,
        slices = 120
    )
    translate([helix_path_radius_mm, 0, 0])
    circle(r = helix_profile_radius_mm);
}

module cyclone_body() {
    difference() {
        cylinder(r = tube_od_mm / 2, h = number_of_complete_revolutions * 18.0 + 10.0, center = false);
        translate([0, 0, -1])
        cylinder(r = tube_id_mm / 2, h = number_of_complete_revolutions * 18.0 + 12.0, center = false);
    }
}

// Assembly
union() {
    cyclone_body();
    corkscrew_vane();
}
