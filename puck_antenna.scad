// puck_antenna.scad
// Parametric model for a ground-level puck antenna with a solar panel array

// === Parameters ===
// Housing
puck_radius = 50.0;
puck_height = 20.0;
ground_clearance = 0.0; // 0 means flush with ground, negative means buried

// Solar Panel
panel_radius = 45.0;
panel_thickness = 3.0;
panel_height_above_puck = 10.0; // Clearance gap

// Antenna Elements
num_antennas = 4; // Supported: 2 or 4
antenna_length = 31.0; // ~1/4 wavelength at 2.4GHz
antenna_width = 4.0;
antenna_thickness = 1.0;
antenna_angle_offset = 0.0; // Degrees to tilt the antenna upwards

$fn = 60;

module solar_panel() {
    translate([0, 0, puck_height + panel_height_above_puck])
    cylinder(h=panel_thickness, r=panel_radius);
}

module puck_housing() {
    translate([0, 0, 0])
    cylinder(h=puck_height, r=puck_radius);
}

module antenna_element() {
    // A simple rectangular patch/trace antenna element extending outwards
    translate([puck_radius, -antenna_width/2, puck_height/2])
    rotate([0, -antenna_angle_offset, 0])
    cube([antenna_length, antenna_width, antenna_thickness]);
}

module antenna_array() {
    angle_step = 360 / num_antennas;
    for (i = [0 : num_antennas - 1]) {
        rotate([0, 0, i * angle_step])
        antenna_element();
    }
}

module assembly() {
    // Translate the entire assembly based on ground clearance
    translate([0, 0, ground_clearance]) {
        union() {
            puck_housing();
            solar_panel();
            antenna_array();
        }
    }
}

// Generate the assembly
assembly();
