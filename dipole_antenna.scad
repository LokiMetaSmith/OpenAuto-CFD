// dipole_antenna.scad

// Parameters
length = 50.0;
thickness = 2.0;

module dipole() {
    // A simple dipole geometry consisting of two cylinders with a gap in the center
    gap = 2.0;
    arm_length = (length - gap) / 2.0;

    // Arm 1
    translate([0, 0, gap/2])
        cylinder(h=arm_length, r=thickness/2, $fn=32);

    // Arm 2
    translate([0, 0, -gap/2 - arm_length])
        cylinder(h=arm_length, r=thickness/2, $fn=32);
}

// Ensure the geometry is aligned along the Z axis for standard EM port orientation
dipole();
