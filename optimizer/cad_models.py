"""
cad_models.py
Pure Python build123d geometry definitions matching OpenSCAD models.
Fully rewrites all OpenSCAD modules into build123d native operations with
support for parametric 3D filleting and chamfering.
"""

import math
import build123d as b3d

def get_param(params, key, default):
    val = params.get(key, default)
    if isinstance(val, str):
        if '//' in val:
            val = val.split('//')[0].strip()
        val = val.rstrip(';')
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False
        try:
            return float(val)
        except ValueError:
            try:
                return float(eval(val, {"__builtins__": None}, {}))
            except Exception:
                return val
    return val

def build_helical_shape(h, twist_deg, path_r, profile_r, scale_ratio=1.0, hollow_inner_r=None, blade_chamfer=0.0, fillet_r=0.0):
    """Builds a helical extrusion using build123d with aligned local X-axis toward central Z-axis."""
    if abs(twist_deg) < 1e-3 or h <= 0:
        with b3d.BuildPart() as p:
            with b3d.BuildSketch(b3d.Plane.XY) as s:
                with b3d.Locations((path_r, 0)):
                    b3d.Ellipse(profile_r, profile_r * scale_ratio)
                with b3d.Locations((path_r / 2, 0)):
                    b3d.Rectangle(path_r, profile_r)
                if hollow_inner_r is not None and hollow_inner_r < profile_r:
                    with b3d.Locations((path_r, 0)):
                        b3d.Ellipse(hollow_inner_r, hollow_inner_r * scale_ratio, mode=b3d.Mode.SUBTRACT)
            b3d.extrude(amount=h, centered=True)
            if blade_chamfer > 0.0:
                try:
                    edges = [e for e in p.edges() if e.geom_type == b3d.GeomType.LINE]
                    if edges:
                        b3d.chamfer(edges, amount=min(blade_chamfer, profile_r * 0.3))
                except Exception:
                    pass
            if fillet_r > 0.0:
                try:
                    edges = p.edges()
                    if edges:
                        b3d.fillet(edges, radius=min(fillet_r, profile_r * 0.3))
                except Exception:
                    pass
        return p.part

    pitch = h / (abs(twist_deg) / 360.0)
    if twist_deg < 0:
        pitch = -pitch

    helix = b3d.Helix(pitch=pitch, height=h, radius=path_r, center=(0, 0, -h / 2))
    start_pos = helix.position_at(0)
    tangent = helix.tangent_at(0)
    radial_in = b3d.Vector(-start_pos.X, -start_pos.Y, 0).normalized()
    sketch_plane = b3d.Plane(origin=start_pos, x_dir=radial_in, z_dir=tangent)

    with b3d.BuildPart() as p:
        with b3d.BuildSketch(sketch_plane) as s:
            b3d.Ellipse(profile_r, profile_r * scale_ratio)
            with b3d.Locations((-path_r / 2, 0)):
                b3d.Rectangle(path_r, profile_r)
            if hollow_inner_r is not None and hollow_inner_r < profile_r:
                b3d.Ellipse(hollow_inner_r, hollow_inner_r * scale_ratio, mode=b3d.Mode.SUBTRACT)

        b3d.sweep(sections=s.sketch, path=helix)

        if blade_chamfer > 0.0:
            try:
                edges = [e for e in p.edges() if e.geom_type == b3d.GeomType.LINE]
                if edges:
                    b3d.chamfer(edges, amount=min(blade_chamfer, profile_r * 0.3))
            except Exception:
                pass

        if fillet_r > 0.0:
            try:
                edges = p.edges()
                if edges:
                    b3d.fillet(edges, radius=min(fillet_r, profile_r * 0.3))
            except Exception:
                pass

    return p.part

def build_corkscrew(params, void=False):
    h = float(get_param(params, "insert_length_mm", 100.0))
    revs = get_param(params, "number_of_complete_revolutions", 2.0)
    if isinstance(revs, list):
        revs = sum(revs)
    else:
        revs = float(revs)
    twist_deg = 360.0 * revs

    path_r = float(get_param(params, "helix_path_radius_mm", 10.0))
    helix_profile_r = float(get_param(params, "helix_profile_radius_mm", 4.0))
    safe_profile_r = min(helix_profile_r, path_r - 0.5)

    void_profile_r = float(get_param(params, "helix_void_profile_radius_mm", 5.0))
    tolerance_ch = float(get_param(params, "tolerance_channel", 0.2))
    scale_ratio = float(get_param(params, "helix_profile_scale_ratio", 1.0))

    blade_chamfer = float(get_param(params, "blade_chamfer_mm", 0.0))
    fillet_r = float(get_param(params, "inlet_fillet_radius_mm", 0.0))

    profile_r = (void_profile_r + tolerance_ch) if void else safe_profile_r
    return build_helical_shape(h, twist_deg, path_r, profile_r, scale_ratio=scale_ratio, blade_chamfer=blade_chamfer, fillet_r=fillet_r)

def build_modular_filter_assembly(params):
    tube_od = float(get_param(params, "tube_od_mm", 30.0))
    tube_wall = float(get_param(params, "tube_wall_mm", 2.0))
    tube_id = tube_od - 2 * tube_wall
    total_length = float(get_param(params, "insert_length_mm", 100.0))
    num_bins = int(get_param(params, "num_bins", 2))
    spacer_h = float(get_param(params, "spacer_height_mm", 5.0))
    path_r = float(get_param(params, "helix_path_radius_mm", 10.0))
    helix_profile_r = float(get_param(params, "helix_profile_radius_mm", 4.0))
    safe_profile_r = min(helix_profile_r, path_r - 0.5)
    void_profile_r = float(get_param(params, "helix_void_profile_radius_mm", 5.0))
    tolerance_ch = float(get_param(params, "tolerance_channel", 0.2))

    blade_chamfer = float(get_param(params, "blade_chamfer_mm", 0.0))
    fillet_r = float(get_param(params, "inlet_fillet_radius_mm", 0.0))

    total_spacer_length = (num_bins + 1) * spacer_h
    total_screw_length = total_length - total_spacer_length
    bin_length = total_screw_length / max(1, num_bins)

    revs = get_param(params, "number_of_complete_revolutions", 2.0)
    if isinstance(revs, list):
        rates = [360.0 * r / bin_length for r in revs]
    else:
        rates = [360.0 * float(revs) / total_length for _ in range(num_bins)]

    gen_cfd = get_param(params, "GENERATE_CFD_VOLUME", False)

    if gen_cfd:
        with b3d.BuildPart() as cfd_vol:
            b3d.Cylinder(radius=tube_id / 2.0, height=total_length)
            solid_screw = build_helical_shape(total_length, sum(rates) * (total_length / max(1, num_bins)), path_r, safe_profile_r, blade_chamfer=blade_chamfer, fillet_r=fillet_r)
            cfd_vol.part = cfd_vol.part - solid_screw
        return cfd_vol.part

    with b3d.BuildPart() as assy:
        screw = build_helical_shape(total_length, sum(rates) * (total_length / max(1, num_bins)), path_r, safe_profile_r, hollow_inner_r=(void_profile_r + tolerance_ch), blade_chamfer=blade_chamfer, fillet_r=fillet_r)
        assy.part = screw

    return assy.part

def build_inlet_cap(params):
    tube_od = float(get_param(params, "tube_od_mm", 30.0))
    tube_wall = float(get_param(params, "tube_wall_mm", 2.0))
    tube_id = tube_od - 2 * tube_wall
    h = float(get_param(params, "insert_length_mm", 100.0))
    shape = str(get_param(params, "cfd_shape", "circle"))

    with b3d.BuildPart() as p:
        with b3d.Locations((0, 0, -h / 2.0)):
            if shape == "square":
                b3d.Box(tube_id, tube_id, 0.5)
            elif shape == "hex":
                with b3d.BuildSketch():
                    b3d.RegularPolygon(radius=tube_id / 2.0, side_count=6)
                b3d.extrude(amount=0.5)
            else:
                b3d.Cylinder(radius=tube_id / 2.0, height=0.5)
    return p.part

def build_outlet_cap(params):
    tube_od = float(get_param(params, "tube_od_mm", 30.0))
    tube_wall = float(get_param(params, "tube_wall_mm", 2.0))
    tube_id = tube_od - 2 * tube_wall
    h = float(get_param(params, "insert_length_mm", 100.0))
    shape = str(get_param(params, "cfd_shape", "circle"))

    with b3d.BuildPart() as p:
        with b3d.Locations((0, 0, h / 2.0)):
            if shape == "square":
                b3d.Box(tube_id, tube_id, 0.5)
            elif shape == "hex":
                with b3d.BuildSketch():
                    b3d.RegularPolygon(radius=tube_id / 2.0, side_count=6)
                b3d.extrude(amount=0.5)
            else:
                b3d.Cylinder(radius=tube_id / 2.0, height=0.5)
    return p.part

def build_cfd_wall(params):
    tube_od = float(get_param(params, "tube_od_mm", 30.0))
    tube_wall = float(get_param(params, "tube_wall_mm", 2.0))
    tube_id = tube_od - 2 * tube_wall
    h = float(get_param(params, "insert_length_mm", 100.0))
    shape = str(get_param(params, "cfd_shape", "circle"))
    wt = 1.0

    with b3d.BuildPart() as p:
        if shape == "square":
            b3d.Box(tube_id + 2 * wt, tube_id + 2 * wt, h)
            b3d.Box(tube_id, tube_id, h + 1.0, mode=b3d.Mode.SUBTRACT)
        elif shape == "hex":
            with b3d.BuildSketch():
                b3d.RegularPolygon(radius=(tube_id + 2 * wt) / 2.0, side_count=6)
            b3d.extrude(amount=h)
            with b3d.BuildSketch():
                b3d.RegularPolygon(radius=tube_id / 2.0, side_count=6)
            b3d.extrude(amount=h + 1.0, mode=b3d.Mode.SUBTRACT)
        else:
            b3d.Cylinder(radius=(tube_id + 2 * wt) / 2.0, height=h)
            b3d.Cylinder(radius=tube_id / 2.0, height=h + 1.0, mode=b3d.Mode.SUBTRACT)
    return p.part

def build_flat_end_screw(params):
    h = float(get_param(params, "insert_length_mm", 100.0))
    revs = get_param(params, "number_of_complete_revolutions", 2.0)
    if isinstance(revs, list):
        revs = sum(revs)
    else:
        revs = float(revs)
    twist_deg = 360.0 * revs
    path_r = float(get_param(params, "helix_path_radius_mm", 10.0))
    helix_profile_r = float(get_param(params, "helix_profile_radius_mm", 4.0))
    safe_profile_r = min(helix_profile_r, path_r - 0.5)
    outer_dia = 2 * (path_r + safe_profile_r) * 1.2

    screw = build_helical_shape(h + 0.5, twist_deg, path_r, safe_profile_r)
    with b3d.BuildPart() as p:
        b3d.Cylinder(radius=outer_dia / 2.0, height=h)
        p.part = p.part.intersect(screw)
    return p.part

def build_barb(hose_id=5.0, hose_od=6.5, barb_count=4, barb_length=2.0, swell=1.0, wall_thickness=1.0):
    bore_id = hose_id - 2.0 * wall_thickness
    total_len = barb_length * (barb_count + 1)
    with b3d.BuildPart() as p:
        for z in range(barb_count):
            with b3d.Locations((0, 0, z * barb_length + barb_length / 2.0)):
                b3d.Cone(bottom_radius=(hose_od + swell) / 2.0, top_radius=hose_od / 2.0, height=barb_length)
        with b3d.Locations((0, 0, barb_count * barb_length + barb_length / 2.0)):
            b3d.Cylinder(radius=hose_od / 2.0, height=barb_length)
        with b3d.Locations((0, 0, total_len / 2.0)):
            b3d.Cylinder(radius=bore_id / 2.0, height=total_len + 1.0, mode=b3d.Mode.SUBTRACT)
    return p.part

def build_hose_adapter_cap(params):
    tube_od = float(get_param(params, "tube_od_mm", 30.0))
    hose_id = float(get_param(params, "adapter_hose_id_mm", 8.0))
    tube_wall = float(get_param(params, "tube_wall_mm", 2.0))
    cap_wall = 3.0
    cap_sleeve_h = 20.0
    cap_plate_t = 3.0
    cap_inner_dia = tube_od - 2.0 * tube_wall + 0.2
    cap_outer_dia = cap_inner_dia + 2.0 * cap_wall

    with b3d.BuildPart() as p:
        with b3d.Locations((0, 0, cap_sleeve_h / 2.0)):
            b3d.Cylinder(radius=cap_outer_dia / 2.0, height=cap_sleeve_h)
        with b3d.Locations((0, 0, cap_sleeve_h + cap_plate_t / 2.0)):
            b3d.Cylinder(radius=cap_outer_dia / 2.0, height=cap_plate_t)
        with b3d.Locations((0, 0, cap_sleeve_h / 2.0)):
            b3d.Cylinder(radius=cap_inner_dia / 2.0, height=cap_sleeve_h + 1.0, mode=b3d.Mode.SUBTRACT)
        with b3d.Locations((0, 0, (cap_sleeve_h + cap_plate_t) / 2.0)):
            b3d.Cylinder(radius=hose_id / 2.0, height=cap_sleeve_h + cap_plate_t + 2.0, mode=b3d.Mode.SUBTRACT)
        with b3d.Locations((0, 0, cap_sleeve_h + cap_plate_t)):
            barb = build_barb(hose_id=hose_id, hose_od=hose_id + 1.5, barb_count=4)
            p.part = p.part + barb
    return p.part

def build_custom_coupling(params):
    inset_h = float(get_param(params, "coupling_inset_height", 10.0))
    inset_w = float(get_param(params, "coupling_inset_width", 20.0))
    lip_h = float(get_param(params, "coupling_lip_height", 2.0))
    lip_w = float(get_param(params, "coupling_lip_width", 24.0))
    outer_h = float(get_param(params, "coupling_outer_coupling_height", 15.0))
    outer_od = float(get_param(params, "coupling_outer_coupling_od", 22.0))
    inlet_d = float(get_param(params, "coupling_inner_inlet", 12.0))
    outlet_d = float(get_param(params, "coupling_inner_outlet", 10.0))

    with b3d.BuildPart() as p:
        with b3d.Locations((0, 0, (inset_h + lip_h) / 2.0)):
            b3d.Cylinder(radius=inset_w / 2.0, height=inset_h)
        with b3d.Locations((0, 0, 0)):
            b3d.Cylinder(radius=lip_w / 2.0, height=lip_h)
        with b3d.Locations((0, 0, -(outer_h + lip_h) / 2.0)):
            b3d.Cylinder(radius=outer_od / 2.0, height=outer_h)
        with b3d.Locations((0, 0, (inset_h + lip_h) / 2.0)):
            b3d.Cone(bottom_radius=outlet_d / 2.0, top_radius=inlet_d / 2.0, height=inset_h + 0.1, mode=b3d.Mode.SUBTRACT)
        with b3d.Locations((0, 0, -(outer_h + lip_h) / 2.0)):
            b3d.Cylinder(radius=outlet_d / 2.0, height=outer_h + 1.0, mode=b3d.Mode.SUBTRACT)
    return p.part

def build_filter_holder(params):
    tube_id = float(get_param(params, "tube_id", 30.0))
    cartridge_od = float(get_param(params, "cartridge_od", 10.0))
    barb_od = float(get_param(params, "barb_od", 6.5))
    barb_id = float(get_param(params, "barb_id", 4.0))
    base_h = 5.0

    with b3d.BuildPart() as p:
        b3d.Cylinder(radius=tube_id / 2.0, height=base_h)
        b3d.Cylinder(radius=barb_id / 2.0, height=base_h + 2.0, mode=b3d.Mode.SUBTRACT)
        with b3d.Locations((0, 0, base_h)):
            barb = build_barb(hose_id=barb_id, hose_od=barb_od, barb_count=3)
            p.part = p.part + barb
    return p.part

def build_single_cell_filter(params):
    cell_len = float(get_param(params, "cell_length", 80.0))
    cell_d = float(get_param(params, "cell_diameter", 20.0))
    num_helices = int(get_param(params, "num_helices", 2))
    tube_od = cell_d + 10.0
    tube_wall = 1.5

    with b3d.BuildPart() as p:
        b3d.Cylinder(radius=tube_od / 2.0, height=cell_len)
        b3d.Cylinder(radius=(tube_od - 2.0 * tube_wall) / 2.0, height=cell_len + 2.0, mode=b3d.Mode.SUBTRACT)
        core = build_helical_shape(cell_len, 360.0 * 2.0, cell_d / 2.0, 3.0)
        p.part = p.part + core
    return p.part

def build_dipole_antenna(params):
    length = float(get_param(params, "length", 50.0))
    thickness = float(get_param(params, "thickness", 2.0))
    gap = float(get_param(params, "gap", 2.0))
    arm_length = (length - gap) / 2.0

    with b3d.BuildPart() as p:
        with b3d.Locations((0, 0, gap / 2.0 + arm_length / 2.0)):
            b3d.Cylinder(radius=thickness / 2.0, height=arm_length)
        with b3d.Locations((0, 0, -gap / 2.0 - arm_length / 2.0)):
            b3d.Cylinder(radius=thickness / 2.0, height=arm_length)
    return p.part

def build_puck_antenna(params):
    puck_radius = float(get_param(params, "puck_radius", 50.0))
    puck_height = float(get_param(params, "puck_height", 20.0))
    ground_clearance = float(get_param(params, "ground_clearance", 0.0))
    panel_radius = float(get_param(params, "panel_radius", 45.0))
    panel_thickness = float(get_param(params, "panel_thickness", 3.0))
    panel_gap = float(get_param(params, "panel_height_above_puck", 10.0))
    num_antennas = int(get_param(params, "num_antennas", 4))
    antenna_len = float(get_param(params, "antenna_length", 31.0))
    antenna_w = float(get_param(params, "antenna_width", 4.0))
    antenna_t = float(get_param(params, "antenna_thickness", 1.0))

    with b3d.BuildPart() as p:
        with b3d.Locations((0, 0, ground_clearance)):
            with b3d.Locations((0, 0, puck_height / 2.0)):
                b3d.Cylinder(radius=puck_radius, height=puck_height)
            with b3d.Locations((0, 0, puck_height + panel_gap + panel_thickness / 2.0)):
                b3d.Cylinder(radius=panel_radius, height=panel_thickness)
            angle_step = 360.0 / num_antennas
            for i in range(num_antennas):
                angle = i * angle_step
                rad = math.radians(angle)
                cx = (puck_radius + antenna_len / 2.0) * math.cos(rad)
                cy = (puck_radius + antenna_len / 2.0) * math.sin(rad)
                cz = puck_height / 2.0
                with b3d.Locations(b3d.Location((cx, cy, cz), (0, 0, angle))):
                    b3d.Box(antenna_len, antenna_w, antenna_t)
    return p.part

def build_trace_optimizer(params, part="all"):
    sub_thickness = float(get_param(params, "substrate_thickness", 1.6))
    sub_width = float(get_param(params, "substrate_width", 20.0))
    sub_length = float(get_param(params, "substrate_length", 100.0))
    cu_thickness = float(get_param(params, "copper_thickness", 0.035))

    with b3d.BuildPart() as p:
        if part in ["all", "substrate"]:
            with b3d.Locations((sub_length / 2.0, sub_width / 2.0, sub_thickness / 2.0)):
                b3d.Box(sub_length, sub_width, sub_thickness)
        if part in ["all", "ground"]:
            with b3d.Locations((sub_length / 2.0, sub_width / 2.0, -cu_thickness / 2.0)):
                b3d.Box(sub_length, sub_width, cu_thickness)
        if part in ["all", "copper"]:
            with b3d.Locations((sub_length / 2.0, sub_width / 2.0, sub_thickness + cu_thickness / 2.0)):
                b3d.Box(sub_length, 2.0, cu_thickness)
        if part == "port1":
            with b3d.Locations((0.25, sub_width / 2.0, sub_thickness / 2.0)):
                b3d.Box(0.5, 4.0, sub_thickness)
        if part == "port2":
            with b3d.Locations((sub_length - 0.25, sub_width / 2.0, sub_thickness / 2.0)):
                b3d.Box(0.5, 4.0, sub_thickness)
    return p.part

def build_cyclone_manifold(params, part_to_generate="solid"):
    cy_d = float(get_param(params, "cyclone_diameter", 100.0))
    cy_r = cy_d / 2.0
    cy_h = float(get_param(params, "cylinder_height", 100.0))
    cone_h = float(get_param(params, "cone_height", 150.0))
    inlet_w = float(get_param(params, "inlet_width", 25.0))
    inlet_h = float(get_param(params, "inlet_height", 50.0))
    vf_d = float(get_param(params, "vortex_finder_diameter", 50.0))
    vf_r = vf_d / 2.0
    vf_len = float(get_param(params, "vortex_finder_length", 75.0))
    dust_d = float(get_param(params, "dust_outlet_diameter", 25.0))
    dust_r = dust_d / 2.0
    wt = float(get_param(params, "wall_thickness", 2.0))

    if part_to_generate in ["corkscrew_fluid", "fluid_volume", "fluid"]:
        with b3d.BuildPart() as p:
            with b3d.Locations((0, 0, cy_h / 2.0)):
                b3d.Cylinder(radius=cy_r, height=cy_h)
            with b3d.Locations((0, 0, -cone_h / 2.0)):
                b3d.Cone(bottom_radius=dust_r, top_radius=cy_r, height=cone_h)
            with b3d.Locations((0, 0, -cone_h - 12.5)):
                b3d.Cylinder(radius=dust_r, height=25.0)
            with b3d.Locations((cy_r - inlet_w / 2.0, cy_r, cy_h - inlet_h / 2.0)):
                b3d.Box(inlet_w, cy_d, inlet_h)
        return p.part

    if part_to_generate == "inlet":
        with b3d.BuildPart() as p:
            with b3d.Locations((cy_r - inlet_w / 2.0, cy_d, cy_h - inlet_h / 2.0)):
                b3d.Box(inlet_w, 1.0, inlet_h)
        return p.part

    if part_to_generate in ["outlet", "clean_outlet"]:
        with b3d.BuildPart() as p:
            with b3d.Locations((0, 0, cy_h + 25.0)):
                b3d.Cylinder(radius=vf_r - wt, height=1.0)
        return p.part

    if part_to_generate == "dust_outlet":
        with b3d.BuildPart() as p:
            with b3d.Locations((0, 0, -cone_h - 25.0)):
                b3d.Cylinder(radius=dust_r, height=1.0)
        return p.part

    fluid = build_cyclone_manifold(params, "fluid")
    with b3d.BuildPart() as shell:
        with b3d.Locations((0, 0, cy_h / 2.0)):
            b3d.Cylinder(radius=cy_r + wt, height=cy_h)
        with b3d.Locations((0, 0, -cone_h / 2.0)):
            b3d.Cone(bottom_radius=dust_r + wt, top_radius=cy_r + wt, height=cone_h)
        with b3d.Locations((0, 0, -cone_h - 12.5)):
            b3d.Cylinder(radius=dust_r + wt, height=25.0)
        with b3d.Locations((cy_r - inlet_w / 2.0, cy_r, cy_h - inlet_h / 2.0)):
            b3d.Box(inlet_w + 2 * wt, cy_d, inlet_h + 2 * wt)
        shell.part = shell.part - fluid
    return shell.part
