# blender_viz.py - Script for Blender/bpy and PyVista 3D Antenna Far-field Pattern Visualizations
import sys
import os
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


def render_pyvista_farfield_pattern(
    vtk_path: str = "artifacts/radiation_pattern_3d.vtk",
    output_png: str = "artifacts/radiation_pattern_render.png"
) -> bool:
    """
    Headless PyVista renderer for openEMS 3D Antenna Far-Field Radiation Pattern VTK PolyData.
    """
    if not HAS_PYVISTA or not os.path.exists(vtk_path):
        print(f"PyVista missing or VTK path '{vtk_path}' not found.")
        return False

    try:
        pv.OFF_SCREEN = True
        mesh = pv.read(vtk_path)

        plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
        plotter.set_background("#0f172a")

        plotter.add_mesh(
            mesh,
            scalars="FarField_Gain_dBi" if "FarField_Gain_dBi" in mesh.point_data else None,
            cmap="turbo",
            show_edges=True,
            edge_color="#334155",
            opacity=0.9,
            scalar_bar_args={"title": "Far-Field Gain (dBi)", "color": "#f8fafc"}
        )

        plotter.add_axes()
        plotter.camera_position = "iso"

        os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
        plotter.screenshot(output_png)
        plotter.close()
        print(f"Rendered PyVista 3D far-field radiation pattern to {output_png}")
        return True
    except Exception as e:
        print(f"Error rendering PyVista farfield pattern: {e}")
        return False


def setup_blender_scene(x3d_file):
    if not HAS_BPY:
        print("bpy (Blender) not available in this environment.")
        return

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(f"Importing X3D: {x3d_file}")
    bpy.ops.import_scene.x3d(filepath=x3d_file)

    imported_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not imported_objs:
        print("No mesh found in X3D file.")
        return

    main_obj = imported_objs[0]
    main_obj.name = "EM_Simulation_Data"

    gn_modifier = main_obj.modifiers.new(name="VizNodes", type='NODES')
    node_group = bpy.data.node_groups.new(name="EM_Viz", type='GeometryNodeTree')
    gn_modifier.node_group = node_group

    nodes = node_group.nodes
    links = node_group.links

    node_in = nodes.new(type='NodeGroupInput')
    node_out = nodes.new(type='NodeGroupOutput')

    if hasattr(node_group, "interface"):
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:
        node_group.inputs.new('NodeSocketGeometry', 'Geometry')
        node_group.outputs.new('NodeSocketGeometry', 'Geometry')

    node_time = nodes.new(type='GeometryNodeInputSceneTime')
    node_math = nodes.new(type='ShaderNodeMath')
    node_math.operation = 'SINE'

    node_combine = nodes.new(type='ShaderNodeCombineXYZ')
    node_transform = nodes.new(type='GeometryNodeTransform')
    node_mat = nodes.new(type='GeometryNodeSetMaterial')

    links.new(node_in.outputs[0], node_transform.inputs[0])
    links.new(node_transform.outputs[0], node_mat.inputs[0])
    links.new(node_mat.outputs[0], node_out.inputs[0])

    links.new(node_time.outputs[1], node_math.inputs[0])
    links.new(node_math.outputs[0], node_combine.inputs[2])
    links.new(node_combine.outputs[0], node_transform.inputs[1])

    rad_x3d = x3d_file.replace(".x3d", "_radiation.x3d")
    if os.path.exists(rad_x3d):
        print(f"Importing Radiation Pattern: {rad_x3d}")
        bpy.ops.import_scene.x3d(filepath=rad_x3d)

    output_blend = x3d_file.replace(".x3d", ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    print(f"Blender scene saved to: {output_blend}")


if __name__ == "__main__":
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if args:
            setup_blender_scene(args[0])
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".vtk"):
        render_pyvista_farfield_pattern(sys.argv[1])
    else:
        print("blender_viz.py ready for Blender/bpy and PyVista 3D far-field pattern rendering.")
