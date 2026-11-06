# paraview_export.py - Script for ParaView/pvpython
import sys
import os

try:
    from paraview.simple import *
except ImportError:
    print("Error: ParaView modules not found. Ensure this is run with pvpython.")
    sys.exit(1)

def run_post_processing(vtk_file, output_x3d):
    print(f"Loading VTK: {vtk_file}")

    # Load data
    data = LegacyVTKReader(FileNames=[vtk_file])

    # Apply Clipping
    clip = Clip(Input=data)
    clip.ClipType = 'Plane'
    clip.ClipType.Origin = [50.0, 10.0, 1.0]
    clip.ClipType.Normal = [0.0, 1.0, 0.0]

    # Apply Warp By Scalar (visualizing E-field magnitude)
    warp = WarpByScalar(Input=clip)
    warp.Scalars = ['POINTS', 'E-field']
    warp.ScaleFactor = 5.0

    # Create a view and show the result
    view = CreateView('RenderView')
    display = Show(warp, view)

    # Set representation to Surface
    display.Representation = 'Surface'

    # Reset camera
    view.ResetCamera()

    # Export as X3D
    print(f"Exporting X3D to: {output_x3d}")
    ExportView(output_x3d, view=view)

    # Handle Radiation Pattern if it exists
    rad_vtk = vtk_file.replace("field_data.vtk", "radiation_pattern.vtk")
    if os.path.exists(rad_vtk):
        rad_data = LegacyVTKReader(FileNames=[rad_vtk])
        rad_view = CreateView('RenderView')
        Show(rad_data, rad_view)
        rad_view.ResetCamera()
        rad_x3d = output_x3d.replace(".x3d", "_radiation.x3d")
        ExportView(rad_x3d, view=rad_view)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: pvpython paraview_export.py <input_vtk> <output_x3d>")
    else:
        run_post_processing(sys.argv[1], sys.argv[2])
