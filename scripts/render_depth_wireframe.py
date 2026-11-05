import trimesh
import numpy as np
from PIL import Image, ImageDraw

def project_points(points, width, height, camera_pos, target, up, fov=60):
    # View Matrix
    z_axis = (camera_pos - target)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    view_matrix = np.eye(4)
    view_matrix[0, :3] = x_axis
    view_matrix[1, :3] = y_axis
    view_matrix[2, :3] = z_axis
    view_matrix[0, 3] = -np.dot(x_axis, camera_pos)
    view_matrix[1, 3] = -np.dot(y_axis, camera_pos)
    view_matrix[2, 3] = -np.dot(z_axis, camera_pos)

    # Transform to Camera Space
    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_cam = (view_matrix @ points_h.T).T

    # Perspective Projection
    aspect = width / height
    f = 1.0 / np.tan(np.radians(fov) / 2)
    z_near = 0.1
    z_far = 1000.0

    projection_matrix = np.zeros((4, 4))
    projection_matrix[0, 0] = f / aspect
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = (z_far + z_near) / (z_near - z_far)
    projection_matrix[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
    projection_matrix[3, 2] = -1

    points_clip = (projection_matrix @ points_cam.T).T

    # Perspective Divide
    w = points_clip[:, 3]
    w[w == 0] = 1e-6 # Avoid division by zero
    points_ndc = points_clip[:, :3] / w[:, np.newaxis]

    # Viewport Transform
    screen_points = np.zeros((len(points), 2))
    screen_points[:, 0] = (points_ndc[:, 0] + 1) * 0.5 * width
    screen_points[:, 1] = (1 - points_ndc[:, 1]) * 0.5 * height

    return screen_points, points_cam[:, 2]

def render_wireframe(stl_path, output_path, casing_path=None):
    print(f"Loading {stl_path}...")
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Collect all edges
    edges = mesh.edges_unique
    vertices = mesh.vertices

    casing_edges = []
    casing_vertices = []
    casing_offset = len(vertices)

    if casing_path:
        print(f"Loading casing {casing_path}...")
        casing_mesh = trimesh.load(casing_path)
        if isinstance(casing_mesh, trimesh.Scene):
            casing_mesh = casing_mesh.dump(concatenate=True)
        casing_edges = casing_mesh.edges_unique
        casing_vertices = casing_mesh.vertices

    # Combine vertices for projection
    all_vertices = np.vstack([vertices, casing_vertices]) if len(casing_vertices) > 0 else vertices
    centroid = mesh.centroid # Center on main object

    # Camera setup
    camera_pos = centroid + np.array([50, -60, 40])
    target = centroid
    up = np.array([0, 0, 1])

    width, height = 1200, 900

    print("Projecting...")
    screen_pts, depths = project_points(all_vertices, width, height, camera_pos, target, up)

    # Process edges
    edge_z_main = []
    for i, edge in enumerate(edges):
        z1 = depths[edge[0]]
        z2 = depths[edge[1]]
        avg_z = (z1 + z2) / 2.0
        edge_z_main.append((avg_z, i, 'main'))

    edge_z_casing = []
    for i, edge in enumerate(casing_edges):
        # Apply offset for casing indices
        idx1 = edge[0] + casing_offset
        idx2 = edge[1] + casing_offset
        z1 = depths[idx1]
        z2 = depths[idx2]
        avg_z = (z1 + z2) / 2.0
        edge_z_casing.append((avg_z, (idx1, idx2), 'casing'))

    # Combine and sort
    all_edges = edge_z_main + edge_z_casing
    all_edges.sort(key=lambda x: x[0]) # Ascending Z (furthest first)

    if not all_edges:
        print("No edges found!")
        return

    min_z = all_edges[0][0]
    max_z = all_edges[-1][0]
    z_range = max_z - min_z if max_z != min_z else 1.0

    print(f"Drawing {len(all_edges)} edges...")
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    for item in all_edges:
        z = item[0]
        type = item[2]

        if type == 'main':
            idx = item[1]
            p1 = screen_pts[edges[idx][0]]
            p2 = screen_pts[edges[idx][1]]

            # Depth Cueing for Main Object
            norm = (z - min_z) / z_range
            line_width = 1 + 3 * norm
            gray = int(180 * (1 - norm)) # 180 to 0
            color = (gray, gray, gray)

        else: # casing
            p_indices = item[1]
            p1 = screen_pts[p_indices[0]]
            p2 = screen_pts[p_indices[1]]

            # Faint style for casing
            line_width = 1
            color = (200, 200, 200) # Light gray

        # Check if roughly on screen
        if (0 <= p1[0] <= width and 0 <= p1[1] <= height) or \
           (0 <= p2[0] <= width and 0 <= p2[1] <= height):

            draw.line([tuple(p1), tuple(p2)], fill=color, width=int(line_width))

    print(f"Saving to {output_path}...")
    img.save(output_path)
    print("Done.")

if __name__ == "__main__":
    import sys
    # Example usage: python render_depth_wireframe.py output.png main.stl [casing.stl]

    # Default behavior for the specific task
    if len(sys.argv) == 1:
        # Render just helix
        render_wireframe("images/temp_helical.stl", "images/Wireframe view of Helical Geometry.png")
        # Render helix + casing
        # Check if casing exists
        try:
            with open("images/temp_casing.stl"):
                render_wireframe("images/temp_helical.stl", "images/Wireframe Corkscrew with Frame.png", "images/temp_casing.stl")
        except FileNotFoundError:
            pass
    elif len(sys.argv) >= 3:
        output = sys.argv[1]
        main_stl = sys.argv[2]
        casing_stl = sys.argv[3] if len(sys.argv) > 3 else None
        render_wireframe(main_stl, output, casing_stl)
