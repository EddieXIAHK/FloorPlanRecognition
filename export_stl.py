"""
STL 3D Model Export Module for Floor Plan Recognition Results

Converts 2D floor plan segmentation to 3D building models with realistic heights.
Exports to STL format and provides interactive 3D visualization.
"""

import numpy as np
from stl import mesh
import plotly.graph_objects as go
from export_dxf import extract_wall_lines, extract_openings, extract_room_polygons


# Standard residential building dimensions (in meters)
DEFAULT_WALL_HEIGHT = 2.7  # Floor to ceiling
DEFAULT_FLOOR_THICKNESS = 0.2  # Concrete slab thickness
DEFAULT_DOOR_HEIGHT = 2.1
DEFAULT_WINDOW_BOTTOM = 0.9
DEFAULT_WINDOW_TOP = 2.1


def create_wall_mesh_from_contour(contour, scale=1.0, wall_height=DEFAULT_WALL_HEIGHT):
    """
    Extrude a 2D wall contour into a 3D mesh.

    Args:
        contour: 2D contour from OpenCV (Nx1x2 array)
        scale: Scale factor for coordinates
        wall_height: Height of the wall in meters

    Returns:
        numpy-stl mesh object for the wall
    """
    # Extract points and scale
    points_2d = contour.squeeze()  # Shape: (N, 2)
    num_points = len(points_2d)

    if num_points < 2:
        return None

    # Create bottom and top vertices
    vertices_bottom = np.column_stack([
        points_2d[:, 0] * scale,  # X
        points_2d[:, 1] * scale,  # Y
        np.zeros(num_points)       # Z = 0 (floor level)
    ])

    vertices_top = np.column_stack([
        points_2d[:, 0] * scale,
        points_2d[:, 1] * scale,
        np.full(num_points, wall_height)  # Z = wall_height
    ])

    # Create faces (2 triangles per wall segment)
    faces = []
    for i in range(num_points - 1):
        # Bottom-left, bottom-right, top-right
        v0 = vertices_bottom[i]
        v1 = vertices_bottom[i + 1]
        v2 = vertices_top[i + 1]
        v3 = vertices_top[i]

        # Triangle 1: bottom-left, bottom-right, top-right
        faces.append([v0, v1, v2])
        # Triangle 2: bottom-left, top-right, top-left
        faces.append([v0, v2, v3])

    if len(faces) == 0:
        return None

    # Create mesh
    wall_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            wall_mesh.vectors[i][j] = face[j]

    return wall_mesh


def create_floor_ceiling_mesh(room_polygons, scale=1.0, z_level=0.0):
    """
    Create horizontal mesh (floor or ceiling) from room polygons.

    Args:
        room_polygons: List of room dictionaries with 'contour' key
        scale: Scale factor
        z_level: Z-coordinate for the plane (0 for floor, wall_height for ceiling)

    Returns:
        numpy-stl mesh object
    """
    all_faces = []

    for room in room_polygons:
        contour = room['contour']
        points_2d = contour.squeeze()

        if len(points_2d) < 3:
            continue

        # Scale points
        points_2d_scaled = points_2d * scale

        # Triangulate polygon using fan triangulation from first vertex
        for i in range(1, len(points_2d_scaled) - 1):
            v0 = np.array([points_2d_scaled[0, 0], points_2d_scaled[0, 1], z_level])
            v1 = np.array([points_2d_scaled[i, 0], points_2d_scaled[i, 1], z_level])
            v2 = np.array([points_2d_scaled[i + 1, 0], points_2d_scaled[i + 1, 1], z_level])

            # Add face (reverse winding for ceiling to face downward)
            if z_level > 0:  # Ceiling
                all_faces.append([v0, v2, v1])
            else:  # Floor
                all_faces.append([v0, v1, v2])

    if len(all_faces) == 0:
        return None

    # Create mesh
    floor_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            floor_mesh.vectors[i][j] = face[j]

    return floor_mesh


def export_to_stl(floorplan_mask, output_path, scale=1.0, wall_height=DEFAULT_WALL_HEIGHT,
                  min_wall_area=200, min_opening_area=100, min_room_area=500,
                  clean_artifacts=True, border_margin=10, include_floor=True, include_ceiling=True):
    """
    Export floor plan to 3D STL format.

    Args:
        floorplan_mask: 2D numpy array with label encoding (0-10)
        output_path: Output STL file path
        scale: Scale factor (pixels to meters)
        wall_height: Height of walls in meters
        min_wall_area: Minimum wall area threshold
        min_opening_area: Minimum opening area threshold
        min_room_area: Minimum room area threshold
        clean_artifacts: Apply cleaning operations
        border_margin: Edge filtering margin
        include_floor: Include floor mesh
        include_ceiling: Include ceiling mesh

    Returns:
        Path to saved STL file
    """
    print(f"\n3D Model Generation:")
    print(f"  Wall height: {wall_height}m")
    print(f"  Scale: {scale} m/pixel")
    print(f"  Including floor: {include_floor}, ceiling: {include_ceiling}")

    # Extract 2D features
    print("  Extracting geometric features...")
    wall_contours = extract_wall_lines(floorplan_mask, min_area=min_wall_area,
                                       clean_artifacts=clean_artifacts, border_margin=border_margin)
    room_polygons = extract_room_polygons(floorplan_mask, min_area=min_room_area,
                                          clean_artifacts=clean_artifacts, border_margin=border_margin)

    print(f"  Found {len(wall_contours)} wall segments, {len(room_polygons)} rooms")

    # Generate 3D meshes
    all_meshes = []

    # 1. Extrude walls to 3D
    print("  Generating wall meshes...")
    for contour in wall_contours:
        wall_mesh = create_wall_mesh_from_contour(contour, scale=scale, wall_height=wall_height)
        if wall_mesh is not None:
            all_meshes.append(wall_mesh)

    # 2. Create floor
    if include_floor and len(room_polygons) > 0:
        print("  Generating floor mesh...")
        floor_mesh = create_floor_ceiling_mesh(room_polygons, scale=scale, z_level=0.0)
        if floor_mesh is not None:
            all_meshes.append(floor_mesh)

    # 3. Create ceiling
    if include_ceiling and len(room_polygons) > 0:
        print("  Generating ceiling mesh...")
        ceiling_mesh = create_floor_ceiling_mesh(room_polygons, scale=scale, z_level=wall_height)
        if ceiling_mesh is not None:
            all_meshes.append(ceiling_mesh)

    if len(all_meshes) == 0:
        print("  ERROR: No valid meshes generated!")
        return None

    # Combine all meshes
    print(f"  Merging {len(all_meshes)} mesh components...")
    combined_mesh = mesh.Mesh(np.concatenate([m.data for m in all_meshes]))

    # Save STL
    combined_mesh.save(output_path)
    print(f"  STL file saved to: {output_path}")
    print(f"  Total triangles: {len(combined_mesh.data)}")

    return output_path


def visualize_stl_interactive(stl_path, html_output=None, auto_open=True):
    """
    Create interactive 3D visualization of STL model using plotly.

    Args:
        stl_path: Path to STL file
        html_output: Optional path to save HTML file
        auto_open: Automatically open in browser

    Returns:
        plotly Figure object
    """
    # Load STL
    stl_mesh = mesh.Mesh.from_file(stl_path)

    # Extract vertices and faces
    vertices = stl_mesh.vectors.reshape(-1, 3)
    num_triangles = len(stl_mesh.vectors)

    # Create indices for triangles
    i_indices = np.arange(0, num_triangles * 3, 3)
    j_indices = np.arange(1, num_triangles * 3, 3)
    k_indices = np.arange(2, num_triangles * 3, 3)

    # Extract coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Create plotly mesh
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i_indices,
            j=j_indices,
            k=k_indices,
            color='lightblue',
            opacity=0.9,
            flatshading=True,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.5,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=100
            )
        )
    ])

    # Update layout
    fig.update_layout(
        title="Floor Plan 3D Model",
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=800
    )

    # Save HTML if requested
    if html_output:
        fig.write_html(html_output)
        print(f"  Interactive 3D viewer saved to: {html_output}")

    # Show in browser
    if auto_open:
        fig.show()

    return fig


def visualize_stl_from_data(floorplan_mask, scale=1.0, wall_height=DEFAULT_WALL_HEIGHT,
                             min_wall_area=200, min_room_area=500,
                             clean_artifacts=True, border_margin=10,
                             html_output=None, auto_open=True):
    """
    Generate and visualize 3D model directly from floor plan mask without saving STL.

    Args:
        floorplan_mask: 2D numpy array with label encoding
        scale: Scale factor
        wall_height: Wall height in meters
        min_wall_area: Minimum wall area threshold
        min_room_area: Minimum room area threshold
        clean_artifacts: Apply cleaning
        border_margin: Edge filtering margin
        html_output: Optional HTML output path
        auto_open: Auto-open in browser

    Returns:
        plotly Figure object
    """
    print("  Generating 3D visualization directly from mask...")

    # Extract features
    wall_contours = extract_wall_lines(floorplan_mask, min_area=min_wall_area,
                                       clean_artifacts=clean_artifacts, border_margin=border_margin)
    room_polygons = extract_room_polygons(floorplan_mask, min_area=min_room_area,
                                          clean_artifacts=clean_artifacts, border_margin=border_margin)

    # Collect all vertices and triangles
    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    # Add walls
    for contour in wall_contours:
        points_2d = contour.squeeze()
        if len(points_2d) < 2:
            continue

        num_points = len(points_2d)

        # Create vertices
        for pt in points_2d:
            all_vertices.append([pt[0] * scale, pt[1] * scale, 0.0])  # Bottom
        for pt in points_2d:
            all_vertices.append([pt[0] * scale, pt[1] * scale, wall_height])  # Top

        # Create triangles
        for i in range(num_points - 1):
            b1 = vertex_offset + i
            b2 = vertex_offset + i + 1
            t1 = vertex_offset + num_points + i
            t2 = vertex_offset + num_points + i + 1

            all_triangles.append([b1, b2, t2])
            all_triangles.append([b1, t2, t1])

        vertex_offset += num_points * 2

    # Add floor and ceiling
    for room in room_polygons:
        contour = room['contour']
        points_2d = contour.squeeze()
        if len(points_2d) < 3:
            continue

        # Floor vertices
        floor_start = vertex_offset
        for pt in points_2d:
            all_vertices.append([pt[0] * scale, pt[1] * scale, 0.0])

        # Floor triangles (fan triangulation)
        for i in range(1, len(points_2d) - 1):
            all_triangles.append([floor_start, floor_start + i, floor_start + i + 1])

        vertex_offset += len(points_2d)

        # Ceiling vertices
        ceiling_start = vertex_offset
        for pt in points_2d:
            all_vertices.append([pt[0] * scale, pt[1] * scale, wall_height])

        # Ceiling triangles (reversed winding)
        for i in range(1, len(points_2d) - 1):
            all_triangles.append([ceiling_start, ceiling_start + i + 1, ceiling_start + i])

        vertex_offset += len(points_2d)

    if len(all_vertices) == 0:
        print("  ERROR: No geometry generated!")
        return None

    # Convert to numpy arrays
    vertices = np.array(all_vertices)
    triangles = np.array(all_triangles)

    # Create plotly figure
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color='lightblue',
            opacity=0.9,
            flatshading=True
        )
    ])

    fig.update_layout(
        title="Floor Plan 3D Model",
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200,
        height=800
    )

    if html_output:
        fig.write_html(html_output)
        print(f"  Interactive viewer saved to: {html_output}")

    if auto_open:
        fig.show()

    return fig
