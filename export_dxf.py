"""
DXF Export Module for Floor Plan Recognition Results

Converts pixel-based segmentation masks to vectorized DXF format
suitable for RSRP simulation and CAD applications.
"""

import numpy as np
import cv2
import ezdxf
from scipy import ndimage


# Room type labels (matching floorplan_fuse_map)
ROOM_TYPES = {
    0: "Background",
    1: "Closet",
    2: "Bathroom",
    3: "Living Room/Kitchen",
    4: "Bedroom",
    5: "Hall",
    6: "Balcony",
    7: "Unused",
    8: "Unused",
    9: "Door/Window",
    10: "Wall"
}


def simplify_contours(contours, epsilon_factor=0.005):
    """
    Simplify contours using Douglas-Peucker algorithm.

    Args:
        contours: List of contours from cv2.findContours
        epsilon_factor: Approximation accuracy as a fraction of perimeter

    Returns:
        List of simplified contours
    """
    simplified = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified.append(approx)
    return simplified


def extract_wall_lines(floorplan_mask, min_area=50):
    """
    Extract wall boundaries as polylines.

    Args:
        floorplan_mask: 2D numpy array with label encoding
        min_area: Minimum contour area to include

    Returns:
        List of wall contours (simplified polylines)
    """
    # Extract wall pixels (label = 10)
    wall_mask = (floorplan_mask == 10).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and simplify
    wall_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            wall_contours.append(contour)

    # Simplify contours
    wall_contours = simplify_contours(wall_contours, epsilon_factor=0.003)

    return wall_contours


def extract_openings(floorplan_mask, min_area=20):
    """
    Extract door/window openings as polylines.

    Args:
        floorplan_mask: 2D numpy array with label encoding
        min_area: Minimum contour area to include

    Returns:
        List of opening contours (simplified polylines)
    """
    # Extract opening pixels (label = 9)
    opening_mask = (floorplan_mask == 9).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(opening_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area and simplify
    opening_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            opening_contours.append(contour)

    # Simplify contours
    opening_contours = simplify_contours(opening_contours, epsilon_factor=0.005)

    return opening_contours


def extract_room_polygons(floorplan_mask, min_area=100):
    """
    Extract room regions as closed polygons with room type labels.

    Args:
        floorplan_mask: 2D numpy array with label encoding
        min_area: Minimum room area to include

    Returns:
        List of tuples: (contour, room_type_id, room_type_name)
    """
    rooms = []

    # Process each room type (1-8, excluding 0=background, 9=door, 10=wall)
    for room_id in range(1, 9):
        if room_id == 7 or room_id == 8:  # Skip unused labels
            continue

        # Create mask for this room type
        room_mask = (floorplan_mask == room_id).astype(np.uint8)

        # Find connected components (separate rooms of same type)
        labeled, num_features = ndimage.label(room_mask)

        for component_id in range(1, num_features + 1):
            component_mask = (labeled == component_id).astype(np.uint8) * 255

            # Find contour of this room
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = contours[0]  # Take largest contour
                area = cv2.contourArea(contour)

                if area >= min_area:
                    # Simplify contour
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Calculate centroid for label placement
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0

                    rooms.append({
                        'contour': approx,
                        'type_id': room_id,
                        'type_name': ROOM_TYPES[room_id],
                        'area': area,
                        'centroid': (cx, cy)
                    })

    return rooms


def export_to_dxf(floorplan_mask, output_path, scale=1.0, units='Pixels'):
    """
    Export floor plan to DXF format with layers and metadata.

    Args:
        floorplan_mask: 2D numpy array with label encoding (0-10)
        output_path: Output DXF file path
        scale: Scale factor (e.g., pixels to meters conversion)
        units: Unit name for documentation (default: 'Pixels')

    Returns:
        Path to saved DXF file
    """
    print(f"Extracting vectorized features from floor plan mask...")

    # Extract geometric features
    wall_contours = extract_wall_lines(floorplan_mask)
    opening_contours = extract_openings(floorplan_mask)
    room_polygons = extract_room_polygons(floorplan_mask)

    print(f"Found {len(wall_contours)} wall segments, {len(opening_contours)} openings, {len(room_polygons)} rooms")

    # Create DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Create layers with properties
    doc.layers.add('WALLS', color=7, linetype='CONTINUOUS')  # White/black walls
    doc.layers.add('OPENINGS', color=1, linetype='CONTINUOUS')  # Red openings
    doc.layers.add('ROOMS', color=3, linetype='CONTINUOUS')  # Green rooms

    # Add walls as polylines
    for wall_contour in wall_contours:
        points = [(pt[0][0] * scale, pt[0][1] * scale) for pt in wall_contour]
        if len(points) >= 2:
            msp.add_lwpolyline(points, dxfattribs={'layer': 'WALLS'})

    # Add openings as polylines
    for opening_contour in opening_contours:
        points = [(pt[0][0] * scale, pt[0][1] * scale) for pt in opening_contour]
        if len(points) >= 2:
            msp.add_lwpolyline(points, dxfattribs={'layer': 'OPENINGS'})

    # Add rooms as closed polylines (no text labels)
    for room in room_polygons:
        contour = room['contour']
        points = [(pt[0][0] * scale, pt[0][1] * scale) for pt in contour]

        if len(points) >= 3:
            # Add closed polyline
            msp.add_lwpolyline(
                points,
                close=True,
                dxfattribs={
                    'layer': 'ROOMS',
                    'color': 3
                }
            )

    # Save DXF file
    doc.saveas(output_path)
    print(f"DXF file saved to: {output_path}")

    return output_path


def apply_post_processing(room_type, room_boundary):
    """
    Apply post-processing to refine predictions before vectorization.

    Args:
        room_type: Room type prediction (2D array, 0-8)
        room_boundary: Boundary prediction (2D array, 0-2)

    Returns:
        Processed floorplan mask (2D array, 0-10)
    """
    from utils.util import fill_break_line, flood_fill, refine_room_region

    # Merge room and boundary into single mask
    floorplan = room_type.copy()
    floorplan[room_boundary == 1] = 9  # Door/window
    floorplan[room_boundary == 2] = 10  # Wall

    # Separate back into components for processing
    rm_ind = floorplan.copy()
    rm_ind[floorplan == 9] = 0
    rm_ind[floorplan == 10] = 0

    bd_ind = np.zeros(floorplan.shape, dtype=np.uint8)
    bd_ind[floorplan == 9] = 9
    bd_ind[floorplan == 10] = 10

    hard_c = (bd_ind > 0).astype(np.uint8)

    # Fill broken wall lines
    cw_mask = fill_break_line(hard_c)

    # Create fused mask
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # Fill holes
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # Refine room regions (one connected component = one room label)
    new_rm_ind = refine_room_region(cw_mask, rm_ind)
    new_rm_ind = fuse_mask * new_rm_ind

    # Merge boundaries back
    new_rm_ind[bd_ind == 9] = 9
    new_rm_ind[bd_ind == 10] = 10

    return new_rm_ind.astype(np.uint8)
