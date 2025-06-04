import numpy as np
from cv2 import pointPolygonTest
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

def find_frame_indices_and_weights(event_ts, timestamps):
    """Find the corresponding frame indices and interpolation weights for event timestamps"""
    idx = np.searchsorted(timestamps, event_ts)
    
    # Handle boundary cases
    if idx == 0:
        return 0, 0, 0.0  # Use first frame
    elif idx >= len(timestamps):
        return len(timestamps)-1, len(timestamps)-1, 0.0  # Use last frame
    
    # Normal case: event is between two frames
    idx_prev, idx_next = idx-1, idx
    
    # Calculate interpolation weight (between 0-1)
    t_prev = timestamps[idx_prev]
    t_next = timestamps[idx_next]
    
    # Prevent division by zero
    alpha = 0.0 if t_next == t_prev else (event_ts - t_prev) / (t_next - t_prev)
    
    return idx_prev, idx_next, alpha


def interpolate_polygon(polygon_prev, polygon_next, alpha):
    """Linearly interpolate polygon vertex coordinates"""
    return (1 - alpha) * polygon_prev + alpha * polygon_next


def interpolate_angle(angle_prev, angle_next, alpha):
    """Linearly interpolate rotation angle"""
    # Handle angle crossing 360°
    diff = angle_next - angle_prev
    if diff > 180:
        angle_next -= 360
    elif diff < -180:
        angle_next += 360
    
    return (1 - alpha) * angle_prev + alpha * angle_next


def interpolate_center(center_prev, center_next, alpha):
    """Linearly interpolate center point coordinates"""
    return (1 - alpha) * center_prev + alpha * center_next


def get_frame_info(idx, crops_info):
    """Get frame information for specified index"""
    frame_info = crops_info[idx]
    if frame_info[0] is None:  # barbara_info is None
        return None, None, None, None
    
    barbara_info = frame_info[0]
    polygon = barbara_info['polygon'] if 'polygon' in barbara_info else None
    angle = barbara_info['angle'] if 'angle' in barbara_info else None
    center = barbara_info['center'] if 'center' in barbara_info else None
    
    return polygon, angle, center, frame_info[2]  # polygon, angle, center, timestamp


def get_interpolated_frame_info(event_ts, crops_info):
    """Get interpolated frame information for event"""
    # Extract all timestamps
    timestamps = np.array([info[2] for info in crops_info if info[2] is not None])
    
    # Find corresponding previous and next frames
    idx_prev, idx_next, alpha = find_frame_indices_and_weights(event_ts, timestamps)
    
    # Get previous and next frame information
    poly_prev, angle_prev, center_prev, _ = get_frame_info(idx_prev, crops_info)
    
    # If previous and next frames are the same, return that frame's information
    if idx_prev == idx_next:
        return poly_prev, angle_prev, center_prev
    
    poly_next, angle_next, center_next, _ = get_frame_info(idx_next, crops_info)
    
    # If one frame's information is missing, use the other frame
    if poly_prev is None:
        return poly_next, angle_next, center_next
    if poly_next is None:
        return poly_prev, angle_prev, center_prev
    
    # Calculate interpolation
    interpolated_poly = interpolate_polygon(poly_prev, poly_next, alpha)
    interpolated_angle = interpolate_angle(angle_prev, angle_next, alpha)
    interpolated_center = interpolate_center(center_prev, center_next, alpha)
    
    return interpolated_poly, interpolated_angle, interpolated_center


def rotate_point(point, center, angle_deg):
    """Rotate a point around a center by specified angle
    
    Args:
        point: Point coordinates (x, y)
        center: Rotation center (cx, cy)
        angle_deg: Rotation angle (degrees)
    
    Returns:
        Rotated point coordinates
    """
    # Ensure numpy arrays
    if isinstance(point, torch.Tensor):
        point = point.cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()
    
    # Convert to radians
    angle_rad = np.radians(angle_deg)
    
    # Translate to origin
    x, y = point
    cx, cy = center
    x_shifted, y_shifted = x - cx, y - cy
    
    # Rotate
    x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
    y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
    
    # Translate back
    return x_rotated + cx, y_rotated + cy


def adjust_polygon_to_target_size(polygon, center, angle, target_size):
    """Adjust polygon size according to target size, maintaining center and rotation angle
    
    Args:
        polygon: Original polygon coordinates
        center: Center point
        angle: Rotation angle
        target_size: Target side length
    
    Returns:
        Adjusted polygon coordinates
    """
    # Calculate original size
    width = np.linalg.norm(polygon[1] - polygon[0])
    height = np.linalg.norm(polygon[2] - polygon[1])
    
    # Use maximum side as reference
    max_size = max(width, height)
    
    # If target size not specified, use original size
    if target_size is None or target_size <= 0:
        return polygon
    
    # Calculate scale factor
    scale = target_size / max_size
    
    # Scale polygon (maintaining center point)
    adjusted_polygon = np.zeros_like(polygon)
    for i, point in enumerate(polygon):
        # Vector from center to vertex
        vector = point - center
        # Scale vector
        scaled_vector = vector * scale
        # New vertex = center + scaled vector
        adjusted_polygon[i] = center + scaled_vector
    
    return adjusted_polygon


def point_in_rotated_box(point, polygon, center, angle):
    """Check if point is inside rotated cropping box
    
    Args:
        point: Point coordinates (x, y)
        polygon: Polygon vertices
        center: Rotation center
        angle: Rotation angle (degrees)
    
    Returns:
        Whether point is inside box
    """
    # Rotate point in reverse direction to make box horizontal
    x, y = rotate_point(point, center, -angle)
    
    # Ensure point coordinates are float type
    point = (float(x), float(y))
    
    # Ensure polygon is float32 type and has correct shape
    polygon = polygon.astype(np.float32).reshape(-1, 1, 2)
    
    # Use OpenCV's pointPolygonTest to check if point is inside polygon
    # Return value > 0 means point is inside, = 0 means on boundary, < 0 means outside
    return pointPolygonTest(polygon, point, False) >= 0


def transform_event_point(point, polygon, center, angle, output_size):
    """Transform event point coordinates to standardized output box
    
    Args:
        point: Event point coordinates (x, y)
        polygon: Polygon vertices
        center: Rotation center
        angle: Rotation angle (degrees)
        output_size: Output box size
    
    Returns:
        transformed_point: Transformed coordinates (x, y)
    """
    # Ensure center is numpy array
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()
    
    # Ensure point is numpy array
    if isinstance(point, torch.Tensor):
        point = point.cpu().numpy()
    
    # Calculate position relative to center
    relative_point = np.array(point) - center
    
    # Calculate rotation matrix to cancel angle
    rot_rad = np.radians(-angle)
    cos_theta = np.cos(rot_rad)
    sin_theta = np.sin(rot_rad)
    rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    # Rotate point
    rotated_point = np.dot(rot_matrix, relative_point)
    
    # Translate to output box center
    output_center = np.array([output_size / 2, output_size / 2])
    transformed_point = rotated_point + output_center
    
    # Clip coordinates and ensure integers
    transformed_point = np.clip(transformed_point, 0, output_size - 1)
    transformed_point = np.floor(transformed_point).astype(int)
    
    return transformed_point


def filter_event(event, crops_info, target_size=None, transform=True):
    """Process single event, determine if valid, considering box rotation
    
    Args:
        event: Single event data, format [timestamp, x, y, polarity]
        crops_info: Cropping box information list
        target_size: Target cropping box size, if None use original box size
        transform: Whether to transform event coordinates
        
    Returns:
        If valid event and transform=False, return original event
        If valid event and transform=True, return transformed event
        Otherwise return None
    """
    # Extract event information, ensure numpy values
    if isinstance(event, torch.Tensor):
        timestamp = event[0].item()
        x = event[1].item()
        y = event[2].item()
        polarity = event[3].item()
    else:
        timestamp, x, y, polarity = event
    
    # Get interpolated cropping box information
    polygon, angle, center = get_interpolated_frame_info(timestamp, crops_info)
    
    # If cannot get valid cropping box information, return None
    if polygon is None or angle is None or center is None:
        return None
    
    # Adjust polygon according to target size
    if target_size is not None:
        polygon = adjust_polygon_to_target_size(polygon, center, angle, target_size)
    
    # Check if event is inside rotated cropping box
    if point_in_rotated_box((x, y), polygon, center, angle):
        if transform:
            # Transform event coordinates
            output_size = target_size if target_size is not None else max(
                np.linalg.norm(polygon[1] - polygon[0]),
                np.linalg.norm(polygon[2] - polygon[1])
            )
            new_x, new_y = transform_event_point((x, y), polygon, center, angle, output_size)
            return [timestamp, new_x, new_y, polarity]
        else:
            # Return original event
            return event.tolist() if isinstance(event, torch.Tensor) else event
    
    return None


def filter_events_for(events_tensor, crops_info, target_size=None, transform=True, start_idx=None, end_idx=None):
    """
    Process events one by one using for loop, return array of valid events
    
    Args:
        events_tensor: Event data tensor [N, 4] (timestamp, x, y, polarity)
        crops_info: Cropping box information list
        target_size: Target cropping box size
        transform: Whether to transform event coordinates
        start_idx: Starting event index (optional)
        end_idx: Ending event index (optional)

    Returns:
        filtered_events: Filtered event data [M, 4]
    """
    # Determine event range to process
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(events_tensor)
    
    # Ensure indices are within valid range
    start_idx = max(0, min(start_idx, len(events_tensor)))
    end_idx = max(0, min(end_idx, len(events_tensor)))
    
    results = []
    # Use tqdm to show progress bar
    for event in tqdm(events_tensor[start_idx:end_idx], desc="Filtering events"):
        filtered = filter_event(event, crops_info, target_size, transform)
        if filtered is not None:
            results.append(filtered)
            
    if results:
        return np.array(results)
    else:
        return np.empty((0, 4))



def find_frame_indices_and_weights_batch(event_timestamps, frame_timestamps):
    """Find corresponding frame indices and interpolation weights for event timestamps in batch
    
    Args:
        event_timestamps: Event timestamp array [N]
        frame_timestamps: Frame timestamp array [M]
        
    Returns:
        idx_prev: Previous frame index array [N]
        idx_next: Next frame index array [N]
        alpha: Interpolation weight array [N]
        valid_mask: Valid event mask [N]
    """
    # Use binary search to determine frame index for each event timestamp
    idx = np.searchsorted(frame_timestamps, event_timestamps, 'right')
    
    # Create valid mask
    valid_mask = (idx > 0) & (idx < len(frame_timestamps))
    
    # Set default values for invalid events
    idx_prev = np.zeros_like(idx)
    idx_next = np.zeros_like(idx)
    alpha = np.zeros_like(event_timestamps, dtype=float)
    
    # Only process valid events
    valid_idx = idx[valid_mask]
    
    # Get previous and next frame indices
    valid_idx_prev = valid_idx - 1
    valid_idx_next = valid_idx
    
    # Save to result arrays
    idx_prev[valid_mask] = valid_idx_prev
    idx_next[valid_mask] = valid_idx_next
    
    # Calculate interpolation weights
    t_prev = frame_timestamps[valid_idx_prev]
    t_next = frame_timestamps[valid_idx_next]
    t_events = event_timestamps[valid_mask]
    
    # Prevent division by zero
    denominator = t_next - t_prev
    nonzero_mask = denominator != 0
    
    # Correction 1: Use combined mask to directly calculate alpha values
    if np.any(nonzero_mask):
        alpha[valid_mask & nonzero_mask] = (t_events[nonzero_mask] - t_prev[nonzero_mask]) / denominator[nonzero_mask]
    
    return idx_prev, idx_next, alpha, valid_mask


def get_frame_info_batch(frame_indices, crops_info):
    """Get frame information for specified indices in batch (optimized version)
    
    Args:
        frame_indices: Frame index array [N]
        crops_info: Cropping box information list
        
    Returns:
        polygons: Polygon array [N, 4, 2] or None
        angles: Angle array [N] or None
        centers: Center point array [N, 2] or None
        valid_mask: Valid information mask [N]
    """
    # Preprocess crops_info to NumPy arrays
    all_polygons = np.array([info[0]['polygon'] if info[0] is not None and 'polygon' in info[0] else None 
                            for info in crops_info], dtype=object)
    all_angles = np.array([info[0]['angle'] if info[0] is not None and 'angle' in info[0] else None 
                          for info in crops_info], dtype=object)
    all_centers = np.array([info[0]['center'] if info[0] is not None and 'center' in info[0] else None 
                           for info in crops_info], dtype=object)
    all_valid = np.array([info[0] is not None for info in crops_info], dtype=bool)
    
    # Initialize result arrays
    n_events = len(frame_indices)
    polygons = np.zeros((n_events, 4, 2), dtype=float)
    angles = np.zeros(n_events, dtype=float)
    centers = np.zeros((n_events, 2), dtype=float)
    valid_mask = np.zeros(n_events, dtype=bool)
    
    # Use indices to directly get corresponding frame information
    valid_indices = (frame_indices >= 0) & (frame_indices < len(crops_info))
    valid_frame_indices = frame_indices[valid_indices].astype(int)
    
    # Find valid indices
    frame_valid = all_valid[valid_frame_indices]
    valid_entries = valid_indices & np.array([frame_valid if isinstance(frame_valid, bool) else True for frame_valid in frame_valid])
    
    if np.any(valid_entries):
        # Extract frame information
        for i, idx in enumerate(frame_indices):
            if not valid_indices[i]:
                continue
                
            idx = int(idx)
            if all_polygons[idx] is not None and all_angles[idx] is not None and all_centers[idx] is not None:
                polygons[i] = all_polygons[idx]
                angles[i] = all_angles[idx]
                centers[i] = all_centers[idx]
                valid_mask[i] = True
    
    return polygons, angles, centers, valid_mask


def interpolate_frame_info_batch(idx_prev, idx_next, alpha, valid_mask, crops_info):
    """Calculate interpolated frame information in batch
    
    Args:
        idx_prev: Previous frame index array [N]
        idx_next: Next frame index array [N]
        alpha: Interpolation weight array [N]
        valid_mask: Valid event mask [N]
        crops_info: Cropping box information list
        
    Returns:
        interp_polygons: Interpolated polygon array [N, 4, 2]
        interp_angles: Interpolated angle array [N]
        interp_centers: Interpolated center point array [N, 2]
        result_valid_mask: Result valid mask [N]
    """
    n_events = len(idx_prev)
    
    # Get previous and next frame information
    poly_prev, angle_prev, center_prev, valid_prev = get_frame_info_batch(idx_prev, crops_info)
    poly_next, angle_next, center_next, valid_next = get_frame_info_batch(idx_next, crops_info)
    
    # Initialize result arrays
    interp_polygons = np.zeros((n_events, 4, 2), dtype=float)
    interp_angles = np.zeros(n_events, dtype=float)
    interp_centers = np.zeros((n_events, 2), dtype=float)
    
    # Calculate final valid mask
    result_valid_mask = valid_mask & (valid_prev | valid_next)
    
    # Handle case where previous and next frames are the same
    same_frame_mask = idx_prev == idx_next
    interp_polygons[result_valid_mask & same_frame_mask] = poly_prev[result_valid_mask & same_frame_mask]
    interp_angles[result_valid_mask & same_frame_mask] = angle_prev[result_valid_mask & same_frame_mask]
    interp_centers[result_valid_mask & same_frame_mask] = center_prev[result_valid_mask & same_frame_mask]
    
    # Handle case where previous and next frames are different
    diff_frame_mask = ~same_frame_mask & result_valid_mask
    
    # Handle case where only previous frame is valid
    prev_only_mask = diff_frame_mask & valid_prev & ~valid_next
    interp_polygons[prev_only_mask] = poly_prev[prev_only_mask]
    interp_angles[prev_only_mask] = angle_prev[prev_only_mask]
    interp_centers[prev_only_mask] = center_prev[prev_only_mask]
    
    # Handle case where only next frame is valid
    next_only_mask = diff_frame_mask & ~valid_prev & valid_next
    interp_polygons[next_only_mask] = poly_next[next_only_mask]
    interp_angles[next_only_mask] = angle_next[next_only_mask]
    interp_centers[next_only_mask] = center_next[next_only_mask]
    
    # Handle case where both frames are valid
    both_valid_mask = diff_frame_mask & valid_prev & valid_next
    if np.any(both_valid_mask):
        # Get interpolation weights
        weights = alpha[both_valid_mask].reshape(-1, 1, 1)
        
        # Linear interpolation of polygon vertices
        interp_polygons[both_valid_mask] = (
            (1 - weights) * poly_prev[both_valid_mask] + 
            weights * poly_next[both_valid_mask]
        )
        
        # Linear interpolation of angles (considering angle wrapping)
        angle_p = angle_prev[both_valid_mask]
        angle_n = angle_next[both_valid_mask]
        
        # Correction 2: Use clearer variable names, handle positive and negative cases separately
        diff = angle_n - angle_p
        
        # Handle positive wrapping case
        wrap_pos = diff > 180
        angle_n[wrap_pos] -= 360
        
        # Handle negative wrapping case
        wrap_neg = diff < -180
        angle_n[wrap_neg] += 360
        
        # Calculate interpolated angles
        interp_angles[both_valid_mask] = angle_p * (1 - alpha[both_valid_mask]) + angle_n * alpha[both_valid_mask]
        
        # Linear interpolation of center points
        interp_centers[both_valid_mask] = (
            (1 - alpha[both_valid_mask].reshape(-1, 1)) * center_prev[both_valid_mask] + 
            alpha[both_valid_mask].reshape(-1, 1) * center_next[both_valid_mask]
        )
    
    return interp_polygons, interp_angles, interp_centers, result_valid_mask


def adjust_polygons_batch(polygons, centers, target_size, valid_mask):
    """Adjust polygon sizes in batch (optimized version)
    
    Args:
        polygons: Polygon array [N, 4, 2]
        centers: Center point array [N, 2]
        target_size: Target side length
        valid_mask: Valid mask [N]
        
    Returns:
        adjusted_polygons: Adjusted polygon array [N, 4, 2]
    """
    if target_size is None or target_size <= 0:
        return polygons
    
    # Only calculate valid polygons
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return polygons
        
    # Calculate relative vectors: vertex positions relative to center [N, 4, 2]
    rel = polygons - centers[:, None, :]
    
    # Calculate scale factors
    width = np.linalg.norm(polygons[:, 1] - polygons[:, 0], axis=1)
    height = np.linalg.norm(polygons[:, 2] - polygons[:, 1], axis=1)
    max_size = np.maximum(width, height)
    scale = target_size / max_size
    
    # Apply scaling, maintaining shape [N, 4, 2]
    rel_scaled = rel * scale[:, None, None]
    
    # Move back to center point
    adjusted_polygons = rel_scaled + centers[:, None, :]
    
    # Ensure invalid polygons are not modified
    adjusted_polygons[~valid_mask] = polygons[~valid_mask]
    
    return adjusted_polygons


def rotate_points_batch(points, centers, angles):
    """Rotate points in batch
    
    Args:
        points: Point coordinate array [N, 2]
        centers: Rotation center array [N, 2]
        angles: Rotation angle array (degrees) [N]
        
    Returns:
        rotated_points: Rotated point coordinate array [N, 2]
    """
    # Convert angles to radians
    angles_rad = np.radians(-angles)
    
    # Calculate rotation matrix elements
    cos_theta = np.cos(angles_rad)
    sin_theta = np.sin(angles_rad)
    
    # Translate to origin
    points_centered = points - centers
    
    # Rotate
    rotated_x = points_centered[:, 0] * cos_theta - points_centered[:, 1] * sin_theta
    rotated_y = points_centered[:, 0] * sin_theta + points_centered[:, 1] * cos_theta
    
    # Translate back
    rotated_points = np.column_stack((rotated_x, rotated_y)) + centers
    
    return rotated_points


def transform_points_batch(points, centers, angles, output_size, valid_mask):
    """Transform point coordinates in batch (vectorized implementation)
    
    Args:
        points: Point coordinate array [N, 2]
        centers: Center point array [N, 2]
        angles: Angle array [N]
        output_size: Output box size
        valid_mask: Valid mask [N]
        
    Returns:
        transformed_points: Transformed point coordinate array [N, 2]
    """
    n_events = len(points)
    transformed_points = np.zeros((n_events, 2), dtype=int)
    
    # Only process valid events
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return transformed_points
    
    # Extract valid data
    valid_points = points[valid_indices]
    valid_centers = centers[valid_indices]
    valid_angles = angles[valid_indices]
    
    # Calculate position relative to center
    points_centered = valid_points - valid_centers
    
    # Create rotation matrices [N, 2, 2]
    angles_rad = np.radians(-valid_angles)
    cos_theta = np.cos(angles_rad)
    sin_theta = np.sin(angles_rad)
    
    # Create rotation matrix array with shape [N, 2, 2]
    rot_matrices = np.zeros((len(valid_indices), 2, 2))
    rot_matrices[:, 0, 0] = cos_theta
    rot_matrices[:, 0, 1] = -sin_theta
    rot_matrices[:, 1, 0] = sin_theta
    rot_matrices[:, 1, 1] = cos_theta
    
    # Expand points_centered to [N, 2, 1] for batch matrix multiplication
    points_centered_exp = points_centered.reshape(-1, 2, 1)
    
    # Batch matrix multiplication: [N, 2, 2] @ [N, 2, 1] -> [N, 2, 1]
    rotated_points = np.matmul(rot_matrices, points_centered_exp).squeeze()
    
    # Translate to output box center
    output_center = np.array([output_size / 2, output_size / 2])
    transformed_valid_points = rotated_points + output_center
    
    # Clip coordinates and ensure integers
    transformed_valid_points = np.clip(transformed_valid_points, 0, output_size - 1)
    transformed_points[valid_indices] = np.floor(transformed_valid_points).astype(int)
    
    return transformed_points


def is_point_in_quad(points, quads):
    """Check if points are inside quadrilaterals using cross product method (fully vectorized)
    
    Args:
        points: Point coordinate array [N, 2]
        quads: Quadrilateral vertex array [N, 4, 2]
        
    Returns:
        in_quad: Mask indicating if points are inside quadrilaterals [N]
    """
    # Extract quadrilateral vertices
    a = quads[:, 0]  # Top-left
    b = quads[:, 1]  # Top-right
    c = quads[:, 2]  # Bottom-right
    d = quads[:, 3]  # Bottom-left
    
    # Calculate cross products to check if points are inside quadrilaterals
    # If all cross products between vertex-to-point vectors and clockwise edges are positive, point is inside
    in_quad = (
        (np.cross(b - a, points - a) >= 0) &
        (np.cross(c - b, points - b) >= 0) &
        (np.cross(d - c, points - c) >= 0) &
        (np.cross(a - d, points - d) >= 0)
    )
    
    return in_quad


def check_points_in_polygons_batch(points, polygons, valid_mask):
    """Check if points are inside polygons in batch (optimized version)
    
    Args:
        points: Point coordinate array [N, 2]
        polygons: Polygon array [N, 4, 2]
        valid_mask: Valid mask [N]
        
    Returns:
        in_polygon: Mask indicating if points are inside polygons [N]
    """
    n_events = len(points)
    in_polygon = np.zeros(n_events, dtype=bool)
    
    # Only process valid events
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return in_polygon
    
    # Extract valid data
    valid_points = points[valid_indices]
    valid_polygons = polygons[valid_indices]
    
    # Use cross product method to check if points are inside quadrilaterals
    valid_results = is_point_in_quad(valid_points, valid_polygons)
    
    # Put results back into original array
    in_polygon[valid_indices] = valid_results
    
    return in_polygon


def preprocess_crops_info(crops_info):
    """Keep only frames with valid polygon/angle/center and timestamp"""
    polygons = []
    angles = []
    centers = []
    timestamps = []

    for info in crops_info:
        # Check if barbara_info (info[0]) and timestamp (info[2]) exist
        if info[0] is None or info[2] is None:
            continue
            
        barbara = info[0]
        # Check if geometric information is complete and valid
        if ('polygon' not in barbara or
            'angle' not in barbara or
            'center' not in barbara or
            barbara['polygon'] is None or
            barbara['angle'] is None or
            barbara['center'] is None):
            continue

        # Ensure polygon is 4x2
        current_polygon = np.asarray(barbara['polygon'], dtype=np.float32)
        if current_polygon.shape != (4, 2):
             # If shape is incorrect, can choose to skip or try to fix, here we skip
             # print(f"Warning: Invalid polygon shape {current_polygon.shape}, skipping frame timestamp={info[2]}")
             continue
             
        polygons.append(current_polygon)
        angles.append(barbara['angle'])
        centers.append(np.asarray(barbara['center'], dtype=np.float32))
        timestamps.append(info[2])

    # Check if valid data was collected
    if not timestamps: # If list is empty
        print("Warning: preprocess_crops_info found no frames with valid geometric information and timestamps.")
        # Return empty numpy arrays, subsequent logic needs to handle this case
        return np.empty((0, 4, 2)), np.empty((0,)), np.empty((0, 2)), np.empty((0,))

    # Convert to numpy
    try:
        polygons_np = np.stack(polygons, axis=0)  # [M_valid, 4, 2]
        angles_np = np.asarray(angles, dtype=np.float32)
        centers_np = np.stack(centers, axis=0)    # [M_valid, 2]
        timestamps_np = np.asarray(timestamps, dtype=np.float64)
    except ValueError as e:
        print(f"Error: Failed to convert lists to NumPy arrays: {e}")
        # For example, if centers are not all 2D, np.stack will fail
        # In this case return empty arrays
        return np.empty((0, 4, 2)), np.empty((0,)), np.empty((0, 2)), np.empty((0,))
        
    # Ensure timestamps are sorted, crucial for np.searchsorted
    sort_indices = np.argsort(timestamps_np)
    
    return polygons_np[sort_indices], angles_np[sort_indices], centers_np[sort_indices], timestamps_np[sort_indices]


def filter_events_all(events_tensor, crops_info, target_size=None, transform=False):
    """
    Fully vectorized event data filtering.
    This function aims to efficiently process large amounts of events through NumPy vectorized operations.
    Core steps include:
    1. Preprocess cropping box information, filter valid frames and sort.
    2. Find corresponding previous and next frame indices and interpolation weights for each event.
    3. Get or interpolate polygon, rotation angle and center point for each event.
    4. (Optional) Adjust polygon size according to target_size.
    5. Reverse rotate event points (relative to their corresponding polygon's rotation) for subsequent point-in-polygon testing.
       This step simulates the logic in filter_event's point_in_rotated_box, where points are reverse rotated
       and then compared with the original (but possibly size-adjusted) rotated polygon.
    6. Use cross product method to check if reverse-rotated event points are inside adjusted polygons,
       considering both clockwise and counterclockwise vertex orders.
    7. (Optional) If transform is True, transform valid event coordinates to normalized target box.
    """
    if not isinstance(events_tensor, np.ndarray):
        if isinstance(events_tensor, torch.Tensor):
            events = events_tensor.cpu().numpy()
        else:
            # Try to convert to NumPy array, if fails user may need to check input type
            try:
                events = np.asarray(events_tensor)
            except Exception as e:
                print(f"Error: Input events_tensor type cannot be directly converted to NumPy array: {type(events_tensor)}, error: {e}")
                return np.empty((0, 4))
    else:
        events = events_tensor

    if events.ndim != 2 or events.shape[1] != 4:
        print(f"Error: events_tensor shape should be [N, 4], actual: {events.shape}")
        return np.empty((0,4))

    if len(events) == 0:
        return np.empty((0, 4))

    # 1. Preprocess crops_info: Get polygons, angles, centers and timestamps for valid frames, sorted by timestamp
    #    Returned polygons_data, angles_data, centers_data, frame_ts all have length n_frames (number of valid frames)
    polygons_data, angles_data, centers_data, frame_ts = preprocess_crops_info(crops_info)
    n_frames = len(frame_ts)
    if n_frames == 0:
        # If no valid reference frame information, cannot process events
        return np.empty((0, 4))

    # 2. Extract basic event data
    timestamps = events[:, 0]
    points = events[:, 1:3]  # Original event (x, y) coordinates
    polarities = events[:, 3]
    n_events = len(events)

    # 3. Find index in sorted frame timestamps for each event timestamp
    #    `idx` indicates where event timestamp should be inserted in frame_ts to maintain sorting
    #    `side='right'` means if timestamps[i] == frame_ts[j], then idx[i] == j + 1
    idx = np.searchsorted(frame_ts, timestamps, side='right')

    #    Calculate previous frame (idx_prev) and next frame (idx_next) indices for each event
    #    Use np.clip to ensure indices are in range [0, n_frames - 1]
    idx_prev = np.clip(idx - 1, 0, n_frames - 1)
    idx_next = np.clip(idx, 0, n_frames - 1) # If idx == n_frames, idx_next will be n_frames - 1

    # 4. Calculate interpolation weights alpha
    #    alpha is used for linear interpolation of polygons, angles and center points between frames
    alpha = np.zeros(n_events, dtype=float)
    #    Only need interpolation when event timestamp is strictly between two frames (idx > 0 and idx < n_frames)
    needs_interp_mask = (idx > 0) & (idx < n_frames)
    if np.any(needs_interp_mask):
        # Extract timestamps of events needing interpolation and their corresponding frame timestamps
        _t_events = timestamps[needs_interp_mask]
        _t_prev = frame_ts[idx_prev[needs_interp_mask]]
        _t_next = frame_ts[idx_next[needs_interp_mask]]
        
        denominator = _t_next - _t_prev
        # Can only calculate valid alpha when frame timestamps are different
        calc_alpha_mask = denominator != 0
        if np.any(calc_alpha_mask):
             # Get global event indices needing alpha update
             alpha_update_indices = np.where(needs_interp_mask)[0][calc_alpha_mask]
             alpha[alpha_update_indices] = (_t_events[calc_alpha_mask] - _t_prev[calc_alpha_mask]) / denominator[calc_alpha_mask]
             # Ensure alpha is in range [0, 1], handle floating point precision issues
             alpha[alpha_update_indices] = np.clip(alpha[alpha_update_indices], 0.0, 1.0)

    # 5. Get or interpolate polygon, angle and center point for each event
    #    First, default to using previous frame (idx_prev) information
    interp_polygons = np.copy(polygons_data[idx_prev])
    interp_angles = np.copy(angles_data[idx_prev])
    interp_centers = np.copy(centers_data[idx_prev])
    
    #    Then, for events with alpha > 0 (i.e., needing interpolation), update with interpolated results
    interp_update_mask = alpha > 0
    if np.any(interp_update_mask):
        interp_indices = np.where(interp_update_mask)[0]
        _alpha_val = alpha[interp_indices]
        _idx_prev_val = idx_prev[interp_indices] # Points to valid frame data in polygons_data etc.
        _idx_next_val = idx_next[interp_indices] # Points to valid frame data in polygons_data etc.

        weights = _alpha_val.reshape(-1, 1, 1)
        interp_polygons[interp_indices] = (1 - weights) * polygons_data[_idx_prev_val] + weights * polygons_data[_idx_next_val]
        
        # Angle interpolation, handle wrapping (e.g., from 350° to 10°)
        angle_diff = (angles_data[_idx_next_val] - angles_data[_idx_prev_val] + 180) % 360 - 180
        interp_angles[interp_indices] = angles_data[_idx_prev_val] + angle_diff * _alpha_val
        
        interp_centers[interp_indices] = (
            (1 - _alpha_val).reshape(-1, 1) * centers_data[_idx_prev_val] +
            _alpha_val.reshape(-1, 1) * centers_data[_idx_next_val]
        )

    # 6. (Optional) Adjust polygon size according to target_size
    #    adjusted_polygons stores final polygons for collision detection
    adjusted_polygons = interp_polygons # If not adjusting, adjusted_polygons equals interp_polygons
    if target_size is not None and target_size > 0:
        _polygons_to_adjust = interp_polygons
        _centers_for_adjust = interp_centers

        # Calculate vectors from center to vertices
        rel_vectors = _polygons_to_adjust - _centers_for_adjust[:, None, :]
        # Calculate original polygon width and height (based on first two edges)
        width = np.linalg.norm(_polygons_to_adjust[:, 1] - _polygons_to_adjust[:, 0], axis=1)
        height = np.linalg.norm(_polygons_to_adjust[:, 2] - _polygons_to_adjust[:, 1], axis=1)
        current_max_size = np.maximum(width, height)
        
        # Calculate scale factors, avoid division by zero
        scale_factors = np.zeros_like(current_max_size)
        valid_scale_mask = current_max_size > 0
        scale_factors[valid_scale_mask] = target_size / current_max_size[valid_scale_mask]
        
        # Apply scaling and move back to center
        rel_vectors_scaled = rel_vectors * scale_factors[:, None, None]
        adjusted_polygons = rel_vectors_scaled + _centers_for_adjust[:, None, :]

    # 7. Prepare event point coordinates for "point in polygon" testing
    #    This logic simulates point_in_rotated_box in filter_event:
    #    Reverse rotate original event points `points` relative to their corresponding box
    #    (defined by interp_centers and interp_angles) to get `points_for_test`.
    #    These points are still in world coordinates.
    points_centered_for_test = points - interp_centers
    angles_rad_for_test = np.radians(-interp_angles) # Use negative polygon angle for reverse rotation
    
    cos_theta_test = np.cos(angles_rad_for_test)
    sin_theta_test = np.sin(angles_rad_for_test)
    
    # Perform reverse rotation
    rotated_x_test = points_centered_for_test[:, 0] * cos_theta_test - points_centered_for_test[:, 1] * sin_theta_test
    rotated_y_test = points_centered_for_test[:, 0] * sin_theta_test + points_centered_for_test[:, 1] * cos_theta_test
    
    # Translate reverse-rotated points back to their original center to get test points
    points_for_test = np.column_stack((rotated_x_test, rotated_y_test)) + interp_centers

    # 8. Use cross product method to check if `points_for_test` are inside `adjusted_polygons`
    #    `adjusted_polygons` are in world coordinates, after interpolation, size adjustment and original rotation.
    #    Consider both clockwise and counterclockwise vertex orders for robustness.
    poly_a = adjusted_polygons[:, 0]; poly_b = adjusted_polygons[:, 1]
    poly_c = adjusted_polygons[:, 2]; poly_d = adjusted_polygons[:, 3]
    
    cross_ab = np.cross(poly_b - poly_a, points_for_test - poly_a)
    cross_bc = np.cross(poly_c - poly_b, points_for_test - poly_b)
    cross_cd = np.cross(poly_d - poly_c, points_for_test - poly_c)
    cross_da = np.cross(poly_a - poly_d, points_for_test - poly_d)
    
    epsilon = 1e-6 # Floating point comparison tolerance

    # Assume counterclockwise vertex order (all cross products >= -epsilon)
    in_polygon_ccw = (
        (cross_ab >= -epsilon) & (cross_bc >= -epsilon) &
        (cross_cd >= -epsilon) & (cross_da >= -epsilon)
    )
    # Assume clockwise vertex order (all cross products <= epsilon)
    in_polygon_cw = (
        (cross_ab <= epsilon) & (cross_bc <= epsilon) &
        (cross_cd <= epsilon) & (cross_da <= epsilon)
    )
    in_polygon_mask = in_polygon_ccw | in_polygon_cw # Satisfy either condition
            
    if not np.any(in_polygon_mask):
        return np.empty((0, 4))

    # 9. (Optional) If transform is True, transform valid event coordinates to target output box
    if transform:
        # Extract information from events that passed point-in-polygon test
        valid_event_indices = np.where(in_polygon_mask)[0]
        
        _points_to_transform = points[valid_event_indices]
        _centers_for_transform = interp_centers[valid_event_indices]
        _angles_for_transform = interp_angles[valid_event_indices]
        
        # Calculate output box size
        # If target_size is specified, use it; otherwise, calculate based on valid polygon sizes
        if target_size is not None and target_size > 0:
            output_size = target_size
        else:
            # Calculate maximum side length from polygons that passed test as output size
            # adjusted_polygons contains all event polygons, we only use valid ones
            valid_output_polygons = adjusted_polygons[in_polygon_mask]
            if len(valid_output_polygons) == 0: # Should not happen as in_polygon_mask is not all False
                output_size = 0
            else:
                output_size = np.max([
                    np.linalg.norm(valid_output_polygons[:, 1] - valid_output_polygons[:, 0], axis=1),
                    np.linalg.norm(valid_output_polygons[:, 2] - valid_output_polygons[:, 1], axis=1)
                ])

        if output_size <= 0:
            # If cannot determine valid output size, cannot perform transformation
            # Can choose to return original untransformed valid events or return empty. Here return empty to match expectations.
            return np.empty((0,4))

        # Perform coordinate transformation (similar to rotation logic in step 7, but purpose is to map to output box)
        points_centered_for_transform = _points_to_transform - _centers_for_transform
        angles_rad_for_transform = np.radians(-_angles_for_transform) # Use negative polygon angle
        
        cos_theta_transform = np.cos(angles_rad_for_transform)
        sin_theta_transform = np.sin(angles_rad_for_transform)
        
        # Construct rotation matrices
        rot_matrices = np.zeros((len(valid_event_indices), 2, 2))
        rot_matrices[:, 0, 0] = cos_theta_transform
        rot_matrices[:, 0, 1] = -sin_theta_transform
        rot_matrices[:, 1, 0] = sin_theta_transform
        rot_matrices[:, 1, 1] = cos_theta_transform
        
        points_centered_exp = points_centered_for_transform.reshape(-1, 2, 1)
        # Rotate centered points
        rotated_points_for_output = np.matmul(rot_matrices, points_centered_exp).squeeze(axis=2)
        
        # Translate to output box center
        output_frame_center = np.array([output_size / 2, output_size / 2])
        transformed_coords = rotated_points_for_output + output_frame_center
        
        # Clip and round
        transformed_coords = np.clip(transformed_coords, 0, output_size - 1)
        final_transformed_coords = np.floor(transformed_coords).astype(int)
        
        # Initialize complete transformed_points array, then only fill valid parts
        # This ensures correct alignment with timestamps[in_polygon_mask] etc.
        transformed_points_output_all = np.zeros((n_events, 2), dtype=int)
        transformed_points_output_all[valid_event_indices] = final_transformed_coords
        
        result_events = np.column_stack((
            timestamps[in_polygon_mask],
            transformed_points_output_all[in_polygon_mask], # Use mask to get values from filled array
            polarities[in_polygon_mask]
        ))
    else:
        # If not transforming, directly return filtered original events
        result_events = events[in_polygon_mask]

    return result_events


def _filter_batch(args):
    """Process single batch of event data (for multi-threading)
    
    Args:
        args: Tuple (batch_events, crops_info, target_size, transform)
    
    Returns:
        Filtered event data
    """
    batch_events, crops_info, target_size, transform = args
    # Note: preprocess_crops_info is now called inside filter_events_all
    return filter_events_all(batch_events, crops_info, target_size, transform)

def filter_events_parallel(events_tensor, crops_info, target_size=None, transform=False, 
                          batch_size=100000, n_workers=4):
    """Process event data in parallel using batch + multi-threading approach
    
    Args:
        events_tensor: Event data tensor [N, 4] (timestamp, x, y, polarity)
        crops_info: Cropping box information list
        target_size: Target cropping box size
        transform: Whether to transform event coordinates
        batch_size: Number of events to process per batch
        n_workers: Number of parallel threads
    
    Returns:
        filtered_events: Filtered event data sorted by timestamp
    """
    # Handle empty input case
    if len(events_tensor) == 0:
        return np.empty((0, 4))
    
    # Split into batches
    batches = []
    for i in range(0, len(events_tensor), batch_size):
        batch_events = events_tensor[i:i+batch_size]
        batches.append((batch_events, crops_info, target_size, transform))
    
    # Use multi-threading for parallel processing, show progress bar
    results = []
    batch_valid_counts = []  # New: record number of valid events per batch

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(_filter_batch, batch) for batch in batches]
        
        # Use tqdm to show progress
        for future in tqdm(
            futures, 
            total=len(batches),
            desc="Processing event batches",
            unit="batch"
        ):
            result = future.result()
            results.append(result)
            batch_valid_counts.append(len(result))  # Record number of valid events
        
    # Output statistics of valid events per batch
    print(f"Total batches: {len(batch_valid_counts)}, Batches with valid events: {sum(1 for c in batch_valid_counts if c > 0)}")
    
    # Merge results
    valid_results = [result for result in results if len(result) > 0]
    if valid_results:
        # Merge results from all batches
        merged_events = np.vstack(valid_results)
        # Sort by timestamp (first column)
        sorted_indices = np.argsort(merged_events[:, 0])
        return merged_events[sorted_indices]
    else:
        return np.empty((0, 4))