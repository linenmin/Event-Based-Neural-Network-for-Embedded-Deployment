#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag Detection Tool Module
Used for detecting AprilTag markers in images and calculating Barbara region
"""

import numpy as np
import cv2
import pupil_apriltags as apriltags

# Basic configuration parameters
TAG_FAMILY = 'tag36h11'  # AprilTag family type
TAG_REF_WIDTH = 287      # AprilTag reference width (pixels)
BARBARA_REF_SIZE = 861   # Barbara reference side length (pixels)
BARBARA_GAP = 82         # Gap between Barbara and tag (pixels)

def create_detector():
    """Create AprilTag detector"""
    return apriltags.Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )

def detect_apriltags(frame, detector):
    """Detect AprilTag markers in image"""
    if frame.dtype != np.uint8:
        frame_8bit = (frame / 256).astype(np.uint8)
    elif len(frame.shape) == 3:
        frame_8bit = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_8bit = frame
    
    return detector.detect(frame_8bit)

def calculate_barbara_region(corners, tag_ref_width=TAG_REF_WIDTH, 
                                    barbara_ref_size=BARBARA_REF_SIZE, 
                           barbara_gap=BARBARA_GAP):
    """Calculate Barbara region
    
    Args:
        corners: Four corner points of AprilTag
        tag_ref_width: AprilTag reference width
        barbara_ref_size: Barbara reference side length
        barbara_gap: Gap between Barbara and tag
        
    Returns:
        barbara_polygon: Four corner point coordinates of Barbara region
    """
    # Get bottom-left corner of tag as reference point
    bl_idx = np.argmin(corners[:, 0] - corners[:, 1])  # Bottom-left
    br_idx = np.argmax(corners[:, 0] + corners[:, 1])  # Bottom-right
    P = corners[bl_idx].astype(np.float32)
    
    # Calculate direction vectors
    vec_bottom = corners[br_idx] - corners[bl_idx]
    norm_bottom = np.linalg.norm(vec_bottom)
    if norm_bottom == 0:
        return None
    
    # Calculate unit vectors
    d_bottom = vec_bottom / norm_bottom
    d_up = np.array([-d_bottom[1], d_bottom[0]])
    
    # Calculate actual dimensions
    s = norm_bottom / tag_ref_width
    B = s * barbara_ref_size
    gap = s * barbara_gap
    
    # Calculate Barbara region
    BR = P - gap * d_bottom
    BL = BR - B * d_bottom
    TR = BR - B * d_up
    TL = BL - B * d_up
    
    return np.array([BL, BR, TR, TL], dtype=np.float32)

def calculate_angle(polygon):
    """Calculate rotation angle of polygon"""
    bottom_left = polygon[0]
    bottom_right = polygon[1]
    dx = bottom_right[0] - bottom_left[0]
    dy = bottom_right[1] - bottom_left[1]
    return np.degrees(np.arctan2(dy, dx))

def process_frame(frame, timestamp, detector, margin_ratio=0.03, tag_ref_width=TAG_REF_WIDTH, 
                 barbara_ref_size=BARBARA_REF_SIZE, barbara_gap=BARBARA_GAP, is_raw=False):
    """Process single frame, detect AprilTag and calculate Barbara region
    
    Args:
        frame: Input image
        detector: AprilTag detector
        margin_ratio: Margin ratio for expanding Barbara region
        tag_ref_width: AprilTag reference width (pixels)
        barbara_ref_size: Barbara reference side length (pixels)
        barbara_gap: Gap between Barbara and tag (pixels)
        is_raw: Whether input is RAW frame (16-bit or 10-bit), if True convert to 8-bit grayscale first
        
    Returns:
        barbara_info: Dictionary containing Barbara region information
        cropped_frame: Cropped image
    """
    # If RAW frame, convert to 8-bit grayscale first
    if is_raw:
        if frame.dtype != np.uint8:
            # Assume RAW is 10-bit or 16-bit, linearly stretch to 8-bit
            frame = ((frame.astype(np.float32) / frame.max()) * 255).astype(np.uint8)
        if len(frame.shape) == 3:
            None

    # Detect AprilTag
    detections = detect_apriltags(frame, detector)
    
    if not detections:
        return None, None, None
    
    # Find tag with ID 0
    barbara_tag = None
    for detection in detections:
        if detection.tag_id == 0:
            barbara_tag = detection
            break
    
    if barbara_tag is None:
        return None, None, None
    
    # Calculate Barbara region
    barbara_polygon = calculate_barbara_region(
        barbara_tag.corners, 
        tag_ref_width=tag_ref_width,
        barbara_ref_size=barbara_ref_size,
        barbara_gap=barbara_gap
    )
    
    if barbara_polygon is None:
        return None, None, None
    
    # Apply margin_ratio to expand Barbara region
    if margin_ratio > 0:
        # Calculate center point
        center = np.mean(barbara_polygon, axis=0)
        
        # Calculate vector from each point to center, expand by margin_ratio
        expanded_polygon = []
        for pt in barbara_polygon:
            vector = pt - center
            expanded_pt = center + vector * (1 + margin_ratio)
            expanded_polygon.append(expanded_pt)
        
        barbara_polygon = np.array(expanded_polygon)
    
    # Calculate rotation angle
    angle = calculate_angle(barbara_polygon)
    
    # Get Barbara region information
    barbara_info = {
        'polygon': barbara_polygon,
        'angle': angle,
        'center': np.mean(barbara_polygon, axis=0)
    }
    
    # Calculate rotated rectangular region
    rect = cv2.minAreaRect(barbara_polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get rotation matrix
    center_np = np.mean(barbara_polygon, axis=0)
    center = (float(center_np[0]), float(center_np[1]))
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate size of rotated image
    height, width = frame.shape[:2]
    rotated_frame = cv2.warpAffine(frame, rot_matrix, (width, height))
    
    # Calculate rotated crop region
    rotated_box = cv2.transform(np.array([box]), rot_matrix)[0]
    x_min = int(np.min(rotated_box[:, 0]))
    x_max = int(np.max(rotated_box[:, 0]))
    y_min = int(np.min(rotated_box[:, 1]))
    y_max = int(np.max(rotated_box[:, 1]))
    
    # Ensure crop region is within image bounds
    x_min = max(0, x_min)
    x_max = min(width, x_max)
    y_min = max(0, y_min)
    y_max = min(height, y_max)
    
    # Crop rotated image
    cropped_frame = rotated_frame[y_min:y_max, x_min:x_max]
    
    return barbara_info, cropped_frame, timestamp


def process_batch(frames, timestamps, margin_ratio=0.03, tag_ref_width=287, barbara_ref_size=861, barbara_gap=82, is_raw=False):
    detector = create_detector()
    batch_results = []
    for frame, ts in zip(frames, timestamps):
        try:
            result = process_frame(
                frame, ts, detector,
                margin_ratio=margin_ratio,
                tag_ref_width=tag_ref_width,
                barbara_ref_size=barbara_ref_size,
                barbara_gap=barbara_gap,
                is_raw=is_raw
            )
            batch_results.append(result)
        except Exception as e:
            raise RuntimeError(f"Error processing frame at timestamp {ts}: {e}")
    return batch_results

# Batch splitting function
def split_batches(data, n_batches):
    return [data[i::n_batches] for i in range(n_batches)]