import math
import numpy as np
import cv2 as cv
from typing import Optional

from config import PERSPECTIVE_PADDING_PX, FIELD_WIDTH_CM, FIELD_HEIGHT_CM
from navigation.calibration import (
    MARKER_HEADING_OFFSET_DEG,
    MARKER_TO_DRIVE_CENTER_FORWARD_PX,
    MARKER_TO_DRIVE_CENTER_RIGHT_PX,
    MARKER_PERSPECTIVE_X_GAIN,
    MARKER_PERSPECTIVE_Y_GAIN,
)


def normalize_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def corrected_robot_pose_values(robot_pose, frame_width: int | None = None, frame_height: int | None = None):
    corrected_heading = normalize_angle_rad(
        robot_pose.heading_rad + math.radians(MARKER_HEADING_OFFSET_DEG)
    )

    # Robot-local unit vectors in image coordinates.
    # These rotate with the ArUco/robot heading.
    forward_x = math.cos(corrected_heading)
    forward_y = math.sin(corrected_heading)
    right_x = math.cos(corrected_heading + math.pi / 2.0)
    right_y = math.sin(corrected_heading + math.pi / 2.0)

    normalized_x = 0.0
    normalized_y = 0.0

    if frame_width is not None and frame_width > 0:
        image_center_x = frame_width / 2.0
        normalized_x = (float(robot_pose.x) - image_center_x) / image_center_x

    if frame_height is not None and frame_height > 0:
        image_center_y = frame_height / 2.0
        normalized_y = (float(robot_pose.y) - image_center_y) / image_center_y

    # 1) Perspective correction: image-space offset caused by marker height.
    # This should NOT rotate with the robot. It only depends on where the elevated marker
    # appears in the camera image.
    perspective_offset_x = MARKER_PERSPECTIVE_X_GAIN * normalized_x
    perspective_offset_y = MARKER_PERSPECTIVE_Y_GAIN * normalized_y

    # 2) Physical mount correction: robot-local offset from marker to drive center.
    # This DOES rotate with the robot.
    mount_offset_x = (
        MARKER_TO_DRIVE_CENTER_FORWARD_PX * forward_x
        + MARKER_TO_DRIVE_CENTER_RIGHT_PX * right_x
    )
    mount_offset_y = (
        MARKER_TO_DRIVE_CENTER_FORWARD_PX * forward_y
        + MARKER_TO_DRIVE_CENTER_RIGHT_PX * right_y
    )

    # Final correction in image coordinates.
    offset_x = perspective_offset_x + mount_offset_x
    offset_y = perspective_offset_y + mount_offset_y

    corrected_x = float(robot_pose.x) + offset_x
    corrected_y = float(robot_pose.y) + offset_y

    # For the UI: express the final image-space correction back in the robot-local axes.
    # This is the part that will appear to "switch" when the robot turns 90 degrees.
    used_forward_px = offset_x * forward_x + offset_y * forward_y
    used_right_px = offset_x * right_x + offset_y * right_y

    return (
        corrected_x,
        corrected_y,
        corrected_heading,
        used_forward_px,
        used_right_px,
        offset_x,
        offset_y,
        normalized_x,
        normalized_y,
    )


def get_scales(frame_width: int, frame_height: int) -> tuple[float, float]:
    width = frame_width - 2 * PERSPECTIVE_PADDING_PX
    height = frame_height - 2 * PERSPECTIVE_PADDING_PX
    scale_x = width / FIELD_WIDTH_CM
    scale_y = height / FIELD_HEIGHT_CM
    return scale_x, scale_y


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, p3, p4):
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def segment_intersects_box(xa, ya, xb, yb, box, margin=8.0):
    x1, y1, x2, y2 = box
    x1_pad = x1 - margin
    y1_pad = y1 - margin
    x2_pad = x2 + margin
    y2_pad = y2 + margin
    
    p1 = (xa, ya)
    p2 = (xb, yb)
    if segments_intersect(p1, p2, (x1_pad, y1_pad), (x1_pad, y2_pad)):
        return True
    if segments_intersect(p1, p2, (x2_pad, y1_pad), (x2_pad, y2_pad)):
        return True
    if segments_intersect(p1, p2, (x1_pad, y1_pad), (x2_pad, y1_pad)):
        return True
    if segments_intersect(p1, p2, (x1_pad, y2_pad), (x2_pad, y2_pad)):
        return True
    return False


def get_middle_obstacles(danger_contours: list[np.ndarray], frame_width: int, frame_height: int, scale_x: float, scale_y: float) -> list[tuple[float, float, float, float]]:
    obstacles = []
    for c in danger_contours:
        x_px, y_px, w_px, h_px = cv.boundingRect(c)
        if w_px >= 0.8 * frame_width or h_px >= 0.8 * frame_height:
            continue
        x1 = (x_px - PERSPECTIVE_PADDING_PX) / scale_x
        y1 = (y_px - PERSPECTIVE_PADDING_PX) / scale_y
        x2 = (x_px + w_px - PERSPECTIVE_PADDING_PX) / scale_x
        y2 = (y_px + h_px - PERSPECTIVE_PADDING_PX) / scale_y
        obstacles.append((x1, y1, x2, y2))
    return obstacles


def check_route_segment_for_obstacles(xa, ya, xb, yb, obstacles, margin=8.0):
    for obs in obstacles:
        if segment_intersects_box(xa, ya, xb, yb, obs, margin):
            return obs
    return None


def find_bypass_waypoint(xa, ya, xb, yb, obstacle, margin=8.0) -> Optional[tuple[float, float]]:
    x1, y1, x2, y2 = obstacle
    
    # Use a smaller wall margin to avoid pushing waypoints inside the obstacle when close to boundaries
    wall_margin = 8.0
    min_x, max_x = wall_margin, FIELD_WIDTH_CM - wall_margin
    min_y, max_y = wall_margin, FIELD_HEIGHT_CM - wall_margin
    
    x1_pad = x1 - margin
    y1_pad = y1 - margin
    x2_pad = x2 + margin
    y2_pad = y2 + margin
    
    corners = [
        (x1_pad, y1_pad),
        (x2_pad, y1_pad),
        (x1_pad, y2_pad),
        (x2_pad, y2_pad),
    ]
    
    candidates = []
    for cx, cy in corners:
        # Clip candidate to valid field boundaries
        cx_clipped = max(min_x, min(cx, max_x))
        cy_clipped = max(min_y, min(cy, max_y))
        
        # Check if the clipped corner lies inside the obstacle (with a small 2.0 cm safety buffer)
        x1_obs = x1 - 2.0
        y1_obs = y1 - 2.0
        x2_obs = x2 + 2.0
        y2_obs = y2 + 2.0
        if x1_obs <= cx_clipped <= x2_obs and y1_obs <= cy_clipped <= y2_obs:
            continue
            
        # Check segment intersection from start (robot) to waypoint
        start_to_wp_clear = not segment_intersects_box(xa, ya, cx_clipped, cy_clipped, obstacle, margin=3.0)
        
        # Check segment intersection from waypoint to end (target)
        wp_to_end_clear = not segment_intersects_box(cx_clipped, cy_clipped, xb, yb, obstacle, margin=3.0)
        
        dist = math.hypot(cx_clipped - xa, cy_clipped - ya) + math.hypot(xb - cx_clipped, yb - cy_clipped)
        
        # Priority mapping:
        # Priority 1: Both segments are clear
        # Priority 2: Only start-to-wp segment is clear (we can drive to it safely, then re-plan)
        # Priority 3: start-to-wp crosses the obstacle (unsafe)
        if start_to_wp_clear and wp_to_end_clear:
            priority = 1
        elif start_to_wp_clear:
            priority = 2
        else:
            priority = 3
            
        candidates.append((priority, dist, (cx_clipped, cy_clipped)))
        
    if candidates:
        # Sort by priority first (lower is better), then by distance (lower is better)
        candidates.sort(key=lambda x: (x[0], x[1]))
        if candidates[0][0] <= 2:
            return candidates[0][2]
            
    return None
