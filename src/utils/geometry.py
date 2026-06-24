import math
import numpy as np
import cv2 as cv
from typing import Optional

from config import PERSPECTIVE_PADDING_PX, FIELD_WIDTH_CM, FIELD_HEIGHT_CM, Settings
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


def segment_intersects_box(xa, ya, xb, yb, box):
    x1, y1, x2, y2 = box
    x1_pad = x1 - Settings.obstacle_avoidance_margin_cm
    y1_pad = y1 - Settings.obstacle_avoidance_margin_cm
    x2_pad = x2 + Settings.obstacle_avoidance_margin_cm
    y2_pad = y2 + Settings.obstacle_avoidance_margin_cm
    
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


def check_route_segment_for_obstacles(xa, ya, xb, yb, obstacles):
    for obs in obstacles:
        if segment_intersects_box(xa, ya, xb, yb, obs):
            return obs
    return None


def find_bypass_waypoint(xa, ya, xb, yb, obstacle) -> Optional[tuple[float, float]]:
    # Obstacle bounding box: (left, top, right, bottom)
    x1, y1, x2, y2 = obstacle

    # Stop short of the target so the robot approaches the ball indirectly
    ball_offset_cm = 25.0

    # Obstacle center point
    obs_cx = (x1 + x2) / 2.0
    obs_cy = (y1 + y2) / 2.0

    # Direction vector from current position to target
    route_dx = xb - xa
    route_dy = yb - ya
    route_len = math.hypot(route_dx, route_dy)

    # No meaningful route if start and target are the same
    if route_len < 1e-6:
        return None

    # Normalize route direction
    route_dx /= route_len
    route_dy /= route_len

    # Create a base point slightly before the target
    # (prevents driving directly onto the ball/target)
    base_x = xb - route_dx * ball_offset_cm
    base_y = yb - route_dy * ball_offset_cm

    # Determine which side of the route the obstacle lies on
    side = (base_x - obs_cx) * route_dy - (base_y - obs_cy) * route_dx

    # Two perpendicular directions to the route
    perp1_x = -route_dy
    perp1_y = route_dx

    perp2_x = route_dy
    perp2_y = -route_dx

    # Choose the perpendicular direction that moves away from the obstacle
    if side >= 0:
        perp_x, perp_y = perp2_x, perp2_y
    else:
        perp_x, perp_y = perp1_x, perp1_y

    # Start waypoint search from the base point
    wp_x = base_x
    wp_y = base_y

    def distance_from_box(px, py):
        """Shortest distance from a point to the obstacle rectangle."""
        closest_x = max(x1, min(px, x2))
        closest_y = max(y1, min(py, y2))
        return math.hypot(px - closest_x, py - closest_y)

    # Move the waypoint sideways until the required clearance is reached
    for _ in range(1000):
        if distance_from_box(wp_x, wp_y) >= Settings.obstacle_avoidance_margin_cm:
            break
        wp_x += perp_x
        wp_y += perp_y

    # Keep waypoint safely inside field boundaries
    wall_margin = Settings.obstacle_avoidance_margin_cm
    wp_x = max(wall_margin, min(wp_x, FIELD_WIDTH_CM - wall_margin))
    wp_y = max(wall_margin, min(wp_y, FIELD_HEIGHT_CM - wall_margin))

    # If the waypoint is still blocked by the obstacle, move it back towards the start
    for _ in range(1000):
        if not segment_intersects_box(xa, ya, wp_x, wp_y, obstacle):
            break
        wp_x = xa + 0.85 * (wp_x - xa)
        wp_y = ya + 0.85 * (wp_y - ya)

    return wp_x, wp_y
