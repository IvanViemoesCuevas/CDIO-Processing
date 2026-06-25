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


def expand_box(box, margin: float):
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def point_in_box(x, y, box, margin: float = 0.0):
    x1, y1, x2, y2 = expand_box(box, margin)
    return x1 <= x <= x2 and y1 <= y <= y2


def segment_intersects_box(xa, ya, xb, yb, box, margin: float | None = None):
    if margin is None:
        margin = Settings.obstacle_avoidance_margin_cm

    x1, y1, x2, y2 = expand_box(box, margin)

    if point_in_box(xa, ya, box, margin) or point_in_box(xb, yb, box, margin):
        return True

    p1 = (xa, ya)
    p2 = (xb, yb)

    edges = [
        ((x1, y1), (x1, y2)),
        ((x2, y1), (x2, y2)),
        ((x1, y1), (x2, y1)),
        ((x1, y2), (x2, y2)),
    ]

    return any(segments_intersect(p1, p2, a, b) for a, b in edges)


def check_route_segment_for_obstacles(xa, ya, xb, yb, obstacles):
    for obs in obstacles:
        if segment_intersects_box(xa, ya, xb, yb, obs):
            return obs
    return None


def clamp_field(x, y, margin: float = 5.0):
    return (
        max(margin, min(x, FIELD_WIDTH_CM - margin)),
        max(margin, min(y, FIELD_HEIGHT_CM - margin)),
    )


def waypoint_from_ball_opposite_cross(ball_x, ball_y, obstacle, distance_cm: float = 20.0):
    x1, y1, x2, y2 = obstacle
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dx = ball_x - cx
    dy = ball_y - cy
    length = math.hypot(dx, dy)

    if length < 1e-6:
        dx, dy = 1.0, 0.0
        length = 1.0

    ux = dx / length
    uy = dy / length

    return clamp_field(
        ball_x + ux * distance_cm,
        ball_y + uy * distance_cm,
    )


def midpoint_perpendicular_waypoint(xa, ya, xb, yb, obstacle, clearance_cm: float = 30.0):
    x1, y1, x2, y2 = obstacle
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    mx = (xa + xb) / 2.0
    my = (ya + yb) / 2.0

    vx = xb - xa
    vy = yb - ya
    length = math.hypot(vx, vy)

    if length < 1e-6:
        vx, vy = 1.0, 0.0
        length = 1.0

    # 90 degrees from route vector
    px = -vy / length
    py = vx / length

    # Pick side pointing away from cross center
    away_dot = (mx - cx) * px + (my - cy) * py
    sign = 1.0 if away_dot >= 0 else -1.0

    # Move along perpendicular until 30 cm outside the RAW cross box
    raw_plus_clearance = expand_box(obstacle, clearance_cm)

    for step in range(0, 200, 2):
        wx = mx + sign * px * step
        wy = my + sign * py * step
        wx, wy = clamp_field(wx, wy)

        if not point_in_box(wx, wy, raw_plus_clearance, 0.0):
            return wx, wy

    # deterministic fallback, still never None
    return clamp_field(mx + sign * px * clearance_cm, my + sign * py * clearance_cm)


def plan_obstacle_safe_waypoints(xa, ya, xb, yb, obstacles, max_depth: int = 8):
    """
    Returns waypoint cm coordinates needed before driving to xb,yb.
    Never returns None.
    """
    obs = check_route_segment_for_obstacles(xa, ya, xb, yb, obstacles)
    if obs is None:
        return []

    if max_depth <= 0:
        wp = midpoint_perpendicular_waypoint(xa, ya, xb, yb, obs)
        return [wp]

    wp_x, wp_y = midpoint_perpendicular_waypoint(xa, ya, xb, yb, obs)

    before = plan_obstacle_safe_waypoints(xa, ya, wp_x, wp_y, obstacles, max_depth - 1)
    after = plan_obstacle_safe_waypoints(wp_x, wp_y, xb, yb, obstacles, max_depth - 1)

    return before + [(wp_x, wp_y)] + after


def obstacle_containing_point(x, y, obstacles):
    margin = Settings.obstacle_avoidance_margin_cm
    for obs in obstacles:
        if point_in_box(x, y, obs, margin):
            return obs
    return None


def find_bypass_waypoint(xa, ya, xb, yb, obstacle) -> Optional[tuple[float, float]]:
    x1, y1, x2, y2 = obstacle

    margin = Settings.obstacle_avoidance_margin_cm
    min_from_current = 30.0
    min_from_target = 25.0
    clearance = margin + 10.0

    # Use expanded obstacle box.
    ex1 = x1 - margin
    ey1 = y1 - margin
    ex2 = x2 + margin
    ey2 = y2 + margin

    candidates = [
        ((ex1 + ex2) / 2.0, ey1 - clearance),  # above
        ((ex1 + ex2) / 2.0, ey2 + clearance),  # below
        (ex1 - clearance, (ey1 + ey2) / 2.0),  # left
        (ex2 + clearance, (ey1 + ey2) / 2.0),  # right

        (ex1 - clearance, ey1 - clearance),    # top-left
        (ex2 + clearance, ey1 - clearance),    # top-right
        (ex1 - clearance, ey2 + clearance),    # bottom-left
        (ex2 + clearance, ey2 + clearance),    # bottom-right
    ]

    valid = []

    for wx, wy in candidates:
        wx = max(margin, min(wx, FIELD_WIDTH_CM - margin))
        wy = max(margin, min(wy, FIELD_HEIGHT_CM - margin))

        dist_from_current = math.hypot(wx - xa, wy - ya)
        dist_from_target = math.hypot(wx - xb, wy - yb)

        if dist_from_current < min_from_current:
            continue

        if dist_from_target < min_from_target:
            continue

        # Waypoint itself must not be inside expanded obstacle box.
        if ex1 <= wx <= ex2 and ey1 <= wy <= ey2:
            continue

        first_leg_blocked = segment_intersects_box(xa, ya, wx, wy, obstacle)
        second_leg_blocked = segment_intersects_box(wx, wy, xb, yb, obstacle)

        # Best case: both legs are clear.
        # Acceptable case: first leg clear, second leg may need another waypoint later.
        if first_leg_blocked:
            continue

        penalty = 0
        if second_leg_blocked:
            penalty += 10000

        total_dist = dist_from_current + dist_from_target
        valid.append((penalty + total_dist, wx, wy))

    if not valid:
        print(
            "[geometry] No valid bypass waypoint found "
            f"from=({xa:.1f},{ya:.1f}) "
            f"to=({xb:.1f},{yb:.1f}) "
            f"obs=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )
        return None

    valid.sort(key=lambda item: item[0])
    return valid[0][1], valid[0][2]
