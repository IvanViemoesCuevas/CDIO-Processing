import math
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from config import (
    Settings,
    RED_RANGE_1,
    RED_RANGE_2,
    PERSPECTIVE_PADDING_PX,
    FIELD_WIDTH_CM,
    FIELD_HEIGHT_CM,
)
from models import (
    DangerFlags,
    DangerState,
    RobotPose,
    BallDetection,
)


def build_danger_mask(frame: np.ndarray) -> np.ndarray:
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red1 = cv.inRange(hsv_frame, RED_RANGE_1.lower, RED_RANGE_1.upper)
    red2 = cv.inRange(hsv_frame, RED_RANGE_2.lower, RED_RANGE_2.upper)
    mask = cv.bitwise_or(red1, red2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask


def detect_danger_zones(
    frame: np.ndarray,
    settings: Settings,
    robot_pose: Optional[RobotPose],
) -> tuple[DangerFlags, DangerState, np.ndarray, list[np.ndarray]]:
    h, w = frame.shape[:2]
    raw_mask = build_danger_mask(frame)

    flags = DangerFlags()
    filtered_mask = np.zeros_like(raw_mask)
    zone_w = max(1, w // 3)

    # Keep only sufficiently large connected red regions while preserving holes.
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(raw_mask, connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area < settings.min_obstacle_area:
            continue
        filtered_mask[labels == label] = 255

    kept_contours_raw, _ = cv.findContours(filtered_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    kept_contours: list[np.ndarray] = list(kept_contours_raw)

    ys, xs = np.where(filtered_mask > 0)
    state = DangerState()
    if xs.size == 0:
        return flags, state, filtered_mask, kept_contours

    if robot_pose is not None:
        width = w - 2 * PERSPECTIVE_PADDING_PX
        height = h - 2 * PERSPECTIVE_PADDING_PX
        scale_x = width / FIELD_WIDTH_CM
        scale_y = height / FIELD_HEIGHT_CM

        dx_img: NDArray[np.float32] = np.asarray(xs, dtype=np.float32) - np.float32(robot_pose.x)
        dy_img: NDArray[np.float32] = np.asarray(ys, dtype=np.float32) - np.float32(robot_pose.y)

        dx_cm = dx_img / scale_x
        dy_cm = dy_img / scale_y

        heading_x = math.cos(robot_pose.heading_rad)
        heading_y = math.sin(robot_pose.heading_rad)
        right_x = -heading_y
        right_y = heading_x

        forward_body_cm: NDArray[np.float32] = np.asarray(
            dx_cm * np.float32(heading_x) + dy_cm * np.float32(heading_y),
            dtype=np.float32,
        )
        right_body_cm: NDArray[np.float32] = np.asarray(
            dx_cm * np.float32(right_x) + dy_cm * np.float32(right_y),
            dtype=np.float32,
        )

        half_length_cm = float(settings.robot_length_cm) * 0.5
        half_width_cm = float(settings.robot_width_cm) * 0.5

        rear_limit_cm = -half_length_cm - float(settings.robot_danger_margin_cm)
        front_limit_cm = (
            half_length_cm
            + float(settings.robot_danger_margin_cm)
            + float(settings.robot_front_extra_margin_cm)
        )
        side_limit_cm = (
            half_width_cm
            + float(settings.robot_danger_margin_cm)
            + float(settings.robot_side_extra_margin_cm)
        )

        # Distance to the expanded rectangular robot footprint.
        # A value of 0 means the obstacle overlaps the safety rectangle.
        outside_forward_cm = np.maximum(
            np.maximum(rear_limit_cm - forward_body_cm, forward_body_cm - front_limit_cm),
            0.0,
        )
        outside_side_cm = np.maximum(np.abs(right_body_cm) - side_limit_cm, 0.0)
        rect_dist2_cm: NDArray[np.float32] = np.asarray(
            outside_forward_cm * outside_forward_cm + outside_side_cm * outside_side_cm,
            dtype=np.float32,
        )

        nearest_index = int(np.argmin(rect_dist2_cm))
        state.nearest_distance_cm = float(np.sqrt(rect_dist2_cm[nearest_index]))
        state.nearest_point = (int(xs[nearest_index]), int(ys[nearest_index]))
        state.nearest_dx_body = float(right_body_cm[nearest_index])
        state.nearest_dy_body = float(forward_body_cm[nearest_index])

        # Anything inside or close to the expanded rectangle is considered relevant danger.
        near_rect = rect_dist2_cm <= float(settings.danger_distance_cm * settings.danger_distance_cm)
        if np.any(near_rect):
            forward_near = forward_body_cm[near_rect]
            right_near = right_body_cm[near_rect]

            front_band = half_length_cm * 0.20
            rear_band = half_length_cm * 0.20
            side_deadband = max(2.0, half_width_cm * 0.35)

            flags.front = bool(np.any(forward_near >= -front_band))
            flags.back = bool(np.any(forward_near <= rear_band))
            flags.center = bool(np.any(np.abs(right_near) <= side_deadband))
            flags.left = bool(np.any(right_near < -side_deadband))
            flags.right = bool(np.any(right_near > side_deadband))

        # too_close now means the obstacle overlaps, or nearly overlaps, the robot's
        # rectangular safety footprint, not just a circular radius around the marker.
        state.too_close = (
            state.nearest_distance_cm <= float(settings.danger_too_close_cm)
            and state.nearest_dy_body >= rear_limit_cm
        )
    else:
        zone_h = max(1, h // 3)
        flags.front = bool(np.any(ys < zone_h))
        flags.back = bool(np.any(ys >= zone_h * 2))
        flags.left = bool(np.any(xs < zone_w))
        flags.center = bool(np.any((xs >= zone_w) & (xs < zone_w * 2)))
        flags.right = bool(np.any(xs >= zone_w * 2))

    return flags, state, filtered_mask, kept_contours


def is_ball_in_danger_zone(ball: BallDetection, danger_contours: list[np.ndarray], scale_x: float, scale_y: float = None) -> bool:
    if not danger_contours:
        return False
    
    # Use average scale for non-square pixels
    scale = scale_x
    if scale_y is not None:
        scale = (scale_x + scale_y) / 2.0
        
    # Estimate frame dimensions from scales and field constants
    frame_width = scale_x * FIELD_WIDTH_CM + 2 * PERSPECTIVE_PADDING_PX
    scale_y_val = scale_y if scale_y is not None else scale_x
    frame_height = scale_y_val * FIELD_HEIGHT_CM + 2 * PERSPECTIVE_PADDING_PX
    
    min_dist_cm = float("inf")
    for contour in danger_contours:
        # Check if the contour is a large boundary (field wall)
        x_px, y_px, w_px, h_px = cv.boundingRect(contour)
        is_boundary = (w_px >= 0.8 * frame_width) or (h_px >= 0.8 * frame_height)
        
        # positive inside, negative outside, 0 on edge
        dist_px = cv.pointPolygonTest(contour, (float(ball.x), float(ball.y)), measureDist=True)
        
        if is_boundary:
            # For a boundary contour, being inside the contour is normal (playable field).
            # The actual distance to the boundary line is what matters.
            raw_dist_cm = abs(dist_px) / scale
            ball_radius_cm = ball.radius / scale
            dist_cm = max(0.0, raw_dist_cm - ball_radius_cm)
        else:
            # For middle solid obstacles, being inside means the ball is in/under the obstacle.
            if dist_px >= 0:
                dist_cm = 0.0
            else:
                raw_dist_cm = abs(dist_px) / scale
                ball_radius_cm = ball.radius / scale
                dist_cm = max(0.0, raw_dist_cm - ball_radius_cm)
            
        if dist_cm < min_dist_cm:
            min_dist_cm = dist_cm
            
    return min_dist_cm <= 7.0
