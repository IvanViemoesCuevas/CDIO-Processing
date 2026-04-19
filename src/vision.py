# All vision detection code

import math

import cv2
import numpy as np

from config import *
from models import *

# TODO - check function
def build_ball_masks(
    hsv_frame: np.ndarray,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build color masks for orange/white balls and a merged mask used for contour search."""
    orange_mask = cv.inRange(hsv_frame, orange_range.lower, orange_range.upper)
    white_mask = cv.inRange(hsv_frame, white_range.lower, white_range.upper)

    # Keep morphology light so small/far balls are not removed.
    kernel = np.ones((3, 3), np.uint8)
    orange_mask = cv.morphologyEx(orange_mask, cv.MORPH_OPEN, kernel)
    orange_mask = cv.morphologyEx(orange_mask, cv.MORPH_CLOSE, kernel)
    white_mask = cv.morphologyEx(white_mask, cv.MORPH_OPEN, kernel)
    white_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)

    mask = cv.bitwise_or(orange_mask, white_mask)
    return orange_mask, white_mask, mask

# TODO - check function
def build_ball_mask(
    hsv_frame: np.ndarray,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
) -> np.ndarray:
    """Convenience wrapper that returns only the merged orange+white mask."""
    orange, white, mask = build_ball_masks(hsv_frame, orange_range=orange_range, white_range=white_range)
    #cv.imshow("orange-mask", orange)
    #cv.imshow("white-mask", white)
    #cv.imshow("combined-mask", orange)
    return mask


def detect_balls(
    frame: np.ndarray,
    settings: Settings,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
    white_sat_split: Optional[float] = None,
) -> list[BallDetection]:
    """Detect candidate ping-pong balls and score them by shape/size confidence."""
    # Work in HSV because thresholding ball colors is more stable than in BGR.
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = build_ball_mask(hsv_frame, orange_range=orange_range, white_range=white_range)
    
    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    balls: list[BallDetection] = []
    
    for contour in contours:
        area = cv.contourArea(contour)
        # Reject tiny noise and large non-ball blobs.
        if area < settings.min_ball_area or area > settings.max_ball_area:
            continue
        
        perimeter = cv.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        # Ideal circle is 1.0; lower values are elongated/irregular shapes.
        if circularity < settings.min_ball_circularity:
            continue

        (x_float, y_float), radius = cv.minEnclosingCircle(contour)
        # Radius gate helps remove shapes that pass circularity but are wrong scale.
        if radius < settings.min_ball_radius or radius > settings.max_ball_radius:
            continue
        
        x = int(x_float)
        y = int(y_float)
        
        sample = hsv_frame[max(0, y-1): y+2, max(0, x-1): x+2]
        if sample.size == 0:
            continue
        # Low saturation at center tends to be white, higher saturation tends to orange.
        sat_mean = float(sample[:, :, 1].mean())
        color_name = "white" if sat_mean < white_sat_split else "orange"
        
        # Blend roundness and contour area into a simple confidence score [0, 1].
        confidence = min(1.0, circularity * 0.65 + min(1.0, area / 2000.0) * 0.35)
        if confidence < settings.min_ball_confidence:
            continue
            
        balls.append(
            BallDetection(
                x=x,
                y=y,
                radius=radius,
                color_name=color_name,
                confidence=confidence,
                circularity=circularity,
            )
        )
    
    return balls


def choose_target_ball(balls: list[BallDetection], robot_pose: Optional[RobotPose]) -> Optional[BallDetection]:
    # If no balls detected, return none
    if not balls:
        return None

    # if no robot is detected, take the ball that we are most sure about
    if robot_pose is None:
        best = balls[0]
        for current in balls[1:]:
            if current.confidence * current.radius > best.confidence * best.radius:
                best = current
            return best

    # Else return ball closest to robot
    return min(balls, key=lambda b: (b.x - robot_pose.x) ** 2 + (b.y - robot_pose.y) ** 2)


def match_candidate_target(
    candidate_target: Optional[BallDetection],
    balls: list[BallDetection],
    max_match_distance_px: float = 90.0,
) -> Optional[BallDetection]:
    # If no candidate/balls return none
    if candidate_target is None or not balls:
        return None

    # Find the ball that is closest to the candidate
    best = min(
        balls,
        key=lambda b: (b.x - candidate_target.x) ** 2 + (b.y - candidate_target.y) ** 2,
    )

    # Calculate distance from best to candidate and check if it's within the max match distance
    dist = math.hypot(float(best.x - candidate_target.x), float(best.y - candidate_target.y))
    if dist < max_match_distance_px:
        return best

    return None


def detect_robot_pose(frame: np.ndarray, settings: Settings) -> Optional[RobotPose]:
    if not hasattr(cv, "aruco"):
        return None
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None

    target_index = 7
    match = np.where(ids.flatten() == target_index)[0]
    if len(match) == 0:
        return None
    target_index = int[match[0]]

    # FIXME maybe use the actual corner locations instead of just creating a square (Could also calculate the confidence)

    pts = corners[target_index][0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    top_mid = 0.5 * (pts[0] + pts[3])
    bottom_mid = 0.5 * (pts[2] + pts[1])
    forward_vec = top_mid - bottom_mid
    heading = math.atan2(float(forward_vec[1]), float(forward_vec[0]))

    return RobotPose(x=cx, y=cy, heading_rad=heading, confidence=-1)

def build_danger_mask(frame: np.ndarray) -> np.ndarray:
    hsv_frame = cv.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

    kept_contours, _ = cv.findContours(filtered_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ys, xs = np.where(filtered_mask > 0)
    state = DangerState()
    if xs.size == 0:
        return flags, state, filtered_mask, kept_contours

    if robot_pose is not None:
        dx_img = xs.astype(np.float32) - float(robot_pose.x)
        dy_img = ys.astype(np.float32) - float(robot_pose.y)
        dist2 = dx_img * dx_img + dy_img * dy_img
        nearest_index = int(np.argmin(dist2))

        state.nearest_distance_px = float(np.sqrt(dist2[nearest_index]))
        state.nearest_point = (int(xs[nearest_index]), int(ys[nearest_index]))

        heading_x = math.cos(robot_pose.heading_rad)
        heading_y = math.sin(robot_pose.heading_rad)
        right_x = -heading_y
        right_y = heading_x

        forward_body = dx_img * heading_x + dy_img * heading_y
        right_body = dx_img * right_x + dy_img * right_y

        near = dist2 <= float(settings.danger_distance_px * settings.danger_distance_px)
        if np.any(near):
            forward_near = forward_body[near]
            right_near = right_body[near]
            flags.front = bool(np.any(forward_near > float(settings.danger_center_deadband_px)))
            flags.back = bool(np.any(forward_near < -float(settings.danger_center_deadband_px)))
            flags.center = bool(np.any(np.abs(right_near) <= float(settings.danger_center_deadband_px)))
            flags.left = bool(np.any(right_near < -float(settings.danger_center_deadband_px)))
            flags.right = bool(np.any(right_near > float(settings.danger_center_deadband_px)))

        state.nearest_dx_body = float(right_body[nearest_index])
        state.nearest_dy_body = float(forward_body[nearest_index])
        state.too_close = (
                state.nearest_distance_px <= float(settings.danger_too_close_px)
                and state.nearest_dy_body >= -float(settings.danger_rear_ignore_px)
        )
    else:
        zone_h = max(1, h // 3)
        flags.front = bool(np.any(ys < zone_h))
        flags.back = bool(np.any(ys >= zone_h * 2))
        flags.left = bool(np.any(xs < zone_w))
        flags.center = bool(np.any((xs >= zone_w) & (xs < zone_w * 2)))
        flags.right = bool(np.any(xs >= zone_w * 2))

    return flags, state, filtered_mask, kept_contours