from __future__ import annotations

# All vision detection code

import math

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - graceful fallback if ultralytics not available
    YOLO = None

from config import *
from models import *

# Path to your custom YOLO model. Replace this placeholder with your model path.
YOLO_MODEL_PATH = "custom_yolo.pt"

# Lazily-loaded YOLO model instance
_yolo_model = None

def _ensure_yolo_model():
    """Lazy-load the YOLO model. Returns None if ultralytics not installed."""
    global _yolo_model
    if YOLO is None:
        return None
    if _yolo_model is None:
        # Instantiate model (user should edit YOLO_MODEL_PATH to point at their weights)
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model

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


class BallDetectionTuner:
    """Debug-only live tuner for the HSV thresholds used by ball detection."""

    WINDOW_NAME = "Ball Detection Tuner"

    def __init__(
        self,
        orange_range: HSVRange = ORANGE_RANGE,
        white_range: HSVRange = WHITE_RANGE,
        white_sat_split: float = 80.0,
    ) -> None:
        cv.namedWindow(self.WINDOW_NAME, cv.WINDOW_NORMAL)

        self._add("OH_lo", orange_range.lower[0], 179)
        self._add("OS_lo", orange_range.lower[1], 255)
        self._add("OV_lo", orange_range.lower[2], 255)
        self._add("OH_hi", orange_range.upper[0], 179)
        self._add("OS_hi", orange_range.upper[1], 255)
        self._add("OV_hi", orange_range.upper[2], 255)

        self._add("WH_lo", white_range.lower[0], 179)
        self._add("WS_lo", white_range.lower[1], 255)
        self._add("WV_lo", white_range.lower[2], 255)
        self._add("WH_hi", white_range.upper[0], 179)
        self._add("WS_hi", white_range.upper[1], 255)
        self._add("WV_hi", white_range.upper[2], 255)

        self._add("white_sat_split", int(white_sat_split), 255)

    def _add(self, name: str, value: int, max_value: int) -> None:
        cv.createTrackbar(name, self.WINDOW_NAME, int(value), max_value, lambda _x: None)

    def _get(self, name: str) -> int:
        return int(cv.getTrackbarPos(name, self.WINDOW_NAME))

    def read(self) -> BallDetectionTuning:
        orange_lo = (self._get("OH_lo"), self._get("OS_lo"), self._get("OV_lo"))
        orange_hi = (self._get("OH_hi"), self._get("OS_hi"), self._get("OV_hi"))
        white_lo = (self._get("WH_lo"), self._get("WS_lo"), self._get("WV_lo"))
        white_hi = (self._get("WH_hi"), self._get("WS_hi"), self._get("WV_hi"))

        # Keep lower <= upper so inRange always receives a valid interval.
        orange_lower = (
            min(orange_lo[0], orange_hi[0]),
            min(orange_lo[1], orange_hi[1]),
            min(orange_lo[2], orange_hi[2]),
        )
        orange_upper = (
            max(orange_lo[0], orange_hi[0]),
            max(orange_lo[1], orange_hi[1]),
            max(orange_lo[2], orange_hi[2]),
        )
        white_lower = (
            min(white_lo[0], white_hi[0]),
            min(white_lo[1], white_hi[1]),
            min(white_lo[2], white_hi[2]),
        )
        white_upper = (
            max(white_lo[0], white_hi[0]),
            max(white_lo[1], white_hi[1]),
            max(white_lo[2], white_hi[2]),
        )

        return BallDetectionTuning(
            orange_range=HSVRange(orange_lower, orange_upper),
            white_range=HSVRange(white_lower, white_upper),
            white_sat_split=float(self._get("white_sat_split")),
        )


def make_ball_debug_view(
    frame: np.ndarray,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
    white_sat_split: float = 80.0,
) -> np.ndarray:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    orange_mask, white_mask, combined_mask = build_ball_masks(
        hsv,
        orange_range=orange_range,
        white_range=white_range,
    )

    orange_vis = cv.cvtColor(orange_mask, cv.COLOR_GRAY2BGR)
    white_vis = cv.cvtColor(white_mask, cv.COLOR_GRAY2BGR)
    combined_vis = cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)
    camera_vis = frame.copy()

    cv.putText(orange_vis, "orange mask", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv.putText(white_vis, "white mask", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(combined_vis, "combined", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(camera_vis, "camera", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    top = np.hstack((camera_vis, orange_vis))
    bottom = np.hstack((white_vis, combined_vis))
    debug_view = np.vstack((top, bottom))

    cv.putText(
        debug_view,
        (
            f"orange={orange_range.lower}-{orange_range.upper} "
            f"white={white_range.lower}-{white_range.upper} sat_split={int(white_sat_split)}"
        ),
        (10, debug_view.shape[0] - 38),
        cv.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        1,
    )
    cv.putText(
        debug_view,
        "q=quit | tune values with Ball Detection Tuner",
        (10, debug_view.shape[0] - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
    return debug_view


def detect_balls(
    frame: np.ndarray,
    settings: Settings,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
    white_sat_split: Optional[float] = None,
) -> list[BallDetection]:
    """Detect candidate ping-pong balls using a YOLO model and convert detections
    into the project's BallDetection dataclass so the navigation code can remain
    unchanged.

    The model path is a placeholder in `YOLO_MODEL_PATH` and must be updated by
    the user to point to their custom yolo26n weights file.
    If the ultralytics package is not available or the model fails to load,
    this function falls back to the original HSV+contour detector.
    """

    # Prefer YOLO if available
    model = _ensure_yolo_model()
    if model is None:
        # Fallback to original contour-based method if YOLO not installed
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = build_ball_mask(hsv_frame, orange_range=orange_range, white_range=white_range)
        sat_split = settings.white_sat_split if white_sat_split is None else white_sat_split

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        balls: list[BallDetection] = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area < settings.min_ball_area or area > settings.max_ball_area:
                continue
            perimeter = cv.arcLength(contour, True)
            if perimeter <= 0.0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < settings.min_ball_circularity:
                continue
            (x_float, y_float), radius = cv.minEnclosingCircle(contour)
            if radius < settings.min_ball_radius or radius > settings.max_ball_radius:
                continue
            x = int(x_float)
            y = int(y_float)
            sample = hsv_frame[max(0, y-1): y+2, max(0, x-1): x+2]
            if sample.size == 0:
                continue
            sat_mean = float(sample[:, :, 1].mean())
            color_name = "white" if sat_mean < sat_split else "orange"
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

    # Run YOLO inference on the frame. The ultralytics model accepts BGR frames.
    try:
        results = model(frame, verbose=False)
    except Exception as e:
        # If inference fails for any reason, return empty list rather than crash.
        print(f"YOLO inference failed: {e}")
        return []

    # Results may be batched; take the first (and only) result for this single frame.
    result = results[0]
    boxes = getattr(result, "boxes", [])
    names = getattr(result, "names", {})

    balls: list[BallDetection] = []
    for box in boxes:
        # Extract coordinates
        try:
            xyxy = box.xyxy[0]
        except Exception:
            # Some ultralytics versions expose xyxy as a plain array
            xyxy = box.xyxy
        try:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        except Exception:
            # If we can't parse box coordinates, skip this detection
            continue

        # Confidence and class
        try:
            conf = float(box.conf[0])
        except Exception:
            try:
                conf = float(box.conf)
            except Exception:
                conf = 0.0

        try:
            cls = int(box.cls[0])
        except Exception:
            try:
                cls = int(box.cls)
            except Exception:
                cls = -1

        class_name = str(names.get(cls, "unknown")) if names is not None else "unknown"

        # Filter by configured confidence threshold
        if conf < settings.min_ball_confidence:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max((x2 - x1), (y2 - y1)) / 2.0

        balls.append(
            BallDetection(
                x=int(cx),
                y=int(cy),
                radius=float(radius),
                color_name=class_name,
                confidence=float(conf),
                circularity=0.0,
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
    target_index = int(match[0])

    # FIXME maybe use the actual corner locations instead of just creating a square (Could also calculate the confidence)

    pts = corners[target_index][0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    top_mid = 0.5 * (pts[2] + pts[1])
    bottom_mid = 0.5 * (pts[0] + pts[3])
    forward_vec = top_mid - bottom_mid
    heading = math.atan2(float(forward_vec[1]), float(forward_vec[0]))

    return RobotPose(x=cx, y=cy, heading_rad=heading, confidence=-1)

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
        dx_img: NDArray[np.float32] = np.asarray(xs, dtype=np.float32) - np.float32(robot_pose.x)
        dy_img: NDArray[np.float32] = np.asarray(ys, dtype=np.float32) - np.float32(robot_pose.y)
        dist2: NDArray[np.float32] = dx_img * dx_img + dy_img * dy_img
        nearest_index = int(np.argmin(dist2))

        state.nearest_distance_px = float(np.sqrt(dist2[nearest_index]))
        state.nearest_point = (int(xs[nearest_index]), int(ys[nearest_index]))

        heading_x = math.cos(robot_pose.heading_rad)
        heading_y = math.sin(robot_pose.heading_rad)
        right_x = -heading_y
        right_y = heading_x

        forward_body: NDArray[np.float32] = np.asarray(
            dx_img * np.float32(heading_x) + dy_img * np.float32(heading_y),
            dtype=np.float32,
        )
        right_body: NDArray[np.float32] = np.asarray(
            dx_img * np.float32(right_x) + dy_img * np.float32(right_y),
            dtype=np.float32,
        )

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