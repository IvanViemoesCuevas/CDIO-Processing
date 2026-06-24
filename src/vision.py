from __future__ import annotations

# All vision detection code

import math

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
import os
from typing import Optional
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - graceful fallback if ultralytics not available
    YOLO = None

from config import *
from models import *

# Path to your custom YOLO model. Replace this placeholder with your model path.
# Set to None to disable YOLO and always use the contour fallback.
YOLO_MODEL_PATH = "my_model.pt"

# Lazily-loaded YOLO model instance
_yolo_model = None

#PERSPECTIVE_SRC_POINTS = np.float32([
#    [350, 130],   # top-left
#    [1540, 150],  # top-right
#    [1550, 1050],  # bottom-right
#    [300, 1030],   # bottom-left
#])

PERSPECTIVE_OUTPUT_SIZE: tuple[int, int] | None = None

# Increase this if the warped image cuts off the border/corners.
# (Imported from config.py)

PERSPECTIVE_UPDATE_INTERVAL = 5
PERSPECTIVE_SMOOTHING = 0.9

_perspective_frame_counter = 0
_smoothed_perspective_points: np.ndarray | None = None

def correct_perspective(frame: np.ndarray) -> np.ndarray:

    #detected_src = find_danger_perspective_points(frame)
    global _perspective_frame_counter, _smoothed_perspective_points

    should_update = (
        _smoothed_perspective_points is None
        or _perspective_frame_counter >= PERSPECTIVE_UPDATE_INTERVAL
    )

    if should_update:
        detected_src = find_danger_perspective_points(frame)

        if detected_src is not None:
            detected_src = np.asarray(detected_src, dtype=np.float32)

            if _smoothed_perspective_points is None:
                _smoothed_perspective_points = detected_src
            else:
                alpha = float(PERSPECTIVE_SMOOTHING)
                _smoothed_perspective_points = (
                    alpha * _smoothed_perspective_points
                    + (1.0 - alpha) * detected_src
                )

        _perspective_frame_counter = 0
    else:
        _perspective_frame_counter += 1

    if _smoothed_perspective_points is None:
        return frame

    src = np.asarray(_smoothed_perspective_points, dtype=np.float32)

    if PERSPECTIVE_OUTPUT_SIZE is None:
        top_width = np.linalg.norm(src[1] - src[0])
        bottom_width = np.linalg.norm(src[2] - src[3])
        left_height = np.linalg.norm(src[3] - src[0])
        right_height = np.linalg.norm(src[2] - src[1])
        width = int(max(top_width, bottom_width))
        height = int(max(left_height, right_height))
    else:
        width, height = PERSPECTIVE_OUTPUT_SIZE

    padding = int(PERSPECTIVE_PADDING_PX)
    output_width = width + padding * 2
    output_height = height + padding * 2

    dst = np.float32([
        [padding, padding],
        [padding + width - 1, padding],
        [padding + width - 1, padding + height - 1],
        [padding, padding + height - 1],
    ])

    matrix = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(frame, matrix, (output_width, output_height))

def _ensure_yolo_model():
    """Lazy-load the YOLO model. Returns None if ultralytics not installed."""
    global _yolo_model
    if YOLO is None:
        return None
    # If user explicitly disabled or didn't set a path, fall back
    if not YOLO_MODEL_PATH:
        print("YOLO model path not set; using contour-based fallback.")
        return None

    # If the file doesn't exist, fall back rather than raising during import
    if not os.path.isfile(YOLO_MODEL_PATH):
        print(f"YOLO model file not found at '{YOLO_MODEL_PATH}'; using contour-based fallback.")
        return None

    if _yolo_model is None:
        try:
            # Instantiate model (user should edit YOLO_MODEL_PATH to point at their weights)
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        except Exception as e:
            # Catch errors such as corrupted/unsupported checkpoint files and fall back
            print(f"Failed to load YOLO model from '{YOLO_MODEL_PATH}': {e}")
            _yolo_model = None
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


class BallHandoffManager:
    """
    Manages the state for ball handoff by tracking when the field is clear of balls.
    Ready for handoff when field is empty (0 balls detected for N consecutive frames).
    """
    def __init__(self, required_empty_frames: int = 1):
        """
        Initializes the manager.
        :param required_empty_frames: Consecutive frames without balls before field considered empty.
        """
        self.required_empty_frames = required_empty_frames
        self._empty_frame_counter = 0
        self._has_seen_ball = False

    def update(self, balls: list[BallDetection]) -> None:
        """
        Updates the manager's state based on current frame's ball detections.
        :param balls: A list of BallDetection objects from the current frame.
        """
        # Track field state: first require that at least one ball has been detected.
        # This prevents the robot from driving to the goal at startup when the field is empty.
        if balls:
            self._has_seen_ball = True
            self._empty_frame_counter = 0
        else:
            self._empty_frame_counter += 1

    def field_is_clear(self) -> bool:
        """
        :return: True if field is empty (sufficient consecutive empty frames detected).
        """
        return self._empty_frame_counter >= self.required_empty_frames

    @property
    def ready_for_handoff(self) -> bool:
        """
        Check if field is clear and ready for handoff.
        :return: True if field is empty for required frames, False otherwise.
        """
        return self._has_seen_ball and self.field_is_clear()

    @property
    def empty_frames_count(self) -> int:
        """
        :return: Current count of consecutive empty frames.
        """
        return self._empty_frame_counter

    def reset(self) -> None:
        """Resets the field-empty counter."""
        self._empty_frame_counter = 0
        self._has_seen_ball = False



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
    if model:
        try:
            results = model(frame, verbose=False)
            result = results[0]
            boxes = getattr(result, "boxes", [])
            names = getattr(result, "names", {})

            balls: list[BallDetection] = []
            for box in boxes:
                try:
                    xyxy = box.xyxy[0]
                except Exception:
                    xyxy = box.xyxy
                try:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                except Exception:
                    continue

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

                raw_name = str(names.get(cls, "unknown")) if names is not None else "unknown"
                raw_name_lower = raw_name.lower()
                if "orange" in raw_name_lower:
                    class_name = "orange"
                elif "white" in raw_name_lower:
                    class_name = "white"
                else:
                    continue

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
        except Exception as e:
            print(f"YOLO inference failed: {e}. Falling back to HSV.")
            # Fall through to HSV method if YOLO fails at runtime

    # Fallback to original contour-based method
    print("Using HSV color detection for balls.")
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    ballMask = build_ball_mask(hsv_frame, orange_range=orange_range, white_range=white_range)
    sat_split = settings.white_sat_split if white_sat_split is None else white_sat_split

    contours, _ = cv.findContours(ballMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
            
    return min_dist_cm <= 5.0

def find_danger_perspective_points(frame: np.ndarray) -> np.ndarray | None:
    mask = build_danger_mask(frame)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    lines = cv.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=max(80, min(frame.shape[:2]) // 4),
        maxLineGap=35,
    )

    if lines is None:
        return None

    horizontal = []
    vertical = []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 80:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle

        if angle < 15:
            horizontal.append((length, (x1, y1, x2, y2)))
        elif angle > 75:
            vertical.append((length, (x1, y1, x2, y2)))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    horizontal = sorted(horizontal, key=lambda l: (l[1][1] + l[1][3]) / 2)
    vertical = sorted(vertical, key=lambda l: (l[1][0] + l[1][2]) / 2)

    top = horizontal[0][1]
    bottom = horizontal[-1][1]
    left = vertical[0][1]
    right = vertical[-1][1]

    def fit_y_line(line):
        x1, y1, x2, y2 = line
        return np.polyfit([x1, x2], [y1, y2], 1)

    def fit_x_line(line):
        x1, y1, x2, y2 = line
        return np.polyfit([y1, y2], [x1, x2], 1)

    top_m, top_b = fit_y_line(top)
    bottom_m, bottom_b = fit_y_line(bottom)
    left_m, left_b = fit_x_line(left)
    right_m, right_b = fit_x_line(right)

    def intersect(h_line, v_line):
        mh, bh = h_line      # y = mh*x + bh
        mv, bv = v_line      # x = mv*y + bv
        x = (mv * bh + bv) / (1 - mv * mh)
        y = mh * x + bh
        return [x, y]

    points = np.float32([
        intersect((top_m, top_b), (left_m, left_b)),
        intersect((top_m, top_b), (right_m, right_b)),
        intersect((bottom_m, bottom_b), (right_m, right_b)),
        intersect((bottom_m, bottom_b), (left_m, left_b)),
    ])

    return order_perspective_points(points)

def order_perspective_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    return np.asarray(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def detect_field_corners(frame: np.ndarray, settings: Settings) -> Optional[FieldCorners]:
    """
    Detects the four corners of the field by finding the centerlines of the red tape
    and calculating their intersections.
    """
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red1 = cv.inRange(hsv_frame, RED_RANGE_1.lower, RED_RANGE_1.upper)
    red2 = cv.inRange(hsv_frame, RED_RANGE_2.lower, RED_RANGE_2.upper)
    mask = cv.bitwise_or(red1, red2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Find the largest contour which represents the red tape
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv.contourArea)

    # Create an empty image to draw the contour skeleton
    skeleton = np.zeros_like(mask)
    cv.drawContours(skeleton, [largest_contour], -1, 255, 1)

    # We will use probabilistic Hough lines to find the 4 main edges from the outline
    # This directly finds lines representing the center/edge of the tape
    lines = cv.HoughLinesP(skeleton, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=20)

    if lines is None:
        return None

    # Categorize lines into roughly vertical and horizontal
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        if -45 < angle < 45 or 135 < angle <= 180 or -180 <= angle < -135:
            horizontal_lines.append(line[0])
        else:
            vertical_lines.append(line[0])

    if not vertical_lines or not horizontal_lines:
        return None

    # Find the extreme lines (leftmost, rightmost, topmost, bottommost)
    # We assume the field is roughly centered

    # Left vertical
    left_line = min(vertical_lines, key=lambda l: (l[0] + l[2])/2)
    # Right vertical
    right_line = max(vertical_lines, key=lambda l: (l[0] + l[2])/2)
    # Top horizontal
    top_line = min(horizontal_lines, key=lambda l: (l[1] + l[3])/2)
    # Bottom horizontal
    bottom_line = max(horizontal_lines, key=lambda l: (l[1] + l[3])/2)

    def get_intersection(line1, line2):
        """Calculates intersection of two line segments given as (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return int(px), int(py)

    # Calculate intersections
    top_left = get_intersection(left_line, top_line)
    top_right = get_intersection(right_line, top_line)
    bottom_left = get_intersection(left_line, bottom_line)
    bottom_right = get_intersection(right_line, bottom_line)

    if None in (top_left, top_right, bottom_left, bottom_right):
        return None

    return FieldCorners(
        topLeft=top_left,
        topRight=top_right,
        bottomLeft=bottom_left,
        bottomRight=bottom_right,
    )

def find_small_goal(frame: np.ndarray, field_corners: Optional[FieldCorners], settings: Settings) -> Optional[GoalDetection]:
    """
    Scans the right vertical edge of the detected field to find the small goal.
    """
    if field_corners is None:
        return None

    try:
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        red1 = cv.inRange(hsv_frame, RED_RANGE_1.lower, RED_RANGE_1.upper)
        red2 = cv.inRange(hsv_frame, RED_RANGE_2.lower, RED_RANGE_2.upper)
        redMaskForBoard = cv.bitwise_or(red1, red2)

        tr = field_corners.topRight
        br = field_corners.bottomRight

        # --- NYT: Begræns søgeområdet til midten af banen ---
        edge_height = br[1] - tr[1]
        if edge_height <= 0: return None # Avoid division by zero or negative height

        y_search_start = tr[1] + int(edge_height * 0.30) # Start 30% nede
        y_search_end = tr[1] + int(edge_height * 0.70)   # Slut 70% nede
        # --- SLUT PÅ NYT ---

        search_width_px = 40
        x_center = int((tr[0] + br[0]) / 2)
        x_start = max(0, x_center - search_width_px // 2)
        x_end = min(redMaskForBoard.shape[1], x_center + search_width_px // 2)

        if not (y_search_start < y_search_end and x_start < x_end):
            return None

        right_strip = redMaskForBoard[y_search_start:y_search_end, x_start:x_end]
        vertical_projection = np.max(right_strip, axis=1)
        gap_indices = np.where(vertical_projection == 0)[0]

        if len(gap_indices) == 0:
            return None

        split_indices = np.where(np.diff(gap_indices) > 1)[0] + 1
        gap_groups = np.split(gap_indices, split_indices)

        if not gap_groups:
            return None

        largest_gap = max(gap_groups, key=len)

        if len(largest_gap) > settings.min_goal_gap_px:
            gap_center_local_y = int(np.mean(largest_gap))
            goal_y = y_search_start + gap_center_local_y
            goal_x = x_center

            smallGoal = GoalDetection(
                x=goal_x,
                y=goal_y,
                size_category="small"
            )
            return smallGoal

    except Exception as e:
        print(f"[ERROR] Could not detect small goal: {e}")

    return None