import math
import cv2 as cv
import numpy as np
from typing import Optional

from config import (
    Settings,
    HSVRange,
    ORANGE_RANGE,
    WHITE_RANGE,
)
from models import BallDetection, RobotPose
from .yolo_helper import _ensure_yolo_model
from .tuner import build_ball_mask


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
