import math
import cv2 as cv
import numpy as np
from typing import Optional

from config import (
    Settings,
    RED_RANGE_1,
    RED_RANGE_2,
)
from models import FieldCorners, GoalDetection


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
