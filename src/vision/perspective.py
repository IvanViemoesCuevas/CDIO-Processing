import math
import cv2 as cv
import numpy as np

from config import (
    PERSPECTIVE_PADDING_PX,
)
from .danger_detector import build_danger_mask

PERSPECTIVE_UPDATE_INTERVAL = 5
PERSPECTIVE_SMOOTHING = 0.9


PERSPECTIVE_OUTPUT_SIZE: tuple[int, int] | None = None

_perspective_frame_counter = 0
_smoothed_perspective_points: np.ndarray | None = None


def correct_perspective(frame: np.ndarray) -> np.ndarray:
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
