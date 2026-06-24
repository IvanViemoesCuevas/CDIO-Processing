import cv2 as cv
import numpy as np

from config import (
    ORANGE_RANGE,
    WHITE_RANGE,
    HSVRange,
    BallDetectionTuning,
)


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
