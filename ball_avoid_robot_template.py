#!/usr/bin/env python3
"""Camera-to-robot template for ping pong ball chase with obstacle avoidance.

- Detects orange/white ping pong balls using HSV + contour circularity.
- Detects near-field obstacles using edge/contour occupancy zones.
- Sends one-byte TCP commands to the robot with safety-first arbitration.
"""

from __future__ import annotations

import argparse
import math
import socket
import time
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class BallDetection:
    x: int
    y: int
    radius: float
    color_name: str
    confidence: float
    circularity: float = 0.0


@dataclass
class RobotPose:
    x: int
    y: int
    heading_rad: float
    confidence: float


@dataclass
class DangerFlags:
    front: bool = False
    back: bool = False
    left: bool = False
    center: bool = False
    right: bool = False


@dataclass
class DangerState:
    nearest_distance_px: float = float("inf")
    nearest_point: Optional[tuple[int, int]] = None
    nearest_dx_body: float = 0.0
    nearest_dy_body: float = 0.0
    too_close: bool = False


@dataclass
class Settings:
    host: str = "172.20.10.2"
    port: int = 12345
    camera_index: int = 0
    send_interval_sec: float = 0.12
    stable_frames_required: int = 2
    reconnect_delay_sec: float = 1.0
    min_ball_area: int = 140
    max_ball_area: int = 8000
    min_ball_circularity: float = 0.65
    min_ball_radius: float = 10.0
    max_ball_radius: float = 20.0
    min_ball_confidence: float = 0.5
    align_deadband_px: int = 35
    target_radius_px: float = 38.0
    obstacle_roi_start_ratio: float = 0.52
    min_obstacle_area: int = 800
    danger_center_deadband_px: int = 60
    danger_distance_px: float = 35.0
    danger_too_close_px: float = 35.0
    danger_rear_ignore_px: float = 45.0
    aruco_dictionary: int = cv.aruco.DICT_4X4_100
    robot_marker_id: int = 7
    use_robot_pose: bool = True
    pose_turn_deadband_deg: float = 12.0
    pose_arrival_distance_px: float = 70.0
    white_sat_split: float = 80.0
    robot_footprint_length_px: float = 70.0
    robot_footprint_width_px: float = 70.0


@dataclass
class HSVTuning:
    orange_range: HSVRange
    white_range: HSVRange
    white_sat_split: float


# Single-byte robot protocol. Change if your robot expects different keys.
CMD_FORWARD = "i"
CMD_LEFT = "j"
CMD_RIGHT = "l"
CMD_STOP = "s"
#CMD_SEARCH = "a"
CMD_QUIT = "q"


class RobotClient:
    def __init__(self, host: str, port: int, reconnect_delay_sec: float) -> None:
        self.host = host
        self.port = port
        self.reconnect_delay_sec = reconnect_delay_sec
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        while self.sock is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((self.host, self.port))
                sock.settimeout(None)
                self.sock = sock
                print(f"Connected to robot: {self.host}:{self.port}")
            except OSError as exc:
                print(f"Connect failed ({exc}); retrying in {self.reconnect_delay_sec:.1f}s")
                time.sleep(self.reconnect_delay_sec)

    def send_char(self, command: str) -> None:
        if len(command) != 1:
            raise ValueError("Command must be a single character")

        if self.sock is None:
            self.connect()

        try:
            assert self.sock is not None
            self.sock.send(bytes([ord(command)]))
        except OSError as exc:
            print(f"Send failed ({exc}); reconnecting...")
            self.close()
            self.connect()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

# (Hue, Saturation, brightness)
# Orange ping pong balls can look darker/less saturated at distance or in shadow.
ORANGE_RANGE = HSVRange((11, 100, 210), (19, 255, 255))
# White ping pong balls under indoor light are often slightly yellow with darker edges.
WHITE_RANGE = HSVRange((0, 0, 189), (179, 95, 255))
# Barrier color is red (camera sees red tops).
RED_RANGE_1 = HSVRange((0, 95, 60), (10, 255, 255))
RED_RANGE_2 = HSVRange((165, 95, 60), (179, 255, 255))


class HSVLiveTuner:
    WINDOW_NAME = "HSV Tuner"

    def __init__(self, orange_range: HSVRange, white_range: HSVRange, white_sat_split: float) -> None:
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

    def read(self) -> HSVTuning:
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

        return HSVTuning(
            orange_range=HSVRange(orange_lower, orange_upper),
            white_range=HSVRange(white_lower, white_upper),
            white_sat_split=float(self._get("white_sat_split")),
        )


def build_ball_masks(
    hsv_frame: np.ndarray,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    _, _, mask = build_ball_masks(hsv_frame, orange_range=orange_range, white_range=white_range)
    return mask


def make_ball_debug_view(
    frame: np.ndarray,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
    white_sat_split: float = 80.0,
) -> np.ndarray:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    orange_mask, white_mask, combined_mask = build_ball_masks(hsv, orange_range=orange_range, white_range=white_range)

    orange_vis = cv.cvtColor(orange_mask, cv.COLOR_GRAY2BGR)
    white_vis = cv.cvtColor(white_mask, cv.COLOR_GRAY2BGR)
    combined_vis = cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)
    overlay = frame.copy()

    cv.putText(orange_vis, "orange mask", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv.putText(white_vis, "white mask", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(combined_vis, "combined", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(overlay, "camera", (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    top = np.hstack((overlay, orange_vis))
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
        "q=quit | tune values with HSV Tuner trackbars",
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
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = build_ball_mask(hsv, orange_range=orange_range, white_range=white_range)

    sat_split = settings.white_sat_split if white_sat_split is None else white_sat_split

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    balls: list[BallDetection] = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < settings.min_ball_area or area > settings.max_ball_area:
            continue

        perimeter = cv.arcLength(cnt, True)
        if perimeter <= 0.0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < settings.min_ball_circularity:
            continue

        (x_f, y_f), radius = cv.minEnclosingCircle(cnt)
        if radius < settings.min_ball_radius or radius > settings.max_ball_radius:
            continue

        x = int(x_f)
        y = int(y_f)

        sample = hsv[max(0, y - 1): y + 2, max(0, x - 1): x + 2]
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


def detect_ball(
    frame: np.ndarray,
    settings: Settings,
    orange_range: HSVRange = ORANGE_RANGE,
    white_range: HSVRange = WHITE_RANGE,
    white_sat_split: Optional[float] = None,
) -> Optional[BallDetection]:
    balls = detect_balls(
        frame,
        settings,
        orange_range=orange_range,
        white_range=white_range,
        white_sat_split=white_sat_split,
    )
    if not balls:
        return None

    # Backward-compatible single target selection when robot pose is unavailable.
    best = balls[0]
    for current in balls[1:]:
        prev_score = best.confidence * best.radius
        new_score = current.confidence * current.radius
        if new_score > prev_score:
            best = current
    return best


def choose_target_ball(balls: list[BallDetection], robot_pose: Optional[RobotPose]) -> Optional[BallDetection]:
    if not balls:
        return None
    if robot_pose is None:
        best = balls[0]
        for current in balls[1:]:
            if current.confidence * current.radius > best.confidence * best.radius:
                best = current
        return best

    # Overhead camera strategy: choose the closest visible ball to robot center.
    return min(balls, key=lambda b: (b.x - robot_pose.x) ** 2 + (b.y - robot_pose.y) ** 2)


def detect_robot_pose(frame: np.ndarray, settings: Settings) -> Optional[RobotPose]:
    if not settings.use_robot_pose:
        return None

    if not hasattr(cv, "aruco"):
        return None

    dictionary = cv.aruco.getPredefinedDictionary(settings.aruco_dictionary)
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None

    target_index = 0
    if settings.robot_marker_id >= 0:
        match = np.where(ids.flatten() == settings.robot_marker_id)[0]
        if len(match) == 0:
            return None
        target_index = int(match[0])

    pts = corners[target_index][0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    top_mid = 0.5 * (pts[0] + pts[1])
    bottom_mid = 0.5 * (pts[2] + pts[3])
    forward_vec = top_mid - bottom_mid
    heading = math.atan2(float(forward_vec[1]), float(forward_vec[0]))

    return RobotPose(x=cx, y=cy, heading_rad=heading, confidence=1.0)


def draw_robot_footprint(frame: np.ndarray, robot_pose: RobotPose, length_px: float, width_px: float) -> None:
    length = max(10.0, float(length_px))
    width = max(10.0, float(width_px))
    angle_deg = math.degrees(robot_pose.heading_rad)
    rect = ((float(robot_pose.x), float(robot_pose.y)), (length, width), angle_deg)
    box = cv.boxPoints(rect).astype(np.int32)
    cv.polylines(frame, [box], True, (0, 255, 255), 2)


def build_danger_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red1 = cv.inRange(hsv, RED_RANGE_1.lower, RED_RANGE_1.upper)
    red2 = cv.inRange(hsv, RED_RANGE_2.lower, RED_RANGE_2.upper)
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


def decide_command(
    frame_width: int,
    ball: Optional[BallDetection],
    danger: DangerFlags,
    settings: Settings,
    robot_pose: Optional[RobotPose] = None,
    frame_height: Optional[int] = None,
    danger_state: Optional[DangerState] = None,
) -> tuple[str, str]:
    if danger_state is not None and danger_state.too_close:
        if abs(danger_state.nearest_dx_body) <= float(settings.danger_center_deadband_px):
            return CMD_STOP, f"danger:too_close d={danger_state.nearest_distance_px:.0f}"
        if danger_state.nearest_dy_body < -float(settings.danger_rear_ignore_px):
            return CMD_FORWARD, f"danger:avoid_back d={danger_state.nearest_distance_px:.0f}"
        if danger_state.nearest_dx_body < 0.0:
            return CMD_RIGHT, f"danger:avoid_left d={danger_state.nearest_distance_px:.0f}"
        return CMD_LEFT, f"danger:avoid_right d={danger_state.nearest_distance_px:.0f}"

    if danger.front and danger.center:
        return CMD_STOP, "danger:front"
    if danger.back and not danger.front and not danger.left and not danger.right:
        return CMD_FORWARD, "danger:back"
    if danger.left and not danger.right:
        return CMD_RIGHT, "danger:left"
    if danger.right and not danger.left:
        return CMD_LEFT, "danger:right"
    if danger.left and danger.right:
        return CMD_STOP, "danger:both"

    if ball is None:
        # No target visible: fail safe and wait for a ball to reappear.
        return CMD_STOP, "no_ball"

    if robot_pose is not None:
        dx = float(ball.x - robot_pose.x)
        dy = float(ball.y - robot_pose.y)
        target_heading = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(target_heading - robot_pose.heading_rad),
            math.cos(target_heading - robot_pose.heading_rad),
        )
        heading_error_deg = math.degrees(heading_error)
        distance_px = math.hypot(dx, dy)

        if heading_error_deg < -settings.pose_turn_deadband_deg:
            return CMD_LEFT, f"pose:left err={heading_error_deg:.1f}"
        if heading_error_deg > settings.pose_turn_deadband_deg:
            return CMD_RIGHT, f"pose:right err={heading_error_deg:.1f}"
        if distance_px > settings.pose_arrival_distance_px:
            return CMD_FORWARD, f"pose:forward d={distance_px:.0f}"
        return CMD_STOP, f"pose:arrived d={distance_px:.0f}"

    center_x = frame_width // 2
    error_x = ball.x - center_x

    if error_x < -settings.align_deadband_px:
        return CMD_LEFT, f"track:{ball.color_name}:left"
    if error_x > settings.align_deadband_px:
        return CMD_RIGHT, f"track:{ball.color_name}:right"

    if ball.radius < settings.target_radius_px:
        return CMD_FORWARD, f"track:{ball.color_name}:forward"

    return CMD_STOP, f"track:{ball.color_name}:arrived"


def annotate(
    frame: np.ndarray,
    balls: list[BallDetection],
    ball: Optional[BallDetection],
    robot_pose: Optional[RobotPose],
    robot_footprint_length_px: float,
    robot_footprint_width_px: float,
    danger_mask: np.ndarray,
    danger_contours: list[np.ndarray],
    danger: DangerFlags,
    danger_state: Optional[DangerState],
    command: str,
    reason: str,
    last_sent: Optional[str],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv.line(out, (w // 3, 0), (w // 3, h), (100, 100, 100), 1)
    cv.line(out, (2 * w // 3, 0), (2 * w // 3, h), (100, 100, 100), 1)

    if danger_mask.size > 0:
        overlay = out.copy()
        overlay[danger_mask > 0] = (255, 0, 255)
        out = cv.addWeighted(overlay, 0.22, out, 0.78, 0.0)
    if danger_contours:
        cv.drawContours(out, danger_contours, -1, (255, 0, 255), 2)

    if (
        robot_pose is not None
        and danger_state is not None
        and danger_state.nearest_point is not None
        and danger_state.too_close
    ):
        px, py = danger_state.nearest_point
        cv.circle(out, (px, py), 5, (255, 0, 255), -1)
        cv.line(out, (robot_pose.x, robot_pose.y), (px, py), (255, 0, 255), 2)
        cv.putText(
            out,
            f"danger_d={danger_state.nearest_distance_px:.0f}px",
            (10, 140),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    for candidate in balls:
        ghost_color = (225, 225, 225) if candidate.color_name == "white" else (80, 120, 255)
        cv.circle(out, (candidate.x, candidate.y), int(candidate.radius), ghost_color, 1)
        label = (
            f"{candidate.color_name} "
            f"conf={candidate.confidence:.2f} "
            f"circ={candidate.circularity:.2f} "
            f"r={candidate.radius:.1f}"
        )
        text_pos = (candidate.x - 36, max(14, candidate.y - int(candidate.radius) - 8))
        cv.putText(out, label, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.45, ghost_color, 2)

    if ball is not None:
        color = (255, 255, 255) if ball.color_name == "white" else (0, 165, 255)
        cv.circle(out, (ball.x, ball.y), int(ball.radius), color, 2)
        cv.circle(out, (ball.x, ball.y), 3, color, -1)
        cv.putText(
            out,
            f"ball={ball.color_name} conf={ball.confidence:.2f}",
            (10, 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    if robot_pose is not None:
        draw_robot_footprint(out, robot_pose, robot_footprint_length_px, robot_footprint_width_px)
        cv.circle(out, (robot_pose.x, robot_pose.y), 8, (0, 255, 0), -1)
        arrow_len = 45
        x2 = int(robot_pose.x + arrow_len * math.cos(robot_pose.heading_rad))
        y2 = int(robot_pose.y + arrow_len * math.sin(robot_pose.heading_rad))
        cv.arrowedLine(out, (robot_pose.x, robot_pose.y), (x2, y2), (0, 255, 0), 2, tipLength=0.25)
        cv.putText(out, "robot", (robot_pose.x + 10, robot_pose.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#    if danger.front:
#        cv.putText(out, "DANGER FRONT", (w // 2 - 95, 28), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
#    if danger.back:
#        cv.putText(out, "DANGER BACK", (w // 2 - 88, h - 24), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
#    if danger.left:
#        cv.putText(out, "DANGER LEFT", (10, h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
#    if danger.right:
#        cv.putText(out, "DANGER RIGHT", (w - 180, h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

    cv.putText(out, f"cmd={command} reason={reason}", (10, 56), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(out, f"last_sent={last_sent}", (10, 84), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv.putText(
        out,
        (
            f"robot_box LxW={int(robot_footprint_length_px)}x{int(robot_footprint_width_px)}px "
            f"([ ]=length ; '=width)"
        ),
        (10, 112),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ping pong ball tracker + obstacle avoid -> robot TCP commands")
    parser.add_argument("--host", default="172.20.10.2", help="Robot IP/hostname")
    parser.add_argument("--port", type=int, default=12345, help="Robot TCP port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--dry-run", action="store_true", help="Do not open socket; only print decisions")
    parser.add_argument("--show-edges", action="store_true", help="Show obstacle edge window")
    parser.add_argument("--show-ball-masks", action="store_true", help="Show orange/white/final ball masks")
    parser.add_argument("--tune-hsv", action="store_true", help="Show HSV trackbars and apply thresholds live")
    parser.add_argument(
        "--min-ball-confidence",
        type=float,
        default=Settings().min_ball_confidence,
        help="Minimum detection confidence for accepting a ball (0.0 to 1.0)",
    )
    parser.add_argument("--no-robot-pose", action="store_true", help="Disable ArUco-based robot pose estimation")
    parser.add_argument("--robot-marker-id", type=int, default=-1, help="Aruco marker id on robot (-1 = first visible)")
    parser.add_argument(
        "--robot-footprint-length-px",
        type=float,
        default=Settings().robot_footprint_length_px,
        help="Robot footprint length in pixels for visualization",
    )
    parser.add_argument(
        "--robot-footprint-width-px",
        type=float,
        default=Settings().robot_footprint_width_px,
        help="Robot footprint width in pixels for visualization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings(
        host=args.host,
        port=args.port,
        camera_index=args.camera,
        min_ball_confidence=max(0.0, min(1.0, args.min_ball_confidence)),
        use_robot_pose=not args.no_robot_pose,
        robot_marker_id=args.robot_marker_id,
        robot_footprint_length_px=max(10.0, float(args.robot_footprint_length_px)),
        robot_footprint_width_px=max(10.0, float(args.robot_footprint_width_px)),
    )

    cap = cv.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        print("Error opening video stream")
        return 1

    client: Optional[RobotClient] = None
    if not args.dry_run:
        client = RobotClient(settings.host, settings.port, settings.reconnect_delay_sec)

    tuner: Optional[HSVLiveTuner] = None
    if args.tune_hsv:
        tuner = HSVLiveTuner(ORANGE_RANGE, WHITE_RANGE, settings.white_sat_split)

    show_ball_masks = args.show_ball_masks or args.tune_hsv

    candidate_command: Optional[str] = None
    candidate_count = 0
    last_sent_command: Optional[str] = None
    last_send_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            tuning = HSVTuning(ORANGE_RANGE, WHITE_RANGE, settings.white_sat_split)
            if tuner is not None:
                tuning = tuner.read()

            robot_pose = detect_robot_pose(frame, settings)
            balls = detect_balls(
                frame,
                settings,
                orange_range=tuning.orange_range,
                white_range=tuning.white_range,
                white_sat_split=tuning.white_sat_split,
            )
            ball = choose_target_ball(balls, robot_pose)
            danger, danger_state, edges, danger_contours = detect_danger_zones(frame, settings, robot_pose)
            command, reason = decide_command(
                frame.shape[1],
                ball,
                danger,
                settings,
                robot_pose=robot_pose,
                frame_height=frame.shape[0],
                danger_state=danger_state,
            )

            if command == candidate_command:
                candidate_count += 1
            else:
                candidate_command = command
                candidate_count = 1

            now = time.time()
            should_send = (
                candidate_count >= settings.stable_frames_required
                and candidate_command != last_sent_command
                and now - last_send_time >= settings.send_interval_sec
            )

            if should_send and candidate_command is not None:
                if client is not None:
                    client.send_char(candidate_command)
                print(f"sent={candidate_command} reason={reason}")
                last_sent_command = candidate_command
                last_send_time = now

            #display = annotate(frame, balls, ball, robot_pose, settings.robot_footprint_length_px, settings.robot_footprint_width_px, danger, command, reason, last_sent_command)
            display = annotate(
                frame,
                balls,
                ball,
                robot_pose,
                settings.robot_footprint_length_px,
                settings.robot_footprint_width_px,
                edges,
                danger_contours,
                danger,
                danger_state,
                command,
                reason,
                last_sent_command,
            )
            cv.imshow("Ball Avoid Robot Template", display)
            if args.show_edges:
                cv.imshow("Obstacle Edges", edges)
            if show_ball_masks:
                cv.imshow(
                    "Ball Masks Debug",
                    make_ball_debug_view(
                        frame,
                        orange_range=tuning.orange_range,
                        white_range=tuning.white_range,
                        white_sat_split=tuning.white_sat_split,
                    ),
                )

            key = cv.waitKey(1) & 0xFF
            if key == ord("["):
                settings.robot_footprint_length_px = max(10.0, settings.robot_footprint_length_px - 2.0)
            elif key == ord("]"):
                settings.robot_footprint_length_px += 2.0
            elif key == ord(";"):
                settings.robot_footprint_width_px = max(10.0, settings.robot_footprint_width_px - 2.0)
            elif key == ord("'"):
                settings.robot_footprint_width_px += 2.0
            elif key == ord("q"):
                if client is not None:
                    client.send_char(CMD_QUIT)
                break

    finally:
        cap.release()
        if client is not None:
            client.close()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



