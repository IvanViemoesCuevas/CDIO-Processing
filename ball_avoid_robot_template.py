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
    left: bool = False
    center: bool = False
    right: bool = False


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
    min_ball_circularity: float = 0.5
    min_ball_radius: float = 20.0
    max_ball_radius: float = 70.0
    min_ball_confidence: float = 0.65
    align_deadband_px: int = 35
    target_radius_px: float = 38.0
    obstacle_roi_start_ratio: float = 0.52
    min_obstacle_area: int = 800
    aruco_dictionary: int = cv.aruco.DICT_4X4_50
    robot_marker_id: int = -1
    use_robot_pose: bool = True
    pose_turn_deadband_deg: float = 12.0
    pose_arrival_distance_px: float = 70.0
    white_sat_split: float = 80.0


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
ORANGE_RANGE = HSVRange((0, 100, 210), (51, 255, 255))
# White ping pong balls under indoor light are often slightly yellow with darker edges.
WHITE_RANGE = HSVRange((19, 0, 0), (179, 99, 255))


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
    corners, ids, _ = detector.detectMarkers(frame)
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


def detect_danger_zones(frame: np.ndarray, settings: Settings) -> tuple[DangerFlags, np.ndarray]:
    h, w = frame.shape[:2]
    y0 = int(h * settings.obstacle_roi_start_ratio)
    roi = frame[y0:h, :]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 60, 140)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    flags = DangerFlags()
    zone_w = max(1, w // 3)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < settings.min_obstacle_area:
            continue

        x, y, cw, ch = cv.boundingRect(cnt)
        if ch < 18 or cw < 18:
            continue

        cx = x + cw // 2
        if cx < zone_w:
            flags.left = True
        elif cx < zone_w * 2:
            flags.center = True
        else:
            flags.right = True

    return flags, edges


def decide_command(
    frame_width: int,
    ball: Optional[BallDetection],
    danger: DangerFlags,
    settings: Settings,
    robot_pose: Optional[RobotPose] = None,
    frame_height: Optional[int] = None,
) -> tuple[str, str]:
    if danger.center:
        return CMD_STOP, "danger:center"
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
    danger: DangerFlags,
    command: str,
    reason: str,
    last_sent: Optional[str],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv.line(out, (w // 3, 0), (w // 3, h), (100, 100, 100), 1)
    cv.line(out, (2 * w // 3, 0), (2 * w // 3, h), (100, 100, 100), 1)

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
        cv.circle(out, (robot_pose.x, robot_pose.y), 8, (0, 255, 0), -1)
        arrow_len = 45
        x2 = int(robot_pose.x + arrow_len * math.cos(robot_pose.heading_rad))
        y2 = int(robot_pose.y + arrow_len * math.sin(robot_pose.heading_rad))
        cv.arrowedLine(out, (robot_pose.x, robot_pose.y), (x2, y2), (0, 255, 0), 2, tipLength=0.25)
        cv.putText(out, "robot", (robot_pose.x + 10, robot_pose.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if danger.left:
        cv.putText(out, "DANGER L", (10, h - 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if danger.center:
        cv.putText(out, "DANGER C", (w // 2 - 65, h - 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if danger.right:
        cv.putText(out, "DANGER R", (w - 145, h - 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.putText(out, f"cmd={command} reason={reason}", (10, 56), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(out, f"last_sent={last_sent}", (10, 84), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
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
            danger, edges = detect_danger_zones(frame, settings)
            command, reason = decide_command(
                frame.shape[1],
                ball,
                danger,
                settings,
                robot_pose=robot_pose,
                frame_height=frame.shape[0],
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

            display = annotate(frame, balls, ball, robot_pose, danger, command, reason, last_sent_command)
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

            if cv.waitKey(1) & 0xFF == ord("q"):
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



