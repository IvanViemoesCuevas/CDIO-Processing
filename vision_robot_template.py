#!/usr/bin/env python3
"""Vision-to-robot command template.

Reads webcam frames, runs YOLO, and sends one-byte commands to a robot over TCP
based on what the camera currently sees.
"""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2 as cv
from ultralytics import YOLO


@dataclass
class Settings:
    host: str = "10.1.1.9"
    port: int = 12345
    model_path: str = "yolov8s.pt"
    camera_index: int = 0
    conf_threshold: float = 0.45
    send_interval_sec: float = 0.15
    stable_frames_required: int = 3
    no_detection_command: str = "s"  # Usually a safe stop command.
    reconnect_delay_sec: float = 1.0


# Map detected class names to single-byte command characters.
# Update this mapping to match your robot protocol.
CLASS_TO_COMMAND: Dict[str, str] = {
    "person": "w",   # Move/track forward
    "bottle": "a",   # Example: turn left
    "chair": "d",    # Example: turn right
}


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


def get_best_detection_command(result, conf_threshold: float, class_to_command: Dict[str, str]) -> Optional[str]:
    """Return command for the highest-confidence mapped class in this frame."""
    best_conf = -1.0
    best_cmd = None

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls = int(box.cls[0])
        class_name = result.names[cls]
        cmd = class_to_command.get(class_name)
        if cmd is None:
            continue

        if conf > best_conf:
            best_conf = conf
            best_cmd = cmd

    return best_cmd


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="YOLO vision -> robot TCP command template")
    parser.add_argument("--host", default="10.1.1.9", help="Robot IP/hostname")
    parser.add_argument("--port", type=int, default=12345, help="Robot TCP port")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--send-interval", type=float, default=0.15, help="Minimum seconds between sends")
    parser.add_argument("--stable-frames", type=int, default=3, help="Frames required before changing command")
    parser.add_argument("--no-detection", default="s", help="Single char command when nothing relevant is seen")
    args = parser.parse_args()

    if len(args.no_detection) != 1:
        raise ValueError("--no-detection must be a single character")

    return Settings(
        host=args.host,
        port=args.port,
        model_path=args.model,
        camera_index=args.camera,
        conf_threshold=args.conf,
        send_interval_sec=args.send_interval,
        stable_frames_required=max(1, args.stable_frames),
        no_detection_command=args.no_detection,
    )


def main() -> int:
    settings = parse_args()

    model = YOLO(settings.model_path)
    cap = cv.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        print("Error opening video stream")
        return 1

    client = RobotClient(settings.host, settings.port, settings.reconnect_delay_sec)

    last_sent_command: Optional[str] = None
    candidate_command: Optional[str] = None
    candidate_count = 0
    last_send_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # stream=False gives a single Results object for one frame.
            result = model(frame, verbose=False)[0]

            command = get_best_detection_command(
                result=result,
                conf_threshold=settings.conf_threshold,
                class_to_command=CLASS_TO_COMMAND,
            )
            if command is None:
                command = settings.no_detection_command

            # Debounce: require command to stay the same for N frames.
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
                client.send_char(candidate_command)
                print(f"Sent command: {candidate_command}")
                last_sent_command = candidate_command
                last_send_time = now

            annotated = result.plot()
            cv.putText(
                annotated,
                f"cmd={command} sent={last_sent_command}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv.imshow("Vision Robot Controller", annotated)

            if cv.waitKey(1) & 0xFF == ord("q"):
                client.send_char("q")
                print("Sent quit command q")
                break

    finally:
        cap.release()
        client.close()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

