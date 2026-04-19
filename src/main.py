import argparse
import time
import cv2 as cv
from typing import Optional

from config import *
from robot_client import RobotClient
from src.models import BallDetection, RobotPose
from src.navigation import decide_command
from src.ui import annotate
from src.vision import detect_balls, choose_target_ball, match_candidate_target, detect_robot_pose, detect_danger_zones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.20.10.2", help="Robot IP/hostname")
    parser.add_argument("--port", type=int, default=12345, help="Robot TCP port")
    parser.add_argument("--dry-run", action="store_true", help="Do not open socket; only print decisions")

    return parser.parse_args()

def main() -> int:
    # Parse arguments from terminal and set the settings
    args = parse_args()
    settings = Settings(
        host=args.host,
        port=args.port,
    )

    # Get the video capture
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream")
        return 1

    # Connect to the client
    client: Optional[RobotClient] = None
    if not args.dry_run:
        client = RobotClient(settings.host, settings.port, settings.reconnect_delay_sec)

    # Setup command variables
    candidate_command: Optional[str] = None
    candidate_count = 0
    last_send_command: Optional[str] = None
    last_send_time = 0.0

    # Setup target variables
    candidate_target: Optional[BallDetection] = None
    hold_command_until = 0.0
    last_target_seen_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error reading frame")
                break

            # Detect robot location and direction
            robot_pose = detect_robot_pose(frame, settings)

            # Detect balls
            balls = detect_balls(
                frame=frame,
                settings=settings,
                orange_range=ORANGE_RANGE,
                white_range=WHITE_RANGE,
                white_sat_split=80.0
            )

            # Choose target
            now = time.monotonic()
            commit_active = now < hold_command_until
            sample_ball = choose_target_ball(balls, robot_pose)

            if candidate_target is not None:
                matched_target = match_candidate_target(candidate_target, balls)
                if matched_target is not None:
                    target_ball = matched_target
                elif commit_active:
                    target_ball = candidate_target
                else:
                    candidate_target = None
                    target_ball = sample_ball
            else:
                target_ball = sample_ball
                if sample_ball is not None:
                    candidate_target = sample_ball

            if sample_ball is not None:
                last_target_seen_time = now
            elif not commit_active:
                candidate_target = None

            # Detect danger zones
            danger, danger_state, edges, danger_contours = detect_danger_zones(frame, settings, robot_pose)

            # Decide command
            command, reason = decide_command(
                frame_width=frame.shape[1],
                target_ball=target_ball,
                settings=settings,
                robot_pose=robot_pose,
                danger=danger,
                danger_state=danger_state,
            )

            # FIXME could probably be moved inside "deciceCommand()"
            #if not danger_state.too_close and not reason.startswith("danger"):
            if reason.endswith(":arrived") and target_ball is not None:
                candidate_target = target_ball
                command = CMD_FORWARD
                reason = f"commit:{reason}"
            elif candidate_target is not None and hold_command_until < now:
                hold_command_until = last_target_seen_time + settings.commit_forward_window_sec
                print("match_committed_target: ", match_candidate_target(candidate_target, balls))
                if not len(balls) == 0 and match_candidate_target(candidate_target, balls) is None:
                    command = CMD_FORWARD
                    reason = "commit:target_lost"
                elif len(balls) == 0 and now <= hold_command_until:
                    command = CMD_FORWARD
                    reason = "commit:no_ball"

            if command == candidate_command:
                candidate_count += 1
            else:
                candidate_command = command
                candidate_count = 1

            # Decide whether to send command
            now = time.time()
            should_send = ( # TODO is this correct?
                candidate_count >= settings.stable_frames_required
                and candidate_command != last_send_command
                and now - last_send_time >= settings.send_interval_sec
            )

            if should_send and candidate_command is not None:
                if client is not None:
                    client.send_char(candidate_command)
                print(f"sent={candidate_command} reason={reason}")
                last_send_command = candidate_command
                last_send_time = now

            # Annotate frame with detections and command
            display = annotate(
                frame=frame,
                robot_pose=robot_pose,
                balls=balls,
                target_ball=target_ball,
                command=command,
                reason=reason,
                last_sent_command=last_send_command,
                # FIXME - Doesn't draw danger zones currently, but they are there otherwise
            )
            cv.imshow("Golfbot", display)

            # Check for the quit key
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                if client is not None:
                    client.send_char(CMD_QUIT)
                break

    finally:
        cap.release()
        if client is not None:
            client.send_char(CMD_QUIT)
            client.close()
        cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())