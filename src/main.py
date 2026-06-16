import argparse
import time
import cv2 as cv
from typing import Optional

from config import *
from robot_client import RobotClient
from src.models import NavigationContext, NavigationState, GoalDetection
from src.navigation import decide_command
from src.ui import annotate
from src.vision import (
    BallDetectionTuner,
    choose_target_ball,
    detect_balls,
    detect_danger_zones,
    detect_robot_pose,
    make_ball_debug_view,
    match_candidate_target,
    detect_field_corners,
    find_small_goal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.20.10.2", help="Robot IP/hostname")
    parser.add_argument("--port", type=int, default=12345, help="Robot TCP port")
    parser.add_argument("--dry-run", action="store_true", help="Do not open socket; only print decisions")
    parser.add_argument(
        "--tune-hsv",
        action="store_true",
        help="Open HSV trackbars and show a live ball-mask debug view",
    )

    return parser.parse_args()

def main() -> int:
    # Parse arguments from terminal and set the settings
    args = parse_args()
    settings = Settings(
        host=args.host,
        port=args.port,
    )

    # Get the video capture
    cap0 = cv.VideoCapture(0)
    if not cap0.isOpened():
        print("Error opening video stream 0")
        return 1

    # Connect to the client
    client: Optional[RobotClient] = None
    if not args.dry_run:
        client = RobotClient(settings.host, settings.port, settings.reconnect_delay_sec)

    tuner: Optional[BallDetectionTuner] = None
    if args.tune_hsv:
        tuner = BallDetectionTuner(
            orange_range=ORANGE_RANGE,
            white_range=WHITE_RANGE,
            white_sat_split=settings.white_sat_split,
        )

    default_tuning = BallDetectionTuning(
        orange_range=ORANGE_RANGE,
        white_range=WHITE_RANGE,
        white_sat_split=settings.white_sat_split,
    )

    # Setup command variables
    candidate_command: Optional[str] = None
    candidate_count = 0
    last_send_command: Optional[str] = None
    last_send_time = 0.0

    # Setup navigation state variables
    nav_state = NavigationState()
    
    # Cache for the small goal
    cached_small_goal: Optional[GoalDetection] = None

    try:
        while True:
            ok, frame = cap0.read()
            if not ok:
                print("Error reading frame")
                break

            # --- Vision Pipeline ---
            
            # 1. Detect field corners (existing logic for green overlay)
            field_corners = detect_field_corners(frame, settings)

            # 2. Calculate small goal ONLY ONCE based on field corners
            if cached_small_goal is None and field_corners is not None:
                tr = field_corners.topRight
                br = field_corners.bottomRight
                
                # Calculate middle of the right edge
                goal_x = int((tr[0] + br[0]) / 2)
                goal_y = int((tr[1] + br[1]) / 2)
                
                # Calculate delivery point (80 pixels inside)
                delivery_x = goal_x - 80
                
                cached_small_goal = GoalDetection(
                    x=goal_x,
                    y=goal_y,
                    size_category="small"
                )
                cached_small_goal.delivery_x = delivery_x
                
                print(f"Small Goal calculated and cached at: ({cached_small_goal.x}, {cached_small_goal.y})")
                print(f"Delivery point for EV3: ({cached_small_goal.delivery_x}, {cached_small_goal.y})")


            # Other detections (unchanged)
            robot_pose = detect_robot_pose(frame, settings)
            tuning = tuner.read() if tuner is not None else default_tuning
            balls = detect_balls(
                frame=frame,
                settings=settings,
                orange_range=tuning.orange_range,
                white_range=tuning.white_range,
                white_sat_split=tuning.white_sat_split,
            )
            danger, danger_state, _, danger_contours = detect_danger_zones(frame, settings, robot_pose)

            # --- Navigation Logic (unchanged) ---
            now = time.monotonic()
            commit_active = now < nav_state.hold_command_until
            sample_ball = choose_target_ball(balls, robot_pose)
            matched_candidate = (
                match_candidate_target(nav_state.candidate_target, balls)
                if nav_state.candidate_target is not None
                else None
            )

            if nav_state.candidate_target is not None:
                if matched_candidate is not None:
                    target_ball = matched_candidate
                elif commit_active:
                    target_ball = nav_state.candidate_target
                else:
                    nav_state.candidate_target = None
                    target_ball = sample_ball
            else:
                target_ball = sample_ball
                if sample_ball is not None:
                    nav_state.candidate_target = sample_ball

            if sample_ball is not None:
                nav_state.last_target_seen_time = now
            elif not commit_active:
                nav_state.candidate_target = None

            decision, nav_state = decide_command(
                context=NavigationContext(
                    frame_width=frame.shape[1],
                    target_ball=target_ball,
                    danger=danger,
                    robot_pose=robot_pose,
                    danger_state=danger_state,
                    now=now,
                    balls_count=len(balls),
                    candidate_target_visible=matched_candidate is not None,
                ),
                state=nav_state,
                settings=settings,
            )
            command = decision.command
            reason = decision.reason

            if command == candidate_command:
                candidate_count += 1
            else:
                candidate_command = command
                candidate_count = 1

            # --- Command Sending (unchanged) ---
            now_time = time.time()
            should_send = (
                candidate_count >= settings.stable_frames_required
                and now_time - last_send_time >= settings.send_interval_sec
            )

            if should_send and candidate_command is not None:
                if client is not None:
                    client.send_char(candidate_command)
                print(f"sent={candidate_command} reason={reason}")
                last_send_command = candidate_command
                last_send_time = now_time

            # --- Annotation and Display ---
            display = annotate(
                frame=frame,
                command=command,
                reason=reason,
                last_sent_command=last_send_command,
                balls=balls,
                target_ball=target_ball,
                robot_pose=robot_pose,
                danger=danger,
                danger_state=danger_state,
                danger_contours=danger_contours,
                field_corners=field_corners,
                small_goal=cached_small_goal,
            )
            cv.imshow("Golfbot", display)

            if tuner is not None:
                cv.imshow(
                    "Ball Masks Debug",
                    make_ball_debug_view(
                        frame,
                        orange_range=tuning.orange_range,
                        white_range=tuning.white_range,
                        white_sat_split=tuning.white_sat_split,
                    ),
                )

            # Check for the quit key
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                if client is not None:
                    client.send_char(CMD_QUIT)
                break

    finally:
        cap0.release()
        if client is not None:
            client.send_char(CMD_QUIT)
            client.close()
        cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())