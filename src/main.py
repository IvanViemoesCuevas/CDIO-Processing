import argparse
import math
import time
import cv2 as cv
from typing import Optional
import platform

from config import *
from robot_client import RobotClient
from models import NavigationContext, NavigationState, GoalDetection, RobotPose, NavigationResult
from navigation import decide_command, decide_immediate_command
from route_manager import RouteManager
from ui import annotate
from vision import (
    BallDetectionTuner,
    BallHandoffManager,
    correct_perspective,
    detect_balls,
    detect_danger_zones,
    detect_robot_pose,
    make_ball_debug_view,
    match_candidate_target,
    detect_field_corners,
)
from utils import get_scales, corrected_robot_pose_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.20.10.9", help="Robot IP/hostname")
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
    if platform.system() == "Windows":
        cap0 = cv.VideoCapture(1, cv.CAP_DSHOW)
    else:
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
    manual_override_until = 0.0

    # Setup navigation state variables
    nav_state = NavigationState()

    # Cache for the small goal
    cached_small_goal: Optional[GoalDetection] = None

    # Setup handoff manager
    handoff_manager = BallHandoffManager(required_empty_frames=11)

    # Setup route manager
    route_manager = RouteManager()

    try:
        while True:
            #image_path = "Test_Image.png"  # Change this to your image path
            #frame = cv.imread(image_path)
            #if frame is None:
            #    print(f"Error reading image from {image_path}")
            #    break

            ok, frame = cap0.read()
            if not ok:
                print("Error reading frame")
                break

            frame = correct_perspective(frame)

            # Detect robot location and direction
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

                # Calculate alignment and delivery points dynamically using get_scales
                scale_x, scale_y = get_scales(frame.shape[1], frame.shape[0])
                alignment_x = goal_x - int(settings.alignment_point_offset_cm * scale_x)
                delivery_x = goal_x - int(settings.delivery_point_offset_cm * scale_x)

                cached_small_goal = GoalDetection(
                    x=goal_x,
                    y=goal_y,
                    size_category="small"
                )
                cached_small_goal.alignment_point_x = alignment_x
                cached_small_goal.alignment_point_y = goal_y
                cached_small_goal.delivery_point_x = delivery_x
                cached_small_goal.delivery_point_y = goal_y

                print(f"Small Goal calculated and cached at: ({cached_small_goal.x}, {cached_small_goal.y})")
                print(f"Alignment point: ({cached_small_goal.alignment_point_x}, {cached_small_goal.alignment_point_y})")
                print(f"Delivery point: ({cached_small_goal.delivery_point_x}, {cached_small_goal.delivery_point_y})")

            robot_pose = detect_robot_pose(frame, settings)
            danger_robot_pose = robot_pose

            if robot_pose is not None:
                corrected_x, corrected_y, corrected_heading, *_ = corrected_robot_pose_values(
                    robot_pose,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
                danger_robot_pose = RobotPose(
                    x=int(round(corrected_x)),
                    y=int(round(corrected_y)),
                    heading_rad=corrected_heading,
                    confidence=robot_pose.confidence,
                )

            tuning = tuner.read() if tuner is not None else default_tuning

            # Detect balls
            balls = detect_balls(
                frame=frame,
                settings=settings,
                orange_range=tuning.orange_range,
                white_range=tuning.white_range,
                white_sat_split=tuning.white_sat_split,
            )
            danger, danger_state, danger_mask, danger_contours = detect_danger_zones(frame, settings, danger_robot_pose)

            # Update handoff manager and check for handoff condition
            handoff_manager.update(balls)

            # Decide target and command override from RouteManager
            scale_x, scale_y = get_scales(frame.shape[1], frame.shape[0])
            now = time.monotonic()

            target_ball, command_override, reason_override = route_manager.update(
                current_time=now,
                balls=balls,
                robot_pose=danger_robot_pose,
                danger_mask=danger_mask,
                scale_x=scale_x,
                scale_y=scale_y,
                settings=settings,
                current_danger_contours=danger_contours,
                small_goal=cached_small_goal,
            )

            # Check waypoint arrival BEFORE deciding command to prevent sending CMD_STOP
            if route_manager.state == "executing" and robot_pose is not None:
                waypoint_pops_this_frame = 0
                while target_ball is not None and target_ball.color_name == "waypoint":
                    waypoint_pops_this_frame += 1
                    if waypoint_pops_this_frame > 3:
                        #print("[main] Waypoint pop guard hit; leaving loop until next frame.")
                        break

                    robot_x, robot_y, robot_heading, *_ = corrected_robot_pose_values(
                        robot_pose,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )
                    dx = float(target_ball.x - robot_x)
                    dy = float(target_ball.y - robot_y)
                    dx_cm = dx / scale_x
                    dy_cm = dy / scale_y
                    distance_cm = math.hypot(dx_cm, dy_cm)

                    waypoint_arrival_dist = getattr(settings, 'waypoint_arrival_distance_cm', 8.0)
                    if distance_cm <= waypoint_arrival_dist:
                        print(f"[main] Arrived at waypoint at distance {distance_cm:.1f} cm (threshold {waypoint_arrival_dist} cm). Popping from queue.")
                        if route_manager.queue:
                            route_manager.queue.pop(0)

                        # Re-update to get the next target ball
                        target_ball, command_override, reason_override = route_manager.update(
                            current_time=now,
                            balls=balls,
                            robot_pose=danger_robot_pose,
                            danger_mask=danger_mask,
                            scale_x=scale_x,
                            scale_y=scale_y,
                            settings=settings,
                            current_danger_contours=danger_contours,
                            small_goal=cached_small_goal,
                        )
                    else:
                        break

            # Manage handoff transitions in main loop
            if route_manager.state == "handoff":
                if nav_state.handoff_phase == "idle":
                    nav_state.handoff_phase = "approaching_alignment"
                    print("[main] RouteManager requested handoff. Initiating handoff...")
                elif nav_state.handoff_phase == "done":
                    print("[main] Handoff complete. Transitioning RouteManager to re-evaluation...")
                    route_manager.state = "re_evaluating"
                    nav_state.handoff_phase = "idle"
                    handoff_manager.reset()

            # Determine command and reason
            if command_override:
                immediate_decision = decide_immediate_command(
                    context=NavigationContext(
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        target_ball=target_ball,
                        danger=danger,
                        robot_pose=danger_robot_pose,
                        danger_state=danger_state,
                        now=now,
                        balls_count=len(balls),
                        candidate_target_visible=False,
                        handoff_manager=handoff_manager,
                        small_goal=cached_small_goal,
                    ),
                    settings=settings,
                    state=nav_state,
                )

                commit_warning_active = (
                    command_override == CMD_FORWARD
                    and danger_state is not None
                    and danger_state.nearest_point is not None
                    and danger_state.nearest_dy_body >= 0.0
                    and danger_state.nearest_distance_cm <= 5.0
                )

                if commit_warning_active and not immediate_decision.reason.startswith("danger"):
                    if danger_state.nearest_dx_body < -2.0:
                        immediate_decision = NavigationResult(
                            CMD_RIGHT,
                            f"danger:commit_warning_left d={danger_state.nearest_distance_cm:.1f}",
                        )
                    elif danger_state.nearest_dx_body > 2.0:
                        immediate_decision = NavigationResult(
                            CMD_LEFT,
                            f"danger:commit_warning_right d={danger_state.nearest_distance_cm:.1f}",
                        )
                    else:
                        immediate_decision = NavigationResult(
                            CMD_BACKWARD,
                            f"danger:commit_warning_front d={danger_state.nearest_distance_cm:.1f}",
                        )

                if immediate_decision.reason.startswith("danger"):
                    command = immediate_decision.command
                    reason = f"override_safety:{immediate_decision.reason}"
                else:
                    command = command_override
                    reason = reason_override
            else:
                matched_candidate = (
                    match_candidate_target(target_ball, balls)
                    if target_ball is not None
                    else None
                )

                decision, nav_state = decide_command(
                    context=NavigationContext(
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        target_ball=target_ball,
                        danger=danger,
                        robot_pose=robot_pose,
                        danger_state=danger_state,
                        now=now,
                        balls_count=len(balls),
                        candidate_target_visible=matched_candidate is not None,
                        handoff_manager=handoff_manager,
                        small_goal=cached_small_goal,
                    ),
                    state=nav_state,
                    settings=settings,
                )
                command = decision.command
                reason = decision.reason
                if "arrived" in reason and route_manager.state == "executing":
                    # Waypoint arrival is handled entirely by the pre-check loop above;
                    # only trigger the commit phase for real ball targets.
                    if target_ball is not None and target_ball.color_name != "waypoint":
                        route_manager.state = "commit"

            if command == candidate_command:
                candidate_count += 1
            else:
                candidate_command = command
                candidate_count = 1

            # --- Command Sending (unchanged) ---
            now_time = time.time()
            is_one_shot_switch = command == CMD_SWITCH and (reason == "handoff:start_unload" or reason == "handoff:done_stop")
            is_safety_override = reason.startswith("override_safety:danger")

            should_send = (
                    is_one_shot_switch
                    or is_safety_override
                    or (
                            candidate_count >= settings.stable_frames_required
                            and now_time - last_send_time >= settings.send_interval_sec
                    )
            )

            if should_send and candidate_command is not None:
                command_to_send = command if (is_one_shot_switch or is_safety_override) else candidate_command
                if client is not None:
                    client.send_char(command_to_send)
                print(f"sent={command_to_send} reason={reason}")
                last_send_command = command_to_send
                last_send_time = now_time

            target_desc = "none"
            if target_ball is not None:
                target_desc = f"{target_ball.color_name}@({target_ball.x},{target_ball.y})"

            danger_desc = (
                f"F{int(danger.front)} B{int(danger.back)} "
                f"L{int(danger.left)} C{int(danger.center)} R{int(danger.right)}"
            )
            debug_signature = (
                route_manager.state,
                len(route_manager.queue),
                target_desc,
                command,
                reason,
                danger_desc,
            )

            # Annotate frame with detections and command
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
                route_manager=route_manager,
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

            # Check for manual key
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                if client is not None:
                    client.send_char(CMD_QUIT)
                break
            elif key == ord('x'):
                if client is not None:
                    client.send_char(CMD_SWITCH)
            elif key == ord('i'):
                if client is not None:
                    client.send_char(CMD_FORWARD)
            elif key == ord('k'):
                if client is not None:
                    client.send_char(CMD_BACKWARD)
            elif key == ord('j'):
                if client is not None:
                    client.send_char(CMD_LEFT)
            elif key == ord('l'):
                if client is not None:
                    client.send_char(CMD_RIGHT)

    finally:
        cap0.release()
        if client is not None:
            client.send_char(CMD_QUIT)
            client.close()
        cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())