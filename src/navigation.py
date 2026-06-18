import math
from dataclasses import replace

from config import *

from models import (
    NavigationContext,
    NavigationResult,
    NavigationState,
    BallDetection,
)

# Calibration offsets between the elevated ArUco marker and the robot's real ground pose.
# Since the marker works when placed on the ground, tune the forward/right pixel offsets first.
# Only tune heading if the marker is physically rotated relative to the robot's forward direction.

MARKER_HEADING_OFFSET_DEG = 0.0

# Physical mount offset from the ArUco marker to the robot's real drive/rotation center.
# These are robot-local offsets and DO rotate when the robot turns:
#   forward_px = along robot forward
#   right_px   = along robot right
MARKER_TO_DRIVE_CENTER_FORWARD_PX = 0.0
MARKER_TO_DRIVE_CENTER_RIGHT_PX = 0.0

# Perspective correction caused by the marker being elevated above the ground.
# These are IMAGE-SPACE corrections and do NOT rotate with the robot.
# Tune these so the elevated marker projects down to the ground point under the marker.
# Positive X gain moves the correction more right when the marker is right of image center.
# Positive Y gain moves the correction more down when the marker is below image center.
MARKER_PERSPECTIVE_X_GAIN = -88.0
MARKER_PERSPECTIVE_Y_GAIN = -53.0

# Backwards-compatible aliases used by ui.py debug text.
MARKER_PERSPECTIVE_RIGHT_GAIN = MARKER_PERSPECTIVE_X_GAIN
MARKER_PERSPECTIVE_FORWARD_GAIN = MARKER_PERSPECTIVE_Y_GAIN


def normalize_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def corrected_robot_pose_values(robot_pose, frame_width: int | None = None, frame_height: int | None = None):
    corrected_heading = normalize_angle_rad(
        robot_pose.heading_rad + math.radians(MARKER_HEADING_OFFSET_DEG)
    )

    # Robot-local unit vectors in image coordinates.
    # These rotate with the ArUco/robot heading.
    forward_x = math.cos(corrected_heading)
    forward_y = math.sin(corrected_heading)
    right_x = math.cos(corrected_heading + math.pi / 2.0)
    right_y = math.sin(corrected_heading + math.pi / 2.0)

    normalized_x = 0.0
    normalized_y = 0.0

    if frame_width is not None and frame_width > 0:
        image_center_x = frame_width / 2.0
        normalized_x = (float(robot_pose.x) - image_center_x) / image_center_x

    if frame_height is not None and frame_height > 0:
        image_center_y = frame_height / 2.0
        normalized_y = (float(robot_pose.y) - image_center_y) / image_center_y

    # 1) Perspective correction: image-space offset caused by marker height.
    # This should NOT rotate with the robot. It only depends on where the elevated marker
    # appears in the camera image.
    perspective_offset_x = MARKER_PERSPECTIVE_X_GAIN * normalized_x
    perspective_offset_y = MARKER_PERSPECTIVE_Y_GAIN * normalized_y

    # 2) Physical mount correction: robot-local offset from marker to drive center.
    # This DOES rotate with the robot.
    mount_offset_x = (
        MARKER_TO_DRIVE_CENTER_FORWARD_PX * forward_x
        + MARKER_TO_DRIVE_CENTER_RIGHT_PX * right_x
    )
    mount_offset_y = (
        MARKER_TO_DRIVE_CENTER_FORWARD_PX * forward_y
        + MARKER_TO_DRIVE_CENTER_RIGHT_PX * right_y
    )

    # Final correction in image coordinates.
    offset_x = perspective_offset_x + mount_offset_x
    offset_y = perspective_offset_y + mount_offset_y

    corrected_x = float(robot_pose.x) + offset_x
    corrected_y = float(robot_pose.y) + offset_y

    # For the UI: express the final image-space correction back in the robot-local axes.
    # This is the part that will appear to "switch" when the robot turns 90 degrees.
    used_forward_px = offset_x * forward_x + offset_y * forward_y
    used_right_px = offset_x * right_x + offset_y * right_y

    return (
        corrected_x,
        corrected_y,
        corrected_heading,
        used_forward_px,
        used_right_px,
        offset_x,
        offset_y,
        normalized_x,
        normalized_y,
    )

def decide_immediate_command(context: NavigationContext, settings: Settings, state: NavigationState, arrival_distance_px=None, turn_deadband_deg=None) -> NavigationResult:
    danger = context.danger
    danger_state = context.danger_state
    target_ball = context.target_ball

    # Use specific tolerances for handoff, otherwise use defaults
    arrival_dist = arrival_distance_px if arrival_distance_px is not None else settings.pose_arrival_distance_px
    turn_deadband = turn_deadband_deg if turn_deadband_deg is not None else settings.pose_turn_deadband_deg

    if danger_state is not None and danger_state.too_close:
        if abs(danger_state.nearest_dx_body) <= float(settings.danger_center_deadband_px):
            return NavigationResult(CMD_STOP, f"danger:too_close d={danger_state.nearest_distance_px:.0f}")
        if danger_state.nearest_dy_body < -float(settings.danger_rear_ignore_px):
            return NavigationResult(CMD_FORWARD, f"danger:avoid_back d={danger_state.nearest_distance_px:.0f}")
        if danger_state.nearest_dx_body < 0.0:
            return NavigationResult(CMD_RIGHT, f"danger:avoid_left d={danger_state.nearest_distance_px:.0f}")
        return NavigationResult(CMD_LEFT, f"danger:avoid_right d={danger_state.nearest_distance_px:.0f}")

    if danger.front and danger.center:
        return NavigationResult(CMD_BACKWARD, "danger:front")
    if danger.back and not danger.front and not danger.left and not danger.right:
        return NavigationResult(CMD_FORWARD, "danger:back")
    if danger.left and not danger.right:
        return NavigationResult(CMD_RIGHT, "danger:left")
    if danger.right and not danger.left:
        return NavigationResult(CMD_LEFT, "danger:right")
    if danger.left and danger.right:
        return NavigationResult(CMD_STOP, "danger:both")

    if target_ball is None:
        return NavigationResult(CMD_STOP, "no_ball")

    if context.robot_pose is not None:
        frame_height = getattr(context, "frame_height", None)
        robot_x, robot_y, robot_heading, *_ = corrected_robot_pose_values(
            context.robot_pose,
            frame_width=context.frame_width,
            frame_height=frame_height,
        )

        dx = float(target_ball.x - robot_x)
        dy = float(target_ball.y - robot_y)
        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle_rad(target_heading - robot_heading)
        heading_error_deg = math.degrees(heading_error)
        distance_px = math.hypot(dx, dy)

        # Hysteresis for turning
        turn_hysteresis_factor = 1.5
        effective_turn_deadband = turn_deadband
        if state.last_command == CMD_FORWARD:
            effective_turn_deadband *= turn_hysteresis_factor

        if heading_error_deg < -effective_turn_deadband:
            return NavigationResult(CMD_LEFT, f"pose:left err={heading_error_deg:.1f}")
        if heading_error_deg > effective_turn_deadband:
            return NavigationResult(CMD_RIGHT, f"pose:right err={heading_error_deg:.1f}")

        if distance_px > arrival_dist:
            return NavigationResult(CMD_FORWARD, f"pose:forward d={distance_px:.0f}")

        return NavigationResult(CMD_STOP, f"pose:arrived d={distance_px:.0f}")

    else:
        center_x = context.frame_width // 2
        error_x = target_ball.x - center_x

        if error_x < -settings.align_deadband_px:
            return NavigationResult(CMD_LEFT, f"track:{target_ball.color_name}:left")
        if error_x > settings.align_deadband_px:
            return NavigationResult(CMD_RIGHT, f"track:{target_ball.color_name}:right")
        if target_ball.radius < settings.target_radius_px:
            return NavigationResult(CMD_FORWARD, f"track:{target_ball.color_name}:forward")
        return NavigationResult(CMD_STOP, f"track:{target_ball.color_name}:arrived")


def decide_command(
    context: NavigationContext,
    state: NavigationState,
    settings: Settings,
) -> tuple[NavigationResult, NavigationState]:
    
    # Handoff Logic
    if context.handoff_manager and context.handoff_manager.ready_for_handoff and state.handoff_phase == "idle":
        state.handoff_phase = "approaching_alignment"
        print("Handoff: Phase -> approaching_alignment")

    if state.handoff_phase != "idle":
        if context.small_goal is None:
            return NavigationResult(CMD_STOP, "handoff:no_goal"), state

        # --- Phase: approaching_alignment ---
        if state.handoff_phase == "approaching_alignment":
            alignment_target = BallDetection(x=context.small_goal.alignment_point_x, y=context.small_goal.alignment_point_y, radius=settings.pose_arrival_distance_px, color_name="align_pt", confidence=1.0)
            nav_context = replace(context, target_ball=alignment_target)
            res = decide_immediate_command(nav_context, settings, state, arrival_distance_px=50) # Tighter arrival
            if "arrived" in res.reason:
                state.handoff_phase = "aligning"
                state.hold_command_until = context.now + 1.0 # Longer pause
                print("Handoff: Arrived at alignment point. Phase -> aligning")
                return NavigationResult(CMD_STOP, "handoff:arrived_align"), state
            return res, state

        # --- Phase: aligning ---
        if state.handoff_phase == "aligning":
            if state.hold_command_until > context.now:
                return NavigationResult(CMD_STOP, "handoff:pausing"), state
            
            goal_target = BallDetection(x=context.small_goal.x, y=context.small_goal.y, radius=0, color_name="goal", confidence=1.0)
            nav_context = replace(context, target_ball=goal_target)
            res = decide_immediate_command(nav_context, settings, state, turn_deadband_deg=5) # Tighter alignment

            if res.command not in [CMD_LEFT, CMD_RIGHT]:
                state.handoff_phase = "approaching_delivery"
                state.hold_command_until = context.now + 1.0 # Longer pause
                print("Handoff: Aligned with goal. Phase -> approaching_delivery")
                return NavigationResult(CMD_STOP, "handoff:aligned"), state
            return res, state

        # --- Phase: approaching_delivery ---
        if state.handoff_phase == "approaching_delivery":
            if state.hold_command_until > context.now:
                return NavigationResult(CMD_STOP, "handoff:pausing"), state

            delivery_target = BallDetection(x=context.small_goal.delivery_point_x, y=context.small_goal.delivery_point_y, radius=settings.pose_arrival_distance_px, color_name="delivery_pt", confidence=1.0)
            nav_context = replace(context, target_ball=delivery_target)
            res = decide_immediate_command(nav_context, settings, state, arrival_distance_px=50) # Tighter arrival

            if "arrived" in res.reason:
                state.handoff_phase = "delivering"
                state.hold_command_until = context.now + 1.0 # Longer pause
                print("Handoff: Arrived at delivery point. Phase -> delivering")
                return NavigationResult(CMD_STOP, "handoff:arrived_delivery"), state
            return res, state

        # --- Phase: delivering ---
        if state.handoff_phase == "delivering":
            if state.hold_command_until > context.now:
                return NavigationResult(CMD_STOP, "handoff:delivered_stop"), state
            state.handoff_phase = "done"
            print("Handoff: Delivery complete. Phase -> done")
            return NavigationResult(CMD_STOP, "handoff:done_stop"), state

        # --- Phase: done ---
        if state.handoff_phase == "done":
            return NavigationResult(CMD_STOP, "handoff:done_stop"), state

    # Regular ball-following logic
    immediate = decide_immediate_command(context, settings, state)
    return apply_commit_transitions(immediate, context, state, settings)


def apply_commit_transitions(
    result: NavigationResult,
    context: NavigationContext,
    state: NavigationState,
    settings: Settings,
) -> tuple[NavigationResult, NavigationState]:
    next_state = NavigationState(
        candidate_target=state.candidate_target,
        hold_command_until=state.hold_command_until,
        last_target_seen_time=state.last_target_seen_time,
        handoff_phase=state.handoff_phase,
        last_command = result.command
    )

    if result.reason.startswith("danger"):
        return result, next_state

    if result.reason.endswith(":arrived") and context.target_ball is not None:
        next_state.candidate_target = context.target_ball
        return NavigationResult(CMD_FORWARD, f"commit:{result.reason}"), next_state

    if next_state.candidate_target is None:
        return result, next_state

    if context.target_ball is None and context.balls_count == 0 and context.now <= next_state.hold_command_until:
        next_state.hold_command_until = context.now
        return NavigationResult(CMD_FORWARD, f"commit:ball_disappeared"), next_state

    if not (next_state.hold_command_until < context.now):
        return result, next_state

    next_state.hold_command_until = (
        next_state.last_target_seen_time + settings.commit_forward_window_sec
    )

    if context.balls_count > 0 and not context.candidate_target_visible:
        return NavigationResult(CMD_FORWARD, "commit:target_lost"), next_state
    if context.balls_count == 0 and context.now <= next_state.hold_command_until:
        return NavigationResult(CMD_FORWARD, "commit:no_ball"), next_state

    return result, next_state