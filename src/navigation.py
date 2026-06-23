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
MARKER_PERSPECTIVE_X_GAIN = -85.0
MARKER_PERSPECTIVE_Y_GAIN = -60.0

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

def get_scales(frame_width: int, frame_height: int) -> tuple[float, float]:
    width = frame_width - 2 * PERSPECTIVE_PADDING_PX
    height = frame_height - 2 * PERSPECTIVE_PADDING_PX
    scale_x = width / FIELD_WIDTH_CM
    scale_y = height / FIELD_HEIGHT_CM
    return scale_x, scale_y


def opposite_turn(command: str) -> str:
    if command == CMD_LEFT:
        return CMD_RIGHT
    if command == CMD_RIGHT:
        return CMD_LEFT
    return ""


def decide_immediate_command(
    context: NavigationContext,
    settings: Settings,
    state: NavigationState,
    arrival_distance_cm=None,
    turn_deadband_deg=None
) -> NavigationResult:
    danger = context.danger
    danger_state = context.danger_state
    target_ball = context.target_ball

    # Use specific tolerances for handoff, otherwise use defaults
    arrival_dist = arrival_distance_cm if arrival_distance_cm is not None else settings.pose_arrival_distance_cm
    turn_deadband = turn_deadband_deg if turn_deadband_deg is not None else settings.pose_turn_deadband_deg

    target_heading_error_deg = None
    if context.robot_pose is not None and target_ball is not None:
        robot_x, robot_y, robot_heading, *_ = corrected_robot_pose_values(
            context.robot_pose,
            frame_width=context.frame_width,
            frame_height=context.frame_height,
        )
        dx_to_target = float(target_ball.x - robot_x)
        dy_to_target = float(target_ball.y - robot_y)
        target_heading = math.atan2(dy_to_target, dx_to_target)
        target_heading_error_deg = math.degrees(normalize_angle_rad(target_heading - robot_heading))

    def choose_deviation_turn(reason_prefix: str) -> NavigationResult:
        if danger.left and not danger.right:
            return NavigationResult(CMD_RIGHT, f"{reason_prefix}:avoid_left")
        if danger.right and not danger.left:
            return NavigationResult(CMD_LEFT, f"{reason_prefix}:avoid_right")

        if target_heading_error_deg is not None:
            if target_heading_error_deg < 0.0:
                return NavigationResult(CMD_LEFT, f"{reason_prefix}:target_left err={target_heading_error_deg:.1f}")
            return NavigationResult(CMD_RIGHT, f"{reason_prefix}:target_right err={target_heading_error_deg:.1f}")

        if state.last_command == CMD_LEFT:
            return NavigationResult(CMD_LEFT, f"{reason_prefix}:keep_left")
        if state.last_command == CMD_RIGHT:
            return NavigationResult(CMD_RIGHT, f"{reason_prefix}:keep_right")

        return NavigationResult(CMD_LEFT, f"{reason_prefix}:default_left")

    def target_turn_command() -> str | None:
        if target_ball is None:
            return None

        if context.robot_pose is not None:
            if target_heading_error_deg is None:
                return None
            if target_heading_error_deg < -turn_deadband:
                return CMD_LEFT
            if target_heading_error_deg > turn_deadband:
                return CMD_RIGHT
            return None

        scale_x, _ = get_scales(context.frame_width, context.frame_height)
        align_deadband_px = settings.align_deadband_cm * scale_x
        center_x = context.frame_width // 2
        error_x = target_ball.x - center_x

        if error_x < -align_deadband_px:
            return CMD_LEFT
        if error_x > align_deadband_px:
            return CMD_RIGHT
        return None

    if danger_state is not None and danger_state.too_close:
        if danger_state.nearest_dy_body < -float(settings.danger_rear_ignore_cm):
            return NavigationResult(CMD_FORWARD, f"danger:avoid_back d={danger_state.nearest_distance_cm:.1f}")

        if danger.front and danger.center:
            return NavigationResult(CMD_BACKWARD, f"danger:too_close_front d={danger_state.nearest_distance_cm:.1f}")

        if danger.left and danger.right:
            return NavigationResult(CMD_BACKWARD, f"danger:too_close_sides d={danger_state.nearest_distance_cm:.1f}")

        if danger_state.nearest_dx_body < 0.0:
            return NavigationResult(CMD_RIGHT, f"danger:too_close_left d={danger_state.nearest_distance_cm:.1f}")

        return NavigationResult(CMD_LEFT, f"danger:too_close_right d={danger_state.nearest_distance_cm:.1f}")

    if danger.front and danger.center:
        if danger.left and danger.right:
            return NavigationResult(CMD_BACKWARD, "danger:front_blocked")
        return choose_deviation_turn("danger:front_deviate")

    if danger.back and not danger.front and not danger.left and not danger.right:
        return NavigationResult(CMD_FORWARD, "danger:back")
    if danger.left and not danger.right:
        return NavigationResult(CMD_RIGHT, "danger:left")
    if danger.right and not danger.left:
        return NavigationResult(CMD_LEFT, "danger:right")
    if danger.left and danger.right:
        return NavigationResult(CMD_BACKWARD, "danger:both_sides")

    target_turn = target_turn_command()
    escape_undo_command = state.escape_undo_command or opposite_turn(state.escape_turn_command)
    if (
        state.escape_turn_command in (CMD_LEFT, CMD_RIGHT)
        and (state.escape_turn_until <= 0.0 or context.now <= state.escape_turn_until)
        and target_turn == escape_undo_command
    ):
        turn_name = "left" if state.escape_turn_command == CMD_LEFT else "right"
        return NavigationResult(state.escape_turn_command, f"escape:{turn_name}")

    if target_ball is None:
        return NavigationResult(CMD_STOP, "no_ball")

    if context.robot_pose is not None:
        robot_x, robot_y, robot_heading, *_ = corrected_robot_pose_values(
            context.robot_pose,
            frame_width=context.frame_width,
            frame_height=context.frame_height,
        )

        dx = float(target_ball.x - robot_x)
        dy = float(target_ball.y - robot_y)
        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle_rad(target_heading - robot_heading)
        heading_error_deg = math.degrees(heading_error)

        scale_x, scale_y = get_scales(context.frame_width, context.frame_height)
        dx_cm = dx / scale_x
        dy_cm = dy / scale_y
        distance_cm = math.hypot(dx_cm, dy_cm)

        # Hysteresis for turning
        turn_hysteresis_factor = 1.5
        effective_turn_deadband = turn_deadband
        if state.last_command == CMD_FORWARD:
            effective_turn_deadband *= turn_hysteresis_factor

        if heading_error_deg < -effective_turn_deadband:
            return NavigationResult(CMD_LEFT, f"pose:left err={heading_error_deg:.1f}")
        if heading_error_deg > effective_turn_deadband:
            return NavigationResult(CMD_RIGHT, f"pose:right err={heading_error_deg:.1f}")

        if distance_cm > arrival_dist:
            return NavigationResult(CMD_FORWARD, f"pose:forward d={distance_cm:.1f}")

        return NavigationResult(CMD_STOP, f"pose:arrived d={distance_cm:.1f}")

    else:
        scale_x, scale_y = get_scales(context.frame_width, context.frame_height)
        align_deadband_px = settings.align_deadband_cm * scale_x
        target_radius_px = settings.target_radius_cm * scale_x

        center_x = context.frame_width // 2
        error_x = target_ball.x - center_x

        if error_x < -align_deadband_px:
            return NavigationResult(CMD_LEFT, f"track:{target_ball.color_name}:left")
        if error_x > align_deadband_px:
            return NavigationResult(CMD_RIGHT, f"track:{target_ball.color_name}:right")
        if target_ball.radius < target_radius_px:
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
            alignment_target = BallDetection(
                x=context.small_goal.alignment_point_x,
                y=context.small_goal.alignment_point_y,
                radius=settings.pose_arrival_distance_cm,
                color_name="align_pt",
                confidence=1.0
            )
            nav_context = replace(context, target_ball=alignment_target)
            res = decide_immediate_command(nav_context, settings, state, arrival_distance_cm=9.0) # Tighter arrival
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
            res = decide_immediate_command(nav_context, settings, state, turn_deadband_deg=3) # Tighter alignment

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

            delivery_target = BallDetection(
                x=context.small_goal.delivery_point_x,
                y=context.small_goal.delivery_point_y,
                radius=settings.pose_arrival_distance_cm,
                color_name="delivery_pt",
                confidence=1.0
            )
            nav_context = replace(context, target_ball=delivery_target)
            res = decide_immediate_command(nav_context, settings, state, arrival_distance_cm=9.0) # Tighter arrival

            if "arrived" in res.reason:
                state.handoff_phase = "final_aligning"
                state.hold_command_until = context.now + 1.0 # Pause before final alignment
                print("Handoff: Arrived at delivery point. Phase -> final_aligning")
                return NavigationResult(CMD_STOP, "handoff:arrived_delivery"), state
            return res, state

        # --- Phase: final_aligning ---
        if state.handoff_phase == "final_aligning":
            if state.hold_command_until > context.now:
                return NavigationResult(CMD_STOP, "handoff:pausing_final_align"), state

            goal_target = BallDetection(
                x=context.small_goal.x,
                y=context.small_goal.y,
                radius=0,
                color_name="goal",
                confidence=1.0
            )
            nav_context = replace(context, target_ball=goal_target)
            res = decide_immediate_command(nav_context, settings, state, turn_deadband_deg=3)

            if res.command not in [CMD_LEFT, CMD_RIGHT]:
                state.handoff_phase = "starting_unload"
                state.hold_command_until = context.now + 10.0 # Unload time
                print("Handoff: Final alignment complete. Phase -> starting_unload")
                return NavigationResult(CMD_STOP, "handoff:ready_unload"), state
            return res, state

        # --- Phase: starting_unload ---
        if state.handoff_phase == "starting_unload":
            state.handoff_phase = "delivering"
            print("Handoff: Starting unload. Phase -> delivering")
            return NavigationResult(CMD_SWITCH, "handoff:start_unload"), state

        # --- Phase: delivering ---
        if state.handoff_phase == "delivering":
            if state.hold_command_until > context.now:
                return NavigationResult(CMD_STOP, "handoff:waiting_unload"), state
            state.handoff_phase = "done"
            print("Handoff: Delivery complete. Phase -> done")
            return NavigationResult(CMD_STOP, "handoff:done_stop"), state

        # --- Phase: done ---
        if state.handoff_phase == "done":
            return NavigationResult(CMD_STOP, "handoff:done_stop"), state

    # Regular ball-following logic
    arrival_dist = None
    if context.target_ball is not None and context.target_ball.color_name == "waypoint":
        arrival_dist = getattr(settings, 'waypoint_arrival_distance_cm', 8.0)
    immediate = decide_immediate_command(context, settings, state, arrival_distance_cm=arrival_dist)
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
        last_command=result.command,
        avoidance_turn_command=state.avoidance_turn_command,
        avoidance_turn_count=state.avoidance_turn_count,
        escape_turn_command=state.escape_turn_command,
        escape_undo_command=state.escape_undo_command,
        escape_turn_until=state.escape_turn_until,
    )

    if result.command == CMD_FORWARD:
        next_state.avoidance_turn_command = ""
        next_state.avoidance_turn_count = 0
        next_state.escape_turn_command = ""
        next_state.escape_undo_command = ""
        next_state.escape_turn_until = 0.0
    elif result.reason.startswith("danger") and result.command in (CMD_LEFT, CMD_RIGHT):
        if state.avoidance_turn_command == result.command:
            next_state.avoidance_turn_count = state.avoidance_turn_count + 1
        else:
            next_state.avoidance_turn_command = result.command
            next_state.avoidance_turn_count = 1

        if next_state.avoidance_turn_count >= settings.avoidance_escape_repeats:
            next_state.escape_turn_command = result.command
            next_state.escape_undo_command = opposite_turn(result.command)
            next_state.escape_turn_until = 0.0
            next_state.avoidance_turn_count = 0
    elif result.reason.startswith("escape"):
        next_state.avoidance_turn_command = result.command
        if next_state.escape_turn_until <= 0.0:
            next_state.escape_turn_until = context.now + settings.avoidance_escape_max_sec
    else:
        next_state.escape_turn_command = ""
        next_state.escape_undo_command = ""
        next_state.escape_turn_until = 0.0

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
