import math
from dataclasses import replace

from config import (
    CMD_LEFT,
    CMD_RIGHT,
    CMD_FORWARD,
    CMD_BACKWARD,
    CMD_STOP,
    CMD_SWITCH,
    Settings,
)
from models import (
    NavigationContext,
    NavigationResult,
    NavigationState,
    BallDetection,
)
from utils.geometry import (
    corrected_robot_pose_values,
    normalize_angle_rad,
    get_scales,
)


def decide_immediate_command(
    context: NavigationContext,
    settings: Settings,
    state: NavigationState,
    arrival_distance_cm=None,
    turn_deadband_deg=None,
    ignore_danger=False,
) -> NavigationResult:
    danger = context.danger
    danger_state = context.danger_state
    target_ball = context.target_ball

    print(
        "[nav danger]",
        f"front={danger.front}",
        f"back={danger.back}",
        f"left={danger.left}",
        f"right={danger.right}",
        f"center={danger.center}",
        f"too_close={danger_state.too_close if danger_state else None}",
    )

    def alt(cmd_a, cmd_b, reason):
        if state.last_command == cmd_a:
            return NavigationResult(cmd_b, reason + ":b")
        if state.last_command == cmd_b:
            return NavigationResult(cmd_a, reason + ":a")

        return NavigationResult(cmd_a, reason + ":a")

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

    if not ignore_danger:
        if danger.back and danger.left:
            return alt(CMD_RIGHT, CMD_FORWARD, "danger:back_left")
        if danger.back and danger.right:
            return alt(CMD_LEFT, CMD_FORWARD, "danger:back_right")
        if danger.front and danger.left:
            return alt(CMD_RIGHT, CMD_BACKWARD, "danger:front_left")
        if danger.front and danger.right:
            return alt(CMD_LEFT, CMD_BACKWARD, "danger:front_right")


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

        # --- Phase: approaching_delivery --- TODO Stop checking danger after reaching this phase
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
            res = decide_immediate_command(nav_context, settings, state, turn_deadband_deg=3, ignore_danger=True)

            if res.command not in [CMD_LEFT, CMD_RIGHT]:
                state.handoff_phase = "starting_unload"
                state.hold_command_until = context.now + 5.0 # Unload time
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
            return NavigationResult(CMD_SWITCH, "handoff:done_stop"), state

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
