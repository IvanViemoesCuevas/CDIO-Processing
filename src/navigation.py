# All decision logic here
import math

from config import *
from models import (
    NavigationContext,
    NavigationResult,
    NavigationState,
)

def decide_immediate_command(context: NavigationContext, settings: Settings) -> NavigationResult:
    # FIXME - Acting on danger is a bit wonky

    danger = context.danger
    danger_state = context.danger_state
    target_ball = context.target_ball

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
        # FIXME - No target visible: stop
        return NavigationResult(CMD_STOP, "no_ball")

    if context.robot_pose is not None:
        # setup vector from robot to target
        dx = float(target_ball.x - context.robot_pose.x)
        dy = float(target_ball.y - context.robot_pose.y)
        target_heading = math.atan2(dy, dx)

        # Compute angle difference between robot and target
        heading_error = math.atan2(
            math.sin(target_heading - context.robot_pose.heading_rad),
            math.cos(target_heading - context.robot_pose.heading_rad),
        )
        heading_error_deg = math.degrees(heading_error)

        # Compute distance from robot to target (straight line)
        distance_px = math.hypot(dx, dy)

        # If target is sufficiently to one side (outside deadband), turn first.
        if heading_error_deg < -settings.pose_turn_deadband_deg:
            return NavigationResult(CMD_LEFT, f"pose:left err={heading_error_deg:.1f}")
        if heading_error_deg > settings.pose_turn_deadband_deg:
            return NavigationResult(CMD_RIGHT, f"pose:right err={heading_error_deg:.1f}")

        # If heading is acceptable and still far away, drive forward.
        if distance_px > settings.pose_arrival_distance_px:
            return NavigationResult(CMD_FORWARD, f"pose:forward d={distance_px:.0f}")

        # If close enough, stop (arrived).
        return NavigationResult(CMD_STOP, f"pose:arrived d={distance_px:.0f}")

    else: # Only use this if we can't find the robot (recognizes it as the robot having the camero on it)
        # Calculate horizontal error from center of frame to target ball
        center_x = context.frame_width // 2
        error_x = target_ball.x - center_x

        # Too far right of image center, turn left
        if error_x < -settings.align_deadband_px:
            return NavigationResult(CMD_LEFT, f"track:{target_ball.color_name}:left")
        # Too far left of image center, turn right
        if error_x > settings.align_deadband_px:
            return NavigationResult(CMD_RIGHT, f"track:{target_ball.color_name}:right")
        # Use radius as distance proxy (small radius = far away -> go forward)
        if target_ball.radius < settings.target_radius_px:
            return NavigationResult(CMD_FORWARD, f"track:{target_ball.color_name}:forward")
        # Otherwise mark as arrived
        return NavigationResult(CMD_STOP, f"track:{target_ball.color_name}:arrived")




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
    )

    # If danger, commit to the command until it's no longer dangerous (don't override with new commands)
    if result.reason.startswith("danger"):
        return result, next_state

    # When the robot reaches a target, it stores that ball as committed and changes command to forward.
    # This helps when the ball disappears under/behind the robot due to camera angle.
    if result.reason.endswith(":arrived") and context.target_ball is not None:
        next_state.candidate_target = context.target_ball
        return NavigationResult(CMD_FORWARD, f"commit:{result.reason}"), next_state

    # If there is no committed target, or it is not yet time to recompute, keep result.
    if next_state.candidate_target is None or not (next_state.hold_command_until < context.now):
        return result, next_state

    # Recompute: Define “continue-forward-until” based on last target seen time + configured window.
    next_state.hold_command_until = (
        next_state.last_target_seen_time + settings.commit_forward_window_sec
    )

    # There are balls, but not the committed one -> keep pushing forward briefly.
    if context.balls_count > 0 and not context.candidate_target_visible:
        return NavigationResult(CMD_FORWARD, "commit:target_lost"), next_state
    # no balls at all, but still inside hold window -> keep pushing forward briefly.
    if context.balls_count == 0 and context.now <= next_state.hold_command_until:
        return NavigationResult(CMD_FORWARD, "commit:no_ball"), next_state

    # Otherwise fall back to original immediate result (which may be to stop or turn or whatever)
    return result, next_state


def decide_command(
    context: NavigationContext,
    state: NavigationState,
    settings: Settings,
) -> tuple[NavigationResult, NavigationState]:
    immediate = decide_immediate_command(context, settings)
    return apply_commit_transitions(immediate, context, state, settings)
