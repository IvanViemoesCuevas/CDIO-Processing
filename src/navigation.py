# All decision logic here
import math

from config import *
from models import *

def decide_command(
    frame_width: int,
    target_ball: Optional[BallDetection],
    danger: DangerFlags,
    settings: Settings,
    robot_pose: Optional[RobotPose] = None,
    danger_state: Optional[DangerState] = None,
) -> tuple[str, str]:
    # FIXME acting on danger is a bit wonky
    if danger_state is not None and danger_state.too_close:
        if abs(danger_state.nearest_dx_body) <= float(settings.danger_center_deadband_px):
            return CMD_STOP, f"danger:too_close d={danger_state.nearest_distance_px:.0f}"
        if danger_state.nearest_dy_body < -float(settings.danger_rear_ignore_px):
            return CMD_FORWARD, f"danger:avoid_back d={danger_state.nearest_distance_px:.0f}"
        if danger_state.nearest_dx_body < 0.0:
            return CMD_RIGHT, f"danger:avoid_left d={danger_state.nearest_distance_px:.0f}"
        return CMD_LEFT, f"danger:avoid_right d={danger_state.nearest_distance_px:.0f}"

    if danger.front and danger.center:
        return CMD_BACKWARD, "danger:front"
    if danger.back and not danger.front and not danger.left and not danger.right:
        return CMD_FORWARD, "danger:back"
    if danger.left and not danger.right:
        return CMD_RIGHT, "danger:left"
    if danger.right and not danger.left:
        return CMD_LEFT, "danger:right"
    if danger.left and danger.right:
        return CMD_STOP, "danger:both"

    if target_ball is None:
        # FIXME - No target visible: stop
        return CMD_STOP, "no_ball"

    # TODO - Check what this does (Seems to take the robots heading error into consideration)
    if robot_pose is not None:
        dx = float(target_ball.x - robot_pose.x)
        dy = float(target_ball.y - robot_pose.y)
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

    # TODO - Check what this actually does
    center_x = frame_width // 2
    error_x = target_ball.x - center_x
    if error_x < -settings.align_deadband_px:
        return CMD_LEFT, f"track:{target_ball.color_name}:left"
    if error_x > settings.align_deadband_px:
        return CMD_RIGHT, f"track:{target_ball.color_name}:right"
    if target_ball.radius < settings.target_radius_px:
        return CMD_FORWARD, f"track:{target_ball.color_name}:forward"
    return CMD_STOP, f"track:{target_ball.color_name}:arrived"