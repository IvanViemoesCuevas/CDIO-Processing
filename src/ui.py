# Drawing and debug windows.
import math

import numpy as np
import cv2 as cv

from models import *

ROBOT_LENGTH_PX = 170
ROBOT_WIDTH_PX = 80

from navigation import (
    corrected_robot_pose_values,
    normalize_angle_rad,
    MARKER_HEADING_OFFSET_DEG,
    MARKER_TO_DRIVE_CENTER_FORWARD_PX,
    MARKER_TO_DRIVE_CENTER_RIGHT_PX,
    MARKER_PERSPECTIVE_FORWARD_GAIN,
    MARKER_PERSPECTIVE_RIGHT_GAIN,
)


# corrected_robot_pose_values and normalize_angle_rad are imported from navigation.py

def draw_robot_footprint(frame: np.ndarray, robot_pose: RobotPose, length_px: float, width_px: float) -> None:
    length = max(10.0, float(length_px))
    width = max(10.0, float(width_px))
    angle_deg = math.degrees(robot_pose.heading_rad)
    rect = ((float(robot_pose.x), float(robot_pose.y)), (length, width), angle_deg)
    box = cv.boxPoints(rect).astype(np.int32)
    cv.polylines(frame, [box], True, (0, 255, 255), 2)

# FIXME - Doesn't draw danger zones, but they are there otherwise
def annotate(
        frame: np.ndarray,
        command: str,
        reason: str,
        last_sent_command: Optional[str],
        balls: list[BallDetection],
        target_ball: Optional[BallDetection],
        robot_pose: Optional[RobotPose],
        #settings: Optional[Settings] = None,
) -> np.ndarray:
    out = frame.copy()

    # Mark the detected balls
    for b in balls:
        ball_outline_color = (225, 225, 225) if b.color_name == "white" else (80, 120, 255)
        cv.circle(out, (b.x, b.y), int(b.radius), ball_outline_color, 1)
        label = (
            f"{b.color_name} "
            f"conf={b.confidence:.2f} "
            f"circ={b.circularity:.2f} "
            f"r={b.radius:.1f}"
        )
        text_pos = (b.x - 36, max(14, b.y - int(b.radius) - 8))
        cv.putText(out, label, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.45, ball_outline_color, 2)

    # Mark target ball
    if target_ball is not None:
        color = (0, 255, 0)
        cv.circle(out, (target_ball.x, target_ball.y), int(target_ball.radius), color, 2)
        cv.circle(out, (target_ball.x, target_ball.y), 3, color, -1)
        cv.putText(
            out,
            f"target_ball={target_ball.color_name} conf={target_ball.confidence:.2f}",
            (10, 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Mark robot and navigation debug info
    debug_lines: list[str] = []
    if robot_pose is not None:
        draw_robot_footprint(out, robot_pose, ROBOT_LENGTH_PX, ROBOT_WIDTH_PX)
        cv.circle(out, (robot_pose.x, robot_pose.y), 8, (0, 255, 0), -1)
        marker_point = (robot_pose.x, robot_pose.y)

        arrow_len = 45
        heading_x = math.cos(robot_pose.heading_rad)
        heading_y = math.sin(robot_pose.heading_rad)
        x2 = int(robot_pose.x + arrow_len * heading_x)
        y2 = int(robot_pose.y + arrow_len * heading_y)
        cv.arrowedLine(out, (robot_pose.x, robot_pose.y), (x2, y2), (0, 255, 0), 2, tipLength=0.25)
        cv.putText(out, "robot", (robot_pose.x + 10, robot_pose.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw corrected drive center and heading
        (
            robot_x,
            robot_y,
            robot_heading,
            used_forward_px,
            used_right_px,
            offset_x,
            offset_y,
            normalized_x,
            normalized_y,
        ) = corrected_robot_pose_values(
            robot_pose,
            frame_width=out.shape[1],
            frame_height=out.shape[0],
        )
        robot_point = (int(round(robot_x)), int(round(robot_y)))

        cv.line(out, marker_point, robot_point, (255, 255, 0), 2)
        cv.circle(out, robot_point, 7, (255, 255, 0), -1)
        cv.putText(out, "raw marker", (marker_point[0] + 10, marker_point[1] + 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(out, "corrected ground point", (robot_point[0] + 10, robot_point[1] + 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw the robot-local axes used for the correction.
        # Blue = robot-local forward, red = robot-local right.
        local_axis_len = 45
        local_forward_end = (
            int(robot_pose.x + local_axis_len * math.cos(robot_heading)),
            int(robot_pose.y + local_axis_len * math.sin(robot_heading)),
        )
        local_right_end = (
            int(robot_pose.x + local_axis_len * math.cos(robot_heading + math.pi / 2.0)),
            int(robot_pose.y + local_axis_len * math.sin(robot_heading + math.pi / 2.0)),
        )
        cv.arrowedLine(out, marker_point, local_forward_end, (255, 0, 0), 2, tipLength=0.25)
        cv.arrowedLine(out, marker_point, local_right_end, (0, 0, 255), 2, tipLength=0.25)
        cv.putText(out, "local forward", (local_forward_end[0] + 5, local_forward_end[1]), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv.putText(out, "local right", (local_right_end[0] + 5, local_right_end[1]), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        heading_deg = math.degrees(robot_pose.heading_rad)
        corrected_heading_deg = math.degrees(robot_heading)
        debug_lines.append(f"marker=({robot_pose.x},{robot_pose.y}) heading={heading_deg:.1f}deg")
        debug_lines.append(f"drive=({robot_x:.0f},{robot_y:.0f}) heading={corrected_heading_deg:.1f}deg")

        if target_ball is not None:
            dx = float(target_ball.x - robot_x)
            dy = float(target_ball.y - robot_y)
            target_heading = math.atan2(dy, dx)
            heading_error = normalize_angle_rad(target_heading - robot_heading)
            heading_error_deg = math.degrees(heading_error)
            distance_px = math.hypot(dx, dy)

            cv.line(out, robot_point, (target_ball.x, target_ball.y), (255, 0, 255), 2)
            cv.circle(out, (target_ball.x, target_ball.y), 5, (255, 0, 255), -1)

            target_arrow_len = min(80.0, max(25.0, distance_px * 0.35))
            tx2 = int(robot_x + target_arrow_len * math.cos(target_heading))
            ty2 = int(robot_y + target_arrow_len * math.sin(target_heading))
            cv.arrowedLine(out, robot_point, (tx2, ty2), (255, 0, 255), 2, tipLength=0.25)

            debug_lines.append(f"target=({target_ball.x},{target_ball.y}) dx={dx:.0f} dy={dy:.0f} d={distance_px:.0f}px")
            debug_lines.append(f"target_heading={math.degrees(target_heading):.1f}deg err={heading_error_deg:.1f}deg")
            debug_lines.append(
                f"calib heading_offset={MARKER_HEADING_OFFSET_DEG:.1f}deg base_fwd={MARKER_TO_DRIVE_CENTER_FORWARD_PX:.0f}px base_right={MARKER_TO_DRIVE_CENTER_RIGHT_PX:.0f}px"
            )
            debug_lines.append(
                f"dynamic local used_fwd={used_forward_px:.0f}px used_right={used_right_px:.0f}px gains fwd={MARKER_PERSPECTIVE_FORWARD_GAIN:.1f} right={MARKER_PERSPECTIVE_RIGHT_GAIN:.1f}"
            )
            debug_lines.append(
                f"rotated image offset=({offset_x:.0f},{offset_y:.0f}) norm_pos=({normalized_x:.2f},{normalized_y:.2f})"
            )
            debug_lines.append("TUNE ORDER: 1) tune base_fwd/base_right near image center")
            debug_lines.append("TUNE ORDER: 2) gains change magnitude; blue/red axes show rotated direction")
    else:
        debug_lines.append("robot_pose=None: using image-center tracking fallback")
        if target_ball is not None:
            center_x = out.shape[1] // 2
            error_x = target_ball.x - center_x
            cv.line(out, (center_x, 0), (center_x, out.shape[0]), (255, 255, 0), 1)
            cv.line(out, (center_x, target_ball.y), (target_ball.x, target_ball.y), (255, 0, 255), 2)
            debug_lines.append(f"target=({target_ball.x},{target_ball.y}) center_x={center_x} error_x={error_x}")
            #if settings is not None:
            #    debug_lines.append(f"align_deadband={settings.align_deadband_px}px target_radius={settings.target_radius_px}px")

    # Add the command info to the screen
    cv.putText(out, f"cmd={command} reason={reason}", (10, 56), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(out, f"last_sent={last_sent_command}", (10, 84), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Add navigation debug text to the screen
    debug_y = 112
    for line in debug_lines:
        cv.putText(out, line, (10, debug_y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        debug_y += 24

    return out