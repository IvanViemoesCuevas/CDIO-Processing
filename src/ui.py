# Drawing and debug windows.
import math

import numpy as np
import cv2 as cv

from models import *

ROBOT_LENGTH_PX = 230
ROBOT_WIDTH_PX = 80

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

    # Mark robot
    if robot_pose is not None:
        draw_robot_footprint(out, robot_pose, ROBOT_LENGTH_PX, ROBOT_WIDTH_PX)
        cv.circle(out, (robot_pose.x, robot_pose.y), 8, (0, 255, 0), -1)
        arrow_len = 45
        x2 = int(robot_pose.x + arrow_len * math.cos(robot_pose.heading_rad))
        y2 = int(robot_pose.y + arrow_len * math.sin(robot_pose.heading_rad))
        cv.arrowedLine(out, (robot_pose.x, robot_pose.y), (x2, y2), (0, 255, 0), 2, tipLength=0.25)
        cv.putText(out, "robot", (robot_pose.x + 10, robot_pose.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add the command info to the screen
    cv.putText(out, f"cmd={command} reason={reason}", (10, 56), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(out, f"last_sent={last_sent_command}", (10, 84), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return out