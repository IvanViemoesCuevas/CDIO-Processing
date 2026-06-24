import math
import numpy as np
import cv2 as cv
from typing import Optional

from models import (
    BallDetection,
    RobotPose,
    DangerFlags,
    DangerState,
    GoalDetection,
    FieldCorners,
)
from config import Settings, PERSPECTIVE_PADDING_PX
from utils.geometry import (
    corrected_robot_pose_values,
    normalize_angle_rad,
    get_scales,
)
from navigation import (
    MARKER_HEADING_OFFSET_DEG,
    MARKER_TO_DRIVE_CENTER_FORWARD_PX,
    MARKER_TO_DRIVE_CENTER_RIGHT_PX,
    MARKER_PERSPECTIVE_FORWARD_GAIN,
    MARKER_PERSPECTIVE_RIGHT_GAIN,
)
from vision import find_danger_perspective_points, is_ball_in_danger_zone

ROBOT_LENGTH_CM = Settings.robot_length_cm
ROBOT_WIDTH_CM = Settings.robot_width_cm


def draw_robot_footprint(frame: np.ndarray, x: float, y: float, heading_rad: float, length_cm: float, width_cm: float) -> None:
    length_cm = max(1.0, float(length_cm))
    width_cm = max(1.0, float(width_cm))
    half_length_cm = length_cm / 2.0
    half_width_cm = width_cm / 2.0

    scale_x, scale_y = get_scales(frame.shape[1], frame.shape[0])

    heading_x = math.cos(heading_rad)
    heading_y = math.sin(heading_rad)
    right_x = -heading_y
    right_y = heading_x

    corners_cm = [
        (half_length_cm, -half_width_cm),
        (half_length_cm, half_width_cm),
        (-half_length_cm, half_width_cm),
        (-half_length_cm, -half_width_cm),
    ]

    corners_px = []
    for forward_cm, right_cm in corners_cm:
        dx_cm = forward_cm * heading_x + right_cm * right_x
        dy_cm = forward_cm * heading_y + right_cm * right_y
        corners_px.append((
            int(round(x + dx_cm * scale_x)),
            int(round(y + dy_cm * scale_y)),
        ))

    box = np.asarray(corners_px, dtype=np.int32)
    cv.polylines(frame, [box], True, (0, 255, 255), 2)


def annotate(
        frame: np.ndarray,
        command: str,
        reason: str,
        last_sent_command: Optional[str],
        balls: list[BallDetection],
        target_ball: Optional[BallDetection],
        robot_pose: Optional[RobotPose],
        danger: Optional[DangerFlags] = None,
        danger_state: Optional[DangerState] = None,
        danger_contours: Optional[list] = None,
        field_corners: Optional[FieldCorners] = None,
        small_goal: Optional[GoalDetection] = None,
        route_manager: Optional[object] = None,
) -> np.ndarray:
    out = frame.copy()

    corners = find_danger_perspective_points(frame)

    if corners is not None:
        for i, (x, y) in enumerate(corners):
            cv.circle(out, (int(x), int(y)), 15, (255, 0, 255), -1)
            cv.putText(
                out,
                str(i),
                (int(x) + 20, int(y)),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )

    scale_x, scale_y = get_scales(out.shape[1], out.shape[0])

    # Mark the detected balls
    for b in balls:
        is_danger = False
        if danger_contours is not None:
            is_danger = is_ball_in_danger_zone(b, danger_contours, scale_x, scale_y)

        if is_danger:
            ball_outline_color = (0, 0, 255)  # Red for danger
        else:
            ball_outline_color = (225, 225, 225) if b.color_name == "white" else (80, 120, 255)

        cv.circle(out, (b.x, b.y), int(b.radius), ball_outline_color, 1)

        status_suffix = " (danger)" if is_danger else ""
        label = (
            f"{b.color_name}{status_suffix} "
            f"conf={b.confidence:.2f} "
            f"r={b.radius:.1f}"
        )
        text_pos = (b.x - 36, max(14, b.y - int(b.radius) - 8))
        cv.putText(out, label, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.45, ball_outline_color, 2)

    # Draw Route Manager overlays (Queue path, numbers, visited marks)
    if route_manager is not None:
        # 1. Draw visited positions
        for vx, vy in route_manager.visited_positions:
            px_x = int(round(PERSPECTIVE_PADDING_PX + vx * scale_x))
            px_y = int(round(PERSPECTIVE_PADDING_PX + vy * scale_y))
            cv.drawMarker(out, (px_x, px_y), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
            cv.putText(out, "tried", (px_x + 8, px_y + 4), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 2. Draw queue trajectory lines
        if route_manager.queue:
            pts = []
            if robot_pose is not None:
                (rx, ry, *_) = corrected_robot_pose_values(robot_pose, frame_width=out.shape[1], frame_height=out.shape[0])
                pts.append((int(round(rx)), int(round(ry))))
            for qb in route_manager.queue:
                pts.append((qb.x, qb.y))

            for i in range(len(pts) - 1):
                cv.line(out, pts[i], pts[i+1], (255, 255, 0), 2, cv.LINE_AA)

            # 3. Draw queue order numbers
            for idx, qb in enumerate(route_manager.queue):
                if qb.color_name == "waypoint":
                    cv.drawMarker(out, (qb.x, qb.y), (0, 255, 255), markerType=cv.MARKER_DIAMOND, markerSize=14, thickness=2)
                    cv.putText(out, f"WP#{idx+1}", (qb.x - 18, qb.y - 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                else:
                    cv.circle(out, (qb.x, qb.y), int(qb.radius) + 5, (0, 255, 255), 2)
                    cv.putText(out, f"#{idx+1}", (qb.x - 12, qb.y + 4), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    # Mark target ball
    if target_ball is not None:
        color = (0, 255, 0)
        if target_ball.color_name == "waypoint":
            cv.drawMarker(out, (target_ball.x, target_ball.y), color, markerType=cv.MARKER_DIAMOND, markerSize=20, thickness=2)
            cv.putText(
                out,
                f"target=WAYPOINT",
                (10, 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            cv.circle(out, (target_ball.x, target_ball.y), int(target_ball.radius) + 2, color, 2)
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

    # Mark field corners (this is the green overlay)
    if field_corners is not None:
        corners_list = [
            field_corners.topLeft,
            field_corners.topRight,
            field_corners.bottomRight, # Note: ordering for polylines drawing
            field_corners.bottomLeft
        ]
        # Draw field boundary
        pts = np.array(corners_list, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(out, [pts], True, (0, 255, 0), 2)

        # Draw corners
        for i, pt in enumerate(corners_list):
            cv.circle(out, pt, 5, (0, 255, 0), -1)
            cv.putText(out, f"C{i+1}", (pt[0]+10, pt[1]+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mark the small goal and its approach/delivery points
    if small_goal is not None:
        # Draw the true goal center
        goal_color = (255, 0, 255)  # Magenta for the true goal
        cv.circle(out, (small_goal.x, small_goal.y), 15, goal_color, 2)
        cv.circle(out, (small_goal.x, small_goal.y), 4, goal_color, -1)
        cv.putText(
            out,
            f"Small Goal ({small_goal.x},{small_goal.y})",
            (small_goal.x - 170, small_goal.y - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            goal_color,
            2,
        )

        alignment_point = None
        delivery_point = None

        # Draw the alignment point
        if small_goal.alignment_point_x is not None and small_goal.alignment_point_y is not None:
            alignment_point = (small_goal.alignment_point_x, small_goal.alignment_point_y)
            alignment_color = (0, 255, 0)  # Green for the alignment point
            cv.circle(out, alignment_point, 10, alignment_color, -1)
            cv.putText(
                out,
                f"Alignment ({alignment_point[0]},{alignment_point[1]})",
                (alignment_point[0] - 120, alignment_point[1] + 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                alignment_color,
                2,
            )
            cv.line(out, alignment_point, (small_goal.x, small_goal.y), (255, 255, 255), 1)

        # Draw the delivery point
        if small_goal.delivery_point_x is not None and small_goal.delivery_point_y is not None:
            delivery_point = (small_goal.delivery_point_x, small_goal.delivery_point_y)
            delivery_color = (255, 0, 128)  # Purple for the delivery point
            cv.circle(out, delivery_point, 10, delivery_color, -1)
            cv.putText(
                out,
                f"Delivery ({delivery_point[0]},{delivery_point[1]})",
                (delivery_point[0] - 110, delivery_point[1] + 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                delivery_color,
                2,
            )

        if alignment_point is not None and delivery_point is not None:
            cv.line(out, alignment_point, delivery_point, (255, 255, 255), 1)
            cv.arrowedLine(out, delivery_point, (small_goal.x, small_goal.y), (255, 255, 255), 1, tipLength=0.25)

    else:
        cv.putText(out, "Small Goal: NOT DETECTED", (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Mark robot and navigation debug info
    debug_lines: list[str] = []
    if route_manager is not None:
        debug_lines.append(f"route_state={route_manager.state.upper()} queue_len={len(route_manager.queue)} visited={len(route_manager.visited_positions)}")
    if robot_pose is not None:
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

        # Draw robot footprint at corrected location and heading
        draw_robot_footprint(out, robot_x, robot_y, robot_heading, ROBOT_LENGTH_CM, ROBOT_WIDTH_CM)

        # Draw marker point and correction line
        cv.circle(out, (robot_pose.x, robot_pose.y), 8, (0, 255, 0), -1)
        marker_point = (robot_pose.x, robot_pose.y)

        arrow_len = 45
        heading_x = math.cos(robot_pose.heading_rad)
        heading_y = math.sin(robot_pose.heading_rad)
        x2 = int(robot_pose.x + arrow_len * heading_x)
        y2 = int(robot_pose.y + arrow_len * heading_y)
        cv.arrowedLine(out, (robot_pose.x, robot_pose.y), (x2, y2), (0, 255, 0), 2, tipLength=0.25)

        cv.line(out, marker_point, robot_point, (255, 255, 0), 2)
        cv.circle(out, robot_point, 7, (255, 255, 0), -1)

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

            scale_x, scale_y = get_scales(out.shape[1], out.shape[0])
            dx_cm = dx / scale_x
            dy_cm = dy / scale_y
            distance_cm = math.hypot(dx_cm, dy_cm)

            cv.line(out, robot_point, (target_ball.x, target_ball.y), (255, 0, 255), 2)
            cv.circle(out, (target_ball.x, target_ball.y), 5, (255, 0, 255), -1)

            target_arrow_len = min(80.0, max(25.0, distance_px * 0.35))
            tx2 = int(robot_x + target_arrow_len * math.cos(target_heading))
            ty2 = int(robot_y + target_arrow_len * math.sin(target_heading))
            cv.arrowedLine(out, robot_point, (tx2, ty2), (255, 0, 255), 2, tipLength=0.25)

            debug_lines.append(f"target=({target_ball.x},{target_ball.y}) dx={dx_cm:.1f} dy={dy_cm:.1f} d={distance_cm:.1f}cm")
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
    else:
        debug_lines.append("robot_pose=None: using image-center tracking fallback")
        if target_ball is not None:
            center_x = out.shape[1] // 2
            error_x = target_ball.x - center_x
            cv.line(out, (center_x, 0), (center_x, out.shape[0]), (255, 255, 0), 1)
            cv.line(out, (center_x, target_ball.y), (target_ball.x, target_ball.y), (255, 0, 255), 2)
            debug_lines.append(f"target=({target_ball.x},{target_ball.y}) center_x={center_x} error_x={error_x}")

    # Draw danger zones
    if danger_contours is not None:
        cv.drawContours(out, danger_contours, -1, (0, 0, 255), 2)

    if danger is not None:
        danger_status = []
        if danger.front:
            danger_status.append("FRONT")
        if danger.back:
            danger_status.append("BACK")
        if danger.left:
            danger_status.append("LEFT")
        if danger.center:
            danger_status.append("CENTER")
        if danger.right:
            danger_status.append("RIGHT")

        if danger_status:
            status_text = "DANGER: " + " | ".join(danger_status)
            cv.putText(out, status_text, (10, out.shape[0] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if danger_state is not None and danger_state.nearest_point is not None:
        cv.circle(out, danger_state.nearest_point, 5, (0, 165, 255), -1)
        danger_text = f"nearest_danger={danger_state.nearest_distance_cm:.1f}cm"
        cv.putText(out, danger_text, (10, out.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Add the command info to the screen
    cv.putText(out, f"cmd={command} reason={reason}", (10, 56), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv.putText(out, f"last_sent={last_sent_command}", (10, 84), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Add navigation debug text to the screen
    debug_y = 112
    for line in debug_lines:
        cv.putText(out, line, (10, debug_y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        debug_y += 24

    return out
