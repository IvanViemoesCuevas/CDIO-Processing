from dataclasses import dataclass

@dataclass
class Settings:
    host: str = "172.20.10.9"
    port: int = 12345
    send_interval_sec: float = 0.12
    stable_frames_required: int = 2
    reconnect_delay_sec: float = 1.0

    # Only required for HSV
    min_ball_area: int = 140
    max_ball_area: int = 8000
    min_ball_circularity: float = 0.0
    min_ball_radius: float = 10.0
    max_ball_radius: float = 20.0
    min_ball_confidence: float = 0.4
    white_sat_split: float = 80.0

    # Aligning and arrival of the robot
    align_deadband_cm: float = 8.0
    target_radius_cm: float = 2.0
    commit_forward_window_sec: float = 1.0  # TODO tweak
    pose_turn_deadband_deg: float = 5.0
    pose_arrival_distance_cm: float = 20.0
    waypoint_arrival_distance_cm: float = 8.0
    obstacle_avoidance_margin_cm: float = 15.0
    target_missing_frames_limit: int = 30

    # Danger detection
    min_obstacle_area: int = 400
    danger_center_deadband_cm: float = 0.0
    danger_distance_cm: float = 2.0
    danger_too_close_cm: float = 2.0
    danger_rear_ignore_cm: float = 0.0
    avoidance_escape_repeats: int = 2
    avoidance_escape_max_sec: float = 20.0

    # Robot footprint used for danger detection.
    robot_length_cm: float = 45.0
    robot_width_cm: float = 18.0
    robot_danger_margin_cm: float = 0.0
    robot_front_extra_margin_cm: float = 0.5
    robot_side_extra_margin_cm: float = 0.0

    # Goal detection settings
    min_goal_gap_px: int = 50  # Minimum pixel height of a gap to be considered a goal
    alignment_point_offset_cm: float = 45.0 # Move alignment point further left from the goal line
    delivery_point_offset_cm: float = 25.0 # Move delivery point further left from the goal line
