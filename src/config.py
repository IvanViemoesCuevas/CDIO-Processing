# Constants & settings
from dataclasses import dataclass

FIELD_WIDTH_CM = 178.0
FIELD_HEIGHT_CM = 133.0
PERSPECTIVE_PADDING_PX = 40


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
    obstacle_avoidance_margin_cm: float = 8.0   # Clearance added around the centre obstacle bounding box
    robot_length_cm: float = 43.0
    robot_width_cm: float = 20.0

    # Danger detection
    min_obstacle_area: int = 400
    # max_obstacle_area_fraction: red blobs covering more than this fraction of the
    # frame are treated as field walls, not obstacles, and are excluded from live
    # danger-flag logic.  Keeps the red boundary tape from being treated as a hazard.
    max_obstacle_area_fraction: float = 0.12
    danger_center_deadband_cm: float = 15.0
    danger_distance_cm: float = 10.0       # Trigger danger flags 10 cm from robot body
    danger_too_close_cm: float = 6.0       # Emergency stop/reverse threshold
    danger_rear_ignore_cm: float = 10.0

    # Heading smoothing (EMA) — prevents ArUco jitter from causing left/right oscillation.
    # 1.0 = no smoothing (raw), 0.0 = frozen. Values around 0.6–0.8 work well.
    heading_ema_alpha: float = 0.65

    # Wall clearance used inside find_bypass_waypoint — increase if robot clips walls.
    waypoint_wall_margin_cm: float = 14.0  # Min distance from field edge for planned waypoints

    # Goal detection settings
    min_goal_gap_px: int = 50  # Minimum pixel height of a gap to be considered a goal
    alignment_point_offset_cm: float = 45.0 # Move alignment point further left from the goal line
    delivery_point_offset_cm: float = 28.0 # Move delivery point further left from the goal line


# (Hue, Saturation, brightness)
@dataclass
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class BallDetectionTuning:
    orange_range: HSVRange
    white_range: HSVRange
    white_sat_split: float


ORANGE_RANGE = HSVRange((0, 191, 204), (179, 255, 255))
WHITE_RANGE = HSVRange((0, 0, 213), (64, 107, 255))
RED_RANGE_1 = HSVRange((0, 95, 60), (10, 255, 255))
RED_RANGE_2 = HSVRange((165, 95, 60), (179, 255, 255))


CMD_FORWARD = "i"
CMD_BACKWARD = "k"
CMD_LEFT = "j"
CMD_RIGHT = "l"
CMD_STOP = "s"
CMD_SWITCH = "x"
CMD_QUIT = "q"