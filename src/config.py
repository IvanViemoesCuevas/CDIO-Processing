# Constants & settings
from dataclasses import dataclass

@dataclass
class Settings:
    host: str = "172.20.10.2"
    port: int = 12345
    send_interval_sec: float = 0.12
    stable_frames_required: int = 2
    reconnect_delay_sec: float = 1.0

    min_ball_area: int = 140
    max_ball_area: int = 8000
    min_ball_circularity: float = 0.65
    min_ball_radius: float = 10.0
    max_ball_radius: float = 70.0
    min_ball_confidence: float = 0.5

    align_deadband_px: int = 35
    target_radius_px: float = 38.0
    commit_forward_window_sec: float = 3  # TODO tweak

    min_obstacle_area: int = 400
    danger_center_deadband_px: int = 60
    danger_distance_px: float = 35.0
    danger_too_close_px: float = 35.0
    danger_rear_ignore_px: float = 35.0
    pose_turn_deadband_deg: float = 12.0
    pose_arrival_distance_px: float = 110.0


# (Hue, Saturation, brightness)
@dataclass
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


ORANGE_RANGE = HSVRange((11, 100, 210), (19, 255, 255))
WHITE_RANGE = HSVRange((0, 0, 189), (179, 22, 255))
RED_RANGE_1 = HSVRange((0, 95, 60), (10, 255, 255))
RED_RANGE_2 = HSVRange((165, 95, 60), (179, 255, 255))


CMD_FORWARD = "i"
CMD_BACKWARD = "k"
CMD_LEFT = "j"
CMD_RIGHT = "l"
CMD_STOP = "s"
CMD_QUIT = "q"