# Only dataclasses
from dataclasses import dataclass
from typing import Optional

@dataclass
class BallDetection:
    x: int
    y: int
    radius: float
    color_name: str
    confidence: float
    circularity: float = 0.0


@dataclass
class RobotPose:
    x: int
    y: int
    heading_rad: float
    confidence: float


@dataclass
class DangerFlags:
    front: bool = False
    back: bool = False
    left: bool = False
    center: bool = False
    right: bool = False


@dataclass
class DangerState:
    nearest_distance_px: float = float("inf")
    nearest_point: Optional[tuple[int, int]] = None
    nearest_dx_body: float = 0.0
    nearest_dy_body: float = 0.0
    too_close: bool = False


@dataclass
class NavigationContext:
    frame_width: int
    target_ball: Optional[BallDetection]
    danger: DangerFlags
    robot_pose: Optional[RobotPose]
    danger_state: Optional[DangerState]
    now: float
    balls_count: int
    candidate_target_visible: bool


@dataclass
class NavigationState:
    candidate_target: Optional[BallDetection] = None
    hold_command_until: float = 0.0
    last_target_seen_time: float = 0.0


@dataclass
class NavigationResult:
    command: str
    reason: str
