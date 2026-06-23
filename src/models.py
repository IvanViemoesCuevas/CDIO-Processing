# Only dataclasses
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vision import BallHandoffManager

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
    nearest_distance_cm: float = float("inf")
    nearest_point: Optional[tuple[int, int]] = None
    nearest_dx_body: float = 0.0
    nearest_dy_body: float = 0.0
    too_close: bool = False


@dataclass
class NavigationContext:
    frame_width: int
    frame_height: int
    target_ball: Optional[BallDetection]
    danger: DangerFlags
    robot_pose: Optional[RobotPose]
    danger_state: Optional[DangerState]
    now: float
    balls_count: int
    candidate_target_visible: bool
    handoff_manager: Optional["BallHandoffManager"] = None
    small_goal: Optional["GoalDetection"] = None


@dataclass
class NavigationState:
    candidate_target: Optional[BallDetection] = None
    hold_command_until: float = 0.0
    last_target_seen_time: float = 0.0
    handoff_phase: str = "idle"  # idle, approaching_alignment, aligning, midway_pause, approaching_delivery, delivering, done
    last_command: str = ""
    avoidance_turn_command: str = ""
    avoidance_turn_count: int = 0
    escape_turn_command: str = ""
    escape_undo_command: str = ""
    escape_turn_until: float = 0.0


@dataclass
class NavigationResult:
    command: str
    reason: str


@dataclass
class GoalDetection:
    """Represents a detected goal with position and size information."""
    x: int
    y: int
    size_category: str
    confidence: float = 1.0
    alignment_point_x: Optional[int] = None
    alignment_point_y: Optional[int] = None
    delivery_point_x: Optional[int] = None
    delivery_point_y: Optional[int] = None


@dataclass
class FieldCorners:
    """Represents the four corners of the golf field/court."""
    topLeft: tuple[int, int]
    topRight: tuple[int, int]
    bottomLeft: tuple[int, int]
    bottomRight: tuple[int, int]
