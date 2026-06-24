from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from .detections import BallDetection, GoalDetection
from .robot import RobotPose

if TYPE_CHECKING:
    from vision import BallHandoffManager


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
    small_goal: Optional[GoalDetection] = None


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
