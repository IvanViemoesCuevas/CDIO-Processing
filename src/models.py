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