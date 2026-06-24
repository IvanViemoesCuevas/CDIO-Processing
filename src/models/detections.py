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
