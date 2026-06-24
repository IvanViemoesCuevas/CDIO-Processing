from dataclasses import dataclass

@dataclass
class RobotPose:
    x: int
    y: int
    heading_rad: float
    confidence: float
