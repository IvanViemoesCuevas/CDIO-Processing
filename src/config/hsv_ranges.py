from dataclasses import dataclass

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
