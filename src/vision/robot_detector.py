import math
import cv2 as cv
import numpy as np
from typing import Optional

from config import Settings
from models import RobotPose


def detect_robot_pose(frame: np.ndarray, settings: Settings) -> Optional[RobotPose]:
    if not hasattr(cv, "aruco"):
        return None
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None

    target_index = 7
    match = np.where(ids.flatten() == target_index)[0]
    if len(match) == 0:
        return None
    target_index = int(match[0])

    pts = corners[target_index][0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    top_mid = 0.5 * (pts[2] + pts[1])
    bottom_mid = 0.5 * (pts[0] + pts[3])
    forward_vec = top_mid - bottom_mid
    heading = math.atan2(float(forward_vec[1]), float(forward_vec[0]))

    return RobotPose(x=cx, y=cy, heading_rad=heading, confidence=-1)
