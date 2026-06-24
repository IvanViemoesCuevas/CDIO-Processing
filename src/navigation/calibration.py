# Calibration offsets between the elevated ArUco marker and the robot's real ground pose.
# Since the marker works when placed on the ground, tune the forward/right pixel offsets first.
# Only tune heading if the marker is physically rotated relative to the robot's forward direction.

MARKER_HEADING_OFFSET_DEG = 0.0

# Physical mount offset from the ArUco marker to the robot's real drive/rotation center.
# These are robot-local offsets and DO rotate when the robot turns:
#   forward_px = along robot forward
#   right_px   = along robot right
MARKER_TO_DRIVE_CENTER_FORWARD_PX = 0.0
MARKER_TO_DRIVE_CENTER_RIGHT_PX = 0.0

# Perspective correction caused by the marker being elevated above the ground.
# These are IMAGE-SPACE corrections and do NOT rotate with the robot.
# Tune these so the elevated marker projects down to the ground point under the marker.
# Positive X gain moves the correction more right when the marker is right of image center.
# Positive Y gain moves the correction more down when the marker is below image center.
MARKER_PERSPECTIVE_X_GAIN = -90.0
MARKER_PERSPECTIVE_Y_GAIN = -60.0

# Backwards-compatible aliases used by ui.py debug text.
MARKER_PERSPECTIVE_RIGHT_GAIN = MARKER_PERSPECTIVE_X_GAIN
MARKER_PERSPECTIVE_FORWARD_GAIN = MARKER_PERSPECTIVE_Y_GAIN
