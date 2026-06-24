from models import BallDetection


class BallHandoffManager:
    """
    Manages the state for ball handoff by tracking when the field is clear of balls.
    Ready for handoff when field is empty (0 balls detected for N consecutive frames).
    """
    def __init__(self, required_empty_frames: int = 1):
        """
        Initializes the manager.
        :param required_empty_frames: Consecutive frames without balls before field considered empty.
        """
        self.required_empty_frames = required_empty_frames
        self._empty_frame_counter = 0
        self._has_seen_ball = False

    def update(self, balls: list[BallDetection]) -> None:
        """
        Updates the manager's state based on current frame's ball detections.
        :param balls: A list of BallDetection objects from the current frame.
        """
        # Track field state: first require that at least one ball has been detected.
        # This prevents the robot from driving to the goal at startup when the field is empty.
        if balls:
            self._has_seen_ball = True
            self._empty_frame_counter = 0
        else:
            self._empty_frame_counter += 1

    def field_is_clear(self) -> bool:
        """
        :return: True if field is empty (sufficient consecutive empty frames detected).
        """
        return self._empty_frame_counter >= self.required_empty_frames

    @property
    def ready_for_handoff(self) -> bool:
        """
        Check if field is clear and ready for handoff.
        :return: True if field is empty for required frames, False otherwise.
        """
        return self._has_seen_ball and self.field_is_clear()

    @property
    def empty_frames_count(self) -> int:
        """
        :return: Current count of consecutive empty frames.
        """
        return self._empty_frame_counter

    def reset(self) -> None:
        """Resets the field-empty counter."""
        self._empty_frame_counter = 0
        self._has_seen_ball = False
