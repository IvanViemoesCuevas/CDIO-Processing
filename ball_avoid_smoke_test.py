#!/usr/bin/env python3
"""Small logic smoke test for ball_avoid_robot_template.py."""

from ball_avoid_robot_template import (
    BallDetection,
    DangerFlags,
    Settings,
    decide_command,
)


def run_case(name: str, ball: BallDetection | None, danger: DangerFlags) -> None:
    settings = Settings()
    command, reason = decide_command(frame_width=640, ball=ball, danger=danger, settings=settings)
    print(f"{name:>20} -> cmd={command} reason={reason}")


def main() -> None:
    run_case("danger center", None, DangerFlags(center=True))
    run_case("danger left", None, DangerFlags(left=True))
    run_case("danger right", None, DangerFlags(right=True))
    run_case("search", None, DangerFlags())

    run_case(
        "ball left",
        BallDetection(x=180, y=240, radius=20.0, color_name="orange", confidence=0.9),
        DangerFlags(),
    )
    run_case(
        "ball right",
        BallDetection(x=490, y=240, radius=20.0, color_name="white", confidence=0.9),
        DangerFlags(),
    )
    run_case(
        "ball forward",
        BallDetection(x=325, y=240, radius=18.0, color_name="orange", confidence=0.9),
        DangerFlags(),
    )
    run_case(
        "ball arrived",
        BallDetection(x=322, y=240, radius=45.0, color_name="white", confidence=0.9),
        DangerFlags(),
    )


if __name__ == "__main__":
    main()

