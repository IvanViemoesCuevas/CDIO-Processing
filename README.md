# OpenCVProject - Vision to Robot Command Template

This project now includes a template script that sends one-byte TCP commands to a robot based on YOLO detections from a webcam.

## Files

- `main.py`: your original YOLO tracking demo.
- `vision_robot_template.py`: camera + YOLO + TCP command sender template.
- `contours_only.py`: contour-only visualization (no YOLO).
- `ball_avoid_robot_template.py`: OpenCV-only orange/white ping pong tracking + obstacle-zone avoidance + TCP sender.
- `ball_avoid_smoke_test.py`: quick logic smoke test for command arbitration.
- `requirements.txt`: minimal Python dependencies.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 vision_robot_template.py --host 10.1.1.9 --port 12345
```

OpenCV-only ping pong ball + barrier avoid template:

```bash
python3 ball_avoid_robot_template.py --host 10.1.1.9 --port 12345 --camera 0
```

Overhead camera mode with robot localization from an ArUco marker:

```bash
python3 ball_avoid_robot_template.py --camera 0 --robot-marker-id 7
```

Dry-run without opening a robot socket:

```bash
python3 ball_avoid_robot_template.py --camera 0 --dry-run --show-edges
```

Live-tune HSV thresholds while watching masks update in real time:

```bash
python3 ball_avoid_robot_template.py --camera 0 --dry-run --tune-hsv
```

Quick logic smoke test:

```bash
python3 ball_avoid_smoke_test.py
```

Optional flags:

```bash
python3 vision_robot_template.py \
  --host 10.1.1.9 \
  --port 12345 \
  --model yolov8s.pt \
  --camera 0 \
  --conf 0.45 \
  --send-interval 0.15 \
  --stable-frames 3 \
  --no-detection s
```

Press `q` in the OpenCV window to quit (the script sends a final `q` command).

## Customize command mapping

Edit `CLASS_TO_COMMAND` in `vision_robot_template.py`.

```python
CLASS_TO_COMMAND = {
    "person": "w",
    "bottle": "a",
    "chair": "d",
}
```

Notes:
- Keys are YOLO class names.
- Values must be single ASCII characters expected by your robot server.
- Unmapped detections fall back to `--no-detection` (default `s`).

## Safety recommendation

Set your fallback (`--no-detection`) to a safe stop command supported by your robot.

## Ball + obstacle behavior notes

- Ball detection uses HSV masks for `orange` and `white` plus circularity/radius checks.
- For overhead stationary cameras, ball target selection switches to nearest-ball-to-robot when ArUco robot pose is available.
- Obstacle detection uses edges + contour occupancy zones in the near-field ROI (not red-only color segmentation).
- Safety priority is: `danger center` -> stop, `danger side` -> steer away, else track ball, else search.
- Tune thresholds in `Settings`, `ORANGE_RANGE`, and `WHITE_RANGE` in `ball_avoid_robot_template.py` for your lighting.

Robot pose flags:
- `--robot-marker-id N`: pick a specific ArUco marker ID on robot (`-1` uses first visible marker).
- `--no-robot-pose`: disable ArUco pose and fall back to image-center ball tracking.

Mask tuning flags:
- `--show-ball-masks`: show camera/orange/white/combined mask debug view.
- `--tune-hsv`: opens HSV trackbars and applies values live (also enables mask debug view).
- `--min-ball-confidence`: reject low-confidence ball detections (default `0.65`).




