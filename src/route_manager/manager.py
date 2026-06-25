import math
import cv2 as cv
import numpy as np
from typing import Optional, List, Tuple

from config import PERSPECTIVE_PADDING_PX, FIELD_WIDTH_CM, FIELD_HEIGHT_CM
from models import BallDetection, RobotPose, GoalDetection
from vision import is_ball_in_danger_zone
from utils.geometry import (
    get_middle_obstacles,
    check_route_segment_for_obstacles,
    plan_obstacle_safe_waypoints,
    waypoint_from_ball_opposite_cross,
    obstacle_containing_point,
)


class RouteManager:
    def __init__(self, scan_duration: float = 2.0, re_eval_duration: float = 2.0):
        self.state = "scanning"  # scanning, planning, executing, commit, handoff, re_evaluating, idle
        self.queue: List[BallDetection] = []
        self.visited_positions: List[Tuple[float, float]] = []  # List of physical (x, y) coordinates in cm
        
        self.scan_start_time: Optional[float] = None
        self.scan_duration = scan_duration
        self.scan_balls: List[BallDetection] = []
        self.cumulative_danger_mask = None
        self.cumulative_contours = []
        
        self.commit_start_time: Optional[float] = None
        self.missing_frames = 0
        
        self.re_eval_start_time: Optional[float] = None
        self.re_eval_duration = re_eval_duration
        self.re_eval_balls: List[BallDetection] = []
        self.re_eval_danger_mask = None

    def _make_waypoint_ball(self, x_cm, y_cm, scale_x, scale_y):
        return BallDetection(
            x=int(round(PERSPECTIVE_PADDING_PX + x_cm * scale_x)),
            y=int(round(PERSPECTIVE_PADDING_PX + y_cm * scale_y)),
            radius=12.0,
            color_name="waypoint",
            confidence=1.0,
            circularity=1.0,
        )

    def update(
        self,
        current_time: float,
        balls: List[BallDetection],
        robot_pose: Optional[RobotPose],
        danger_mask: Optional[np.ndarray],
        scale_x: float,
        scale_y: float,
        settings,
        current_danger_contours: list = [],
        small_goal: Optional[GoalDetection] = None,
    ) -> Tuple[Optional[BallDetection], str, str]:
        print(self.state)
        """
        Updates the route planning state machine.
        Returns:
            Tuple[Optional[BallDetection], str, str]: (target_ball, command_override, reason_override)
            command_override and reason_override are non-empty if the RouteManager overrides navigation commands.
        """
        if self.state == "scanning":
            if self.scan_start_time is None:
                self.scan_start_time = current_time
                self.scan_balls = []
                self.cumulative_danger_mask = None
                self.cumulative_contours = []
                print("[RouteManager] Starting scan phase...")
            
            # Accumulate danger mask
            if danger_mask is not None:
                if self.cumulative_danger_mask is None:
                    self.cumulative_danger_mask = np.zeros_like(danger_mask)
                elif self.cumulative_danger_mask.shape != danger_mask.shape:
                    self.cumulative_danger_mask = cv.resize(
                        self.cumulative_danger_mask,
                        (danger_mask.shape[1], danger_mask.shape[0]),
                        interpolation=cv.INTER_NEAREST
                    )
                self.cumulative_danger_mask = cv.bitwise_or(self.cumulative_danger_mask, danger_mask)
            
            # Accumulate balls detected in this frame
            self.scan_balls.extend(balls)
            
            if current_time - self.scan_start_time >= self.scan_duration:
                self.state = "planning"
                self.scan_start_time = None
            
            return None, "s", "scanning"

        elif self.state == "planning":
            print(f"[RouteManager] Planning phase. Total raw detections: {len(self.scan_balls)}")
            
            # Extract contours from cumulative danger mask
            if self.cumulative_danger_mask is not None:
                cnts, _ = cv.findContours(self.cumulative_danger_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                self.cumulative_contours = list(cnts)
                print(f"[RouteManager] Extracted {len(self.cumulative_contours)} cumulative danger contours.")
            else:
                self.cumulative_contours = []

            # Group/cluster the accumulated balls to get unique positions
            unique_balls = []
            for b in self.scan_balls:
                cm_x = (b.x - PERSPECTIVE_PADDING_PX) / scale_x
                cm_y = (b.y - PERSPECTIVE_PADDING_PX) / scale_y
                
                matched = False
                for ub in unique_balls:
                    dist = math.hypot(cm_x - ub['x_cm'], cm_y - ub['y_cm'])
                    if dist < 3.0:  # 10 cm clustering threshold
                        n = ub['count']
                        ub['x_cm'] = (ub['x_cm'] * n + cm_x) / (n + 1)
                        ub['y_cm'] = (ub['y_cm'] * n + cm_y) / (n + 1)
                        ub['count'] += 1
                        if b.color_name == "orange":
                            ub['color'] = "orange"
                        matched = True
                        break
                if not matched:
                    unique_balls.append({
                        'x_cm': cm_x,
                        'y_cm': cm_y,
                        'color': b.color_name,
                        'count': 1
                    })
            
            # Filter and convert back to BallDetection in pixel coordinates
            planned_balls = []
            contours_to_use = self.cumulative_contours if self.cumulative_contours else current_danger_contours
            for ub in unique_balls:
                # Keep balls detected in at least 2 frames to filter transient noise
                if ub['count'] >= 2:
                    pixel_x = int(round(PERSPECTIVE_PADDING_PX + ub['x_cm'] * scale_x))
                    pixel_y = int(round(PERSPECTIVE_PADDING_PX + ub['y_cm'] * scale_y))
                    
                    # Create temporary BallDetection object for danger zone check
                    temp_ball = BallDetection(
                        x=pixel_x,
                        y=pixel_y,
                        radius=12.0,
                        color_name=ub['color'],
                        confidence=1.0,
                        circularity=1.0
                    )
                    
                    # Check danger zone using the averaged position
                    if is_ball_in_danger_zone(temp_ball, contours_to_use, scale_x, scale_y):
                        print(f"[RouteManager] Filtering out danger zone ball at ({temp_ball.x}, {temp_ball.y}) color={temp_ball.color_name}")
                        continue
                    
                    planned_balls.append(temp_ball)
            
            # Separate white and orange balls
            white_balls = [b for b in planned_balls if b.color_name == "white"]
            orange_balls = [b for b in planned_balls if b.color_name == "orange"]
            
            # Start position in cm
            if robot_pose is not None:
                start_x = (robot_pose.x - PERSPECTIVE_PADDING_PX) / scale_x
                start_y = (robot_pose.y - PERSPECTIVE_PADDING_PX) / scale_y
            else:
                # Fallback to field center
                start_x = 178.0 / 2.0
                start_y = 133.0 / 2.0
                
            # Nearest neighbor planning for white balls
            route = []
            curr_x, curr_y = start_x, start_y
            remaining_white = list(white_balls)
            
            while remaining_white:
                best_idx = 0
                best_dist = float("inf")
                for i, wb in enumerate(remaining_white):
                    wb_x = (wb.x - PERSPECTIVE_PADDING_PX) / scale_x
                    wb_y = (wb.y - PERSPECTIVE_PADDING_PX) / scale_y
                    dist = math.hypot(wb_x - curr_x, wb_y - curr_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                closest = remaining_white.pop(best_idx)
                route.append(closest)
                curr_x = (closest.x - PERSPECTIVE_PADDING_PX) / scale_x
                curr_y = (closest.y - PERSPECTIVE_PADDING_PX) / scale_y
            
            # Append orange ball at the end if detected
            if orange_balls:
                route.append(orange_balls[0])
                print(f"[RouteManager] Planned route with {len(white_balls)} white balls and 1 orange ball.")
            else:
                print(f"[RouteManager] Planned route with {len(white_balls)} white balls (no orange ball detected).")



            # Detect middle obstacles and build route queue with bypass waypoints
            frame_width = danger_mask.shape[1] if danger_mask is not None else 1280
            frame_height = danger_mask.shape[0] if danger_mask is not None else 960
            obstacles = get_middle_obstacles(contours_to_use, frame_width, frame_height, scale_x, scale_y)
            print(f"[RouteManager] Detected {len(obstacles)} middle obstacles.")
            for obs_idx, obs in enumerate(obstacles):
                print(
                    f"[RouteManager] obstacle[{obs_idx}] "
                    f"x=({obs[0]:.1f},{obs[2]:.1f}) y=({obs[1]:.1f},{obs[3]:.1f}) cm"
                )

            final_queue = []
            curr_x, curr_y = start_x, start_y

            # FIXME FROM HERE

            for target in route:
                target_x_cm = (target.x - PERSPECTIVE_PADDING_PX) / scale_x
                target_y_cm = (target.y - PERSPECTIVE_PADDING_PX) / scale_y

                containing_obs = obstacle_containing_point(target_x_cm, target_y_cm, obstacles)

                if containing_obs is not None:
                    approach_x, approach_y = waypoint_from_ball_opposite_cross(
                        target_x_cm,
                        target_y_cm,
                        containing_obs,
                        distance_cm=20.0,
                    )

                    for wp_x, wp_y in plan_obstacle_safe_waypoints(
                            curr_x, curr_y, approach_x, approach_y, obstacles
                    ):
                        final_queue.append(self._make_waypoint_ball(wp_x, wp_y, scale_x, scale_y))

                    final_queue.append(self._make_waypoint_ball(approach_x, approach_y, scale_x, scale_y))
                    final_queue.append(target)

                    print(
                        "[RouteManager] Ball inside margined obstacle box. "
                        f"Inserted opposite-cross approach waypoint cm=({approach_x:.1f},{approach_y:.1f})"
                    )
                    continue

                else:
                    for wp_x, wp_y in plan_obstacle_safe_waypoints(
                            curr_x, curr_y, target_x_cm, target_y_cm, obstacles
                    ):
                        final_queue.append(self._make_waypoint_ball(wp_x, wp_y, scale_x, scale_y))
                        print(f"[RouteManager] Inserted obstacle waypoint cm=({wp_x:.1f},{wp_y:.1f})")

                    final_queue.append(target)

                curr_x, curr_y = target_x_cm, target_y_cm

            #FIXME TO HERE

            self.queue = final_queue
            self.visited_positions = []
            
            if not self.queue:
                print("[RouteManager] No balls detected. Transitioning to executing to verify path to handoff.")
                self.state = "executing"
                self.missing_frames = 0
            else:
                self.state = "executing"
                self.missing_frames = 0
                
            return None, "s", "planning"

        elif self.state == "executing":
            if not self.queue:
                print("[RouteManager] Queue empty. Checking path and transitioning to handoff phase.")
                # Before entering handoff, check whether the straight path to the
                # alignment point crosses the centre obstacle. If it does, insert a
                # bypass waypoint so the robot doesn't drive through the cross.
                if robot_pose is not None and small_goal is not None and small_goal.alignment_point_x is not None:
                    r_x_cm = (robot_pose.x - PERSPECTIVE_PADDING_PX) / scale_x
                    r_y_cm = (robot_pose.y - PERSPECTIVE_PADDING_PX) / scale_y
                    align_x_cm = (small_goal.alignment_point_x - PERSPECTIVE_PADDING_PX) / scale_x
                    align_y_cm = (small_goal.alignment_point_y - PERSPECTIVE_PADDING_PX) / scale_y

                    frame_width = danger_mask.shape[1] if danger_mask is not None else 1280
                    frame_height = danger_mask.shape[0] if danger_mask is not None else 960
                    contours_to_use = self.cumulative_contours if self.cumulative_contours else current_danger_contours
                    obstacles = get_middle_obstacles(contours_to_use, frame_width, frame_height, scale_x, scale_y)

                    obs = check_route_segment_for_obstacles(r_x_cm, r_y_cm, align_x_cm, align_y_cm, obstacles)
                    if obs is not None:
                        print(
                            "[RouteManager] Handoff segment crosses obstacle: "
                            f"from=({r_x_cm:.1f},{r_y_cm:.1f}) to=({align_x_cm:.1f},{align_y_cm:.1f}) "
                            f"obs=({obs[0]:.1f},{obs[1]:.1f},{obs[2]:.1f},{obs[3]:.1f})"
                        )
                        # FIXME AGAIN FROM HERE

                        wps = plan_obstacle_safe_waypoints(
                            r_x_cm, r_y_cm, align_x_cm, align_y_cm, obstacles
                        )

                        print(f"[RouteManager] Handoff waypoint candidates: {wps}")

                        if wps:
                            wp_x, wp_y = wps[0]
                            wp_ball = self._make_waypoint_ball(wp_x, wp_y, scale_x, scale_y)
                            self.queue.append(wp_ball)

                            print(
                                f"[RouteManager] Inserted handoff waypoint cm=({wp_x:.1f},{wp_y:.1f}) "
                                f"dist_from_robot={math.hypot(wp_x - r_x_cm, wp_y - r_y_cm):.1f}"
                            )

                            # Stay in executing. Do NOT enter handoff yet.
                            return wp_ball, "", ""

                        print("[RouteManager] Handoff path blocked, but no waypoint was generated.")

                        #FIXME AGAIN TO HERE
                self.state = "handoff"
                return None, "s", "executing_done"
            
            target = self.queue[0]
            
            # --- Dynamic Queue Updating ---
            # Match current frame's detections to the queue and update their coordinates
            matched_queued_indices = set()
            contours_to_use = self.cumulative_contours if self.cumulative_contours else current_danger_contours
            for db in balls:
                db_x_cm = (db.x - PERSPECTIVE_PADDING_PX) / scale_x
                db_y_cm = (db.y - PERSPECTIVE_PADDING_PX) / scale_y
                
                if is_ball_in_danger_zone(db, contours_to_use, scale_x, scale_y):
                    continue
                
                # Check if it matches a queued ball (non-visited)
                best_match_idx = None
                best_match_dist = float("inf")
                for idx, qb in enumerate(self.queue):
                    if idx in matched_queued_indices:
                        continue
                    qb_x_cm = (qb.x - PERSPECTIVE_PADDING_PX) / scale_x
                    qb_y_cm = (qb.y - PERSPECTIVE_PADDING_PX) / scale_y
                    dist = math.hypot(db_x_cm - qb_x_cm, db_y_cm - qb_y_cm)
                    if dist < 15.0:  # 15 cm matching threshold
                        if dist < best_match_dist:
                            best_match_dist = dist
                            best_match_idx = idx
                
                if best_match_idx is not None:
                    # Update queued ball's coordinate to keep it accurate
                    self.queue[best_match_idx].x = db.x
                    self.queue[best_match_idx].y = db.y
                    matched_queued_indices.add(best_match_idx)
                else:
                    # Do not dynamically add new balls to the queue during the run
                    pass
            
            # --- Check if Target Ball is Still There & Dynamic Waypoint Check ---
            if robot_pose is not None:
                r_x_cm = (robot_pose.x - PERSPECTIVE_PADDING_PX) / scale_x
                r_y_cm = (robot_pose.y - PERSPECTIVE_PADDING_PX) / scale_y
                t_x_cm = (target.x - PERSPECTIVE_PADDING_PX) / scale_x
                t_y_cm = (target.y - PERSPECTIVE_PADDING_PX) / scale_y
                dist_to_target = math.hypot(t_x_cm - r_x_cm, t_y_cm - r_y_cm)
                
                # Check for dynamic obstacle waypoint insertion
                if target.color_name != "waypoint":
                    frame_width = danger_mask.shape[1] if danger_mask is not None else 1280
                    frame_height = danger_mask.shape[0] if danger_mask is not None else 960
                    obstacles = get_middle_obstacles(contours_to_use, frame_width, frame_height, scale_x, scale_y)

                    #FIXME AGAIN AGAIN FROM HERE
                    obs = check_route_segment_for_obstacles(r_x_cm, r_y_cm, t_x_cm, t_y_cm, obstacles)
                    if obs is not None:
                        wps = plan_obstacle_safe_waypoints(
                            r_x_cm, r_y_cm, t_x_cm, t_y_cm, obstacles
                        )

                        if wps:
                            wp_x, wp_y = wps[0]
                            waypoint_arrival_dist = getattr(settings, 'waypoint_arrival_distance_cm', 8.0)

                            #if math.hypot(wp_x - r_x_cm, wp_y - r_y_cm) > waypoint_arrival_dist + 4.0:
                            #    wp_ball = self._make_waypoint_ball(wp_x, wp_y, scale_x, scale_y)
                            #    self.queue.insert(0, wp_ball)
                            #
                            #    print(f"[RouteManager] Dynamic waypoint inserted cm=({wp_x:.1f},{wp_y:.1f})")
                            #
                            #    target = wp_ball
                            #    t_x_cm, t_y_cm = wp_x, wp_y
                            #    dist_to_target = math.hypot(t_x_cm - r_x_cm, t_y_cm - r_y_cm)
                    #FIXME AGAIN AGAIN TO HERE
                
                if target.color_name != "waypoint" and dist_to_target < 30.0:
                    # Is target currently matched by any detection?
                    target_matched = (0 in matched_queued_indices)
                    if not target_matched:
                        self.missing_frames += 1
                        missing_limit = getattr(settings, 'target_missing_frames_limit', 10)
                        if self.missing_frames >= missing_limit:
                            print(f"[RouteManager] Target missing for {missing_limit} frames at close range. Skipping target.")
                            self.visited_positions.append((t_x_cm, t_y_cm))
                            self.queue.pop(0)
                            self.missing_frames = 0
                            return None, "s", "target_skipped"
                    else:
                        self.missing_frames = 0
            
            # Return target to follow
            return self.queue[0], "", ""

        elif self.state == "commit":
            # Driving forward during commit phase
            if self.commit_start_time is None:
                self.commit_start_time = current_time
                print("[RouteManager] Entering commit phase (driving forward)...")
                
            if current_time - self.commit_start_time >= settings.commit_forward_window_sec:
                # Commit phase complete
                if self.queue:
                    target = self.queue.pop(0)
                    t_x_cm = (target.x - PERSPECTIVE_PADDING_PX) / scale_x
                    t_y_cm = (target.y - PERSPECTIVE_PADDING_PX) / scale_y
                    if target.color_name != "waypoint":
                        self.visited_positions.append((t_x_cm, t_y_cm))
                    print(f"[RouteManager] Completed attempt for ball. Remaining queue size: {len(self.queue)}")
                
                self.commit_start_time = None
                self.missing_frames = 0
                
                self.state = "executing"
                
                return None, "s", "commit_done"
                
            return None, "i", "commit_active"

        elif self.state == "handoff":
            # Handled in main loop by monitoring and overriding handoff phase
            return None, "", ""

        elif self.state == "re_evaluating":
            if self.re_eval_start_time is None:
                self.re_eval_start_time = current_time
                self.re_eval_balls = []
                self.re_eval_danger_mask = None
                print("[RouteManager] Re-evaluating field after handoff...")
            
            # Accumulate danger mask
            if danger_mask is not None:
                if self.re_eval_danger_mask is None:
                    self.re_eval_danger_mask = np.zeros_like(danger_mask)
                elif self.re_eval_danger_mask.shape != danger_mask.shape:
                    self.re_eval_danger_mask = cv.resize(
                        self.re_eval_danger_mask,
                        (danger_mask.shape[1], danger_mask.shape[0]),
                        interpolation=cv.INTER_NEAREST
                    )
                self.re_eval_danger_mask = cv.bitwise_or(self.re_eval_danger_mask, danger_mask)

            self.re_eval_balls.extend(balls)
            
            if current_time - self.re_eval_start_time >= self.re_eval_duration:
                # Extract contours from cumulative re-evaluation danger mask
                re_eval_contours = []
                if self.re_eval_danger_mask is not None:
                    cnts, _ = cv.findContours(self.re_eval_danger_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    re_eval_contours = list(cnts)
                
                # Evaluate if any balls are left outside danger zones
                unique_balls = []
                for b in self.re_eval_balls:
                    cm_x = (b.x - PERSPECTIVE_PADDING_PX) / scale_x
                    cm_y = (b.y - PERSPECTIVE_PADDING_PX) / scale_y
                    
                    matched = False
                    for ub in unique_balls:
                        dist = math.hypot(cm_x - ub['x_cm'], cm_y - ub['y_cm'])
                        if dist < 10.0:
                            n = ub['count']
                            ub['x_cm'] = (ub['x_cm'] * n + cm_x) / (n + 1)
                            ub['y_cm'] = (ub['y_cm'] * n + cm_y) / (n + 1)
                            ub['count'] += 1
                            if b.color_name == "orange":
                                ub['color'] = "orange"
                            matched = True
                            break
                    if not matched:
                        unique_balls.append({
                            'x_cm': cm_x,
                            'y_cm': cm_y,
                            'color': b.color_name,
                            'count': 1
                        })
                
                # Check danger zone on the clustered unique balls
                contours_to_use = re_eval_contours if re_eval_contours else current_danger_contours
                collectible_count = 0
                for ub in unique_balls:
                    if ub['count'] >= 2:
                        pixel_x = int(round(PERSPECTIVE_PADDING_PX + ub['x_cm'] * scale_x))
                        pixel_y = int(round(PERSPECTIVE_PADDING_PX + ub['y_cm'] * scale_y))
                        
                        temp_ball = BallDetection(
                            x=pixel_x,
                            y=pixel_y,
                            radius=12.0,
                            color_name=ub.get('color', 'white'),
                            confidence=1.0,
                            circularity=1.0
                        )
                        
                        if is_ball_in_danger_zone(temp_ball, contours_to_use, scale_x, scale_y):
                            print(f"[RouteManager] Re-eval filtering out danger zone ball at ({temp_ball.x}, {temp_ball.y})")
                            continue
                        
                        collectible_count += 1
                
                print(f"[RouteManager] Re-evaluation finished. Found {collectible_count} collectible balls.")
                
                self.re_eval_start_time = None
                
                if collectible_count > 0:
                    print("[RouteManager] Collectible balls remain. Starting a new run.")
                    self.state = "scanning"
                else:
                    print("[RouteManager] No collectible balls remain. Entering idle state.")
                    self.state = "idle"
            
            return None, "s", "re_evaluating"

        elif self.state == "idle":
            return None, "s", "idle"

        return None, "", ""
