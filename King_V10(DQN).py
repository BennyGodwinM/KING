import math
import time
import csv
import os
import serial
import cv2
import numpy as np
import pyrealsense2 as rs
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


SERIAL_PORT = "COM3"
BAUD_RATE = 9600

LOWER_HSV = np.array([145, 80, 80])
UPPER_HSV = np.array([179, 255, 255])

MIN_AREA = 400
PATCH_RADIUS = 3
CENTER_DRAW_RADIUS = 6

CAM_W_D = 480
CAM_H_D = 270
CAM_W_C = 640
CAM_H_C = 480
CAM_FPS = 30

WHEEL_RADIUS = 0.039
CPR = 615

GYRO_SIGN = 1.0
GYRO_YAW_AXIS = "y"
MAX_GYRO_DT = 0.05
GYRO_DEADBAND = 0.01

MAX_FORWARD_STEP = 0.20
MAX_TURN_STEP_DEG = 20.0

TARGET_REACHED_DIST_M = 0.3
SUCCESS_ANGLE_THRESH_DEG = 10.0

FRONT_OBSTACLE_DISTANCE = 0.20
AVOIDANCE_TRIGGER_DISTANCE = 0.3
FRONT_INVALID_THRESH = 0.50
UNKNOWN_FRAMES_NEEDED = 4

MAX_OBS_DISTANCE = 10.0
IGNORE_INVALIDS_NEAR_TARGET_DIST = 0.4
DISABLE_AVOIDANCE_NEAR_TARGET_DIST = 0.4

TARGET_DEPTH_MATCH_THRESH = 0.10
TARGET_CENTER_MATCH_DEG = 18.0

ACTION_MEANINGS = {
    0: "F",
    1: "L",
    2: "R",
}

CONTROL_DT = 0.15

SUCCESS_REWARD = 10.0
ANGLE_BONUS_REWARD = 3.0
COLLISION_PENALTY = -15.0
DISTANCE_DELTA_GAIN = 3.0
ANGLE_DELTA_GAIN = 0.015
TIME_PENALTY_GAIN = 0.15

CENTERED_FORWARD_ANGLE_DEG = 10.0
FORWARD_CLEAR_BONUS = 0.25
PATH_CLEAR_CENTER_DANGER_THRESH = 0.25

MAX_R_DELTA_PER_STEP = 0.10

INVALID_DANGER_GAIN = 0.65
DEPTH_DANGER_GAIN = 0.35
AVOIDANCE_REWARD_GAIN = 0.35
FORWARD_DANGER_PENALTY_GAIN = 0.20
TURN_TO_CLEAR_CENTER_BONUS_GAIN = 0.08
STOP_DANGER_PENALTY_GAIN = 0.04

AVOIDANCE_ON_UNKNOWN = True

MODEL_PATH = "dqn_target_nav_model_5"
LOG_PATH = "dqn_step_log_5.csv"

last_cmd = None

SESSION_ID = ""
PHASE = ""
SCENARIO = ""
RUN_COMMENT = ""


def sanitize_label(text):
    if text is None:
        return ""
    return str(text).strip()


def prompt_run_metadata():
    global SESSION_ID, PHASE, SCENARIO, RUN_COMMENT

    print("\nENTER RUN INFO")
    SESSION_ID = sanitize_label(input("Session ID: "))
    PHASE = sanitize_label(input("Phase (training/testing/etc): "))
    SCENARIO = sanitize_label(input("Scenario (no_obstacle/one_obstacle/etc): "))
    RUN_COMMENT = sanitize_label(input("Run comment: "))

    if SESSION_ID == "":
        SESSION_ID = "default_session"
    if PHASE == "":
        PHASE = "unspecified_phase"
    if SCENARIO == "":
        SCENARIO = "unspecified_scenario"

    print("\nRUN INFO SAVED")
    print(f"SESSION_ID = {SESSION_ID}")
    print(f"PHASE      = {PHASE}")
    print(f"SCENARIO   = {SCENARIO}")
    print(f"COMMENT    = {RUN_COMMENT}")


def ensure_log_file():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id",
                "phase",
                "scenario",
                "run_comment",
                "episode",
                "step_in_episode",
                "global_step",
                "action",
                "reward_total",
                "reward_success",
                "reward_collision",
                "reward_distance",
                "reward_angle",
                "reward_forward",
                "reward_avoidance",
                "reward_time",
                "R_est",
                "theta_obs_used_deg",
                "theta_real_deg",
                "theta_est_deg",
                "theta_pure_deg",
                "dpsi_step_deg",
                "robot_heading_total_deg",
                "target_visible",
                "ignore_invalids_near_target",
                "disable_avoidance_near_target",
                "target_like_front_object",
                "depth_avoidance_active",
                "avoidance_active",
                "front_state",
                "front_min_depth_m",
                "front_invalid_ratio",
                "left_depth",
                "center_depth",
                "right_depth",
                "left_invalid_ratio",
                "center_invalid_ratio",
                "right_invalid_ratio",
                "left_danger",
                "center_danger",
                "right_danger",
                "success",
                "collision",
                "done"
            ])


def send_cmd(ser, cmd):
    global last_cmd
    if ser is None:
        return
    try:
        ser.write(cmd.encode())
        last_cmd = cmd
    except Exception as e:
        print("Serial write failed:", e)


def wrap_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rad_to_deg(x):
    return x * 180.0 / math.pi


def deg_to_rad(x):
    return x * math.pi / 180.0


def clip_obs_distance(x):
    if x is None or not np.isfinite(x):
        return MAX_OBS_DISTANCE
    return float(np.clip(x, 0.0, MAX_OBS_DISTANCE))


def clip_ratio(x):
    return float(np.clip(x, 0.0, 1.0))


def get_gyro_yaw_rate(gyro_xyz, axis_name="z"):
    if axis_name == "x":
        return gyro_xyz[0]
    elif axis_name == "y":
        return gyro_xyz[1]
    else:
        return gyro_xyz[2]


def measure_theta_R_from_pixel(u, v, Z, intr):
    point = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(Z))
    X = point[0]
    Zc = point[2]
    theta = math.atan2(X, Zc)
    R = math.sqrt(X**2 + Zc**2)
    return theta, R, X, Zc


def xz_to_theta_R(x, z):
    theta = math.atan2(x, z)
    R = math.sqrt(x**2 + z**2)
    return theta, R


def propagate_hidden_target_midpoint(x_old, z_old, d_forward, d_yaw):
    dpsi_half = 0.5 * d_yaw

    c1 = math.cos(dpsi_half)
    s1 = math.sin(dpsi_half)

    x_mid = c1 * x_old - s1 * z_old
    z_mid = s1 * x_old + c1 * z_old

    x_mid2 = x_mid
    z_mid2 = z_mid - d_forward

    c2 = math.cos(dpsi_half)
    s2 = math.sin(dpsi_half)

    x_new = c2 * x_mid2 - s2 * z_mid2
    z_new = s2 * x_mid2 + c2 * z_mid2

    return x_new, z_new


def get_valid_depth_average(depth_frame, u, v, patch_radius=3):
    values = []
    h = depth_frame.get_height()
    w = depth_frame.get_width()

    u = int(u)
    v = int(v)

    for yy in range(max(0, v - patch_radius), min(h, v + patch_radius + 1)):
        for xx in range(max(0, u - patch_radius), min(w, u + patch_radius + 1)):
            d = depth_frame.get_distance(xx, yy)
            if d > 0:
                values.append(d)

    if len(values) == 0:
        return None

    return float(np.mean(values))


def detect_target(color_image, lower_hsv, upper_hsv, min_area):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(largest)

    return {
        "u": u,
        "v": v,
        "area": area,
        "bbox": (x, y, w, h),
    }, mask


def integrate_gyro_nonblocking(imu_pipeline, last_gyro_ts, axis_name="y"):
    frames = imu_pipeline.poll_for_frames()

    total_dpsi = 0.0
    latest_gyro = None
    latest_ts = last_gyro_ts
    total_dt = 0.0

    if not frames:
        return total_dpsi, latest_gyro, latest_ts, total_dt

    for f in frames:
        if f.get_profile().stream_type() != rs.stream.gyro:
            continue

        data = f.as_motion_frame().get_motion_data()
        gyro_xyz = np.array([data.x, data.y, data.z], dtype=float)
        gyro_ts = f.get_timestamp() * 1e-3

        latest_gyro = gyro_xyz

        if last_gyro_ts is not None:
            dt = gyro_ts - last_gyro_ts

            if 0.0 < dt < MAX_GYRO_DT:
                yaw_rate = get_gyro_yaw_rate(gyro_xyz, axis_name)

                if abs(yaw_rate) < GYRO_DEADBAND:
                    yaw_rate = 0.0

                total_dpsi += GYRO_SIGN * yaw_rate * dt
                total_dt += dt

        last_gyro_ts = gyro_ts
        latest_ts = gyro_ts

    max_turn_step = deg_to_rad(MAX_TURN_STEP_DEG)
    total_dpsi = max(-max_turn_step, min(max_turn_step, total_dpsi))

    return total_dpsi, latest_gyro, latest_ts, total_dt


def read_latest_encoder_counts(ser):
    if ser is None:
        return None, None

    latest_right = None
    latest_left = None

    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 2:
                continue

            try:
                latest_right = int(parts[0].strip())
                latest_left = int(parts[1].strip())
            except ValueError:
                continue
    except Exception:
        return None, None

    return latest_right, latest_left


def counts_to_forward_step(dR, dL, wheel_radius, cpr):
    sR = 2.0 * math.pi * wheel_radius * (dR / cpr)
    sL = 2.0 * math.pi * wheel_radius * (dL / cpr)
    s_forward = 0.5 * (sR + sL)
    return s_forward, sR, sL


def invalid_ratio(roi):
    if roi is None or roi.size == 0:
        return 1.0
    return float(np.mean(roi == 0))


def nearest_valid_depth_m(roi):
    if roi is None or roi.size == 0:
        return None
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(np.min(valid) / 1000.0)


def depth_danger_from_distance(depth_m, danger_distance):
    if depth_m is None or not np.isfinite(depth_m):
        return 0.0
    if depth_m >= danger_distance:
        return 0.0
    return float(np.clip((danger_distance - depth_m) / danger_distance, 0.0, 1.0))


def combined_direction_danger(depth_m, inv_ratio, danger_distance):
    inv_part = INVALID_DANGER_GAIN * clip_ratio(inv_ratio)
    depth_part = DEPTH_DANGER_GAIN * depth_danger_from_distance(depth_m, danger_distance)
    return float(np.clip(inv_part + depth_part, 0.0, 1.0))


class StopRobotDuringDQNUpdateCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env_ref = env

    def _on_rollout_end(self) -> None:
        # DQN callback placeholder. Not used during DQN training.
        try:
            self.env_ref.stop_robot()
            print("[CALLBACK] DQN rollout ended -> sent STOP")
        except Exception as e:
            print("[CALLBACK] Failed to stop robot:", e)

    def _on_step(self) -> bool:
        return True


class RealRobotDQNEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=True):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)

        low = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -180.0,
            0.0,
            0.0,
            0.0
        ], dtype=np.float32)

        high = np.array([
            MAX_OBS_DISTANCE,
            MAX_OBS_DISTANCE,
            MAX_OBS_DISTANCE,
            MAX_OBS_DISTANCE,
            1.0,
            1.0,
            180.0,
            1.0,
            1.0,
            1.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.ser = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            time.sleep(2)
            print("Arduino connected")
        except Exception as e:
            print("Serial not connected:", e)
            self.ser = None

        self.imu_pipeline = rs.pipeline()
        imu_config = rs.config()
        imu_config.enable_stream(rs.stream.gyro)
        self.imu_pipeline.start(imu_config)

        self.cam_pipeline = rs.pipeline()
        cam_config = rs.config()
        cam_config.enable_stream(rs.stream.depth, CAM_W_D, CAM_H_D, rs.format.z16, CAM_FPS)
        cam_config.enable_stream(rs.stream.color, CAM_W_C, CAM_H_C, rs.format.bgr8, CAM_FPS)
        self.cam_pipeline.start(cam_config)

        profile = self.cam_pipeline.get_active_profile()
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        try:
            depth_sensor.set_option(rs.option.visual_preset, 5)
        except Exception as e:
            print("Preset not set:", e)

        self.align = rs.align(rs.stream.color)
        time.sleep(2)

        cam_profile = self.cam_pipeline.get_active_profile()
        color_profile = cam_profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_profile.get_intrinsics()

        self.reset_tracking_state()
        self.episode_step = 0
        self.prev_R_est = None
        self.prev_theta_est_deg = None
        self.last_obs = None
        self.last_info = None

        self.global_step = 0
        self.episode_reward = 0.0
        self.episode_index = 0

    def reset_tracking_state(self):
        self.x_est = None
        self.z_est = None
        self.theta_est = None
        self.R_est = None

        self.x_pure = None
        self.z_pure = None
        self.theta_pure = None
        self.R_pure = None

        self.latest_gyro = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_gyro_ts = None
        self.dt_gyro = 0.0
        self.dpsi_step = 0.0
        self.robot_heading_total = 0.0

        self.right_count = None
        self.left_count = None
        self.prev_right_count = None
        self.prev_left_count = None
        self.dR_counts = 0
        self.dL_counts = 0
        self.s_forward_step = 0.0
        self.sR_step = 0.0
        self.sL_step = 0.0

        self.unknown_counter = 0

    def _update_gyro(self):
        self.dt_gyro = 0.0
        self.dpsi_step = 0.0

        dpsi_total, latest_gyro, latest_ts, total_dt = integrate_gyro_nonblocking(
            self.imu_pipeline,
            self.last_gyro_ts,
            axis_name=GYRO_YAW_AXIS
        )

        if latest_gyro is not None:
            self.latest_gyro = latest_gyro

        if latest_ts is not None:
            self.last_gyro_ts = latest_ts

        self.dt_gyro = total_dt
        self.dpsi_step = dpsi_total
        self.robot_heading_total = wrap_angle_rad(self.robot_heading_total + self.dpsi_step)

    def _update_encoders(self):
        new_right_count, new_left_count = read_latest_encoder_counts(self.ser)

        if new_right_count is not None and new_left_count is not None:
            self.right_count = new_right_count
            self.left_count = new_left_count

        self.s_forward_step = 0.0
        self.sR_step = 0.0
        self.sL_step = 0.0
        self.dR_counts = 0
        self.dL_counts = 0

        if self.right_count is not None and self.left_count is not None:
            if self.prev_right_count is not None and self.prev_left_count is not None:
                self.dR_counts = self.right_count - self.prev_right_count
                self.dL_counts = -(self.left_count - self.prev_left_count)

                self.s_forward_step, self.sR_step, self.sL_step = counts_to_forward_step(
                    self.dR_counts,
                    self.dL_counts,
                    WHEEL_RADIUS,
                    CPR
                )

                self.s_forward_step = max(-MAX_FORWARD_STEP, min(MAX_FORWARD_STEP, self.s_forward_step))

            self.prev_right_count = self.right_count
            self.prev_left_count = self.left_count

    def _get_camera_and_estimate(self):
        cam_frames = self.cam_pipeline.wait_for_frames()
        aligned_frames = self.align.process(cam_frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        ch, cw, _ = color_image.shape

        detection, mask = detect_target(color_image, LOWER_HSV, UPPER_HSV, MIN_AREA)

        target_visible = False
        theta_real = None
        R_real = None
        x_real = None
        z_real = None

        if detection is not None:
            u = detection["u"]
            v = detection["v"]
            depth_used = get_valid_depth_average(depth_frame, u, v, PATCH_RADIUS)

            if depth_used is not None and depth_used > 0:
                try:
                    theta_real, R_real, x_real, z_real = measure_theta_R_from_pixel(u, v, depth_used, self.intr)
                    target_visible = True
                except Exception:
                    target_visible = False

        if target_visible and self.x_est is None and self.z_est is None:
            self.x_est = x_real
            self.z_est = z_real
            self.theta_est, self.R_est = xz_to_theta_R(self.x_est, self.z_est)

        if target_visible and self.x_pure is None and self.z_pure is None:
            self.x_pure = x_real
            self.z_pure = z_real
            self.theta_pure, self.R_pure = xz_to_theta_R(self.x_pure, self.z_pure)

        x_pred_est = None
        z_pred_est = None
        theta_pred_est = None
        R_pred_est = None

        if self.x_est is not None and self.z_est is not None:
            x_pred_est, z_pred_est = propagate_hidden_target_midpoint(
                self.x_est,
                self.z_est,
                self.s_forward_step,
                self.dpsi_step
            )
            theta_pred_est, R_pred_est = xz_to_theta_R(x_pred_est, z_pred_est)

        x_pred_pure = None
        z_pred_pure = None
        theta_pred_pure = None
        R_pred_pure = None

        if self.x_pure is not None and self.z_pure is not None:
            x_pred_pure, z_pred_pure = propagate_hidden_target_midpoint(
                self.x_pure,
                self.z_pure,
                self.s_forward_step,
                self.dpsi_step
            )
            theta_pred_pure, R_pred_pure = xz_to_theta_R(x_pred_pure, z_pred_pure)

        if target_visible:
            self.x_est = x_real
            self.z_est = z_real
            self.theta_est = theta_real
            self.R_est = R_real
        else:
            if x_pred_est is not None and z_pred_est is not None:
                self.x_est = x_pred_est
                self.z_est = z_pred_est
                self.theta_est = theta_pred_est
                self.R_est = R_pred_est

        if x_pred_pure is not None and z_pred_pure is not None:
            self.x_pure = x_pred_pure
            self.z_pure = z_pred_pure
            self.theta_pure = theta_pred_pure
            self.R_pure = R_pred_pure

        if self.R_est is None:
            R_est_obs = MAX_OBS_DISTANCE
            theta_est_deg_obs = 0.0
        else:
            R_est_obs = clip_obs_distance(self.R_est)
            theta_est_deg_obs = float(np.clip(rad_to_deg(self.theta_est), -180.0, 180.0))

        target_range_for_compare = R_real if R_real is not None else self.R_est

        ignore_invalids_near_target = (
            target_range_for_compare is not None and
            target_range_for_compare < IGNORE_INVALIDS_NEAR_TARGET_DIST
        )

        disable_avoidance_near_target = (
            target_range_for_compare is not None and
            target_range_for_compare < DISABLE_AVOIDANCE_NEAR_TARGET_DIST
        )

        box_w = int(cw * 0.75)
        box_h = int(ch * 0.3)
        fx1 = cw // 2 - box_w // 2
        fx2 = cw // 2 + box_w // 2
        fy1 = ch // 2 - box_h // 2
        fy2 = ch // 2 + box_h // 2

        front_roi = depth_image[fy1:fy2, fx1:fx2]

        front_invalid_ratio = invalid_ratio(front_roi)
        front_min_depth_m = nearest_valid_depth_m(front_roi)

        third = cw // 3
        strip_y1 = int(ch * 0.35)
        strip_y2 = int(ch * 0.75)

        left_roi = depth_image[strip_y1:strip_y2, 0:third]
        center_roi = depth_image[strip_y1:strip_y2, third:2 * third]
        right_roi = depth_image[strip_y1:strip_y2, 2 * third:cw]

        left_depth = nearest_valid_depth_m(left_roi)
        center_depth = nearest_valid_depth_m(center_roi)
        right_depth = nearest_valid_depth_m(right_roi)

        left_invalid_ratio = invalid_ratio(left_roi)
        center_invalid_ratio = invalid_ratio(center_roi)
        right_invalid_ratio = invalid_ratio(right_roi)

        if ignore_invalids_near_target:
            front_invalid_ratio = 0.0
            left_invalid_ratio = 0.0
            center_invalid_ratio = 0.0
            right_invalid_ratio = 0.0
            self.unknown_counter = 0

        if front_min_depth_m is not None and front_min_depth_m < FRONT_OBSTACLE_DISTANCE:
            front_state = "OBSTACLE"
            self.unknown_counter = 0
        elif (not ignore_invalids_near_target) and (front_invalid_ratio > FRONT_INVALID_THRESH):
            self.unknown_counter += 1
            if self.unknown_counter >= UNKNOWN_FRAMES_NEEDED:
                front_state = "UNKNOWN"
            else:
                front_state = "SAFE"
        else:
            front_state = "SAFE"
            self.unknown_counter = 0

        side_danger_distance = AVOIDANCE_TRIGGER_DISTANCE * 1.6

        left_danger = combined_direction_danger(left_depth, left_invalid_ratio, side_danger_distance)
        center_danger = combined_direction_danger(center_depth, center_invalid_ratio, AVOIDANCE_TRIGGER_DISTANCE * 1.8)
        right_danger = combined_direction_danger(right_depth, right_invalid_ratio, side_danger_distance)

        target_like_front_object = False

        if target_visible and target_range_for_compare is not None and front_min_depth_m is not None:
            if abs(front_min_depth_m - target_range_for_compare) < TARGET_DEPTH_MATCH_THRESH:
                if abs(theta_est_deg_obs) < TARGET_CENTER_MATCH_DEG:
                    target_like_front_object = True

        depth_avoidance_active = (
            front_min_depth_m is not None and
            front_min_depth_m < AVOIDANCE_TRIGGER_DISTANCE
        )

        obs = np.array([
            clip_obs_distance(R_est_obs),
            clip_obs_distance(left_depth),
            clip_obs_distance(center_depth),
            clip_obs_distance(right_depth),
            clip_ratio(front_invalid_ratio),
            1.0 if target_visible else 0.0,
            theta_est_deg_obs,
            clip_ratio(left_invalid_ratio),
            clip_ratio(center_invalid_ratio),
            clip_ratio(right_invalid_ratio)
        ], dtype=np.float32)

        info = {
            "color_image": color_image,
            "depth_image": depth_image,
            "mask": mask,
            "detection": detection,
            "target_visible": target_visible,
            "theta_real": theta_real,
            "R_real": R_real,
            "x_real": x_real,
            "z_real": z_real,
            "theta_est": self.theta_est,
            "R_est": self.R_est,
            "theta_pure": self.theta_pure,
            "R_pure": self.R_pure,
            "ignore_invalids_near_target": ignore_invalids_near_target,
            "disable_avoidance_near_target": disable_avoidance_near_target,
            "target_like_front_object": target_like_front_object,
            "depth_avoidance_active": depth_avoidance_active,
            "front_state": front_state,
            "front_invalid_ratio": front_invalid_ratio,
            "front_min_depth_m": front_min_depth_m,
            "left_depth": left_depth,
            "center_depth": center_depth,
            "right_depth": right_depth,
            "left_invalid_ratio": left_invalid_ratio,
            "center_invalid_ratio": center_invalid_ratio,
            "right_invalid_ratio": right_invalid_ratio,
            "left_danger": left_danger,
            "center_danger": center_danger,
            "right_danger": right_danger,
            "box": (fx1, fy1, fx2, fy2),
            "gyro_yaw_rate": get_gyro_yaw_rate(self.latest_gyro, GYRO_YAW_AXIS),
            "dt_gyro": self.dt_gyro,
            "dpsi_step": self.dpsi_step,
            "robot_heading_total": self.robot_heading_total,
            "right_count": self.right_count,
            "left_count": self.left_count,
            "dR_counts": self.dR_counts,
            "dL_counts": self.dL_counts,
            "s_forward_step": self.s_forward_step,
            "sR_step": self.sR_step,
            "sL_step": self.sL_step,
        }

        return obs, info

    def reset(self, seed=None, options=None):
        global last_cmd
        super().reset(seed=seed)

        last_cmd = None

        self.episode_step = 0
        self.reset_tracking_state()
        self.prev_R_est = None
        self.prev_theta_est_deg = None
        self.episode_reward = 0.0
        self.episode_index += 1

        send_cmd(self.ser, "S")
        time.sleep(0.3)

        print("\nRESET EPISODE")
        print("Place robot and target, then press ENTER.")
        input()

        obs, info = None, None
        while obs is None:
            self._update_gyro()
            self._update_encoders()
            obs, info = self._get_camera_and_estimate()

        self.prev_R_est = float(obs[0])
        self.prev_theta_est_deg = float(obs[6])

        self.last_obs = obs
        self.last_info = info

        if self.render_mode:
            self._render(info, obs, reward=None, action_char="S", done=False, success=False, collision=False)

        return obs, {}

    def step(self, action):
        self.episode_step += 1

        action_char = ACTION_MEANINGS[int(action)]
        send_cmd(self.ser, action_char)

        t0 = time.time()
        obs = None
        info = None

        while time.time() - t0 < CONTROL_DT:
            self._update_gyro()
            self._update_encoders()
            obs, info = self._get_camera_and_estimate()

            if self.render_mode and obs is not None and info is not None:
                self._render(
                    info,
                    obs,
                    reward=None,
                    action_char=action_char,
                    done=False,
                    success=False,
                    collision=False
                )

        dt = time.time() - t0

        if obs is None:
            obs = np.array([
                MAX_OBS_DISTANCE,
                MAX_OBS_DISTANCE,
                MAX_OBS_DISTANCE,
                MAX_OBS_DISTANCE,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0
            ], dtype=np.float32)

            reward = -1.0
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, {"camera_fail": True}

        curr_R_est = float(obs[0])
        curr_theta_est_deg = float(obs[6])

        reward = 0.0
        success = False
        collision = False

        reward_success = 0.0
        reward_collision = 0.0
        reward_distance = 0.0
        reward_angle = 0.0
        reward_forward = 0.0
        reward_avoidance = 0.0
        reward_time = 0.0

        if curr_R_est < TARGET_REACHED_DIST_M and abs(curr_theta_est_deg) < SUCCESS_ANGLE_THRESH_DEG:
            reward_success = SUCCESS_REWARD + ANGLE_BONUS_REWARD
            success = True
        else:
            unsafe_forward = (
                action_char == "F" and
                (info["front_state"] == "OBSTACLE" or info["front_state"] == "UNKNOWN")
            )

            if unsafe_forward:
                reward_collision = COLLISION_PENALTY
                collision = True

        if self.prev_R_est is not None:
            delta_R = self.prev_R_est - curr_R_est
            delta_R = float(np.clip(delta_R, -MAX_R_DELTA_PER_STEP, MAX_R_DELTA_PER_STEP))
            reward_distance = DISTANCE_DELTA_GAIN * delta_R

        if self.prev_theta_est_deg is not None:
            reward_angle = ANGLE_DELTA_GAIN * (
                abs(self.prev_theta_est_deg) - abs(curr_theta_est_deg)
            )

            if curr_theta_est_deg < 0 and action_char == "L":
                reward_angle += 0.005
            elif curr_theta_est_deg > 0 and action_char == "R":
                reward_angle += 0.005

        if AVOIDANCE_ON_UNKNOWN:
            base_avoidance_active = (
                info["depth_avoidance_active"] or
                info["front_state"] == "OBSTACLE" or
                info["front_state"] == "UNKNOWN"
            )
        else:
            base_avoidance_active = (
                info["depth_avoidance_active"] or
                info["front_state"] == "OBSTACLE"
            )

        avoidance_active = (
            base_avoidance_active and
            (not info["target_like_front_object"]) and
            (not info["disable_avoidance_near_target"])
        )

        path_clear = (
            info["front_state"] == "SAFE"
            and (not avoidance_active)
            and info["center_danger"] < PATH_CLEAR_CENTER_DANGER_THRESH
        )

        if (
            abs(curr_theta_est_deg) < CENTERED_FORWARD_ANGLE_DEG
            and action_char == "F"
            and path_clear
        ):
            reward_forward = FORWARD_CLEAR_BONUS

        left_danger = info["left_danger"]
        center_danger = info["center_danger"]
        right_danger = info["right_danger"]

        if avoidance_active:
            if action_char == "L":
                reward_avoidance += AVOIDANCE_REWARD_GAIN * (right_danger - left_danger)
                reward_avoidance += TURN_TO_CLEAR_CENTER_BONUS_GAIN * center_danger
            elif action_char == "R":
                reward_avoidance += AVOIDANCE_REWARD_GAIN * (left_danger - right_danger)
                reward_avoidance += TURN_TO_CLEAR_CENTER_BONUS_GAIN * center_danger
            elif action_char == "F":
                reward_avoidance -= FORWARD_DANGER_PENALTY_GAIN * center_danger
                reward_avoidance -= 0.20 * (left_danger + right_danger)
            elif action_char == "S":
                reward_avoidance -= STOP_DANGER_PENALTY_GAIN * (left_danger + center_danger + right_danger)
        else:
            reward_avoidance = 0.0

        reward_time = -TIME_PENALTY_GAIN * dt

        reward = (
            reward_success
            + reward_collision
            + reward_distance
            + reward_angle
            + reward_forward
            + reward_avoidance
            + reward_time
        )

        self.episode_reward += reward
        self.global_step += 1

        terminated = success or collision
        truncated = False
        done_flag = terminated or truncated

        theta_real_deg = ""
        if info["theta_real"] is not None:
            theta_real_deg = rad_to_deg(info["theta_real"])

        theta_est_deg = ""
        if info["theta_est"] is not None:
            theta_est_deg = rad_to_deg(info["theta_est"])

        theta_pure_deg = ""
        if info["theta_pure"] is not None:
            theta_pure_deg = rad_to_deg(info["theta_pure"])

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                SESSION_ID,
                PHASE,
                SCENARIO,
                RUN_COMMENT,
                self.episode_index,
                self.episode_step,
                self.global_step,
                action_char,
                reward,
                reward_success,
                reward_collision,
                reward_distance,
                reward_angle,
                reward_forward,
                reward_avoidance,
                reward_time,
                curr_R_est,
                curr_theta_est_deg,
                theta_real_deg,
                theta_est_deg,
                theta_pure_deg,
                rad_to_deg(info["dpsi_step"]),
                rad_to_deg(info["robot_heading_total"]),
                int(info["target_visible"]),
                int(info["ignore_invalids_near_target"]),
                int(info["disable_avoidance_near_target"]),
                int(info["target_like_front_object"]),
                int(info["depth_avoidance_active"]),
                int(avoidance_active),
                info["front_state"],
                info["front_min_depth_m"] if info["front_min_depth_m"] is not None else "",
                info["front_invalid_ratio"],
                info["left_depth"] if info["left_depth"] is not None else "",
                info["center_depth"] if info["center_depth"] is not None else "",
                info["right_depth"] if info["right_depth"] is not None else "",
                info["left_invalid_ratio"],
                info["center_invalid_ratio"],
                info["right_invalid_ratio"],
                info["left_danger"],
                info["center_danger"],
                info["right_danger"],
                int(success),
                int(collision),
                int(done_flag)
            ])

        print(
            f"ep={self.episode_index} step={self.episode_step} action={action_char} "
            f"total={reward:.3f} success={reward_success:.3f} collision={reward_collision:.3f} "
            f"distance={reward_distance:.3f} angle={reward_angle:.3f} "
            f"forward={reward_forward:.3f} avoid={reward_avoidance:.3f} time={reward_time:.3f} | "
            f"Ld={left_danger:.2f} Cd={center_danger:.2f} Rd={right_danger:.2f} | "
            f"front={info['front_state']} depth_avoid={info['depth_avoidance_active']} "
            f"disable_avoid_near_target={info['disable_avoidance_near_target']} "
            f"target_like_front={info['target_like_front_object']} "
            f"avoid_active={avoidance_active} ignore_invalids={info['ignore_invalids_near_target']} | "
            f"s_forward={self.s_forward_step:.4f} m | dpsi={rad_to_deg(self.dpsi_step):.2f} deg"
        )

        if terminated or truncated:
            print(f"Episode {self.episode_index} done | total reward = {self.episode_reward:.3f}")
            send_cmd(self.ser, "S")

        self.prev_R_est = curr_R_est
        self.prev_theta_est_deg = curr_theta_est_deg
        self.last_obs = obs
        self.last_info = info

        if self.render_mode:
            self._render(
                info,
                obs,
                reward=reward,
                action_char=action_char,
                done=(terminated or truncated),
                success=success,
                collision=collision
            )

        return obs, reward, terminated, truncated, info

    def _render(self, info, obs, reward, action_char, done, success, collision):
        display = info["color_image"].copy()
        mask = info["mask"]
        depth_image = info["depth_image"]

        h_img, w_img = display.shape[:2]
        cx_draw = int(self.intr.ppx)
        fx = self.intr.fx
        cx = self.intr.ppx

        cv2.line(display, (cx_draw, 0), (cx_draw, h_img), (255, 0, 0), 1)

        if info["detection"] is not None:
            x, y, w, h = info["detection"]["bbox"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display, (info["detection"]["u"], info["detection"]["v"]), CENTER_DRAW_RADIUS, (0, 0, 255), -1)

        if info["x_real"] is not None and info["z_real"] is not None and info["z_real"] > 0:
            xpix_real = int(cx + fx * (info["x_real"] / info["z_real"]))
            xpix_real = max(0, min(w_img - 1, xpix_real))
            cv2.line(display, (xpix_real, 0), (xpix_real, h_img), (0, 255, 0), 2)

        if self.x_est is not None and self.z_est is not None and self.z_est > 0:
            xpix_est = int(cx + fx * (self.x_est / self.z_est))
            xpix_est = max(0, min(w_img - 1, xpix_est))
            cv2.line(display, (xpix_est, 0), (xpix_est, h_img), (0, 255, 255), 2)

        if self.x_pure is not None and self.z_pure is not None and self.z_pure > 0:
            xpix_pure = int(cx + fx * (self.x_pure / self.z_pure))
            xpix_pure = max(0, min(w_img - 1, xpix_pure))
            cv2.line(display, (xpix_pure, 0), (xpix_pure, h_img), (255, 0, 255), 2)

        fx1, fy1, fx2, fy2 = info["box"]
        if info["front_state"] == "OBSTACLE":
            front_color = (0, 0, 255)
        elif info["front_state"] == "UNKNOWN":
            front_color = (0, 165, 255)
        else:
            front_color = (0, 255, 0)
        cv2.rectangle(display, (fx1, fy1), (fx2, fy2), front_color, 2)

        theta_real_disp = "NA" if info["theta_real"] is None else f"{rad_to_deg(info['theta_real']):.2f}"
        theta_est_disp = "NA" if info["theta_est"] is None else f"{rad_to_deg(info['theta_est']):.2f}"
        theta_pure_disp = "NA" if info["theta_pure"] is None else f"{rad_to_deg(info['theta_pure']):.2f}"

        if AVOIDANCE_ON_UNKNOWN:
            base_avoidance_active_disp = (
                info["depth_avoidance_active"] or
                info["front_state"] == "OBSTACLE" or
                info["front_state"] == "UNKNOWN"
            )
        else:
            base_avoidance_active_disp = (
                info["depth_avoidance_active"] or
                info["front_state"] == "OBSTACLE"
            )

        avoidance_active_disp = (
            base_avoidance_active_disp and
            (not info["target_like_front_object"]) and
            (not info["disable_avoidance_near_target"])
        )

        lines = [
            f"Session: {SESSION_ID}",
            f"Phase: {PHASE}",
            f"Scenario: {SCENARIO}",
            f"Action: {action_char}",
            f"Reward: {reward if reward is not None else 'NA'}",
            f"Visible: {info['target_visible']}",
            f"Ignore invalids near target: {info['ignore_invalids_near_target']}",
            f"Disable avoidance near target: {info['disable_avoidance_near_target']}",
            f"Target-like front object: {info['target_like_front_object']}",
            f"Depth avoidance active: {info['depth_avoidance_active']}",
            f"Avoidance active: {avoidance_active_disp}",
            f"Obs R_est: {obs[0]:.3f} m",
            f"Obs angle used: {obs[6]:.2f} deg",
            f"Theta real: {theta_real_disp} deg",
            f"Theta est: {theta_est_disp} deg",
            f"Theta pure: {theta_pure_disp} deg",
            f"Left depth: {obs[1]:.3f} m",
            f"Center depth: {obs[2]:.3f} m",
            f"Right depth: {obs[3]:.3f} m",
            f"Front invalid: {obs[4]:.2f}",
            f"L/C/R invalid: {obs[7]:.2f} / {obs[8]:.2f} / {obs[9]:.2f}",
            f"L/C/R danger: {info['left_danger']:.2f} / {info['center_danger']:.2f} / {info['right_danger']:.2f}",
            f"Front state: {info['front_state']}",
            f"Gyro: {info['gyro_yaw_rate']:.3f} rad/s  dt={info['dt_gyro']:.4f}",
            f"dpsi_step: {rad_to_deg(info['dpsi_step']):.2f} deg",
            f"Heading total: {rad_to_deg(info['robot_heading_total']):.2f} deg",
            f"Enc: R={info['right_count']} L={info['left_count']} dR={info['dR_counts']} dL={info['dL_counts']}",
            f"s_forward={info['s_forward_step']:.4f} m",
            f"Done: {done}  Success: {success}  Collision: {collision}"
        ]

        y0 = 25
        for i, line in enumerate(lines):
            cv2.putText(
                display,
                line,
                (10, y0 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2
            )

        depth_vis = depth_image.copy()
        depth_vis = cv2.convertScaleAbs(depth_vis, alpha=0.08)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.rectangle(depth_vis, (fx1, fy1), (fx2, fy2), front_color, 2)

        if collision:
            collision_text = "COLLISION = YES"
            collision_color = (0, 0, 255)
        else:
            collision_text = "COLLISION = NO"
            collision_color = (0, 255, 0)

        cv2.putText(
            depth_vis,
            collision_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            collision_color,
            2
        )

        cv2.putText(
            depth_vis,
            f"Front state: {info['front_state']}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        front_min_str = "None" if info["front_min_depth_m"] is None else f"{info['front_min_depth_m']:.3f}"
        cv2.putText(
            depth_vis,
            f"Front min depth: {front_min_str} m",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Color View", display)
        cv2.imshow("Mask", mask)
        cv2.imshow("Depth View", depth_vis)
        cv2.waitKey(1)


    def stop_robot(self):
        send_cmd(self.ser, "S")

    def close(self):
        try:
            send_cmd(self.ser, "S")
        except Exception:
            pass

        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass

        try:
            self.imu_pipeline.stop()
        except Exception:
            pass

        try:
            self.cam_pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()


def train_model():
    env = RealRobotDQNEnv(render_mode=True)

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing DQN model...")
        model = DQN.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("Creating new DQN model...")
        model = DQN(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=500,
            batch_size=64,
            gamma=0.99,
            train_freq=2,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.4,
            exploration_initial_eps=0.5,
            exploration_final_eps=0.1,
            tensorboard_log="./dqn_tensorboard/"
        )

    try:
        model.learn(
            total_timesteps=3000,
            reset_num_timesteps=False
        )
        model.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")
        print(f"Step log saved to {LOG_PATH}")
    finally:
        env.close()


def run_model(model_path=MODEL_PATH, episodes=5):
    env = RealRobotDQNEnv(render_mode=True)
    model = DQN.load(model_path)

    try:
        for ep in range(episodes):
            print(f"\nRUN EPISODE {ep + 1}")
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        print(f"Step log saved to {LOG_PATH}")
    finally:
        env.close()

# this is a test comment
if __name__ == "__main__":
    prompt_run_metadata()
    ensure_log_file()

    MODE = "train"   # "train" or "run"

    if MODE == "train":
        train_model()
    else:
        run_model()