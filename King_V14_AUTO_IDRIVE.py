import math
import time
import csv
import os
import select
import sys
import serial
import cv2
import numpy as np
import pyrealsense2 as rs
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

LOWER_HSV = np.array([145, 80, 80])
UPPER_HSV = np.array([179, 255, 255])

MIN_AREA = 200
PATCH_RADIUS = 3
CENTER_DRAW_RADIUS = 6

CAM_W_D = 480
CAM_H_D = 270
CAM_W_C = 640
CAM_H_C = 480
CAM_FPS = 30

WHEEL_RADIUS = 0.039
CPR = 620

GYRO_SIGN = 1.0
GYRO_YAW_AXIS = "y"
MAX_GYRO_DT = 0.05
GYRO_DEADBAND = 0.01

MAX_FORWARD_STEP = 0.20
MAX_TURN_STEP_DEG = 45.0

TARGET_REACHED_DIST_M = 0.35
SUCCESS_ANGLE_THRESH_DEG = 15.0

FRONT_OBSTACLE_DISTANCE = 0.30
AVOIDANCE_TRIGGER_DISTANCE = 0.55
FRONT_INVALID_THRESH = 0.50
UNKNOWN_FRAMES_NEEDED = 4

MAX_OBS_DISTANCE = 10.0
IGNORE_INVALIDS_NEAR_TARGET_DIST = 0.4
DISABLE_AVOIDANCE_NEAR_TARGET_DIST = 0.57

TARGET_DEPTH_MATCH_THRESH = 0.25
TARGET_CENTER_MATCH_DEG = 18.0

TARGET_BAND_Y1_FRAC = 0.00
TARGET_BAND_Y2_FRAC = 0.35

ACTION_MEANINGS = {
    0: "F",
    1: "L",
    2: "R",
}

CONTROL_DT = 0.15

SUCCESS_REWARD = 25.0
ANGLE_BONUS_REWARD = 25.0
COLLISION_PENALTY = -35.0
DISTANCE_DELTA_GAIN = 8.0
ANGLE_DELTA_GAIN = 0.015
TIME_PENALTY_GAIN = 0.35

CENTERED_FORWARD_ANGLE_DEG = 7.0
FORWARD_CLEAR_BONUS = 0.15
PATH_CLEAR_CENTER_DANGER_THRESH = 0.40

MAX_R_DELTA_PER_STEP = 0.10
MAX_MEASURED_R_JUMP = 0.40

# R confidence filtering for partial target visibility / corner reacquisition.
# Angle is trusted immediately when target is visible, but R must look stable.
# R measurement trust rules for partial target / corner cases.
# Angle is trusted immediately when target color is visible.
# Distance R is accepted only if it is stable, close to prediction,
# and not suspiciously closer than prediction.
MAX_R_REAL_FRAME_JUMP = 0.15
MIN_STABLE_R_FRAMES = 2
SUSPICIOUS_CLOSER_MARGIN = 0.20

INVALID_DANGER_GAIN = 0.40
DEPTH_DANGER_GAIN = 0.75
AVOIDANCE_REWARD_GAIN = 0.02
FORWARD_DANGER_PENALTY_GAIN = 0.30
TURN_TO_CLEAR_CENTER_BONUS_GAIN = 0.03
STOP_DANGER_PENALTY_GAIN = 0.04

AVOIDANCE_ON_UNKNOWN = True
AVOIDANCE_CLEAR_FORWARD_STEPS = 2

MODEL_PATH = "dqn_target_nav_model_GAMBLE"
LOG_PATH = "dqn_step_log_GAMBLE.csv"

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
    """
    Short single-read gyro update used during reset only.
    The actual step() function uses integrate_gyro_blocking() so the robot
    does not miss most of the rotation during each control action.
    """
    try:
        frames = imu_pipeline.wait_for_frames(timeout_ms=5)
    except Exception:
        frames = None

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
    total_dpsi = float(np.clip(total_dpsi, -max_turn_step, max_turn_step))

    return total_dpsi, latest_gyro, latest_ts, total_dt


def integrate_gyro_blocking(imu_pipeline, last_gyro_ts, axis_name="y", duration=CONTROL_DT):
    """
    Integrate gyro continuously over one full control action.

    This fixes the repeated dpsi=0.00 problem that happens when the code
    samples the IMU in tiny chunks and misses gyro frames while the robot is
    physically turning.
    """
    t_start = time.time()

    total_dpsi = 0.0
    total_dt = 0.0
    latest_gyro = None
    latest_ts = last_gyro_ts

    while time.time() - t_start < duration:
        try:
            frames = imu_pipeline.wait_for_frames(timeout_ms=10)
        except Exception:
            continue

        if not frames:
            continue

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
    total_dpsi = float(np.clip(total_dpsi, -max_turn_step, max_turn_step))

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
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0:
        return 1.0
    if depth_m >= danger_distance:
        return 0.0
    return float(np.clip((danger_distance - depth_m) / danger_distance, 0.0, 1.0))


def combined_direction_danger(depth_m, inv_ratio, danger_distance):
    inv_ratio = clip_ratio(inv_ratio)

    # If the ROI is mostly invalid, treat it as a high-danger region.
    # This prevents danger from collapsing when a close obstacle causes
    # the depth camera to lose almost all valid pixels.
    if inv_ratio >= 0.75:
        return 0.75

    inv_part = INVALID_DANGER_GAIN * inv_ratio
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
            -1.0,
            -1.0,
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
            1.0,
            1.0,
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

        # Used to decide when measured target distance R is trustworthy.
        # This prevents obstacle edges/corners from corrupting R when the target
        # first becomes partially visible.
        self.prev_R_real = None
        self.stable_R_frames = 0
        self.last_R_filter_reason = ""

        self.prev_action_char = None
        self.avoidance_timer = 0

        # Once avoidance starts, keep it active until the robot completes
        # two consecutive safe Forward actions.
        self.avoidance_maneuver_active = False
        self.avoidance_clear_forward_count = 0

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

        # IMPORTANT SENSOR-FUSION FIX:
        # If the target is visible, ALWAYS trust the measured camera angle.
        # But only trust measured R after it looks believable/stable.
        #
        # This handles corner/partial-visibility cases:
        #   - target color becomes visible first -> theta_real is useful
        #   - depth may still hit the obstacle edge -> R_real may be too close
        #   - keep predicted R until measured R stabilizes
        if target_visible:
            self.theta_est = theta_real

            use_measured_R = True
            r_filter_reasons = []

            if R_pred_est is not None:
                R_jump = abs(R_real - R_pred_est)

                if R_jump > MAX_MEASURED_R_JUMP:
                    use_measured_R = False
                    r_filter_reasons.append(f"R_jump={R_jump:.3f}")

                if R_real < (R_pred_est - SUSPICIOUS_CLOSER_MARGIN):
                    use_measured_R = False
                    r_filter_reasons.append(
                        f"R_real_too_close={R_real:.3f}<pred-{SUSPICIOUS_CLOSER_MARGIN:.2f}"
                    )

            if self.prev_R_real is not None:
                R_real_frame_jump = abs(R_real - self.prev_R_real)
                if R_real_frame_jump > MAX_R_REAL_FRAME_JUMP:
                    use_measured_R = False
                    self.stable_R_frames = 0
                    r_filter_reasons.append(f"R_real_frame_jump={R_real_frame_jump:.3f}")
                else:
                    self.stable_R_frames += 1
            else:
                # First visible frame after being hidden. Do not trust R immediately.
                use_measured_R = False
                self.stable_R_frames = 0
                r_filter_reasons.append("first_visible_R_frame")

            if self.stable_R_frames < MIN_STABLE_R_FRAMES:
                use_measured_R = False
                r_filter_reasons.append(f"stable_R_frames={self.stable_R_frames}")

            if use_measured_R or R_pred_est is None:
                # R is stable/believable, or there is no prediction available.
                self.R_est = R_real
                self.x_est = x_real
                self.z_est = z_real
                self.last_R_filter_reason = "R_measured_used"
            else:
                # Keep predicted distance, but snap bearing to real measured angle.
                self.R_est = R_pred_est
                self.x_est = self.R_est * math.sin(self.theta_est)
                self.z_est = self.R_est * math.cos(self.theta_est)
                self.last_R_filter_reason = ";".join(r_filter_reasons)
                print(
                    f"[R FILTER] Kept predicted R, used real theta. "
                    f"R_real={R_real:.3f}, R_pred={R_pred_est:.3f}, "
                    f"stable={self.stable_R_frames}, reason={self.last_R_filter_reason}, "
                    f"theta_real={rad_to_deg(theta_real):.2f} deg"
                )

            self.prev_R_real = R_real
        else:
            self.prev_R_real = None
            self.stable_R_frames = 0
            self.last_R_filter_reason = "target_not_visible"

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

        # Use filtered estimate for near-target/avoidance decisions.
        # Do not use raw R_real here because it may be polluted by an obstacle edge.
        target_range_for_compare = self.R_est

        ignore_invalids_near_target = (
            target_range_for_compare is not None and
            target_range_for_compare < IGNORE_INVALIDS_NEAR_TARGET_DIST
        )

        disable_avoidance_near_target = (
                target_visible and
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

        target_band_y1 = int(ch * TARGET_BAND_Y1_FRAC)
        target_band_y2 = int(ch * TARGET_BAND_Y2_FRAC)

        left_roi = depth_image[strip_y1:strip_y2, 0:third]
        center_roi = depth_image[strip_y1:strip_y2, third:2 * third]
        right_roi = depth_image[strip_y1:strip_y2, 2 * third:cw]
        target_band_roi = depth_image[target_band_y1:target_band_y2, 0:cw]
        target_band_min_depth_m = nearest_valid_depth_m(target_band_roi)

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

        side_danger_distance = AVOIDANCE_TRIGGER_DISTANCE * 1.2

        left_danger = combined_direction_danger(left_depth, left_invalid_ratio, side_danger_distance)
        center_danger = combined_direction_danger(center_depth, center_invalid_ratio, AVOIDANCE_TRIGGER_DISTANCE)
        right_danger = combined_direction_danger(right_depth, right_invalid_ratio, side_danger_distance)

        target_like_front_object = False
        target_depth_match_disable_avoidance = False
        target_in_band_roi = False

        if detection is not None:
            target_u = detection["u"]
            target_v = detection["v"]

            target_in_band_roi = (
                0 <= target_u < cw and
                target_band_y1 <= target_v < target_band_y2
            )

        depth_avoidance_active = (
            front_min_depth_m is not None and
            front_min_depth_m < AVOIDANCE_TRIGGER_DISTANCE
        )

        # HARD TARGET DEPTH OVERRIDE:
        # Disable avoidance if either:
        # 1. R_est is below the old near-target distance, or
        # 2. the target is visible and the target-depth match logic says the close object is the target.
        if disable_avoidance_near_target:
            disable_avoidance_near_target = True
            ignore_invalids_near_target = True
            depth_avoidance_active = False
            front_state = "SAFE"
            front_invalid_ratio = 0.0
            left_invalid_ratio = 0.0
            center_invalid_ratio = 0.0
            right_invalid_ratio = 0.0
            left_danger = 0.0
            center_danger = 0.0
            right_danger = 0.0
            self.unknown_counter = 0

        theta_rad_obs = deg_to_rad(theta_est_deg_obs)

        obs = np.array([
            clip_obs_distance(R_est_obs),
            clip_obs_distance(left_depth),
            clip_obs_distance(center_depth),
            clip_obs_distance(right_depth),
            clip_ratio(front_invalid_ratio),
            1.0 if target_visible else 0.0,
            math.sin(theta_rad_obs),
            math.cos(theta_rad_obs),
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
            "stable_R_frames": self.stable_R_frames,
            "R_filter_reason": self.last_R_filter_reason,
            "ignore_invalids_near_target": ignore_invalids_near_target,
            "disable_avoidance_near_target": disable_avoidance_near_target,
            "target_like_front_object": target_like_front_object,
            "target_depth_match_disable_avoidance": target_depth_match_disable_avoidance,
            "target_in_band_roi": target_in_band_roi,
            "target_band_y1": target_band_y1,
            "target_band_y2": target_band_y2,
            "target_band_min_depth_m": target_band_min_depth_m,
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

        print("\nRESET EPISODE - TWO STAGE MANUAL SETUP")
        print("STAGE 1: Place robot and make the target clearly visible.")
        print("Press ENTER to acquire the target R_est/theta before adding obstacles.")
        input()

        acquire_obs, acquire_info = None, None
        while acquire_obs is None or acquire_info is None or not acquire_info.get("target_visible", False):
            self._update_gyro()
            self._update_encoders()
            acquire_obs, acquire_info = self._get_camera_and_estimate()

            if acquire_info is None or not acquire_info.get("target_visible", False):
                print("Target not visible yet. Adjust target/robot and press ENTER to try again.")
                input()

        captured_R = self.R_est
        captured_theta = self.theta_est
        captured_x = self.x_est
        captured_z = self.z_est

        captured_R_disp = "NA" if captured_R is None else f"{captured_R:.3f} m"
        captured_theta_disp = "NA" if captured_theta is None else f"{rad_to_deg(captured_theta):.2f} deg"

        print("\nTARGET ACQUIRED")
        print(f"Captured R_est = {captured_R_disp}")
        print(f"Captured theta = {captured_theta_disp}")
        print("STAGE 2: Now place the obstacle(s) in front, even if they totally block the target.")
        print("Press ENTER when ready to start manual data collection.")
        input()

        # Keep the acquired target estimate as the hidden-target initial state.
        # After obstacles are placed, the target may be fully blocked, so the
        # next camera update will use this saved R/theta estimate instead of
        # needing the target to still be visible.
        self.R_est = captured_R
        self.theta_est = captured_theta
        self.x_est = captured_x
        self.z_est = captured_z

        self.R_pure = captured_R
        self.theta_pure = captured_theta
        self.x_pure = captured_x
        self.z_pure = captured_z

        obs, info = None, None
        while obs is None:
            self._update_gyro()
            self._update_encoders()
            obs, info = self._get_camera_and_estimate()

        self.prev_R_est = float(obs[0])
        self.prev_theta_est_deg = rad_to_deg(math.atan2(obs[6], obs[7]))

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

        # Integrate gyro continuously over the whole action window.
        # This avoids missing rotation and getting dpsi=0.00 while turning.
        dpsi_accum, latest_gyro, latest_ts, dt_gyro_accum = integrate_gyro_blocking(
            self.imu_pipeline,
            self.last_gyro_ts,
            axis_name=GYRO_YAW_AXIS,
            duration=CONTROL_DT
        )

        # IMPORTANT MANUAL CONTROL SAFETY:
        # The Arduino treats F/L/R as continuous commands, so stop after
        # each control window. This makes every manual keypress a short pulse:
        # F/L/R for CONTROL_DT seconds, then S.
        send_cmd(self.ser, "S")

        if latest_gyro is not None:
            self.latest_gyro = latest_gyro

        if latest_ts is not None:
            self.last_gyro_ts = latest_ts

        self.dpsi_step = dpsi_accum
        self.dt_gyro = dt_gyro_accum
        self.robot_heading_total = wrap_angle_rad(self.robot_heading_total + self.dpsi_step)

        # Now update encoders and camera once using the full dpsi for this action.
        self._update_encoders()
        obs, info = self._get_camera_and_estimate()

        if info is not None:
            info["dpsi_step"] = self.dpsi_step
            info["dt_gyro"] = self.dt_gyro
            info["robot_heading_total"] = self.robot_heading_total

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
                1.0,
                1.0
            ], dtype=np.float32)

            reward = -1.0
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, {"camera_fail": True}

        curr_R_est = float(obs[0])
        curr_theta_est_deg = rad_to_deg(math.atan2(obs[6], obs[7]))

        reward = 0.0
        success = False
        collision = False

        # Keep the old CSV columns so your plotting/Excel sheets do not break,
        # but these three shaping rewards are now always zero.
        reward_success = 0.0
        reward_collision = 0.0
        reward_distance = 0.0
        reward_angle = 0.0
        reward_forward = 0.0
        reward_avoidance = 0.0
        reward_time = 0.0

        # SUCCESS REWARD ONLY:
        # No distance progress reward and no angle shaping reward.
        if curr_R_est < TARGET_REACHED_DIST_M and action_char == "F":
            if abs(curr_theta_est_deg) < SUCCESS_ANGLE_THRESH_DEG:
                reward_success = SUCCESS_REWARD + ANGLE_BONUS_REWARD
                success = True
            else:
                reward_success = SUCCESS_REWARD
                success = True
        else:
            # COLLISION / UNSAFE FORWARD TERMINATION:
            # This is kept from the old code. It ends the episode if the robot
            # chooses forward while the front is an obstacle or unknown.
            wheel_collision_risk = (
                    info["left_danger"] > 1.0 or
                    info["right_danger"] > 1.0
            )

            unsafe_forward = (
                    action_char == "F" and
                    (not info["disable_avoidance_near_target"]) and
                    (
                            info["front_state"] == "OBSTACLE"
                            or info["front_state"] == "UNKNOWN"
                            or wheel_collision_risk
                    )
            )

            if unsafe_forward:
                reward_collision = COLLISION_PENALTY
                collision = True

        # AVOIDANCE ACTIVATION LOGIC:
        # This is still the same basic avoidance/collision sensor logic.
        # Avoidance stays active for a few steps after it is triggered.
        if AVOIDANCE_ON_UNKNOWN:
            base_avoidance_active = (
                    info["depth_avoidance_active"] or
                    info["front_state"] == "OBSTACLE" or
                    info["front_state"] == "UNKNOWN" or
                    info["left_danger"] > 0.65 or
                    info["right_danger"] > 0.65
            )
        else:
            base_avoidance_active = (
                    info["depth_avoidance_active"] or
                    info["front_state"] == "OBSTACLE" or
                    info["left_danger"] > 0.65 or
                    info["right_danger"] > 0.65
            )

        base_avoidance_active = (
                base_avoidance_active and
                (not info["disable_avoidance_near_target"])
        )

        if info["disable_avoidance_near_target"]:
            self.avoidance_timer = 0
            self.avoidance_maneuver_active = False
            self.avoidance_clear_forward_count = 0
            avoidance_active = False

        else:
            if base_avoidance_active:
                self.avoidance_timer = 0
                self.avoidance_maneuver_active = True
                self.avoidance_clear_forward_count = 0

            elif self.avoidance_timer > 0:
                self.avoidance_timer -= 1

            path_clear_for_forward = (
                    info["front_state"] == "SAFE"
                    and not info["depth_avoidance_active"]
                    and info["center_danger"] < 0.15
                    and info["left_danger"] < 0.35
                    and info["right_danger"] < 0.35
            )

            avoidance_active = self.avoidance_maneuver_active

            if self.avoidance_maneuver_active:
                if action_char == "F" and path_clear_for_forward:
                    self.avoidance_clear_forward_count += 1

                    if (
                            self.avoidance_clear_forward_count
                            >= AVOIDANCE_CLEAR_FORWARD_STEPS
                    ):
                        self.avoidance_maneuver_active = False
                        self.avoidance_clear_forward_count = 0

                elif action_char == "F":
                    self.avoidance_clear_forward_count = 0

        left_danger = info["left_danger"]
        center_danger = info["center_danger"]
        right_danger = info["right_danger"]

        # AVOIDANCE REWARD ONLY:
        # No bonus for getting closer to the target.
        # No angle bonus.
        # No clear-path forward bonus.
        if avoidance_active:
            prev_center_danger = 0.0
            if self.last_info is not None:
                prev_center_danger = self.last_info.get("center_danger", center_danger)

            center_danger_delta = prev_center_danger - center_danger

            if (
                    (self.prev_action_char == "L" and action_char == "R") or
                    (self.prev_action_char == "R" and action_char == "L")
            ):
                reward_avoidance -= 0.10

            if action_char == "L":
                reward_avoidance += 0.10 * (right_danger - left_danger)
                reward_avoidance += 0.04 * center_danger
                reward_avoidance += 0.08 * max(0.0, center_danger_delta)

            elif action_char == "R":
                reward_avoidance += 0.10 * (left_danger - right_danger)
                reward_avoidance += 0.04 * center_danger
                reward_avoidance += 0.08 * max(0.0, center_danger_delta)

            elif action_char == "F":
                reward_avoidance -= 0.60 * center_danger
                reward_avoidance -= 0.20 * (left_danger + right_danger)

                if center_danger < 0.15 and left_danger < 0.35 and right_danger < 0.35:
                    reward_avoidance += 0.20

            elif action_char == "S":
                reward_avoidance -= STOP_DANGER_PENALTY_GAIN * (left_danger + center_danger + right_danger)
        else:
            reward_avoidance = 0.0

        # TIME PENALTY ONLY:
        reward_time = -TIME_PENALTY_GAIN * dt

        reward = (
            reward_success
            + reward_collision
            + reward_avoidance
            + reward_time
        )

        self.episode_reward += reward
        self.global_step += 1

        manual_stop = False

        # PuTTY terminal manual stop:
        # Type q then press ENTER to immediately end the current episode.
        # This does not add any new reward. The step keeps only the normal reward
        # already calculated above, usually just the time penalty unless success/collision happened.
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip().lower()

            if cmd == "q":
                print("MANUAL EPISODE STOP")
                manual_stop = True
                send_cmd(self.ser, "S")

        terminated = success or collision or manual_stop
        truncated = False
        done_flag = terminated or truncated

        # Let the automatic demonstration loop distinguish q + ENTER
        # from a normal success or collision.
        info["manual_stop"] = manual_stop

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

        self.prev_action_char = action_char

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
                info["front_state"] == "UNKNOWN" or
                info["left_danger"] > 0.35 or
                info["right_danger"] > 0.35
            )
        else:
            base_avoidance_active_disp = (
                info["depth_avoidance_active"] or
                info["front_state"] == "OBSTACLE" or
                info["left_danger"] > 0.35 or
                info["right_danger"] > 0.35
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
            f"R filter: {info.get('R_filter_reason', '')}",
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



# Manual control learning settings matched to the main DQN setup.
# Main-code equivalent settings:
# learning_starts=500, train_freq=2, gradient_steps=1,
# batch_size=64, target_update_interval=500.
DEMO_TRAINING_ENABLED = True
DEMO_LEARNING_STARTS = 64
DEMO_TRAIN_FREQ = 2
DEMO_GRADIENT_STEPS = 1
DEMO_BATCH_SIZE = 64
DEMO_TARGET_UPDATE_INTERVAL = 500
SAVE_EVERY_MANUAL_STEPS = 25

KEY_TO_ACTION = {
    "f": 0,
    "l": 1,
    "r": 2,
}


def make_dqn_model(env):
    buffer_path = MODEL_PATH + "_replay_buffer.pkl"

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing DQN model for manual demonstrations...")
        model = DQN.load(MODEL_PATH, env=env, device="cpu")

        if os.path.exists(buffer_path):
            print("Loading existing replay buffer...")
            model.load_replay_buffer(buffer_path)
        else:
            print("No replay buffer found. Starting with empty buffer.")

    else:
        print("Creating new DQN model for manual demonstrations...")
        model = DQN(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            learning_rate=1e-4,
            buffer_size=500000,
            learning_starts=64,
            batch_size=64,
            gamma=0.99,
            train_freq=2,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.6,
            exploration_initial_eps=0.6,
            exploration_final_eps=0.15,
            tensorboard_log="./dqn_tensorboard/"
        )

    new_logger = configure("./logs/", ["stdout"])
    model.set_logger(new_logger)

    return model


def add_manual_transition_to_replay(model, obs, action, reward, next_obs, done, info):
    """
    Add one manually driven transition to the DQN replay buffer.

    This lets your manual driving become demonstration data using the exact same
    observation, reward, done, and info logic as the normal robot environment.
    """
    try:
        model.replay_buffer.add(
            np.array(obs, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array([action]),
            np.array([reward], dtype=np.float32),
            np.array([done]),
            [info]
        )
        model.num_timesteps += 1
        return True
    except Exception as e:
        print("Replay buffer add failed:", e)
        return False


def train_from_manual_buffer_if_ready(model, manual_step_count):
    """
    Train with the same timing as the main DQN settings:
    learning_starts=500, train_freq=2, gradient_steps=1, batch_size=64.

    The only intended difference from normal training is action selection:
    - Normal code: DQN/exploration chooses the action.
    - Manual code: you choose the action.
    """
    if not DEMO_TRAINING_ENABLED:
        return

    try:
        buffer_size_now = model.replay_buffer.size()
    except Exception:
        return

    if buffer_size_now < DEMO_LEARNING_STARTS:
        return

    if manual_step_count % DEMO_TRAIN_FREQ != 0:
        return

    try:
        model.train(
            gradient_steps=DEMO_GRADIENT_STEPS,
            batch_size=DEMO_BATCH_SIZE
        )
    except Exception as e:
        print("Manual control train step failed:", e)


def update_target_network_if_needed(model, manual_step_count):
    """
    DQN uses a separate target network. In normal model.learn(), SB3 updates
    it internally. Since this manual loop replaces model.learn(), we update it
    here on the same 500-step interval used by target_update_interval=500.
    """
    if manual_step_count <= 0:
        return

    if manual_step_count % DEMO_TARGET_UPDATE_INTERVAL != 0:
        return

    try:
        model.q_net_target.load_state_dict(model.q_net.state_dict())
        print("Updated DQN target network")
    except Exception as e:
        print("Target network update failed:", e)


def print_manual_controls():
    print("\nMANUAL DEMO CONTROLS")
    print("f    = forward step")
    print("l    = left step")
    print("r    = right step")
    print("fff  = three forward steps")
    print("llff = two left steps, then two forward steps")
    print("stop = send stop command only, no replay transition")
    print("n    = reset episode")
    print("save = save model")
    print("q    = save model and quit")
    print("\nEach f/l/r command uses env.step(), writes to the normal CSV log,")
    print("and adds the transition to the DQN replay buffer.")
    print(f"DEMO_TRAINING_ENABLED = {DEMO_TRAINING_ENABLED}")
    print(f"DEMO_LEARNING_STARTS = {DEMO_LEARNING_STARTS}")
    print(f"DEMO_TRAIN_FREQ = {DEMO_TRAIN_FREQ}")
    print(f"DEMO_GRADIENT_STEPS = {DEMO_GRADIENT_STEPS}")
    print(f"DEMO_TARGET_UPDATE_INTERVAL = {DEMO_TARGET_UPDATE_INTERVAL}\n")


def run_manual_demo():
    env = RealRobotDQNEnv(render_mode=False)
    model = make_dqn_model(env)

    obs = None
    manual_step_count = 0

    try:
        print_manual_controls()
        obs, _ = env.reset()
        done = False

        while True:
            cmd = input("manual command [f/l/r strings/stop/n/save/q]: ").strip().lower()

            if cmd == "":
                continue

            if cmd == "q":
                env.stop_robot()
                model.save(MODEL_PATH)
                model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
                print(f"Saved model to {MODEL_PATH}")
                print(f"Saved replay buffer.")
                print(f"Step log saved to {LOG_PATH}")
                break

            if cmd == "save":
                model.save(MODEL_PATH)
                model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
                print(f"Saved model to {MODEL_PATH}")
                print("Saved replay buffer.")
                continue

            if cmd == "stop" or cmd == "s":
                env.stop_robot()
                print("Sent STOP. No replay transition added.")
                continue

            if cmd == "n":
                env.stop_robot()
                obs, _ = env.reset()
                done = False
                continue

            # Allow strings of actions, for example:
            #   fff      -> F, F, F
            #   llffff   -> L, L, F, F, F, F
            #   rff      -> R, F, F
            # Each character is still one normal env.step(), one reward,
            # one CSV row, and one replay-buffer transition.
            bad_chars = [c for c in cmd if c not in KEY_TO_ACTION]
            if bad_chars:
                print(f"Unknown command character(s): {bad_chars}. Use only f, l, r, stop, n, save, or q.")
                continue

            for action_cmd in cmd:
                if done:
                    print("Episode was done. Resetting before next manual action.")
                    obs, _ = env.reset()
                    done = False

                action = KEY_TO_ACTION[action_cmd]
                prev_obs = np.array(obs, dtype=np.float32).copy()

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                add_ok = add_manual_transition_to_replay(
                    model=model,
                    obs=prev_obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    info=info
                )

                manual_step_count += 1

                if add_ok:
                    train_from_manual_buffer_if_ready(model, manual_step_count)
                    update_target_network_if_needed(model, manual_step_count)

                obs = next_obs

                print(
                    f"manual_step={manual_step_count} action={ACTION_MEANINGS[action]} "
                    f"reward={reward:.3f} done={done} replay_size={model.replay_buffer.size()}"
                )

                if manual_step_count % SAVE_EVERY_MANUAL_STEPS == 0:
                    model.save(MODEL_PATH)
                    model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
                    print(f"Auto-saved model to {MODEL_PATH}")

                if done:
                    print("Episode ended during command string. Remaining characters were skipped.")
                    break

    finally:
        try:
            env.stop_robot()
        except Exception:
            pass

        try:
            model.save(MODEL_PATH)
            model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
            print(f"Saved model to {MODEL_PATH}")
            print("Saved replay buffer.")
        except Exception as e:
            print("Final model/replay buffer save failed:", e)

        env.close()




# =============================================================================
# AUTOMATIC EXPERT-DEMONSTRATION CONTROLLER
# =============================================================================
#
# This third script uses the same environment, rewards, model, replay buffer,
# CSV log, and online DQN training as IDRIVE.
#
# Action selection is the only major difference:
#   - Outside avoidance: the trained DQN drives deterministically.
#   - During avoidance: a scripted expert chooses the lower-danger side and
#     commits to that turn until the path has remained clear.
#
# Type q then press ENTER while it is running to stop, save, and exit.
# =============================================================================

AUTO_CLEAR_FRAMES_NEEDED = 2
AUTO_CLEAR_CENTER_DANGER = 0.15
AUTO_CLEAR_SIDE_DANGER = 0.35
AUTO_TIE_DANGER_MARGIN = 0.03
AUTO_FORWARD_STEPS_AFTER_CLEAR = 2
AUTO_TARGET_CENTER_DEADBAND_DEG = 12
SAVE_EVERY_AUTO_STEPS = 25


def current_auto_avoidance_trigger(info):
    """Mirror the avoidance trigger used by env.step(), using the current frame."""
    if info is None:
        return False

    if info.get("disable_avoidance_near_target", False):
        return False

    if AVOIDANCE_ON_UNKNOWN:
        return bool(
            info.get("depth_avoidance_active", False)
            or info.get("front_state") == "OBSTACLE"
            or info.get("front_state") == "UNKNOWN"
            or info.get("left_danger", 0.0) > 0.65
            or info.get("right_danger", 0.0) > 0.65
        )

    return bool(
        info.get("depth_avoidance_active", False)
        or info.get("front_state") == "OBSTACLE"
        or info.get("left_danger", 0.0) > 0.65
        or info.get("right_danger", 0.0) > 0.65
    )


def auto_path_is_clear(info):
    """
    Require a genuinely clear forward path before ending a committed turn.

    Requiring multiple consecutive clear frames prevents one noisy depth frame
    from making the controller immediately drive forward again.
    """
    if info is None:
        return False

    return bool(
        info.get("front_state") == "SAFE"
        and not info.get("depth_avoidance_active", False)
        and info.get("center_danger", 1.0) < AUTO_CLEAR_CENTER_DANGER
        and info.get("left_danger", 1.0) < AUTO_CLEAR_SIDE_DANGER
        and info.get("right_danger", 1.0) < AUTO_CLEAR_SIDE_DANGER
    )


def choose_new_expert_turn(info, episode_index):
    """
    Choose the direction with lower danger.

    If danger values are almost tied, use the measured side depths as a
    secondary cue. If those are also tied/unavailable, alternate by episode
    rather than permanently biasing the demonstration data toward one side.
    """
    left_danger = float(info.get("left_danger", 0.0))
    right_danger = float(info.get("right_danger", 0.0))
    danger_difference = right_danger - left_danger

    if danger_difference > AUTO_TIE_DANGER_MARGIN:
        return 1, "lower left danger"

    if danger_difference < -AUTO_TIE_DANGER_MARGIN:
        return 2, "lower right danger"

    left_depth = info.get("left_depth")
    right_depth = info.get("right_depth")

    left_depth_valid = left_depth is not None and np.isfinite(left_depth)
    right_depth_valid = right_depth is not None and np.isfinite(right_depth)

    if left_depth_valid and right_depth_valid:
        if left_depth > right_depth + 0.05:
            return 1, "danger tie; more left clearance"
        if right_depth > left_depth + 0.05:
            return 2, "danger tie; more right clearance"

    if episode_index % 2 == 0:
        return 1, "full tie; alternating left"
    return 2, "full tie; alternating right"


def choose_auto_action(model, obs, info, env, controller_state):
    """
    Select one action and update the persistent expert-controller state.

    controller_state keys:
        committed_action: None, 1 (L), or 2 (R)
        clear_frames: consecutive clear observations while committed
        forced_forward_steps: clean forward steps immediately after clearance
    """
    if info is None:
        theta_deg = math.degrees(math.atan2(float(obs[6]), float(obs[7])))

        if theta_deg > AUTO_TARGET_CENTER_DEADBAND_DEG:
            return 2, "EXPERT: center target right"
        elif theta_deg < -AUTO_TARGET_CENTER_DEADBAND_DEG:
            return 1, "EXPERT: center target left"
        else:
            return 0, "EXPERT: target centered -> forward"

    if info.get("disable_avoidance_near_target", False):
        controller_state["committed_action"] = None
        controller_state["clear_frames"] = 0
        controller_state["forced_forward_steps"] = 0
        controller_state["recenter_after_avoidance"] = True

    # Finish the clean demonstration sequence with a couple forward actions
    # after the obstacle has been cleared.
    if controller_state["forced_forward_steps"] > 0:
        forward_is_unsafe = (
                info.get("front_state") == "OBSTACLE"
                or info.get("front_state") == "UNKNOWN"
                or info.get("depth_avoidance_active", False)
                or info.get("center_danger", 1.0) >= AUTO_CLEAR_CENTER_DANGER
        )

        if forward_is_unsafe:
            controller_state["forced_forward_steps"] = 0
        else:
            controller_state["forced_forward_steps"] -= 1
            return 0, "EXPERT: forward after clearance"

    committed_action = controller_state["committed_action"]

    if committed_action is not None:
        if auto_path_is_clear(info):
            controller_state["clear_frames"] += 1
        else:
            controller_state["clear_frames"] = 0

        if controller_state["clear_frames"] >= AUTO_CLEAR_FRAMES_NEEDED:
            controller_state["committed_action"] = None
            controller_state["clear_frames"] = 0
            controller_state["recenter_after_avoidance"] = False

            # Use the first forced-forward action immediately.
            controller_state["forced_forward_steps"] = max(
                0, AUTO_FORWARD_STEPS_AFTER_CLEAR - 1
            )
            return 0, "EXPERT: first forward after clearance"

        else:
            return int(committed_action), "EXPERT: committed turn"

    if controller_state["recenter_after_avoidance"]:
        # If avoidance becomes active again, stop recentering and handle the obstacle.
        if current_auto_avoidance_trigger(info):
            controller_state["recenter_after_avoidance"] = False
        else:
            theta_deg = math.degrees(
                math.atan2(float(obs[6]), float(obs[7]))
            )

            if theta_deg > AUTO_TARGET_CENTER_DEADBAND_DEG:
                return 2, f"EXPERT: recenter right after avoidance theta={theta_deg:.1f}"

            elif theta_deg < -AUTO_TARGET_CENTER_DEADBAND_DEG:
                return 1, f"EXPERT: recenter left after avoidance theta={theta_deg:.1f}"

            else:
                controller_state["recenter_after_avoidance"] = False
                controller_state["forced_forward_steps"] = max(
                    0, AUTO_FORWARD_STEPS_AFTER_CLEAR - 1
                )
                return 0, f"EXPERT: recentered -> forward theta={theta_deg:.1f}"

    if current_auto_avoidance_trigger(info):
        action, reason = choose_new_expert_turn(info, env.episode_index)
        controller_state["committed_action"] = int(action)
        controller_state["clear_frames"] = 0
        return int(action), f"EXPERT: {reason}"

    theta_deg = math.degrees(
        math.atan2(float(obs[6]), float(obs[7]))
    )

    if theta_deg > AUTO_TARGET_CENTER_DEADBAND_DEG:
        return 2, f"EXPERT: target right -> turn right theta={theta_deg:.1f}"

    elif theta_deg < -AUTO_TARGET_CENTER_DEADBAND_DEG:
        return 1, f"EXPERT: target left -> turn left theta={theta_deg:.1f}"

    else:
        return 0, f"EXPERT: target centered -> forward theta={theta_deg:.1f}"


def print_auto_controls():
    print("\nAUTOMATIC DEMONSTRATION MODE")
    print("Outside avoidance: deterministic DQN action")
    print("During avoidance: lower-danger committed expert turn")
    print(
        f"Turn releases after {AUTO_CLEAR_FRAMES_NEEDED} consecutive clear frames: "
        f"center<{AUTO_CLEAR_CENTER_DANGER}, sides<{AUTO_CLEAR_SIDE_DANGER}"
    )
    print(f"Forced forward steps after clearance: {AUTO_FORWARD_STEPS_AFTER_CLEAR}")
    print("Type q then press ENTER at any time to save and quit.")
    print(f"DEMO_TRAINING_ENABLED = {DEMO_TRAINING_ENABLED}")
    print(f"DEMO_LEARNING_STARTS = {DEMO_LEARNING_STARTS}")
    print(f"DEMO_TRAIN_FREQ = {DEMO_TRAIN_FREQ}")
    print(f"DEMO_GRADIENT_STEPS = {DEMO_GRADIENT_STEPS}")
    print(f"DEMO_TARGET_UPDATE_INTERVAL = {DEMO_TARGET_UPDATE_INTERVAL}\n")


def run_auto_demo():
    env = RealRobotDQNEnv(render_mode=False)
    model = make_dqn_model(env)

    obs = None
    auto_step_count = 0
    quit_requested = False

    controller_state = {
        "committed_action": None,
        "clear_frames": 0,
        "forced_forward_steps": 0,
        "recenter_after_avoidance": False,
    }

    try:
        print_auto_controls()
        obs, _ = env.reset()
        done = False

        while not quit_requested:
            if done:
                print("\nEpisode ended. Starting the normal two-stage setup for the next episode.")
                obs, _ = env.reset()
                done = False
                controller_state = {
                    "committed_action": None,
                    "clear_frames": 0,
                    "forced_forward_steps": 0,
                    "recenter_after_avoidance": False,
                }

            current_info = env.last_info

            action, action_source = choose_auto_action(
                model=model,
                obs=obs,
                info=current_info,
                env=env,
                controller_state=controller_state,
            )

            prev_obs = np.array(obs, dtype=np.float32).copy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            add_ok = add_manual_transition_to_replay(
                model=model,
                obs=prev_obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                info=info,
            )

            auto_step_count += 1

            if add_ok:
                train_from_manual_buffer_if_ready(model, auto_step_count)
                update_target_network_if_needed(model, auto_step_count)

            obs = next_obs

            committed_char = (
                "NONE"
                if controller_state["committed_action"] is None
                else ACTION_MEANINGS[controller_state["committed_action"]]
            )

            print(
                f"auto_step={auto_step_count} action={ACTION_MEANINGS[action]} "
                f"source='{action_source}' committed={committed_char} "
                f"clear_frames={controller_state['clear_frames']} "
                f"forced_forward={controller_state['forced_forward_steps']} "
                f"recenter={controller_state['recenter_after_avoidance']} "
                f"reward={reward:.3f} done={done} "
                f"replay_size={model.replay_buffer.size()}"
            )

            if auto_step_count % SAVE_EVERY_AUTO_STEPS == 0:
                model.save(MODEL_PATH)
                model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
                print(f"Auto-saved model and replay buffer to {MODEL_PATH}")

            if info.get("manual_stop", False):
                print("q received. Saving and exiting automatic demonstration mode.")
                quit_requested = True

    finally:
        try:
            env.stop_robot()
        except Exception:
            pass

        try:
            model.save(MODEL_PATH)
            model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
            print(f"Saved model to {MODEL_PATH}")
            print("Saved replay buffer.")
            print(f"Step log saved to {LOG_PATH}")
        except Exception as e:
            print("Final model/replay buffer save failed:", e)

        env.close()


# This is the automatic expert-demonstration version of King_V14_IDRIVE.py.
if __name__ == "__main__":
    prompt_run_metadata()
    ensure_log_file()
    run_auto_demo()
