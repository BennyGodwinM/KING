"""
RL_follower_train.py  —  Stable Baselines 3 / PPO target-following robot

Observation vector (7 floats):
  [0]  distance_to_target      metres  (MAX_DEPTH_M when invisible)
  [1]  nearest_left_depth       metres  (nearest obstacle in left depth zone)
  [2]  nearest_center_depth     metres  (nearest obstacle in centre depth zone)
  [3]  nearest_right_depth      metres  (nearest obstacle in right depth zone)
  [4]  invalid_pixel_ratio      0–1     (% invalid pixels in centre box)
  [5]  target_visible           0 or 1
  [6]  angle_to_target          radians (−π … +π, positive = target to the right)

Reward:
  +10.0               distance < 0.2 m (goal reached)
  +3.0                |angle| < 10 deg
  −5.0                collision  (centre depth < MIN_DEPTH or FORWARD into >50% invalid)
  +1.0 * Δdist        progress toward target   (prev_dist − curr_dist)
  +0.05 * Δangle      angle improvement        (|prev_angle| − |curr_angle|)
  −0.01 * dt          time penalty per step

Actions (Discrete 6):
  0  FORWARD
  1  TURN_LEFT_BIG
  2  TURN_LEFT_SMALL
  3  TURN_RIGHT_SMALL
  4  TURN_RIGHT_BIG
  5  STOP

Usage:
  # Train from scratch
  python RL_follower_train.py

  # Continue training from a saved model
  python RL_follower_train.py --load rl_follower_ppo.zip

  # Inference only (no weight updates)
  python RL_follower_train.py --inference

  # Override total timesteps
  python RL_follower_train.py --timesteps 100000
"""

import argparse
import math
import os
import time

import cv2
import gymnasium as gym
import numpy as np
import pyrealsense2 as rs
import serial
import serial.tools.list_ports
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Serial
SERIAL_PORT    = "COM3"        # overridden by auto-detect if Arduino found
BAUD_RATE      = 9600
SERIAL_TIMEOUT = 0.01

CMD_FORWARD = "F"
CMD_LEFT    = "L"
CMD_RIGHT   = "R"
CMD_STOP    = "S"

# Camera
COLOR_W, COLOR_H = 640, 480
DEPTH_W, DEPTH_H = 640, 480   # after alignment to colour frame
FPS              = 30
FRAME_TIMEOUT_MS = 200

# Depth zones — horizontal strip spanning ±15 % of frame height around centre
# Divided into equal left / centre / right thirds
STRIP_TOP    = int(COLOR_H * 0.35)
STRIP_BOTTOM = int(COLOR_H * 0.65)
ZONE_LEFT_X   = (0,              COLOR_W // 3)
ZONE_CENTER_X = (COLOR_W // 3,   2 * COLOR_W // 3)
ZONE_RIGHT_X  = (2 * COLOR_W // 3, COLOR_W)

MIN_DEPTH_M = 0.20    # collision threshold (metres)
MAX_DEPTH_M = 4.00    # used as "no reading" sentinel

# HSV target detection — pink / magenta (consistent with existing files)
LOWER_HSV = np.array([160, 150, 120], dtype=np.uint8)
UPPER_HSV = np.array([179, 255, 255], dtype=np.uint8)
MIN_AREA  = 60        # minimum contour area in px²

# Action durations (seconds)
ACTION_CMDS = {
    0: CMD_FORWARD,
    1: CMD_LEFT,
    2: CMD_LEFT,
    3: CMD_RIGHT,
    4: CMD_RIGHT,
    5: CMD_STOP,
}
ACTION_DURATIONS = {
    0: 0.10,    # FORWARD
    1: 0.10,    # TURN_LEFT_BIG
    2: 0.025,   # TURN_LEFT_SMALL
    3: 0.025,   # TURN_RIGHT_SMALL
    4: 0.10,    # TURN_RIGHT_BIG
    5: 0.02,    # STOP
}
ACTION_NAMES = {
    0: "FORWARD",
    1: "LEFT_BIG",
    2: "LEFT_SMALL",
    3: "RIGHT_SMALL",
    4: "RIGHT_BIG",
    5: "STOP",
}

SETTLE_TIME = 0.01   # seconds after hard-stop before observation

# Episode limits
MAX_STEPS        = 200
GOAL_DIST_M      = 0.20    # target reached
GOAL_ANGLE_DEG   = 10.0    # heading bonus threshold
LOST_FRAMES_MAX  = 10      # consecutive invisible steps → terminate

# Model paths
MODEL_PATH       = "rl_follower_ppo.zip"
CHECKPOINT_DIR   = "./checkpoints/"
CHECKPOINT_FREQ  = 5_000   # timesteps between checkpoint saves

# Display
SHOW_WINDOW      = True
PRINT_STEP_INFO  = True

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_arduino_port():
    """Auto-detect Arduino serial port; fall back to SERIAL_PORT constant."""
    for port_info in serial.tools.list_ports.comports():
        desc = port_info.description or ""
        dev  = port_info.device or ""
        if any(kw in desc for kw in ("Arduino", "CH340", "CH341", "FTDI")) or \
           any(kw in dev  for kw in ("ttyUSB", "ttyACM")):
            return port_info.device
    return SERIAL_PORT


def connect_serial():
    port = find_arduino_port()
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT,
                            write_timeout=SERIAL_TIMEOUT)
        time.sleep(2.0)   # Arduino bootloader reset
        print(f"[HW] Arduino connected on {port}")
        return ser
    except Exception as exc:
        print(f"[HW] Serial connection failed ({exc}) — running without motor control")
        return None


def start_realsense():
    """Start RealSense with colour + depth; return (pipeline, intrinsics)."""
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16,  FPS)
    profile  = pipeline.start(cfg)
    time.sleep(1.0)

    # camera intrinsics from the colour stream (needed for angle calculation)
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr   = stream.get_intrinsics()
    print(f"[HW] RealSense started  fx={intr.fx:.1f}  ppx={intr.ppx:.1f}")
    return pipeline, intr


def restart_realsense(pipeline):
    print("[HW] Restarting RealSense …")
    try:
        pipeline.stop()
    except Exception:
        pass
    time.sleep(0.3)

    new_pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16,  FPS)
    new_pipeline.start(cfg)
    time.sleep(0.5)
    print("[HW] RealSense restarted")
    return new_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# PERCEPTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_valid_depth_average(depth_m, u, v, patch_radius=3):
    """
    Return the mean of non-zero depth values in a square patch of radius
    `patch_radius` centred at pixel (u, v) in the depth image (metres).
    Returns None if no valid readings are found.
    """
    h, w = depth_m.shape[:2]
    r  = patch_radius
    u0 = max(0, u - r);  u1 = min(w, u + r + 1)
    v0 = max(0, v - r);  v1 = min(h, v + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[patch > 0.0]
    return float(valid.mean()) if valid.size > 0 else None


def detect_target(color_bgr):
    """
    HSV-based pink/magenta target detection.

    Returns:
        result  – (u, v, area) pixel centroid + area, or None
        mask    – binary HSV mask (for visualisation)
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    if area < MIN_AREA:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return (u, v, area), mask


def compute_depth_zones(depth_m):
    """
    Analyse a horizontal strip of the depth image and return:
      left_min       – nearest valid depth (m) in left third
      center_min     – nearest valid depth (m) in centre third
      right_min      – nearest valid depth (m) in right third
      invalid_ratio  – fraction of zero-depth pixels in centre zone

    Returns MAX_DEPTH_M for zones with no valid readings.
    """
    strip = depth_m[STRIP_TOP:STRIP_BOTTOM, :]

    def zone_stats(col_start, col_end):
        zone  = strip[:, col_start:col_end]
        valid = zone[zone > 0.0]
        total = zone.size
        inv_r = float(np.sum(zone == 0)) / total if total > 0 else 1.0
        d_min = float(valid.min()) if valid.size > 0 else MAX_DEPTH_M
        return d_min, inv_r

    left_min,   _          = zone_stats(*ZONE_LEFT_X)
    center_min, inv_ratio  = zone_stats(*ZONE_CENTER_X)
    right_min,  _          = zone_stats(*ZONE_RIGHT_X)
    return left_min, center_min, right_min, inv_ratio

# ─────────────────────────────────────────────────────────────────────────────
# GYM ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class TargetFollowerEnv(gym.Env):
    """
    Gymnasium environment for a differential-drive robot following a
    coloured target using RealSense RGB-D and an Arduino motor controller.

    Observation space:  Box(7,)  — see module docstring
    Action space:       Discrete(6)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, pipeline, intr, ser):
        super().__init__()

        self.pipeline = pipeline
        self.intr     = intr          # rs.intrinsics (colour stream)
        self.ser      = ser           # serial.Serial or None

        # ── spaces ──────────────────────────────────────────────────────────
        obs_low  = np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -math.pi], dtype=np.float32)
        obs_high = np.array([MAX_DEPTH_M, MAX_DEPTH_M, MAX_DEPTH_M, MAX_DEPTH_M,
                             1.0, 1.0, math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Discrete(6)

        # ── episode state ────────────────────────────────────────────────────
        self._prev_dist      = MAX_DEPTH_M
        self._prev_angle_rad = math.pi
        self._lost_count     = 0
        self._step_count     = 0
        self._last_cmd       = None
        self._frame_fails    = 0
        self._last_obs       = np.zeros(7, dtype=np.float32)
        self._last_obs[0]    = MAX_DEPTH_M
        self._last_obs[2]    = MAX_DEPTH_M

        if SHOW_WINDOW:
            cv2.namedWindow("RL Follower – Feed",  cv2.WINDOW_NORMAL)
            cv2.namedWindow("RL Follower – Depth", cv2.WINDOW_NORMAL)

    # ── serial helpers ───────────────────────────────────────────────────────

    def _send_cmd(self, cmd: str):
        if self.ser is None:
            return
        try:
            if cmd == CMD_STOP or cmd != self._last_cmd:
                self.ser.write(cmd.encode())
                self._last_cmd = cmd
        except Exception as exc:
            print(f"[Serial] write failed: {exc}")
            self.ser = None

    def _hard_stop(self, n: int = 3):
        for _ in range(n):
            self._send_cmd(CMD_STOP)
            time.sleep(0.005)
        self._last_cmd = CMD_STOP

    # ── camera helpers ───────────────────────────────────────────────────────

    def _get_frames(self):
        """
        Return (color_bgr, depth_m) as numpy arrays aligned to the colour
        frame, or (None, None) on failure.
        """
        try:
            frames    = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            align_op  = rs.align(rs.stream.color)
            frames    = align_op.process(frames)
            color_f   = frames.get_color_frame()
            depth_f   = frames.get_depth_frame()
            if not color_f or not depth_f:
                return None, None

            color   = np.asanyarray(color_f.get_data())                    # BGR uint8
            depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) \
                      * depth_f.get_units()                                  # metres
            self._frame_fails = 0
            return color, depth_m

        except Exception:
            self._frame_fails += 1
            if self._frame_fails >= 3:
                try:
                    self.pipeline = restart_realsense(self.pipeline)
                    self._frame_fails = 0
                except Exception as exc:
                    print(f"[Camera] restart failed: {exc}")
            return None, None

    # ── observation builder ──────────────────────────────────────────────────

    def _observe(self):
        """
        Build the 7-element observation vector from a fresh camera frame.
        Returns (obs, color_bgr, depth_m, mask).
        """
        color, depth_m = self._get_frames()

        if color is None:
            # sensor failure — return last known observation
            return self._last_obs.copy(), None, None, None

        target, mask = detect_target(color)
        left_d, center_d, right_d, inv_ratio = compute_depth_zones(depth_m)

        if target is None:
            obs = np.array([MAX_DEPTH_M, left_d, center_d, right_d,
                            inv_ratio, 0.0, 0.0], dtype=np.float32)
            self._last_obs = obs.copy()
            return obs, color, depth_m, mask

        u, v, _ = target

        # distance: patch average at target pixel
        dist_raw = get_valid_depth_average(depth_m, u, v)
        dist     = float(dist_raw) if dist_raw is not None else MAX_DEPTH_M

        # angle: horizontal pixel offset → angle via camera intrinsics
        # positive when target is to the right of centre
        angle_rad = math.atan2(float(u) - self.intr.ppx, self.intr.fx)

        obs = np.array([dist, left_d, center_d, right_d,
                        inv_ratio, 1.0, angle_rad], dtype=np.float32)
        self._last_obs = obs.copy()
        return obs, color, depth_m, mask

    # ── reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, obs, action: int, dt: float) -> float:
        dist      = float(obs[0])
        center_d  = float(obs[2])
        inv_ratio = float(obs[4])
        visible   = bool(obs[5] > 0.5)
        angle_rad = float(obs[6])

        reward = 0.0

        # Time penalty (always applied)
        reward -= 0.01 * dt

        if not visible:
            # no other terms when target is lost
            return reward

        # Goal bonus: close enough to target
        if dist < GOAL_DIST_M:
            reward += 10.0

        # Heading bonus: pointing toward target
        if abs(math.degrees(angle_rad)) < GOAL_ANGLE_DEG:
            reward += 3.0

        # Collision penalty
        # — obstacle closer than MIN_DEPTH in the centre zone, or
        # — attempting FORWARD (action 0) into a region with >50 % invalid depth
        if (center_d < MIN_DEPTH_M) or (action == 0 and inv_ratio > 0.50):
            reward -= 5.0

        # Distance progress reward
        reward += 1.0 * (self._prev_dist - dist)

        # Angle progress reward
        reward += 0.05 * (abs(self._prev_angle_rad) - abs(angle_rad))

        return reward

    # ── visualisation ────────────────────────────────────────────────────────

    def _render(self, color, depth_m, obs, action, reward):
        if not SHOW_WINDOW or color is None:
            return

        dist      = float(obs[0])
        angle_deg = math.degrees(float(obs[6]))
        visible   = bool(obs[5] > 0.5)
        inv_ratio = float(obs[4])

        vis = color.copy()
        cx  = COLOR_W // 2

        # centre line
        cv2.line(vis, (cx, 0), (cx, COLOR_H), (255, 255, 255), 1)

        # centre-box outline (matching STRIP_TOP/BOTTOM and ZONE_CENTER_X)
        cv2.rectangle(vis,
                      (ZONE_CENTER_X[0], STRIP_TOP),
                      (ZONE_CENTER_X[1], STRIP_BOTTOM),
                      (0, 165, 255), 1)

        cv2.putText(vis, f"Action: {ACTION_NAMES[action]}",
                    (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(vis, f"Dist: {dist:.2f}m  Ang: {angle_deg:+.1f}deg",
                    (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255),  2)
        cv2.putText(vis, f"Reward: {reward:+.3f}  Inv: {inv_ratio:.2f}",
                    (10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255),  2)
        cv2.putText(vis, f"Step: {self._step_count}  Visible: {visible}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 0),  2)

        cv2.imshow("RL Follower – Feed", vis)

        # depth colourmap
        if depth_m is not None:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(
                    np.clip(depth_m, 0, MAX_DEPTH_M) / MAX_DEPTH_M * 255
                ),
                cv2.COLORMAP_JET
            )
            cv2.imshow("RL Follower – Depth", depth_vis)

        cv2.waitKey(1)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._hard_stop()
        self._step_count     = 0
        self._lost_count     = 0
        self._prev_dist      = MAX_DEPTH_M
        self._prev_angle_rad = math.pi

        print("\n" + "=" * 60)
        print("RESET — place target in view within 3 seconds …")
        print("=" * 60)
        time.sleep(3.0)
        self._hard_stop()
        time.sleep(SETTLE_TIME)

        obs, _, _, _ = self._observe()
        if obs[5] > 0.5:
            self._prev_dist      = float(obs[0])
            self._prev_angle_rad = float(obs[6])

        return obs, {}

    def step(self, action: int):
        t0 = time.time()
        self._step_count += 1

        # ── execute action ────────────────────────────────────────────────────
        cmd = ACTION_CMDS[action]
        dur = ACTION_DURATIONS[action]
        self._send_cmd(cmd)
        time.sleep(dur)
        self._hard_stop()
        time.sleep(SETTLE_TIME)

        dt = time.time() - t0

        # ── observe & reward ─────────────────────────────────────────────────
        obs, color, depth_m, mask = self._observe()
        reward  = self._compute_reward(obs, action, dt)

        visible   = bool(obs[5] > 0.5)
        dist      = float(obs[0])
        angle_rad = float(obs[6])

        # ── update lost-target counter ────────────────────────────────────────
        if not visible:
            self._lost_count += 1
        else:
            self._lost_count = 0

        # ── termination conditions ────────────────────────────────────────────
        terminated = False
        if visible and dist < GOAL_DIST_M:
            terminated = True      # SUCCESS: reached target
            print(f"[ENV] Goal reached! dist={dist:.3f}m")
        elif self._lost_count >= LOST_FRAMES_MAX:
            terminated = True      # FAILURE: target permanently lost
            print(f"[ENV] Target lost for {self._lost_count} steps — terminating")

        truncated = (self._step_count >= MAX_STEPS)

        # ── update running state (only when target is visible) ────────────────
        if visible:
            self._prev_dist      = dist
            self._prev_angle_rad = angle_rad

        if PRINT_STEP_INFO:
            print(
                f"step={self._step_count:3d}  action={ACTION_NAMES[action]:<14s}"
                f"reward={reward:+7.3f}  dist={dist:.2f}m  "
                f"ang={math.degrees(angle_rad):+6.1f}deg  "
                f"vis={int(visible)}"
            )

        self._render(color, depth_m, obs, action, reward)

        info = {
            "visible":    visible,
            "dist_m":     dist,
            "angle_deg":  math.degrees(angle_rad),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._hard_stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(env):
    """Create a fresh PPO model with tuned hyperparameters."""
    return PPO(
        policy        = "MlpPolicy",
        env           = env,
        verbose       = 1,
        device        = "cpu",
        learning_rate = 3e-4,
        n_steps       = 256,       # rollout length before each update
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.20,
        ent_coef      = 0.01,      # entropy bonus to encourage exploration
        policy_kwargs = dict(net_arch=[128, 128]),
    )


def load_model(path, env):
    print(f"[RL] Loading model from {path}")
    return PPO.load(path, env=env, device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL target-follower (PPO / SB3)")
    parser.add_argument("--timesteps", type=int,  default=50_000,
                        help="Total SB3 training timesteps (default: 50000)")
    parser.add_argument("--load",      type=str,  default=None,
                        help="Path to an existing .zip model to continue training")
    parser.add_argument("--inference", action="store_true",
                        help="Run inference only — no weight updates")
    args = parser.parse_args()

    # ── hardware init ─────────────────────────────────────────────────────────
    pipeline, intr = start_realsense()
    ser            = connect_serial()

    # ── environment ───────────────────────────────────────────────────────────
    env = TargetFollowerEnv(pipeline, intr, ser)

    if not args.inference:
        # sanity-check observation / action space definitions
        check_env(env, warn=True)

    # ── model ─────────────────────────────────────────────────────────────────
    if args.load:
        model = load_model(args.load, env)
    elif os.path.exists(MODEL_PATH):
        print(f"[RL] Found existing model {MODEL_PATH} — loading to continue training")
        model = load_model(MODEL_PATH, env)
    else:
        print("[RL] Creating new PPO model")
        model = build_model(env)

    # ── inference loop ────────────────────────────────────────────────────────
    if args.inference:
        print("[RL] Running inference …  press Ctrl-C to stop")
        obs, _ = env.reset()
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    obs, _ = env.reset()
        except KeyboardInterrupt:
            print("\n[RL] Stopped by user")
        finally:
            env.close()
            pipeline.stop()
            if ser:
                ser.close()
        return

    # ── training loop ─────────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq    = CHECKPOINT_FREQ,
        save_path    = CHECKPOINT_DIR,
        name_prefix  = "rl_follower_ppo",
        verbose      = 1,
    )

    print(f"[RL] Training for {args.timesteps:,} timesteps …")
    try:
        model.learn(
            total_timesteps    = args.timesteps,
            callback           = checkpoint_cb,
            reset_num_timesteps= False,
            log_interval       = 1,
        )
        model.save(MODEL_PATH)
        print(f"[RL] Training complete — model saved to {MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n[RL] Training interrupted — saving current model …")
        model.save(MODEL_PATH)
        print(f"[RL] Model saved to {MODEL_PATH}")

    except Exception as exc:
        print(f"[RL] Training error: {exc}")
        env._hard_stop()
        raise

    finally:
        env.close()
        pipeline.stop()
        if ser:
            ser.close()


if __name__ == "__main__":
    main()
