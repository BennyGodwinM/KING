import os
import time
import serial
import numpy as np
import cv2
import pyrealsense2 as rs
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

# ============================================================
# USER SETTINGS
# ============================================================

SERIAL_PORT = "COM3"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.01

CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_STOP = "S"

# Action durations
LEFT_BIG_TIME = 0.10
LEFT_SMALL_TIME = 0.025
STOP_TIME = 0.02
RIGHT_SMALL_TIME = 0.025
RIGHT_BIG_TIME = 0.10
SETTLE_TIME = 0.01

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 30
FRAME_TIMEOUT_MS = 100

LOWER_HSV = np.array([160, 150, 120], dtype=np.uint8)
UPPER_HSV = np.array([179, 255, 255], dtype=np.uint8)

MIN_AREA = 60
MORPH_KERNEL_SIZE = 5

MAX_STEPS_PER_EPISODE = 20
CENTER_THRESH = 0.08
CENTER_HOLD_STEPS = 5
LOST_TARGET_LIMIT = 6

STEP_TIMEOUT_SEC = 0.60

MODEL_PATH = "dqn_sticky_note_nonblocking.zip"

SHOW_WINDOW = True
PRINT_STEP_INFO = True

NUM_CHUNKS = 100
TIMESTEPS_PER_CHUNK = 100

# ============================================================
# SERIAL
# ============================================================

def connect_serial():
    try:
        ser = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT,
            write_timeout=SERIAL_TIMEOUT
        )
        time.sleep(2.0)
        print(f"Connected to Arduino on {SERIAL_PORT}")
        return ser
    except Exception as e:
        print("Serial connection failed:", e)
        return None


# ============================================================
# REALSENSE
# ============================================================

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    time.sleep(1.0)
    print("RealSense color stream started")
    return pipeline


def restart_realsense(pipeline):
    print("Restarting RealSense pipeline...")
    try:
        pipeline.stop()
    except Exception:
        pass

    time.sleep(0.3)

    new_pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    new_pipeline.start(config)
    time.sleep(0.5)
    print("RealSense restarted")
    return new_pipeline


# ============================================================
# DETECTION
# ============================================================

def detect_sticky_note(color_bgr, action_name="NONE", reward_text="0.000"):
    h, w = color_bgr.shape[:2]
    center_x = w // 2

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visible = False
    x_error_norm = 0.0
    vis_img = color_bgr.copy()

    cv2.line(vis_img, (center_x, 0), (center_x, h), (255, 255, 255), 2)

    if len(contours) > 0:
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)

        if area >= MIN_AREA:
            x, y, bw, bh = cv2.boundingRect(best)
            cx = x + bw // 2
            cy = y + bh // 2

            visible = True
            x_error_pixels = cx - center_x
            x_error_norm = x_error_pixels / (w / 2.0)
            x_error_norm = float(np.clip(x_error_norm, -1.0, 1.0))

            cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(vis_img, (cx, cy), 5, (0, 255, 255), -1)
            cv2.line(vis_img, (center_x, cy), (cx, cy), (255, 255, 0), 2)

    cv2.putText(vis_img, f"Visible: {visible}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Err: {x_error_norm:+.3f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Action: {action_name}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Reward: {reward_text}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return visible, x_error_norm, mask, vis_img


# ============================================================
# ENV
# ============================================================

class StickyNoteCenterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, pipeline, ser):
        super().__init__()
        self.pipeline = pipeline
        self.ser = ser

        # 0=LEFT_BIG, 1=LEFT_SMALL, 2=STOP, 3=RIGHT_SMALL, 4=RIGHT_BIG
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.last_cmd = None
        self.step_count = 0
        self.centered_count = 0
        self.lost_count = 0
        self.prev_error = 0.0
        self.frame_fail_count = 0

        self.last_action_name = "NONE"
        self.last_reward = 0.0
        self.latest_vis_img = None
        self.latest_mask = None
        self.last_obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if SHOW_WINDOW:
            cv2.namedWindow("Sticky Note Feed", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Sticky Note Mask", cv2.WINDOW_NORMAL)

    def send_cmd(self, cmd):
        if self.ser is None:
            return
        try:
            if cmd == CMD_STOP or cmd != self.last_cmd:
                self.ser.write(cmd.encode())
                self.last_cmd = cmd
        except Exception as e:
            print("Serial write failed:", e)
            self.ser = None

    def hard_stop(self):
        for _ in range(3):
            self.send_cmd(CMD_STOP)
            time.sleep(0.01)

    def apply_action(self, action):
        action_names = {
            0: "LEFT_BIG",
            1: "LEFT_SMALL",
            2: "STOP",
            3: "RIGHT_SMALL",
            4: "RIGHT_BIG",
        }
        self.last_action_name = action_names.get(int(action), "UNKNOWN")

        try:
            if action == 0:
                self.send_cmd(CMD_LEFT)
                time.sleep(LEFT_BIG_TIME)

            elif action == 1:
                self.send_cmd(CMD_LEFT)
                time.sleep(LEFT_SMALL_TIME)

            elif action == 2:
                self.send_cmd(CMD_STOP)
                time.sleep(STOP_TIME)

            elif action == 3:
                self.send_cmd(CMD_RIGHT)
                time.sleep(RIGHT_SMALL_TIME)

            elif action == 4:
                self.send_cmd(CMD_RIGHT)
                time.sleep(RIGHT_BIG_TIME)

        finally:
            self.hard_stop()
            time.sleep(SETTLE_TIME)

    def get_color_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            self.frame_fail_count = 0
            return np.asanyarray(color_frame.get_data())

        except Exception:
            self.frame_fail_count += 1
            if self.frame_fail_count >= 3:
                try:
                    self.pipeline = restart_realsense(self.pipeline)
                    self.frame_fail_count = 0
                except Exception as e:
                    print("Camera restart failed:", e)
            return None

    def render_windows(self):
        if SHOW_WINDOW and self.latest_vis_img is not None:
            cv2.imshow("Sticky Note Feed", self.latest_vis_img)

        if SHOW_WINDOW and self.latest_mask is not None:
            cv2.imshow("Sticky Note Mask", self.latest_mask)

        if SHOW_WINDOW:
            cv2.waitKey(1)

    def get_observation(self):
        color_img = self.get_color_frame()

        if color_img is None:
            return self.last_obs.copy()

        visible, x_error_norm, mask, vis_img = detect_sticky_note(
            color_img,
            action_name=self.last_action_name,
            reward_text=f"{self.last_reward:.3f}"
        )

        self.latest_vis_img = vis_img
        self.latest_mask = mask
        self.render_windows()

        obs = np.array([
            1.0 if visible else 0.0,
            x_error_norm,
            self.prev_error
        ], dtype=np.float32)

        self.last_obs = obs.copy()
        return obs

    def compute_reward(self, obs, action):
        visible = bool(obs[0] > 0.5)
        err = float(obs[1])
        act = int(action)

        reward = 0.0

        if not visible:
            reward -= 5.0
            self.prev_error = err
            self.last_reward = reward
            return float(reward)

        prev_abs = abs(self.prev_error)
        curr_abs = abs(err)
        delta = prev_abs - curr_abs

        # Base shaping
        reward += 12.0 * delta
        reward -= 2.0 * curr_abs

        # =====================================================
        # FIXED DIRECTION LOGIC
        # positive err -> target is to the RIGHT -> turn RIGHT
        # negative err -> target is to the LEFT  -> turn LEFT
        # =====================================================

        good_big = ((err > 0 and act == 4) or (err < 0 and act == 0))
        good_small = ((err > 0 and act == 3) or (err < 0 and act == 1))
        wrong_big = ((err > 0 and act == 0) or (err < 0 and act == 4))
        wrong_small = ((err > 0 and act == 1) or (err < 0 and act == 3))
        stop_action = (act == 2)

        # Large error: strongly prefer big corrective turn
        if curr_abs > 0.40:
            if good_big:
                reward += 1.5
            elif good_small:
                reward -= 0.5
            elif wrong_big:
                reward -= 1.5
            elif wrong_small:
                reward -= 1.0
            elif stop_action:
                reward -= 1.0

        # Medium error: prefer small corrective turn
        elif curr_abs > 0.15:
            if good_small:
                reward += 0.8
            elif good_big:
                reward += 0.2
            elif wrong_big:
                reward -= 1.0
            elif wrong_small:
                reward -= 0.8
            elif stop_action:
                reward -= 0.5

        # Near center: prefer stop
        else:
            if stop_action:
                reward += 2.0
            elif good_small:
                reward += 0.3
            elif good_big:
                reward -= 0.4
            elif wrong_big or wrong_small:
                reward -= 0.8

        # Center bonuses
        if curr_abs < 0.10:
            reward += 5.0
        if curr_abs < 0.05:
            reward += 8.0

        # Penalty for doing basically nothing when not centered
        if curr_abs > 0.08 and delta < 0.002:
            reward -= 0.3

        # Extra penalty if error clearly got worse
        if delta < -0.01:
            reward -= 0.5

        self.prev_error = err
        self.last_reward = reward
        return float(reward)

    def check_done(self, obs):
        visible = bool(obs[0] > 0.5)
        err = float(obs[1])

        if visible and abs(err) < CENTER_THRESH:
            self.centered_count += 1
        else:
            self.centered_count = 0

        if not visible:
            self.lost_count += 1
        else:
            self.lost_count = 0

        if self.centered_count >= CENTER_HOLD_STEPS:
            return True
        if self.lost_count >= LOST_TARGET_LIMIT:
            return True
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            return True

        return False

    def step(self, action):
        step_start = time.time()
        self.step_count += 1

        try:
            self.apply_action(int(action))
            obs = self.get_observation()

            if time.time() - step_start > STEP_TIMEOUT_SEC:
                print("Step timeout")
                self.hard_stop()
                self.last_reward = -5.0
                return self.last_obs.copy(), -5.0, True, False, {"timeout": True}

            reward = self.compute_reward(obs, action)
            done = self.check_done(obs)

            info = {
                "visible": bool(obs[0] > 0.5),
                "x_error_norm": float(obs[1]),
                "prev_error": float(obs[2]),
            }

            if PRINT_STEP_INFO:
                print(
                    f"step={self.step_count}, action={self.last_action_name}, reward={reward:.3f}, "
                    f"visible={info['visible']}, err={info['x_error_norm']:+.3f}, "
                    f"prev={info['prev_error']:+.3f}"
                )

            return obs, reward, done, False, info

        except Exception as e:
            print("Step error:", e)
            self.hard_stop()
            self.last_reward = -5.0
            return self.last_obs.copy(), -5.0, True, False, {"error": str(e)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.hard_stop()

        self.step_count = 0
        self.centered_count = 0
        self.lost_count = 0
        self.prev_error = 0.0
        self.last_reward = 0.0
        self.last_action_name = "RESET"
        self.last_obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        print("\n==================================================")
        print("RESET EPISODE")
        print("Put the sticky in view before the 3 second countdown ends.")
        time.sleep(3.0)

        self.hard_stop()
        time.sleep(0.05)

        obs = self.get_observation()
        if obs[0] > 0.5:
            self.prev_error = float(obs[1])
            obs[2] = self.prev_error
            self.last_obs = obs.copy()

        return obs, {}

    def close(self):
        try:
            self.hard_stop()
        except Exception:
            pass

        try:
            self.pipeline.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============================================================
# MODEL
# ============================================================

def build_or_load_model(env):
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        return DQN.load(MODEL_PATH, env=env, device="cpu")

    print("Creating new DQN model")
    return DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=5e-4,
        buffer_size=10000,
        learning_starts=500,
        batch_size=32,
        tau=1.0,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=250,
        exploration_fraction=0.8,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        policy_kwargs=dict(net_arch=[64, 64])
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ser = connect_serial()
    pipeline = start_realsense()
    env = StickyNoteCenterEnv(pipeline=pipeline, ser=ser)
    model = build_or_load_model(env)

    try:
        for chunk in range(1, NUM_CHUNKS + 1):
            print(f"\n########## DQN CHUNK {chunk}/{NUM_CHUNKS} ##########")
            model.learn(
                total_timesteps=TIMESTEPS_PER_CHUNK,
                reset_num_timesteps=False,
                log_interval=1
            )
            env.hard_stop()
            model.save(MODEL_PATH)
            print(f"Saved model after chunk {chunk}")

        print("\nTraining finished.")

    except KeyboardInterrupt:
        print("\nStopped by user")
        env.hard_stop()

    except Exception as e:
        print("\nMain loop error:", e)
        env.hard_stop()

    finally:
        env.close()


if __name__ == "__main__":
    main()