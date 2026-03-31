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

MODEL_PATH = "dqn_sticky_note_nonblocking.zip"

SHOW_WINDOW = True
PRINT_INFO = True


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

def detect_sticky_note(color_bgr, action_name="NONE"):
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

    return visible, x_error_norm, mask, vis_img


# ============================================================
# SIMPLE ENV-LIKE WRAPPER
# ============================================================

class StickyFollower:
    def __init__(self, pipeline, ser):
        self.pipeline = pipeline
        self.ser = ser
        self.last_cmd = None
        self.prev_error = 0.0
        self.frame_fail_count = 0
        self.last_action_name = "NONE"

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)

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

    def get_observation(self):
        color_img = self.get_color_frame()
        if color_img is None:
            return np.array([0.0, 0.0, self.prev_error], dtype=np.float32), None, None, None

        visible, x_error_norm, mask, vis_img = detect_sticky_note(
            color_img,
            action_name=self.last_action_name
        )

        obs = np.array([
            1.0 if visible else 0.0,
            x_error_norm,
            self.prev_error
        ], dtype=np.float32)

        if visible:
            self.prev_error = x_error_norm

        return obs, mask, vis_img, visible

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
# MAIN TEST LOOP
# ============================================================

def main():
    ser = connect_serial()
    pipeline = start_realsense()
    follower = StickyFollower(pipeline, ser)

    model = DQN.load(MODEL_PATH, device="cpu")
    print(f"Loaded model from {MODEL_PATH}")

    try:
        while True:
            obs, mask, vis_img, visible = follower.get_observation()

            if vis_img is not None and SHOW_WINDOW:
                cv2.imshow("Sticky Note Feed", vis_img)
            if mask is not None and SHOW_WINDOW:
                cv2.imshow("Sticky Note Mask", mask)

            if SHOW_WINDOW:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

            # If not visible, do not keep blindly moving
            if not visible:
                follower.last_action_name = "STOP_NO_TARGET"
                follower.hard_stop()
                if PRINT_INFO:
                    print("No target visible -> STOP")
                time.sleep(0.03)
                continue

            action, _ = model.predict(obs, deterministic=True)

            if PRINT_INFO:
                print(
                    f"visible={bool(obs[0] > 0.5)}, "
                    f"err={float(obs[1]):+.3f}, "
                    f"prev={float(obs[2]):+.3f}, "
                    f"action={int(action)}"
                )

            follower.apply_action(int(action))

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        follower.close()


if __name__ == "__main__":
    main()