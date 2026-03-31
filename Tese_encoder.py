import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
import serial

# ============================================================
# USER SETTINGS
# ============================================================

SERIAL_PORT = "COM3"
BAUD_RATE = 9600

LOWER_HSV = np.array([145, 80, 80])
UPPER_HSV = np.array([179, 255, 255])

MIN_AREA = 200
PATCH_RADIUS = 3
CENTER_DRAW_RADIUS = 6

# ----------------------------
# Encoder settings
# ----------------------------
WHEEL_RADIUS = 0.04   # meters
CPR = 620             # counts per wheel revolution

# ----------------------------
# Gyro settings
# ----------------------------
GYRO_SIGN = 1.0
GYRO_YAW_AXIS = "y"       # try "z", then "y", then "x"
MAX_GYRO_DT = 0.05
GYRO_DEADBAND = 0.01      # rad/s

# ----------------------------
# Prediction settings
# ----------------------------
MAX_FORWARD_STEP = 0.20   # meters, safety clamp per frame
MAX_TURN_STEP_DEG = 20.0  # degrees, safety clamp per frame

# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def get_gyro_yaw_rate(gyro_xyz, axis_name="z"):
    if axis_name == "x":
        return gyro_xyz[0]
    elif axis_name == "y":
        return gyro_xyz[1]
    else:
        return gyro_xyz[2]


def measure_theta_R_from_pixel(u, v, Z, intr):
    """
    Real measurement from camera when target is visible.

    X = horizontal offset
    Zc = forward distance
    theta = atan2(X, Zc)
    R = sqrt(X^2 + Zc^2)
    """
    point = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(Z))
    X = point[0]
    Zc = point[2]

    theta = math.atan2(X, Zc)
    R = math.sqrt(X**2 + Zc**2)
    return theta, R, X, Zc


def theta_R_to_xz(theta, R):
    x = R * math.sin(theta)
    z = R * math.cos(theta)
    return x, z


def xz_to_theta_R(x, z):
    theta = math.atan2(x, z)
    R = math.sqrt(x**2 + z**2)
    return theta, R


def propagate_hidden_target_midpoint(x_old, z_old, d_forward, d_yaw):
    """
    Midpoint integration version.

    Instead of doing:
        full rotate -> full translate

    do:
        half rotate -> translate -> half rotate

    This better approximates simultaneous turning + forward motion.
    """

    dpsi_half = 0.5 * d_yaw

    # first half-rotation
    c1 = math.cos(dpsi_half)
    s1 = math.sin(dpsi_half)

    x_mid = c1 * x_old - s1 * z_old
    z_mid = s1 * x_old + c1 * z_old

    # forward translation
    x_mid2 = x_mid
    z_mid2 = z_mid - d_forward

    # second half-rotation
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

    result = {
        "u": u,
        "v": v,
        "area": area,
        "bbox": (x, y, w, h),
    }

    return result, mask


last_cmd = None

def send_cmd(ser, cmd):
    global last_cmd

    if ser is None:
        return

    if cmd == last_cmd:
        return

    try:
        ser.write(cmd.encode())
        last_cmd = cmd
    except Exception:
        pass


def get_latest_gyro_sample_nonblocking(imu_pipeline):
    """
    Returns latest gyro sample and timestamp in seconds.
    If nothing available, returns (None, None).
    """
    frames = imu_pipeline.poll_for_frames()

    if not frames:
        return None, None

    latest_gyro = None
    latest_ts = None

    for f in frames:
        if f.get_profile().stream_type() == rs.stream.gyro:
            data = f.as_motion_frame().get_motion_data()
            latest_gyro = np.array([data.x, data.y, data.z], dtype=float)
            latest_ts = f.get_timestamp() * 1e-3

    return latest_gyro, latest_ts


def read_latest_encoder_counts(ser):
    """
    Expects lines like:
        right_count,left_count
    """
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
                right_count = int(parts[0].strip())
                left_count = int(parts[1].strip())
                latest_right = right_count
                latest_left = left_count
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


# ============================================================
# SERIAL SETUP
# ============================================================

ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
except Exception:
    ser = None

# ============================================================
# IMU PIPELINE
# ============================================================

imu_pipeline = rs.pipeline()
imu_config = rs.config()
imu_config.enable_stream(rs.stream.gyro)
imu_pipeline.start(imu_config)

# ============================================================
# CAMERA PIPELINE
# ============================================================

cam_pipeline = rs.pipeline()
cam_config = rs.config()
cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cam_pipeline.start(cam_config)

align = rs.align(rs.stream.color)
time.sleep(2)

cam_profile = cam_pipeline.get_active_profile()
color_profile = cam_profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()

# ============================================================
# STATE
# ============================================================

# corrected estimate track
x_est = None
z_est = None
theta_est = None
R_est = None

# pure prediction track
x_pure = None
z_pure = None
theta_pure = None
R_pure = None

# predicted values each frame
x_pred_est = None
z_pred_est = None
theta_pred_est = None
R_pred_est = None

x_pred_pure = None
z_pred_pure = None
theta_pred_pure = None
R_pred_pure = None

# real measured values
theta_real = None
R_real = None
x_real = None
z_real = None

# gyro state
latest_gyro = np.array([0.0, 0.0, 0.0], dtype=float)
last_gyro_ts = None
dt_gyro = 0.0
dpsi_step = 0.0
robot_heading_total = 0.0

# encoder state
right_count = None
left_count = None
prev_right_count = None
prev_left_count = None

dR_counts = 0
dL_counts = 0
s_forward_step = 0.0
sR_step = 0.0
sL_step = 0.0

# motion mode
motion_mode = 'S'

send_cmd(ser, 'S')

# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:
        # ------------------------------------------------
        # Read latest gyro
        # ------------------------------------------------
        gyro, gyro_ts = get_latest_gyro_sample_nonblocking(imu_pipeline)

        dt_gyro = 0.0
        dpsi_step = 0.0

        if gyro is not None:
            latest_gyro = gyro

        if gyro_ts is not None:
            if last_gyro_ts is not None:
                new_dt = gyro_ts - last_gyro_ts

                if 0.0 < new_dt < MAX_GYRO_DT:
                    dt_gyro = new_dt
                    yaw_rate = get_gyro_yaw_rate(latest_gyro, GYRO_YAW_AXIS)

                    if abs(yaw_rate) < GYRO_DEADBAND:
                        yaw_rate = 0.0

                    dpsi_step = GYRO_SIGN * yaw_rate * dt_gyro

                    max_turn_step = deg_to_rad(MAX_TURN_STEP_DEG)
                    dpsi_step = max(-max_turn_step, min(max_turn_step, dpsi_step))

                    robot_heading_total = wrap_angle_rad(robot_heading_total + dpsi_step)

            last_gyro_ts = gyro_ts

        # ------------------------------------------------
        # Read latest encoders
        # ------------------------------------------------
        new_right_count, new_left_count = read_latest_encoder_counts(ser)

        if new_right_count is not None and new_left_count is not None:
            right_count = new_right_count
            left_count = new_left_count

        s_forward_step = 0.0
        sR_step = 0.0
        sL_step = 0.0
        dR_counts = 0
        dL_counts = 0

        if right_count is not None and left_count is not None:
            if prev_right_count is not None and prev_left_count is not None:
                dR_counts = right_count - prev_right_count
                dL_counts = -(left_count - prev_left_count)  # left CPR is negative

                s_forward_step, sR_step, sL_step = counts_to_forward_step(
                    dR_counts, dL_counts, WHEEL_RADIUS, CPR
                )

                s_forward_step = max(-MAX_FORWARD_STEP, min(MAX_FORWARD_STEP, s_forward_step))

                # Fix 2: debug forward step
                print(
                    f"Forward step: {s_forward_step:.4f} m | "
                    f"sR: {sR_step:.4f} m | sL: {sL_step:.4f} m | "
                    f"dR: {dR_counts} | dL: {dL_counts}"
                )

            prev_right_count = right_count
            prev_left_count = left_count

        # ------------------------------------------------
        # Camera frame
        # ------------------------------------------------
        cam_frames = cam_pipeline.wait_for_frames()
        aligned_frames = align.process(cam_frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # ------------------------------------------------
        # Keyboard control
        # ------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            motion_mode = 'S'
            send_cmd(ser, 'S')
            break

        elif key == ord('f'):
            motion_mode = 'F'
            send_cmd(ser, 'F')

        elif key == ord('l'):
            motion_mode = 'L'
            send_cmd(ser, 'L')

        elif key == ord('r'):
            motion_mode = 'R'
            send_cmd(ser, 'R')

        elif key == ord('s'):
            motion_mode = 'S'
            send_cmd(ser, 'S')

        # ------------------------------------------------
        # Detect target and compute REAL measurement
        # ------------------------------------------------
        detection, mask = detect_target(color_image, LOWER_HSV, UPPER_HSV, MIN_AREA)

        target_visible = False
        theta_real = None
        R_real = None
        x_real = None
        z_real = None
        depth_used = None

        if detection is not None:
            u = detection["u"]
            v = detection["v"]

            depth_used = get_valid_depth_average(depth_frame, u, v, PATCH_RADIUS)

            if depth_used is not None and depth_used > 0:
                try:
                    theta_real, R_real, x_real, z_real = measure_theta_R_from_pixel(u, v, depth_used, intr)
                    target_visible = True
                except Exception:
                    target_visible = False

        # ------------------------------------------------
        # Initialize tracks from first visible target
        # ------------------------------------------------
        if target_visible and x_est is None and z_est is None:
            x_est = x_real
            z_est = z_real
            theta_est, R_est = xz_to_theta_R(x_est, z_est)

        if target_visible and x_pure is None and z_pure is None:
            x_pure = x_real
            z_pure = z_real
            theta_pure, R_pure = xz_to_theta_R(x_pure, z_pure)

        # ------------------------------------------------
        # Predict corrected track
        # ------------------------------------------------
        x_pred_est = None
        z_pred_est = None
        theta_pred_est = None
        R_pred_est = None

        if x_est is not None and z_est is not None:
            x_pred_est, z_pred_est = propagate_hidden_target_midpoint(
                x_est, z_est, s_forward_step, dpsi_step
            )
            theta_pred_est, R_pred_est = xz_to_theta_R(x_pred_est, z_pred_est)

        # ------------------------------------------------
        # Predict pure track
        # ------------------------------------------------
        x_pred_pure = None
        z_pred_pure = None
        theta_pred_pure = None
        R_pred_pure = None

        if x_pure is not None and z_pure is not None:
            x_pred_pure, z_pred_pure = propagate_hidden_target_midpoint(
                x_pure, z_pure, s_forward_step, dpsi_step
            )
            theta_pred_pure, R_pred_pure = xz_to_theta_R(x_pred_pure, z_pred_pure)

        # ------------------------------------------------
        # Update corrected estimate
        # ------------------------------------------------
        if target_visible:
            x_est = x_real
            z_est = z_real
            theta_est = theta_real
            R_est = R_real
        else:
            if x_pred_est is not None and z_pred_est is not None:
                x_est = x_pred_est
                z_est = z_pred_est
                theta_est = theta_pred_est
                R_est = R_pred_est

        # ------------------------------------------------
        # Update pure prediction track
        # ------------------------------------------------
        if x_pred_pure is not None and z_pred_pure is not None:
            x_pure = x_pred_pure
            z_pure = z_pred_pure
            theta_pure = theta_pred_pure
            R_pure = R_pred_pure

        # ------------------------------------------------
        # Draw display
        # ------------------------------------------------
        display = color_image.copy()

        if detection is not None:
            x, y, w, h = detection["bbox"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display, (detection["u"], detection["v"]), CENTER_DRAW_RADIUS, (0, 0, 255), -1)

        h_img, w_img = display.shape[:2]
        cx_draw = int(intr.ppx)
        cv2.line(display, (cx_draw, 0), (cx_draw, h_img), (255, 0, 0), 1)

        yaw_rate_disp = get_gyro_yaw_rate(latest_gyro, GYRO_YAW_AXIS)

        fx = intr.fx
        cx = intr.ppx

        if x_real is not None and z_real is not None and z_real > 0:
            xpix_real = int(cx + fx * (x_real / z_real))
            xpix_real = max(0, min(w_img - 1, xpix_real))
            cv2.line(display, (xpix_real, 0), (xpix_real, h_img), (0, 255, 0), 2)

        if x_est is not None and z_est is not None and z_est > 0:
            xpix_est = int(cx + fx * (x_est / z_est))
            xpix_est = max(0, min(w_img - 1, xpix_est))
            cv2.line(display, (xpix_est, 0), (xpix_est, h_img), (0, 255, 255), 2)

        if x_pure is not None and z_pure is not None and z_pure > 0:
            xpix_pure = int(cx + fx * (x_pure / z_pure))
            xpix_pure = max(0, min(w_img - 1, xpix_pure))
            cv2.line(display, (xpix_pure, 0), (xpix_pure, h_img), (255, 0, 255), 2)

        line1 = f"Visible: {target_visible}"
        line2 = f"Mode: {motion_mode}"
        line3 = f"Gyro: {yaw_rate_disp:.3f} rad/s   dt_gyro: {dt_gyro:.4f} s"
        line4 = f"dpsi_step: {rad_to_deg(dpsi_step):.2f} deg   heading_total: {rad_to_deg(robot_heading_total):.2f} deg"
        line5 = f"Enc: R={right_count} L={left_count}   dR={dR_counts} dL={dL_counts}"
        line6 = f"sR={sR_step:.4f} m   sL={sL_step:.4f} m   s={s_forward_step:.4f} m"

        if theta_real is None or R_real is None:
            line7 = "Real:      th=NA   R=NA"
        else:
            line7 = f"Real:      th={rad_to_deg(theta_real):.2f} deg   R={R_real:.2f} m"

        if theta_est is None or R_est is None:
            line8 = "Estimate:  th=NA   R=NA"
        else:
            line8 = f"Estimate:  th={rad_to_deg(theta_est):.2f} deg   R={R_est:.2f} m"

        if theta_pure is None or R_pure is None:
            line9 = "Pred only: th=NA   R=NA"
        else:
            line9 = f"Pred only: th={rad_to_deg(theta_pure):.2f} deg   R={R_pure:.2f} m"

        if x_est is None or z_est is None:
            line10 = "Est XZ:    x=NA   z=NA"
        else:
            line10 = f"Est XZ:    x={x_est:.3f} m   z={z_est:.3f} m"

        line11 = f"Yaw axis: {GYRO_YAW_AXIS}   Sign: {GYRO_SIGN:+.1f}"

        cv2.putText(display, line1,  (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(display, line2,  (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(display, line3,  (10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(display, line4,  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(display, line5,  (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(display, line6,  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(display, line7,  (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        cv2.putText(display, line8,  (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        cv2.putText(display, line9,  (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        cv2.putText(display, line10, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(display, line11, (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        cv2.imshow("Color View", display)
        cv2.imshow("Mask", mask)

finally:
    try:
        if ser is not None:
            ser.write(b'S')
            ser.close()
    except:
        pass

    try:
        imu_pipeline.stop()
    except:
        pass

    try:
        cam_pipeline.stop()
    except:
        pass

    cv2.destroyAllWindows()