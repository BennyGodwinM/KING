import pyrealsense2 as rs
import numpy as np
import cv2
import time
import serial

# ----------------------------
# Serial
# ----------------------------
ser = None
try:
    ser = serial.Serial("COM3", 9600, timeout=1)
    time.sleep(2)
    print("Arduino connected")
except Exception as e:
    print("Serial not connected:", e)

last_cmd = None


def send_cmd(cmd):
    global last_cmd, ser
    if ser is not None and cmd != last_cmd:
        try:
            ser.write(cmd.encode())
            last_cmd = cmd
            print("Sent:", cmd)
        except Exception as e:
            print("Serial write failed:", e)
            ser = None


# ----------------------------
# RealSense
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
time.sleep(2)

# ----------------------------
# Pink target HSV range
# ----------------------------
lower = np.array([160, 150, 120])
upper = np.array([178, 255, 255])

# ----------------------------
# Tuning values
# ----------------------------
CENTER_TOL = 60
MIN_AREA = 20
MIN_DISTANCE = 0.25              # target ROI stop distance
FRONT_OBSTACLE_DISTANCE = 0.20   # front zone obstacle threshold
FRONT_INVALID_THRESH = 0.50      # if >50% invalid, front is UNKNOWN

# optional persistence for UNKNOWN state
unknown_counter = 0
UNKNOWN_FRAMES_NEEDED = 2

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            print("No frames received")
            send_cmd('S')
            continue

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Missing frame")
            send_cmd('S')
            continue

        depth_image = np.asanyarray(depth_frame.get_data())   # (270, 480)
        color_image = np.asanyarray(color_frame.get_data())   # (480, 640, 3)

        ch, cw, _ = color_image.shape
        dh, dw = depth_image.shape

        cx = cw // 2
        cy = ch // 2

        # ----------------------------
        # Pink target detection
        # ----------------------------
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cmd = 'S'
        status = "NO TARGET"

        # ----------------------------
        # Depth displays
        # ----------------------------
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        depth_resized = cv2.resize(depth_image, (cw, ch), interpolation=cv2.INTER_NEAREST)
        depth_colormap_big = cv2.resize(depth_colormap, (cw, ch), interpolation=cv2.INTER_NEAREST)

        # ----------------------------
        # Invalid depth overlay
        # ----------------------------
        invalid_mask = (depth_resized == 0)

        invalid_overlay = color_image.copy()
        invalid_overlay[invalid_mask] = (0, 0, 255)  # red
        invalid_display = cv2.addWeighted(color_image, 0.7, invalid_overlay, 0.3, 0)

        # ----------------------------
        # Front danger zone
        # ----------------------------
        box_w = int(cw * 0.5)
        box_h = int(ch * 0.3)

        fx1 = cw // 2 - box_w // 2
        fx2 = cw // 2 + box_w // 2
        fy1 = ch // 2 - box_h // 2
        fy2 = ch // 2 + box_h // 2

        front_roi = depth_resized[fy1:fy2, fx1:fx2]
        front_valid = front_roi[front_roi > 0]

        front_invalid_ratio = float(np.mean(front_roi == 0))

        front_min_depth_m = None
        if front_valid.size > 0:
            front_min_depth_m = float(np.min(front_valid) / 1000.0)

        if front_min_depth_m is not None and front_min_depth_m < FRONT_OBSTACLE_DISTANCE:
            front_state = "OBSTACLE"
            front_color = (0, 0, 255)
            unknown_counter = 0
        elif front_invalid_ratio > FRONT_INVALID_THRESH:
            unknown_counter += 1
            if unknown_counter >= UNKNOWN_FRAMES_NEEDED:
                front_state = "UNKNOWN"
                front_color = (0, 165, 255)
            else:
                front_state = "SAFE"
                front_color = (0, 255, 0)
        else:
            front_state = "SAFE"
            front_color = (0, 255, 0)
            unknown_counter = 0

        # ----------------------------
        # Target following logic
        # ----------------------------
        if contours:
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > MIN_AREA:
                x, y, ww, hh = cv2.boundingRect(largest)
                obj_cx = x + ww // 2
                obj_cy = y + hh // 2
                error = obj_cx - cx

                cv2.rectangle(color_image, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                cv2.circle(color_image, (obj_cx, obj_cy), 6, (0, 0, 255), -1)

                # Convert target center from color coordinates to depth coordinates
                depth_x = int(obj_cx * dw / cw)
                depth_y = int(obj_cy * dh / ch)

                # ROI around target in depth image
                roi_half_w = 40
                roi_half_h = 30

                x1 = max(0, depth_x - roi_half_w)
                x2 = min(dw, depth_x + roi_half_w)
                y1 = max(0, depth_y - roi_half_h)
                y2 = min(dh, depth_y + roi_half_h)

                roi = depth_image[y1:y2, x1:x2]
                valid_mask = roi > 0

                cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(depth_colormap, (depth_x, depth_y), 4, (255, 255, 255), -1)

                min_depth_m = None
                blocked = True

                if np.any(valid_mask):
                    roi_valid = roi[valid_mask]
                    min_depth_m = np.min(roi_valid) / 1000.0
                    blocked = min_depth_m < MIN_DISTANCE

                if min_depth_m is None:
                    status = "NO DEPTH - STOP"
                    cmd = 'S'
                elif blocked:
                    status = f"BLOCKED TARGET {min_depth_m:.2f}m"
                    cmd = 'S'
                else:
                    if error > CENTER_TOL:
                        cmd = 'R'
                        status = f"TURN RIGHT {error}  target {min_depth_m:.2f}m"
                    elif error < -CENTER_TOL:
                        cmd = 'L'
                        status = f"TURN LEFT {error}  target {min_depth_m:.2f}m"
                    else:
                        cmd = 'F'
                        status = f"FORWARD  target {min_depth_m:.2f}m"

        # ----------------------------
        # Front safety override
        # ----------------------------
        if cmd == 'F':
            if front_state == "OBSTACLE":
                cmd = 'S'
                if front_min_depth_m is not None:
                    status = f"FRONT OBSTACLE {front_min_depth_m:.2f}m - STOP"
                else:
                    status = "FRONT OBSTACLE - STOP"

            elif front_state == "UNKNOWN":
                cmd = 'S'
                status = f"FRONT UNKNOWN ({front_invalid_ratio:.2f} invalid) - STOP"

        print(
            f"{status} | cmd={cmd} | front_state={front_state} | "
            f"front_invalid={front_invalid_ratio:.2f} | "
            f"front_min={front_min_depth_m if front_min_depth_m is not None else 'None'}"
        )

        send_cmd(cmd)

        # ----------------------------
        # Draw overlays
        # ----------------------------
        displays = [
            ("Color", color_image.copy()),
            ("Invalid Depth Highlighted", invalid_display.copy()),
            ("Depth", depth_colormap_big.copy())
        ]

        for name, img in displays:
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
            cv2.line(img, (cx, 0), (cx, ch), (255, 0, 0), 2)
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), front_color, 2)

            cv2.putText(
                img,
                status,
                (20, ch - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Front state: {front_state}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                front_color,
                2
            )

            cv2.putText(
                img,
                f"Front invalid ratio: {front_invalid_ratio:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            front_depth_text = "None" if front_min_depth_m is None else f"{front_min_depth_m:.2f} m"
            cv2.putText(
                img,
                f"Front min depth: {front_depth_text}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"Unknown count: {unknown_counter}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(name, img)

        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) == 27:
            send_cmd('S')
            break

finally:
    send_cmd('S')
    pipeline.stop()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()