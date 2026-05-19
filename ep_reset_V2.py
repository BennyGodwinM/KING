import cv2
import serial
import time
import pyrealsense2 as rs
import numpy as np

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Change these if your Arduino expects different letters
CMD_FORWARD = "F"
CMD_BACKWARD = "B"
CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_STOP = "S"

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
time.sleep(2)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

last_cmd = None

def send_cmd(cmd):
    global last_cmd

    # Drain encoder / serial spam so buffer never fills up
    while ser.in_waiting > 0:
        try:
            junk = ser.readline().decode("utf-8", errors="ignore").strip()

            # Optional: print encoder lines
            # print("RX:", junk)

        except:
            break

    if cmd != last_cmd:
        ser.write((cmd + "\n").encode("utf-8"))
        print("Sent:", cmd)
        last_cmd = cmd

try:
    print("Click the camera window, then use WASD.")
    print("W = forward")
    print("A = left")
    print("S = backward")
    print("D = right")
    print("Space = stop")
    print("Q = quit")

    while True:

        # Continuously clear incoming encoder data
        while ser.in_waiting > 0:
            try:
                ser.readline()
            except:
                break

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.putText(
            color_image,
            "WASD drive | SPACE stop | Q quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Robot Camera View", color_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("w"):
            send_cmd(CMD_FORWARD)

        elif key == ord("s"):
            send_cmd(CMD_BACKWARD)

        elif key == ord("a"):
            send_cmd(CMD_LEFT)

        elif key == ord("d"):
            send_cmd(CMD_RIGHT)

        elif key == ord(" "):
            send_cmd(CMD_STOP)

        elif key == ord("q"):
            send_cmd(CMD_STOP)
            break

finally:
    send_cmd(CMD_STOP)
    pipeline.stop()
    ser.close()
    cv2.destroyAllWindows()