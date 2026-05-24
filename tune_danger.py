import cv2
import numpy as np
import pyrealsense2 as rs

CAM_W_D = 480
CAM_H_D = 270
CAM_W_C = 640
CAM_H_C = 480
CAM_FPS = 30

AVOIDANCE_TRIGGER_DISTANCE = 0.55
SIDE_DANGER_DISTANCE = AVOIDANCE_TRIGGER_DISTANCE * 1.6

INVALID_DANGER_GAIN = 0.25
DEPTH_DANGER_GAIN = 0.75


def clip_ratio(x):
    return float(np.clip(x, 0.0, 1.0))


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
    inv_part = INVALID_DANGER_GAIN * clip_ratio(inv_ratio)
    depth_part = DEPTH_DANGER_GAIN * depth_danger_from_distance(depth_m, danger_distance)
    return float(np.clip(inv_part + depth_part, 0.0, 1.0))


def fmt_depth(d):
    if d is None:
        return "None"
    return f"{d:.3f} m"


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, CAM_W_D, CAM_H_D, rs.format.z16, CAM_FPS)
    config.enable_stream(rs.stream.color, CAM_W_C, CAM_H_C, rs.format.bgr8, CAM_FPS)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            h, w = depth_image.shape

            strip_y1 = int(h * 0.35)
            strip_y2 = int(h * 0.75)

            third = w // 3

            left_roi = depth_image[strip_y1:strip_y2, 0:third]
            center_roi = depth_image[strip_y1:strip_y2, third:2 * third]
            right_roi = depth_image[strip_y1:strip_y2, 2 * third:w]

            left_depth = nearest_valid_depth_m(left_roi)
            center_depth = nearest_valid_depth_m(center_roi)
            right_depth = nearest_valid_depth_m(right_roi)

            left_invalid = invalid_ratio(left_roi)
            center_invalid = invalid_ratio(center_roi)
            right_invalid = invalid_ratio(right_roi)

            left_danger = combined_direction_danger(left_depth, left_invalid, SIDE_DANGER_DISTANCE)
            center_danger = combined_direction_danger(center_depth, center_invalid, AVOIDANCE_TRIGGER_DISTANCE)
            right_danger = combined_direction_danger(right_depth, right_invalid, SIDE_DANGER_DISTANCE)

            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.08)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.rectangle(depth_vis, (0, strip_y1), (third, strip_y2), (255, 255, 255), 2)
            cv2.rectangle(depth_vis, (third, strip_y1), (2 * third, strip_y2), (255, 255, 255), 2)
            cv2.rectangle(depth_vis, (2 * third, strip_y1), (w, strip_y2), (255, 255, 255), 2)

            lines = [
                f"GAINS: depth={DEPTH_DANGER_GAIN:.2f}, invalid={INVALID_DANGER_GAIN:.2f}",
                f"LEFT   depth={fmt_depth(left_depth)}   invalid={left_invalid:.3f}   danger={left_danger:.3f}",
                f"CENTER depth={fmt_depth(center_depth)}   invalid={center_invalid:.3f}   danger={center_danger:.3f}",
                f"RIGHT  depth={fmt_depth(right_depth)}   invalid={right_invalid:.3f}   danger={right_danger:.3f}",
                "Press q to quit"
            ]

            y = 30
            for line in lines:
                cv2.putText(
                    depth_vis,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )
                y += 30

            cv2.imshow("Color View", color_image)
            cv2.imshow("Depth Danger Tuning", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()