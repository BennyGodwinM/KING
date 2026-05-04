import time
import torch
import csv
from stable_baselines3 import DQN

from King_V10_dqn import RealRobotDQNEnv

MODEL_PATH = "dqn_target_nav_model_6.zip"
OUTPUT_FILE = "q_values_log_2.csv"


def safe_info(info, key, default=""):
    if info is None:
        return default
    value = info.get(key, default)
    if value is None:
        return default
    return value


def main():
    env = RealRobotDQNEnv(render_mode=True)

    model = DQN.load(MODEL_PATH, env=env, device="cpu")

    obs, _ = env.reset()
    last_info = env.last_info

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "R_est",
            "theta_deg",
            "Q_F",
            "Q_L",
            "Q_R",
            "chosen_action",
            "chosen_action_name",
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
            "target_visible",
            "target_like_front_object",
            "depth_avoidance_active",
            "disable_avoidance_near_target",
            "ignore_invalids_near_target",
            "s_forward_step",
            "dpsi_step_deg",
            "reward"
        ])

        step = 0

        while True:
            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32,
                device=model.device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = model.q_net(obs_tensor).cpu().numpy()[0]

            q_f, q_l, q_r = q_values

            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            action_name = ["F", "L", "R"][action_int]

            front_state = safe_info(last_info, "front_state")
            left_danger = safe_info(last_info, "left_danger")
            center_danger = safe_info(last_info, "center_danger")
            right_danger = safe_info(last_info, "right_danger")
            depth_avoidance_active = safe_info(last_info, "depth_avoidance_active")
            target_like_front_object = safe_info(last_info, "target_like_front_object")

            print(
                f"[{step}] R={obs[0]:.2f} theta={obs[6]:.1f} | "
                f"Q: F={q_f:.3f} L={q_l:.3f} R={q_r:.3f} | "
                f"action={action_name} | "
                f"front={front_state} "
                f"Ld={left_danger} Cd={center_danger} Rd={right_danger} "
                f"depth_avoid={depth_avoidance_active} "
                f"target_like={target_like_front_object}"
            )

            obs, reward, terminated, truncated, info = env.step(action_int)

            writer.writerow([
                time.time(),
                obs[0],
                obs[6],
                q_f,
                q_l,
                q_r,
                action_int,
                action_name,
                safe_info(info, "front_state"),
                safe_info(info, "front_min_depth_m"),
                safe_info(info, "front_invalid_ratio"),
                safe_info(info, "left_depth"),
                safe_info(info, "center_depth"),
                safe_info(info, "right_depth"),
                safe_info(info, "left_invalid_ratio"),
                safe_info(info, "center_invalid_ratio"),
                safe_info(info, "right_invalid_ratio"),
                safe_info(info, "left_danger"),
                safe_info(info, "center_danger"),
                safe_info(info, "right_danger"),
                int(bool(safe_info(info, "target_visible", False))),
                int(bool(safe_info(info, "target_like_front_object", False))),
                int(bool(safe_info(info, "depth_avoidance_active", False))),
                int(bool(safe_info(info, "disable_avoidance_near_target", False))),
                int(bool(safe_info(info, "ignore_invalids_near_target", False))),
                safe_info(info, "s_forward_step"),
                safe_info(info, "dpsi_step"),
                reward
            ])

            f.flush()

            last_info = info
            step += 1

            if terminated or truncated:
                print("Episode ended. Resetting...")
                obs, _ = env.reset()
                last_info = env.last_info


if __name__ == "__main__":
    main()