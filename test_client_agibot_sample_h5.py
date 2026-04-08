#!/usr/bin/env python3
"""Offline validation client for DreamZero AgiBot using sample_dataset episodes.

Reads one AgiBot sample episode from:
- observations/<task_id>/<episode_id>/videos/*.mp4
- proprio_stats/<task_id>/<episode_id>/proprio_stats.h5
- task_info/task_<task_id>.json

Then replays synchronized frames and states through the websocket inference server,
and optionally compares the first predicted action step against the sample dataset's
recorded robot action stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path

import av
import cv2
import h5py
import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy

SAMPLE_ROOT = Path("/root/autodl-tmp/agibot_sample/extracted/sample_dataset")
VIDEO_FILENAMES = {
    "observation/top_head": "head_color.mp4",
    "observation/hand_left": "hand_left_color.mp4",
    "observation/hand_right": "hand_right_color.mp4",
}
GT_ACTION_KEYS = [
    "action.left_arm_joint_position",
    "action.right_arm_joint_position",
    "action.left_effector_position",
    "action.right_effector_position",
    "action.head_position",
    "action.waist_pitch",
    "action.waist_lift",
    "action.robot_velocity",
]


def _load_all_frames_opencv(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _load_all_frames_pyav(video_path: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def load_all_frames(video_path: str) -> np.ndarray:
    frames = _load_all_frames_opencv(video_path)
    if not frames:
        logging.info("OpenCV failed to decode %s, falling back to PyAV...", video_path)
        frames = _load_all_frames_pyav(video_path)
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames(episode_video_dir: Path) -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for obs_key, filename in VIDEO_FILENAMES.items():
        path = episode_video_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required camera video: {path}")
        camera_frames[obs_key] = load_all_frames(str(path))
        logging.info("Loaded %s: %s", obs_key, camera_frames[obs_key].shape)
    return camera_frames


def load_proprio(proprio_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with h5py.File(proprio_path, "r") as f:
        state_joint_position = np.asarray(f["state/joint/position"], dtype=np.float32)
        state_effector_position = np.asarray(f["state/effector/position"], dtype=np.float32)
        state_head_position = np.asarray(f["state/head/position"], dtype=np.float32)
        state_waist_position = np.asarray(f["state/waist/position"], dtype=np.float32)
        timestamp = np.asarray(f["timestamp"])

        action_joint_position = np.asarray(f["action/joint/position"], dtype=np.float32)
        action_effector_position = np.asarray(f["action/effector/position"], dtype=np.float32)
        action_head_position = np.asarray(f["action/head/position"], dtype=np.float32)
        action_waist_position = np.asarray(f["action/waist/position"], dtype=np.float32)
        action_robot_velocity = np.asarray(f["action/robot/velocity"], dtype=np.float32)

    proprio = {
        "observation/left_arm_joint_position": state_joint_position[:, :7],
        "observation/right_arm_joint_position": state_joint_position[:, 7:],
        "observation/left_effector_position": state_effector_position[:, 0:1],
        "observation/right_effector_position": state_effector_position[:, 1:2],
        "observation/head_position": state_head_position,
        "observation/waist_pitch": state_waist_position[:, 0:1],
        "observation/waist_lift": state_waist_position[:, 1:2],
        "timestamp": timestamp,
    }

    gt_action_dict = {
        "action.left_arm_joint_position": action_joint_position[:, :7],
        "action.right_arm_joint_position": action_joint_position[:, 7:],
        "action.left_effector_position": action_effector_position[:, 0:1],
        "action.right_effector_position": action_effector_position[:, 1:2],
        "action.head_position": action_head_position,
        "action.waist_pitch": action_waist_position[:, 0:1],
        "action.waist_lift": action_waist_position[:, 1:2],
        "action.robot_velocity": action_robot_velocity,
    }

    for key, value in proprio.items():
        if key == "timestamp":
            continue
        logging.info("Loaded %s: %s", key, value.shape)
    for key, value in gt_action_dict.items():
        logging.info("Loaded %s: %s", key, value.shape)

    return proprio, gt_action_dict


def flatten_gt_actions(gt_action_dict: dict[str, np.ndarray]) -> np.ndarray:
    chunks = [gt_action_dict[key].reshape(gt_action_dict[key].shape[0], -1) for key in GT_ACTION_KEYS]
    return np.concatenate(chunks, axis=-1).astype(np.float32)


def load_task_info(task_info_path: Path, episode_id: int) -> dict:
    with open(task_info_path, "r", encoding="utf-8") as f:
        task_info = json.load(f)
    for item in task_info:
        if int(item["episode_id"]) == episode_id:
            return item
    raise ValueError(f"Episode {episode_id} not found in {task_info_path}")


def build_prompt(task_item: dict, frame_idx: int, prompt_mode: str) -> str:
    task_name = task_item.get("task_name", "")
    init_scene_text = task_item.get("init_scene_text", "")
    action_config = task_item.get("label_info", {}).get("action_config", [])

    current_skill = ""
    for segment in action_config:
        if int(segment.get("start_frame", -1)) <= frame_idx < int(segment.get("end_frame", -1)):
            current_skill = segment.get("skill", "")
            action_text = segment.get("action_text", "")
            if action_text:
                current_skill = action_text
            break

    if prompt_mode == "task_name":
        return task_name
    if prompt_mode == "scene":
        return init_scene_text or task_name
    if prompt_mode == "skill":
        return current_skill or task_name
    if prompt_mode == "task_and_skill":
        if current_skill and task_name:
            return f"{task_name}. Current skill: {current_skill}"
        return current_skill or task_name
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def build_obs(
    camera_frames: dict[str, np.ndarray],
    proprio: dict[str, np.ndarray],
    frame_idx: int,
    prompt: str,
    session_id: str,
) -> dict:
    obs: dict[str, object] = {}
    for obs_key, frames in camera_frames.items():
        obs[obs_key] = frames[frame_idx]

    for key, values in proprio.items():
        if key == "timestamp":
            continue
        obs[key] = values[frame_idx].copy()

    obs["prompt"] = prompt
    obs["session_id"] = session_id

    state_shapes = {k: np.asarray(obs[k]).shape for k in proprio.keys() if k != "timestamp"}
    logging.info("State shapes for frame %s: %s", frame_idx, state_shapes)
    logging.info("Prompt for frame %s: %s", frame_idx, prompt)
    return obs


def analyze_action_horizon(actions: np.ndarray) -> dict[str, object]:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (T, D), got {actions.shape}")

    if actions.shape[0] < 2:
        step_delta_norms = np.zeros((0,), dtype=np.float32)
        mean_step_delta = 0.0
        max_step_delta = 0.0
    else:
        step_deltas = actions[1:] - actions[:-1]
        step_delta_norms = np.linalg.norm(step_deltas, axis=1)
        mean_step_delta = float(np.mean(step_delta_norms))
        max_step_delta = float(np.max(step_delta_norms))

    first_last_l2 = float(np.linalg.norm(actions[-1] - actions[0]))
    mean_action_l2 = float(np.linalg.norm(np.mean(actions, axis=0) - actions[0]))
    per_dim_range = np.max(actions, axis=0) - np.min(actions, axis=0)

    return {
        "horizon": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "first_last_l2": first_last_l2,
        "mean_action_l2_from_first": mean_action_l2,
        "mean_step_delta_l2": mean_step_delta,
        "max_step_delta_l2": max_step_delta,
        "mean_per_dim_range": float(np.mean(per_dim_range)),
        "max_per_dim_range": float(np.max(per_dim_range)),
        "per_dim_range": per_dim_range.astype(np.float32),
    }


def summarize_action_comparison(
    predicted_first_steps: np.ndarray,
    gt_actions: np.ndarray,
    horizon_stats: list[dict[str, object]],
    summary_path: Path | None,
) -> None:
    diff = predicted_first_steps - gt_actions
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    rmse_per_dim = np.sqrt(np.mean(diff * diff, axis=0))
    pred_mean = predicted_first_steps.mean(axis=0)
    gt_mean = gt_actions.mean(axis=0)
    pred_std = predicted_first_steps.std(axis=0)
    gt_std = gt_actions.std(axis=0)

    corr_per_dim = []
    for dim in range(predicted_first_steps.shape[1]):
        pred_col = predicted_first_steps[:, dim]
        gt_col = gt_actions[:, dim]
        if np.std(pred_col) < 1e-8 or np.std(gt_col) < 1e-8:
            corr_per_dim.append(None)
        else:
            corr_per_dim.append(float(np.corrcoef(pred_col, gt_col)[0, 1]))

    overall_mae = float(np.mean(np.abs(diff)))
    overall_rmse = float(np.sqrt(np.mean(diff * diff)))

    horizon_first_last = np.asarray([s["first_last_l2"] for s in horizon_stats], dtype=np.float32)
    horizon_mean_step = np.asarray([s["mean_step_delta_l2"] for s in horizon_stats], dtype=np.float32)
    horizon_max_step = np.asarray([s["max_step_delta_l2"] for s in horizon_stats], dtype=np.float32)
    horizon_mean_range = np.asarray([s["mean_per_dim_range"] for s in horizon_stats], dtype=np.float32)
    horizon_max_range = np.asarray([s["max_per_dim_range"] for s in horizon_stats], dtype=np.float32)

    logging.info("Action comparison summary:")
    logging.info("  compared_steps: %s", predicted_first_steps.shape[0])
    logging.info("  action_dim: %s", predicted_first_steps.shape[1])
    logging.info("  overall_mae: %.6f", overall_mae)
    logging.info("  overall_rmse: %.6f", overall_rmse)
    logging.info("Action horizon summary:")
    logging.info("  avg first_last_l2: %.6f", float(np.mean(horizon_first_last)))
    logging.info("  avg mean_step_delta_l2: %.6f", float(np.mean(horizon_mean_step)))
    logging.info("  avg max_step_delta_l2: %.6f", float(np.mean(horizon_max_step)))
    logging.info("  avg mean_per_dim_range: %.6f", float(np.mean(horizon_mean_range)))
    logging.info("  avg max_per_dim_range: %.6f", float(np.mean(horizon_max_range)))

    for dim in range(predicted_first_steps.shape[1]):
        logging.info(
            "  dim %02d: pred_mean=%.5f gt_mean=%.5f pred_std=%.5f gt_std=%.5f mae=%.5f rmse=%.5f corr=%s",
            dim,
            float(pred_mean[dim]),
            float(gt_mean[dim]),
            float(pred_std[dim]),
            float(gt_std[dim]),
            float(mae_per_dim[dim]),
            float(rmse_per_dim[dim]),
            "None" if corr_per_dim[dim] is None else f"{corr_per_dim[dim]:.5f}",
        )

    if summary_path is not None:
        payload = {
            "compared_steps": int(predicted_first_steps.shape[0]),
            "action_dim": int(predicted_first_steps.shape[1]),
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "dim_names": GT_ACTION_KEYS,
            "mae_per_dim": mae_per_dim.tolist(),
            "rmse_per_dim": rmse_per_dim.tolist(),
            "pred_mean_per_dim": pred_mean.tolist(),
            "gt_mean_per_dim": gt_mean.tolist(),
            "pred_std_per_dim": pred_std.tolist(),
            "gt_std_per_dim": gt_std.tolist(),
            "corr_per_dim": corr_per_dim,
            "horizon_summary": {
                "avg_first_last_l2": float(np.mean(horizon_first_last)),
                "avg_mean_step_delta_l2": float(np.mean(horizon_mean_step)),
                "avg_max_step_delta_l2": float(np.mean(horizon_max_step)),
                "avg_mean_per_dim_range": float(np.mean(horizon_mean_range)),
                "avg_max_per_dim_range": float(np.mean(horizon_max_range)),
            },
            "horizon_stats_per_frame": [
                {
                    "horizon": int(s["horizon"]),
                    "action_dim": int(s["action_dim"]),
                    "first_last_l2": float(s["first_last_l2"]),
                    "mean_action_l2_from_first": float(s["mean_action_l2_from_first"]),
                    "mean_step_delta_l2": float(s["mean_step_delta_l2"]),
                    "max_step_delta_l2": float(s["max_step_delta_l2"]),
                    "mean_per_dim_range": float(s["mean_per_dim_range"]),
                    "max_per_dim_range": float(s["max_per_dim_range"]),
                    "per_dim_range": np.asarray(s["per_dim_range"]).tolist(),
                }
                for s in horizon_stats
            ],
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info("Saved action comparison summary to %s", summary_path)


def run_client(
    host: str,
    port: int,
    sample_root: Path,
    task_id: int,
    episode_id: int,
    prompt_mode: str,
    max_steps: int | None,
    frame_stride: int,
    save_every: int,
    compare_actions: bool,
    summary_path: Path | None,
) -> None:
    logging.info("Connecting to AgiBot server at %s:%s...", host, port)
    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    episode_video_dir = sample_root / "observations" / str(task_id) / str(episode_id) / "videos"
    proprio_path = sample_root / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
    task_info_path = sample_root / "task_info" / f"task_{task_id}.json"

    camera_frames = load_camera_frames(episode_video_dir)
    proprio, gt_action_dict = load_proprio(proprio_path)
    gt_actions = flatten_gt_actions(gt_action_dict)
    task_item = load_task_info(task_info_path, episode_id)

    total_frames = min(
        min(v.shape[0] for v in camera_frames.values()),
        proprio["observation/left_arm_joint_position"].shape[0],
        gt_actions.shape[0],
    )
    frame_indices = list(range(0, total_frames, frame_stride))
    if max_steps is not None:
        frame_indices = frame_indices[:max_steps]

    logging.info("Episode task_id=%s episode_id=%s", task_id, episode_id)
    logging.info("Task name: %s", task_item.get("task_name", ""))
    logging.info("Total synchronized steps: %s", len(frame_indices))
    logging.info("Saving one video segment every %s steps", save_every)
    logging.info("Compare predicted first-step actions to sample GT: %s", compare_actions)

    compared_predicted: list[np.ndarray] = []
    compared_gt: list[np.ndarray] = []
    horizon_stats: list[dict[str, object]] = []

    session_id = str(uuid.uuid4())
    segment_idx = 0
    segment_step_count = 0
    logging.info("Segment %s session ID: %s", segment_idx, session_id)

    for step_idx, frame_idx in enumerate(frame_indices):
        prompt = build_prompt(task_item, frame_idx, prompt_mode)
        obs = build_obs(camera_frames, proprio, frame_idx, prompt, session_id)
        logging.info("=== Step %s/%s: frame %s ===", step_idx + 1, len(frame_indices), frame_idx)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        actions = np.asarray(actions, dtype=np.float32)
        logging.info(
            "  Action shape: %s, range: [%.4f, %.4f], time: %.2fs",
            actions.shape,
            float(actions.min()),
            float(actions.max()),
            dt,
        )

        horizon_info = analyze_action_horizon(actions)
        horizon_stats.append(horizon_info)
        logging.info(
            "  Horizon dynamics: first_last_l2=%.5f mean_step_delta_l2=%.5f max_step_delta_l2=%.5f mean_per_dim_range=%.5f max_per_dim_range=%.5f",
            float(horizon_info["first_last_l2"]),
            float(horizon_info["mean_step_delta_l2"]),
            float(horizon_info["max_step_delta_l2"]),
            float(horizon_info["mean_per_dim_range"]),
            float(horizon_info["max_per_dim_range"]),
        )

        if compare_actions:
            pred_first = actions[0]
            gt_current = gt_actions[frame_idx]
            compared_predicted.append(pred_first)
            compared_gt.append(gt_current)
            step_mae = float(np.mean(np.abs(pred_first - gt_current)))
            step_rmse = float(np.sqrt(np.mean((pred_first - gt_current) ** 2)))
            logging.info(
                "  First-step vs GT: mae=%.5f rmse=%.5f pred[:6]=%s gt[:6]=%s",
                step_mae,
                step_rmse,
                np.array2string(pred_first[:6], precision=4, separator=", "),
                np.array2string(gt_current[:6], precision=4, separator=", "),
            )

        segment_step_count += 1
        is_segment_boundary = segment_step_count >= save_every
        is_last_step = step_idx == len(frame_indices) - 1
        if is_segment_boundary or is_last_step:
            logging.info(
                "Saving segment %s after %s step(s) via reset...",
                segment_idx,
                segment_step_count,
            )
            client.reset({})
            logging.info("Segment %s saved.", segment_idx)

            if not is_last_step:
                segment_idx += 1
                segment_step_count = 0
                session_id = str(uuid.uuid4())
                logging.info("Segment %s session ID: %s", segment_idx, session_id)

    if compare_actions and compared_predicted:
        summarize_action_comparison(
            np.stack(compared_predicted, axis=0),
            np.stack(compared_gt, axis=0),
            horizon_stats,
            summary_path,
        )
    elif horizon_stats:
        logging.info("Action horizon summary without GT comparison:")
        logging.info("  analyzed_steps: %s", len(horizon_stats))
        logging.info("  avg first_last_l2: %.6f", float(np.mean([s["first_last_l2"] for s in horizon_stats])))
        logging.info("  avg mean_step_delta_l2: %.6f", float(np.mean([s["mean_step_delta_l2"] for s in horizon_stats])))
        logging.info("  avg max_step_delta_l2: %.6f", float(np.mean([s["max_step_delta_l2"] for s in horizon_stats])))

    logging.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay AgiBot sample_dataset episode through DreamZero")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--task-id", type=int, default=355)
    parser.add_argument("--episode-id", type=int, default=662854)
    parser.add_argument(
        "--prompt-mode",
        choices=["task_name", "scene", "skill", "task_and_skill"],
        default="task_and_skill",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--no-compare-actions", action="store_true")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON path for action comparison summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_client(
        host=args.host,
        port=args.port,
        sample_root=args.sample_root,
        task_id=args.task_id,
        episode_id=args.episode_id,
        prompt_mode=args.prompt_mode,
        max_steps=args.max_steps,
        frame_stride=args.frame_stride,
        save_every=args.save_every,
        compare_actions=not args.no_compare_actions,
        summary_path=args.summary_path,
    )


if __name__ == "__main__":
    main()
