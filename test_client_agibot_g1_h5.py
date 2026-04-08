#!/usr/bin/env python3
"""Offline validation client for DreamZero AgiBot using G1 raw_joints recordings."""

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

DEBUG_VIDEO_DIR = Path("/root/autodl-tmp/dreamzero/debug_image")
VIDEO_FILENAMES = {
    "observation/top_head": "head_color.mp4",
    "observation/hand_left": "hand_left_color.mp4",
    "observation/hand_right": "hand_right_color.mp4",
}
ACTION_DIM_LAYOUT = [
    ("action.left_arm_joint_position", 7),
    ("action.right_arm_joint_position", 7),
    ("action.left_effector_position", 1),
    ("action.right_effector_position", 1),
    ("action.head_position", 2),
    ("action.waist_pitch", 1),
    ("action.waist_lift", 1),
    ("action.robot_velocity", 2),
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


def _resize_frames(frames: np.ndarray, width: int, height: int) -> np.ndarray:
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    resized = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(resized, axis=0)


def load_all_frames(video_path: str, expected_resolution: tuple[int, int]) -> np.ndarray:
    frames = _load_all_frames_opencv(video_path)
    if not frames:
        logging.info("OpenCV failed to decode %s, falling back to PyAV...", video_path)
        frames = _load_all_frames_pyav(video_path)
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    arr = np.stack(frames, axis=0)
    width, height = expected_resolution
    arr = _resize_frames(arr, width=width, height=height)
    return arr


def load_camera_frames(video_dir: Path, expected_resolution: tuple[int, int]) -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for obs_key, filename in VIDEO_FILENAMES.items():
        path = video_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required video: {path}")
        camera_frames[obs_key] = load_all_frames(str(path), expected_resolution)
        logging.info("Loaded %s: %s", obs_key, camera_frames[obs_key].shape)
    return camera_frames


def _normalized_query_grid(num_steps: int) -> np.ndarray:
    if num_steps <= 1:
        return np.zeros((num_steps,), dtype=np.float64)
    return np.linspace(0.0, 1.0, num_steps, dtype=np.float64)


def _normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        return timestamps
    if timestamps.size == 1:
        return np.zeros((1,), dtype=np.float64)
    t0 = timestamps[0]
    t1 = timestamps[-1]
    if abs(t1 - t0) < 1e-12:
        return np.zeros_like(timestamps, dtype=np.float64)
    return (timestamps - t0) / (t1 - t0)


def _interp_values(values: np.ndarray, timestamps: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps)
    target_times = np.asarray(target_times, dtype=np.float64)

    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return np.zeros((target_times.shape[0], values.shape[1]), dtype=np.float32)
    if values.shape[0] == 1:
        return np.repeat(values.astype(np.float32), target_times.shape[0], axis=0)

    src_t = _normalize_timestamps(timestamps)
    out = np.empty((target_times.shape[0], values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(
            target_times,
            src_t,
            values[:, dim].astype(np.float64),
            left=float(values[0, dim]),
            right=float(values[-1, dim]),
        ).astype(np.float32)
    return out


def _load_recording_meta(record_root: Path) -> dict:
    with open(record_root / "meta_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _resample_clip_state_and_actions(record_root: Path, clip_steps: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    meta = _load_recording_meta(record_root)
    raw_path = record_root / "record" / "raw_joints.h5"
    target_t = _normalized_query_grid(clip_steps)

    with h5py.File(raw_path, "r") as f:
        state_joint = _interp_values(f["state/joint/position"][:], f["state/joint/timestamp"][:], target_t)
        state_left_eff = _interp_values(f["state/left_effector/position"][:], f["state/left_effector/timestamp"][:], target_t)
        state_right_eff = _interp_values(f["state/right_effector/position"][:], f["state/right_effector/timestamp"][:], target_t)
        state_head = _interp_values(f["state/head/position"][:], f["state/head/timestamp"][:], target_t)
        state_waist = _interp_values(f["state/waist/position"][:], f["state/waist/timestamp"][:], target_t)

        action_joint = _interp_values(f["action/joint/position"][:], f["action/joint/timestamp"][:], target_t)
        action_left_eff = _interp_values(f["action/left_effector/position"][:], f["action/left_effector/timestamp"][:], target_t)
        action_right_eff = _interp_values(f["action/right_effector/position"][:], f["action/right_effector/timestamp"][:], target_t)
        action_robot_vel = _interp_values(f["action/robot/velocity"][:], f["action/robot/timestamp"][:], target_t)

        if "action/head/position" in f and f["action/head/position"].shape[0] > 0:
            action_head = _interp_values(f["action/head/position"][:], f["action/head/timestamp"][:], target_t)
            action_head_mask = np.ones((clip_steps, 2), dtype=bool)
        else:
            action_head = np.zeros((clip_steps, 2), dtype=np.float32)
            action_head_mask = np.zeros((clip_steps, 2), dtype=bool)

        if "action/waist/position" in f and f["action/waist/position"].shape[0] > 0:
            action_waist = _interp_values(f["action/waist/position"][:], f["action/waist/timestamp"][:], target_t)
            action_waist_mask = np.ones((clip_steps, 2), dtype=bool)
        else:
            action_waist = np.zeros((clip_steps, 2), dtype=np.float32)
            action_waist_mask = np.zeros((clip_steps, 2), dtype=bool)

    proprio = {
        "observation/left_arm_joint_position": state_joint[:, :7],
        "observation/right_arm_joint_position": state_joint[:, 7:],
        "observation/left_effector_position": state_left_eff,
        "observation/right_effector_position": state_right_eff,
        "observation/head_position": state_head,
        "observation/waist_pitch": state_waist[:, 0:1],
        "observation/waist_lift": state_waist[:, 1:2],
    }
    gt_actions = np.concatenate(
        [
            action_joint[:, :7],
            action_joint[:, 7:],
            action_left_eff,
            action_right_eff,
            action_head,
            action_waist[:, 0:1],
            action_waist[:, 1:2],
            action_robot_vel,
        ],
        axis=-1,
    ).astype(np.float32)
    gt_mask = np.concatenate(
        [
            np.ones((clip_steps, 7), dtype=bool),
            np.ones((clip_steps, 7), dtype=bool),
            np.ones((clip_steps, 1), dtype=bool),
            np.ones((clip_steps, 1), dtype=bool),
            action_head_mask,
            action_waist_mask[:, 0:1],
            action_waist_mask[:, 1:2],
            np.ones((clip_steps, 2), dtype=bool),
        ],
        axis=-1,
    )
    logging.info("Loaded %s: duration=%ss, task_id=%s, allocated_steps=%s", record_root, meta.get("duration"), meta.get("task_id"), clip_steps)
    return proprio, gt_actions, gt_mask


def load_g1_sequence(record_roots: list[Path], total_steps: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    metas = [_load_recording_meta(root) for root in record_roots]
    durations = np.asarray([float(meta.get("duration", 0.0)) for meta in metas], dtype=np.float64)
    if np.any(durations <= 0):
        raise ValueError(f"Invalid recording durations: {durations.tolist()}")

    raw_counts = durations / durations.sum() * total_steps
    clip_steps = np.floor(raw_counts).astype(int)
    clip_steps = np.maximum(clip_steps, 1)
    remainder = total_steps - int(clip_steps.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw_counts - np.floor(raw_counts)))
        for idx in frac_order[:remainder]:
            clip_steps[idx] += 1
    elif remainder < 0:
        frac_order = np.argsort(raw_counts - np.floor(raw_counts))
        for idx in frac_order[: -remainder]:
            if clip_steps[idx] > 1:
                clip_steps[idx] -= 1
    clip_steps[-1] += total_steps - int(clip_steps.sum())
    logging.info("Clip allocation across %s recording(s): %s", len(record_roots), clip_steps.tolist())

    proprio_chunks = {k: [] for k in [
        "observation/left_arm_joint_position",
        "observation/right_arm_joint_position",
        "observation/left_effector_position",
        "observation/right_effector_position",
        "observation/head_position",
        "observation/waist_pitch",
        "observation/waist_lift",
    ]}
    action_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    for root, num_steps in zip(record_roots, clip_steps.tolist()):
        clip_proprio, clip_actions, clip_mask = _resample_clip_state_and_actions(root, num_steps)
        for key in proprio_chunks:
            proprio_chunks[key].append(clip_proprio[key])
        action_chunks.append(clip_actions)
        mask_chunks.append(clip_mask)

    proprio = {key: np.concatenate(chunks, axis=0).astype(np.float32) for key, chunks in proprio_chunks.items()}
    gt_actions = np.concatenate(action_chunks, axis=0).astype(np.float32)
    gt_mask = np.concatenate(mask_chunks, axis=0).astype(bool)
    return proprio, gt_actions, gt_mask


def build_obs(camera_frames: dict[str, np.ndarray], proprio: dict[str, np.ndarray], frame_idx: int, prompt: str, session_id: str) -> dict:
    obs: dict[str, object] = {}
    for obs_key, frames in camera_frames.items():
        obs[obs_key] = frames[frame_idx]
    for key, values in proprio.items():
        obs[key] = values[frame_idx].copy()
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    state_shapes = {k: np.asarray(obs[k]).shape for k in proprio.keys()}
    logging.info("State shapes for frame %s: %s", frame_idx, state_shapes)
    logging.info("Prompt for frame %s: %s", frame_idx, prompt)
    return obs


def analyze_action_horizon(actions: np.ndarray) -> dict[str, object]:
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (T, D), got {actions.shape}")
    if actions.shape[0] < 2:
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


def summarize_action_comparison(predicted_first_steps: np.ndarray, gt_actions: np.ndarray, gt_mask: np.ndarray, horizon_stats: list[dict[str, object]], summary_path: Path | None) -> None:
    diff = predicted_first_steps - gt_actions
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    valid = gt_mask.astype(np.float32)
    valid_counts = np.sum(valid, axis=0)
    mae_per_dim = np.divide(np.sum(abs_diff * valid, axis=0), np.maximum(valid_counts, 1.0))
    rmse_per_dim = np.sqrt(np.divide(np.sum(sq_diff * valid, axis=0), np.maximum(valid_counts, 1.0)))
    pred_mean = np.divide(np.sum(predicted_first_steps * valid, axis=0), np.maximum(valid_counts, 1.0))
    gt_mean = np.divide(np.sum(gt_actions * valid, axis=0), np.maximum(valid_counts, 1.0))
    pred_std = np.zeros(predicted_first_steps.shape[1], dtype=np.float32)
    gt_std = np.zeros(gt_actions.shape[1], dtype=np.float32)
    corr_per_dim: list[float | None] = []
    for dim in range(predicted_first_steps.shape[1]):
        mask = gt_mask[:, dim]
        if np.count_nonzero(mask) < 2:
            corr_per_dim.append(None)
            continue
        pred_col = predicted_first_steps[mask, dim]
        gt_col = gt_actions[mask, dim]
        pred_std[dim] = float(np.std(pred_col))
        gt_std[dim] = float(np.std(gt_col))
        if np.std(pred_col) < 1e-8 or np.std(gt_col) < 1e-8:
            corr_per_dim.append(None)
        else:
            corr_per_dim.append(float(np.corrcoef(pred_col, gt_col)[0, 1]))
    overall_mae = float(np.sum(abs_diff * valid) / np.maximum(np.sum(valid), 1.0))
    overall_rmse = float(np.sqrt(np.sum(sq_diff * valid) / np.maximum(np.sum(valid), 1.0)))
    horizon_first_last = np.asarray([s["first_last_l2"] for s in horizon_stats], dtype=np.float32)
    horizon_mean_step = np.asarray([s["mean_step_delta_l2"] for s in horizon_stats], dtype=np.float32)
    horizon_max_step = np.asarray([s["max_step_delta_l2"] for s in horizon_stats], dtype=np.float32)
    horizon_mean_range = np.asarray([s["mean_per_dim_range"] for s in horizon_stats], dtype=np.float32)
    horizon_max_range = np.asarray([s["max_per_dim_range"] for s in horizon_stats], dtype=np.float32)
    logging.info("Action comparison summary:")
    logging.info("  compared_steps: %s", predicted_first_steps.shape[0])
    logging.info("  valid_action_dims: %s/%s", int(np.count_nonzero(valid_counts)), predicted_first_steps.shape[1])
    logging.info("  overall_mae: %.6f", overall_mae)
    logging.info("  overall_rmse: %.6f", overall_rmse)
    logging.info("Action horizon summary:")
    logging.info("  avg first_last_l2: %.6f", float(np.mean(horizon_first_last)))
    logging.info("  avg mean_step_delta_l2: %.6f", float(np.mean(horizon_mean_step)))
    logging.info("  avg max_step_delta_l2: %.6f", float(np.mean(horizon_max_step)))
    logging.info("  avg mean_per_dim_range: %.6f", float(np.mean(horizon_mean_range)))
    logging.info("  avg max_per_dim_range: %.6f", float(np.mean(horizon_max_range)))
    dim_labels: list[str] = []
    for name, width in ACTION_DIM_LAYOUT:
        if width == 1:
            dim_labels.append(name)
        else:
            for i in range(width):
                dim_labels.append(f"{name}[{i}]")
    for dim, label in enumerate(dim_labels):
        logging.info("  dim %02d %-32s available=%s pred_mean=%.5f gt_mean=%.5f pred_std=%.5f gt_std=%.5f mae=%.5f rmse=%.5f corr=%s", dim, label, bool(valid_counts[dim] > 0), float(pred_mean[dim]), float(gt_mean[dim]), float(pred_std[dim]), float(gt_std[dim]), float(mae_per_dim[dim]), float(rmse_per_dim[dim]), "None" if corr_per_dim[dim] is None else f"{corr_per_dim[dim]:.5f}")
    if summary_path is not None:
        payload = {
            "compared_steps": int(predicted_first_steps.shape[0]),
            "action_dim": int(predicted_first_steps.shape[1]),
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "dim_labels": dim_labels,
            "available_gt_per_dim": (valid_counts > 0).tolist(),
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
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info("Saved action comparison summary to %s", summary_path)


def run_client(host: str, port: int, video_dir: Path, record_roots: list[Path], prompt: str, max_steps: int | None, frame_stride: int, save_every: int, compare_actions: bool, summary_path: Path | None) -> None:
    logging.info("Connecting to AgiBot server at %s:%s...", host, port)
    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    expected_resolution = tuple(int(x) for x in metadata.get("image_resolution", [640, 480]))
    logging.info("Resizing all input frames to expected resolution: %s", expected_resolution)
    camera_frames = load_camera_frames(video_dir, expected_resolution)
    total_frames = min(v.shape[0] for v in camera_frames.values())
    proprio, gt_actions, gt_mask = load_g1_sequence(record_roots, total_frames)
    frame_indices = list(range(0, total_frames, frame_stride))
    if max_steps is not None:
        frame_indices = frame_indices[:max_steps]
    logging.info("Video dir: %s", video_dir)
    logging.info("Record roots: %s", [str(p) for p in record_roots])
    logging.info("Total synchronized steps: %s", len(frame_indices))
    logging.info("Saving one video segment every %s steps", save_every)
    logging.info("Compare predicted first-step actions to G1 GT: %s", compare_actions)
    compared_predicted: list[np.ndarray] = []
    compared_gt: list[np.ndarray] = []
    compared_mask: list[np.ndarray] = []
    horizon_stats: list[dict[str, object]] = []
    session_id = str(uuid.uuid4())
    segment_idx = 0
    segment_step_count = 0
    logging.info("Segment %s session ID: %s", segment_idx, session_id)
    for step_idx, frame_idx in enumerate(frame_indices):
        obs = build_obs(camera_frames, proprio, frame_idx, prompt, session_id)
        logging.info("=== Step %s/%s: frame %s ===", step_idx + 1, len(frame_indices), frame_idx)
        t0 = time.time()
        actions = np.asarray(client.infer(obs), dtype=np.float32)
        dt = time.time() - t0
        logging.info("  Action shape: %s, range: [%.4f, %.4f], time: %.2fs", actions.shape, float(actions.min()), float(actions.max()), dt)
        horizon_info = analyze_action_horizon(actions)
        horizon_stats.append(horizon_info)
        logging.info("  Horizon dynamics: first_last_l2=%.5f mean_step_delta_l2=%.5f max_step_delta_l2=%.5f mean_per_dim_range=%.5f max_per_dim_range=%.5f", float(horizon_info["first_last_l2"]), float(horizon_info["mean_step_delta_l2"]), float(horizon_info["max_step_delta_l2"]), float(horizon_info["mean_per_dim_range"]), float(horizon_info["max_per_dim_range"]))
        if compare_actions:
            pred_first = actions[0]
            gt_current = gt_actions[frame_idx]
            gt_current_mask = gt_mask[frame_idx]
            compared_predicted.append(pred_first)
            compared_gt.append(gt_current)
            compared_mask.append(gt_current_mask)
            if np.any(gt_current_mask):
                masked_diff = pred_first[gt_current_mask] - gt_current[gt_current_mask]
                step_mae = float(np.mean(np.abs(masked_diff)))
                step_rmse = float(np.sqrt(np.mean(masked_diff ** 2)))
            else:
                step_mae = 0.0
                step_rmse = 0.0
            logging.info("  First-step vs G1 GT: mae=%.5f rmse=%.5f pred[:6]=%s gt[:6]=%s", step_mae, step_rmse, np.array2string(pred_first[:6], precision=4, separator=", "), np.array2string(gt_current[:6], precision=4, separator=", "))
        segment_step_count += 1
        is_segment_boundary = segment_step_count >= save_every
        is_last_step = step_idx == len(frame_indices) - 1
        if is_segment_boundary or is_last_step:
            logging.info("Saving segment %s after %s step(s) via reset...", segment_idx, segment_step_count)
            client.reset({})
            logging.info("Segment %s saved.", segment_idx)
            if not is_last_step:
                segment_idx += 1
                segment_step_count = 0
                session_id = str(uuid.uuid4())
                logging.info("Segment %s session ID: %s", segment_idx, session_id)
    if compare_actions and compared_predicted:
        summarize_action_comparison(np.stack(compared_predicted, axis=0), np.stack(compared_gt, axis=0), np.stack(compared_mask, axis=0), horizon_stats, summary_path)
    logging.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay G1 recordings through DreamZero with debug videos")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--video-dir", type=Path, default=DEBUG_VIDEO_DIR)
    parser.add_argument("--record-roots", nargs="+", type=Path, default=[Path("/root/autodl-tmp/G1-1"), Path("/root/autodl-tmp/G1-2")], help="One or more G1 recording roots, concatenated in the provided order.")
    parser.add_argument("--prompt", default="Manipulate the object safely.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--no-compare-actions", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_client(args.host, args.port, args.video_dir, args.record_roots, args.prompt, args.max_steps, args.frame_stride, args.save_every, not args.no_compare_actions, args.summary_path)


if __name__ == "__main__":
    main()
