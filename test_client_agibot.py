#!/usr/bin/env python3
"""Test client for AgiBot policy server using three debug videos.

Reads videos from debug_image/ and sends one synchronized frame per inference step
using the AgiBot observation schema expected by socket_test_optimized_AR.py.
"""

import argparse
import logging
import os
import time
import uuid

import av
import cv2
import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")
CAMERA_FILES = {
    "observation/top_head": "head_color.mp4",
    "observation/hand_left": "hand_left_color.mp4",
    "observation/hand_right": "hand_right_color.mp4",
}

STATE_DEFAULTS = {
    "observation/left_arm_joint_position": np.zeros(7, dtype=np.float32),
    "observation/right_arm_joint_position": np.zeros(7, dtype=np.float32),
    "observation/left_effector_position": np.zeros(1, dtype=np.float32),
    "observation/right_effector_position": np.zeros(1, dtype=np.float32),
    "observation/head_position": np.zeros(2, dtype=np.float32),
    "observation/waist_pitch": np.zeros(1, dtype=np.float32),
    "observation/waist_lift": np.zeros(1, dtype=np.float32),
}


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


def load_camera_frames() -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for obs_key, filename in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, filename)
        camera_frames[obs_key] = load_all_frames(path)
        logging.info("Loaded %s: %s", obs_key, camera_frames[obs_key].shape)
    return camera_frames


def build_obs(camera_frames: dict[str, np.ndarray], frame_idx: int, prompt: str, session_id: str) -> dict:
    obs: dict[str, object] = {}
    for obs_key, frames in camera_frames.items():
        obs[obs_key] = frames[frame_idx]
    for key, value in STATE_DEFAULTS.items():
        obs[key] = value.copy()
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    state_shapes = {k: np.asarray(obs[k]).shape for k in STATE_DEFAULTS}
    logging.info("State shapes for frame %s: %s", frame_idx, state_shapes)
    return obs


def run_client(
    host: str,
    port: int,
    prompt: str,
    max_steps: int | None,
    frame_stride: int,
    save_every: int,
) -> None:
    logging.info("Connecting to AgiBot server at %s:%s...", host, port)
    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    camera_frames = load_camera_frames()
    total_frames = min(v.shape[0] for v in camera_frames.values())
    frame_indices = list(range(0, total_frames, frame_stride))
    if max_steps is not None:
        frame_indices = frame_indices[:max_steps]

    logging.info("Total synchronized steps: %s", len(frame_indices))
    logging.info("Saving one video segment every %s steps", save_every)

    session_id = str(uuid.uuid4())
    segment_idx = 0
    segment_step_count = 0
    logging.info("Segment %s session ID: %s", segment_idx, session_id)

    for step_idx, frame_idx in enumerate(frame_indices):
        obs = build_obs(camera_frames, frame_idx, prompt, session_id)
        logging.info("=== Step %s/%s: frame %s ===", step_idx + 1, len(frame_indices), frame_idx)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        actions = np.asarray(actions)
        logging.info(
            "  Action shape: %s, range: [%.4f, %.4f], time: %.2fs",
            actions.shape,
            float(actions.min()),
            float(actions.max()),
            dt,
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

    logging.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test AgiBot server with videos from debug_image/")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--prompt", default="Pick up the object")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_client(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        max_steps=args.max_steps,
        frame_stride=args.frame_stride,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
