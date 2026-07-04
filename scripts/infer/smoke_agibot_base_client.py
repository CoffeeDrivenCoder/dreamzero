#!/usr/bin/env python3
"""Smoke-test a DreamZero AgiBot websocket server with synthetic observations."""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from eval_utils.policy_client import WebsocketClientPolicy


def _make_frames(history: int, height: int, width: int, camera_index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for t in range(history):
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[..., 0] = (xx // 3 + 17 * t + 31 * camera_index) % 256
        image[..., 1] = (yy // 2 + 11 * t + 47 * camera_index) % 256
        image[..., 2] = ((xx + yy) // 5 + 23 * t + 19 * camera_index) % 256
        frames.append(image)
    return np.stack(frames, axis=0)


def _encode_video_observation(frames: np.ndarray, image_transport: str, jpeg_quality: int) -> object:
    if image_transport == "raw":
        return frames
    quality = int(np.clip(jpeg_quality, 1, 100))
    encoded_frames: list[bytes] = []
    for frame in frames:
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise RuntimeError("Failed to JPEG-encode smoke-test frame")
        encoded_frames.append(encoded.tobytes())
    return {
        "__dreamzero_image_encoding__": "jpeg_sequence",
        "shape": tuple(int(dim) for dim in frames.shape),
        "dtype": str(frames.dtype),
        "quality": quality,
        "frames": encoded_frames,
    }


def build_obs(args: argparse.Namespace, session_id: str) -> dict[str, object]:
    obs: dict[str, object] = {
        "observation/top_head": _encode_video_observation(
            _make_frames(args.history, args.height, args.width, 0),
            args.image_transport,
            args.image_jpeg_quality,
        ),
        "observation/hand_left": _encode_video_observation(
            _make_frames(args.history, args.height, args.width, 1),
            args.image_transport,
            args.image_jpeg_quality,
        ),
        "observation/hand_right": _encode_video_observation(
            _make_frames(args.history, args.height, args.width, 2),
            args.image_transport,
            args.image_jpeg_quality,
        ),
        "observation/left_arm_joint_position": np.zeros(7, dtype=np.float32),
        "observation/right_arm_joint_position": np.zeros(7, dtype=np.float32),
        "observation/left_effector_position": np.zeros(1, dtype=np.float32),
        "observation/right_effector_position": np.zeros(1, dtype=np.float32),
        "observation/head_position": np.zeros(2, dtype=np.float32),
        "observation/waist_pitch": np.zeros(1, dtype=np.float32),
        "observation/waist_lift": np.zeros(1, dtype=np.float32),
        "prompt": args.prompt,
        "session_id": session_id,
    }
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test DreamZero AgiBot base inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument("--prompt", default="拿起桌上的杯子。")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--image-transport", choices=["raw", "jpeg"], default="jpeg")
    parser.add_argument("--image-jpeg-quality", type=int, default=80)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    session_id = str(uuid.uuid4())
    records = []
    for step in range(args.steps):
        obs = build_obs(args, session_id)
        started = time.time()
        actions = np.asarray(client.infer(obs), dtype=np.float32)
        dt = time.time() - started
        if actions.ndim != 2 or actions.shape[1] < 22:
            raise RuntimeError(f"Unexpected action shape from server: {actions.shape}")
        record = {
            "step": step,
            "duration_s": dt,
            "shape": list(actions.shape),
            "min": float(actions.min()),
            "max": float(actions.max()),
            "mean": float(actions.mean()),
            "first_action": actions[0, :22].round(6).tolist(),
            "last_action": actions[-1, :22].round(6).tolist(),
        }
        records.append(record)
        logging.info(
            "Step %s action shape=%s range=[%.4f, %.4f] mean=%.4f dt=%.2fs",
            step,
            actions.shape,
            record["min"],
            record["max"],
            record["mean"],
            dt,
        )

    reset_response = client.reset({})
    logging.info("Reset response: %s", reset_response)

    payload = {"metadata": metadata, "records": records}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
