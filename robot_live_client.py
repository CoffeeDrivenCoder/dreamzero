#!/usr/bin/env python3
"""Live client for AgiBot policy server using GDK Python APIs (CosineCamera + RobotDds).

This client reads real-time images and robot states, maps them to the AgiBot
observation schema, and sends them to the websocket policy server.
"""

import argparse
import logging
import time
import uuid
from typing import Tuple

import numpy as np
import cv2

from eval_utils.policy_client import WebsocketClientPolicy

try:
    from a2d_sdk.robot import CosineCamera as Camera
    from a2d_sdk.robot import RobotDds as Robot
except Exception as exc:  # pragma: no cover - depends on robot runtime
    raise RuntimeError(
        "Failed to import GDK python SDK. Make sure a2d_sdk is installed and env.sh is sourced."
    ) from exc


def _wait_for_image(camera: Camera, name: str, timeout_s: float = 2.0) -> Tuple[np.ndarray, int]:
    """Wait for a non-None image from the camera."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        img, ts = camera.get_latest_image(name)
        last = img
        if img is not None:
            return img, ts
        time.sleep(0.01)
    raise RuntimeError(f"Camera '{name}' did not return a valid frame within {timeout_s}s (last={last}).")


def _build_obs(
    head_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    arm_pos: list,
    head_pos: list,
    waist_pos: list,
    gripper_pos: list,
    prompt: str,
    session_id: str,
) -> dict:
    obs: dict[str, object] = {}
    obs["observation/top_head"] = head_img
    obs["observation/hand_left"] = left_img
    obs["observation/hand_right"] = right_img

    arm_pos = np.asarray(arm_pos, dtype=np.float32)
    if arm_pos.shape[0] != 14:
        raise ValueError(f"arm_joint_states length must be 14, got {arm_pos.shape[0]}")
    obs["observation/left_arm_joint_position"] = arm_pos[:7]
    obs["observation/right_arm_joint_position"] = arm_pos[7:]

    gripper_pos = np.asarray(gripper_pos, dtype=np.float32)
    if gripper_pos.shape[0] != 2:
        raise ValueError(f"gripper_states length must be 2, got {gripper_pos.shape[0]}")
    obs["observation/left_effector_position"] = np.asarray([gripper_pos[0]], dtype=np.float32)
    obs["observation/right_effector_position"] = np.asarray([gripper_pos[1]], dtype=np.float32)

    head_pos = np.asarray(head_pos, dtype=np.float32)
    waist_pos = np.asarray(waist_pos, dtype=np.float32)
    if head_pos.shape[0] != 2:
        raise ValueError(f"head_joint_states length must be 2, got {head_pos.shape[0]}")
    if waist_pos.shape[0] != 2:
        raise ValueError(f"waist_joint_states length must be 2, got {waist_pos.shape[0]}")
    obs["observation/head_position"] = head_pos
    obs["observation/waist_pitch"] = np.asarray([waist_pos[0]], dtype=np.float32)
    obs["observation/waist_lift"] = np.asarray([waist_pos[1]], dtype=np.float32)

    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def run_client(
    host: str,
    port: int,
    prompt: str,
    hz: float,
    max_steps: int | None,
) -> None:
    logging.info("Connecting to AgiBot server at %s:%s...", host, port)
    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    image_resolution = metadata.get("image_resolution")
    if image_resolution is not None:
        target_w, target_h = int(image_resolution[0]), int(image_resolution[1])
    else:
        target_w = target_h = None

    camera = Camera(["head", "hand_left", "hand_right"])
    robot = Robot()
    time.sleep(0.5)

    _wait_for_image(camera, "head")
    _wait_for_image(camera, "hand_left")
    _wait_for_image(camera, "hand_right")

    session_id = str(uuid.uuid4())
    step = 0
    period = 1.0 / max(hz, 1e-6)
    logging.info("Starting live loop at %.2f Hz, session_id=%s", hz, session_id)

    try:
        while True:
            t0 = time.time()

            head_img, _ = camera.get_latest_image("head")
            left_img, _ = camera.get_latest_image("hand_left")
            right_img, _ = camera.get_latest_image("hand_right")
            if head_img is None or left_img is None or right_img is None:
                logging.warning("Camera returned None frame(s); skipping step.")
                time.sleep(0.01)
                continue

            if target_w is not None and target_h is not None:
                head_img = cv2.resize(head_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                left_img = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                right_img = cv2.resize(right_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            arm_pos, _ = robot.arm_joint_states()
            head_pos, _ = robot.head_joint_states()
            waist_pos, _ = robot.waist_joint_states()
            gripper_pos, _ = robot.gripper_states()

            obs = _build_obs(
                head_img=head_img,
                left_img=left_img,
                right_img=right_img,
                arm_pos=arm_pos,
                head_pos=head_pos,
                waist_pos=waist_pos,
                gripper_pos=gripper_pos,
                prompt=prompt,
                session_id=session_id,
            )

            actions = client.infer(obs)
            actions = np.asarray(actions)
            dt = time.time() - t0
            logging.info(
                "Step %s | action shape=%s range=[%.4f, %.4f] dt=%.3fs",
                step,
                actions.shape,
                float(actions.min()),
                float(actions.max()),
                dt,
            )

            step += 1
            if max_steps is not None and step >= max_steps:
                break

            sleep_s = period - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        camera.close()
        robot.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live AgiBot client using GDK Python APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument("--prompt", default="Pick up the object")
    parser.add_argument("--hz", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_client(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        hz=args.hz,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
