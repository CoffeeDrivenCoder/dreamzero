#!/usr/bin/env python3
"""Live client for AgiBot policy server using GDK Python APIs (CosineCamera + RobotDds).

This client reads real-time images and robot states, maps them to the AgiBot
observation schema, and sends them to the websocket policy server.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import select
import sys
import termios
import time
import tty
import uuid
from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import numpy as np
import cv2
import ruckig

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


ArmOrder = Literal["left_right", "right_left"]
DEFAULT_SDK_ARM_ORDER: ArmOrder = "left_right"
ARM_JOINT_COUNT = 14
RIGHT_ARM_START = 7
MOVE_ARM_INTERVAL_S = 0.01
MOVE_ARM_TIMEOUT_S = 5.0
POSITION_TOLERANCE_RAD = 5e-3
RUCKIG_MAX_VELOCITY = 2.0
RUCKIG_MAX_ACCELERATION = 1.0
RUCKIG_MAX_JERK = 5.0
GRIPPER_OBSERVATION_MAX = 120.0


class _StopRequested(Exception):
    """Raised when the operator requests the client to stop."""


class _KeyboardMonitor:
    """Non-blocking keyboard monitor for stop hotkeys."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs: list | None = None
        self._enabled = False

    def __enter__(self) -> "_KeyboardMonitor":
        if not sys.stdin.isatty():
            logging.warning("stdin is not a TTY; hotkey stop is disabled.")
            return self
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        self._enabled = False
        self._fd = None
        self._old_attrs = None

    def consume_stop_request(self) -> bool:
        if not self._enabled:
            return False
        assert self._fd is not None
        ready, _, _ = select.select([self._fd], [], [], 0.0)
        if not ready:
            return False
        key = sys.stdin.read(1).lower()
        return key == "q"


def _sdk_to_policy_arm(arm_pos: list | np.ndarray, sdk_arm_order: ArmOrder) -> np.ndarray:
    """Map SDK-native arm order to policy order [left_arm, right_arm]."""
    arm = np.asarray(arm_pos, dtype=np.float32)
    if arm.shape[0] != 14:
        raise ValueError(f"arm_joint_states length must be 14, got {arm.shape[0]}")
    if sdk_arm_order == "left_right":
        return arm.copy()
    if sdk_arm_order == "right_left":
        return np.concatenate([arm[7:], arm[:7]], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported sdk_arm_order: {sdk_arm_order}")


def _policy_to_sdk_arm(policy_arm: np.ndarray, sdk_arm_order: ArmOrder) -> np.ndarray:
    """Map policy order [left_arm, right_arm] back to SDK-native arm order."""
    arm = np.asarray(policy_arm, dtype=np.float32)
    if arm.shape[0] != 14:
        raise ValueError(f"policy arm vector length must be 14, got {arm.shape[0]}")
    if sdk_arm_order == "left_right":
        return arm.copy()
    if sdk_arm_order == "right_left":
        return np.concatenate([arm[7:], arm[:7]], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported sdk_arm_order: {sdk_arm_order}")


def _sdk_gripper_to_policy(values: list | np.ndarray) -> np.ndarray:
    """Map GDK gripper states to policy order [left, right].

    Per the GDK guide, gripper_states() and move_gripper() both use fixed
    [left, right] order and should not inherit arm_joint_states ordering.
    """
    pair = np.asarray(values, dtype=np.float32)
    if pair.shape[0] != 2:
        raise ValueError(f"gripper_states length must be 2, got {pair.shape[0]}")
    return pair.copy()


def _sdk_gripper_to_policy_obs(values: list | np.ndarray) -> np.ndarray:
    """Map GDK gripper states [0, 1] to the training-time observation scale [0, 120]."""
    pair = _sdk_gripper_to_policy(values)
    pair = np.clip(pair, 0.0, 1.0)
    return (pair * GRIPPER_OBSERVATION_MAX).astype(np.float32)


def _policy_gripper_to_sdk(values: list | np.ndarray) -> np.ndarray:
    """Map policy-order gripper command [left, right] back to GDK order."""
    pair = np.asarray(values, dtype=np.float32)
    if pair.shape[0] != 2:
        raise ValueError(f"gripper_cmd length must be 2, got {pair.shape[0]}")
    return pair.copy()


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
    sdk_arm_order: ArmOrder,
    obs_flip_config: ObsFlipConfig,
) -> dict:
    obs: dict[str, object] = {}
    obs["observation/top_head"] = head_img
    obs["observation/hand_left"] = left_img
    obs["observation/hand_right"] = right_img

    # Arm joints (14) -> left/right 7 each
    arm_pos = _sdk_to_policy_arm(arm_pos, sdk_arm_order)
    arm_pos = _apply_obs_joint_sign_flips(arm_pos, obs_flip_config)
    obs["observation/left_arm_joint_position"] = arm_pos[:7]
    obs["observation/right_arm_joint_position"] = arm_pos[7:]

    # The GDK exposes gripper states in fixed [left, right] order.
    # Keep that order independent from arm joint ordering.
    gripper_pos_policy = _sdk_gripper_to_policy_obs(gripper_pos)
    obs["observation/left_effector_position"] = np.asarray([gripper_pos_policy[0]], dtype=np.float32)
    obs["observation/right_effector_position"] = np.asarray([gripper_pos_policy[1]], dtype=np.float32)

    # Head and waist states
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


@dataclass
class Limits:
    arm_delta: float
    head_delta: float
    waist_pitch_delta: float
    waist_lift_delta: float


@dataclass(frozen=True)
class GripperConfig:
    close_threshold: float = 0.5
    close_when_high: bool = True
    open_value: float = 0.0
    closed_value: float = 1.0


@dataclass(frozen=True)
class ObsFlipConfig:
    left_joint_indices: tuple[int, ...] = ()
    right_joint_indices: tuple[int, ...] = ()


def _parse_joint_index_list(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw:
        return ()

    indices: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if not 1 <= value <= 7:
            raise ValueError(f"Joint index must be in 1..7, got {value}")
        if value not in indices:
            indices.append(value)
    return tuple(indices)


def _apply_obs_joint_sign_flips(policy_arm: np.ndarray, flip_config: ObsFlipConfig) -> np.ndarray:
    arm = np.asarray(policy_arm, dtype=np.float32).copy()
    for joint_index in flip_config.left_joint_indices:
        arm[joint_index - 1] *= -1.0
    for joint_index in flip_config.right_joint_indices:
        arm[RIGHT_ARM_START + joint_index - 1] *= -1.0
    return arm


def _clip_delta(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    delta = np.clip(delta, -max_delta, max_delta)
    return np.asarray(current, dtype=np.float32) + delta


def _sleep_with_stop(duration_s: float, should_stop: Callable[[], bool]) -> None:
    deadline = time.time() + max(duration_s, 0.0)
    while time.time() < deadline:
        if should_stop():
            raise _StopRequested
        time.sleep(min(0.02, max(deadline - time.time(), 0.0)))


def _parse_action_first(actions: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected action shape (T, D), got {arr.shape}")
    if arr.shape[1] < 22:
        raise ValueError(f"Expected at least 22 dims, got {arr.shape}")

    first = arr[0]
    return {
        "left_arm": first[0:7],
        "right_arm": first[7:14],
        "gripper": first[14:16],
        "head": first[16:18],
        "waist": first[18:20],
        "wheel": first[20:22],
        "horizon": np.asarray([arr.shape[0]], dtype=np.int32),
    }


def _select_gripper_command(actions: np.ndarray, gripper_config: GripperConfig) -> dict[str, np.ndarray]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 16:
        raise ValueError(f"Expected action shape (T, >=16), got {arr.shape}")

    first_policy = arr[0, 14:16].astype(np.float32)
    last_policy = arr[-1, 14:16].astype(np.float32)
    if gripper_config.close_when_high:
        close_mask = last_policy >= gripper_config.close_threshold
    else:
        close_mask = last_policy <= gripper_config.close_threshold

    command_policy = np.where(
        close_mask,
        gripper_config.closed_value,
        gripper_config.open_value,
    ).astype(np.float32)
    return {
        "first_policy": first_policy,
        "last_policy": last_policy,
        "command_policy": command_policy,
    }


def _build_safe_arm_targets(
    actions: np.ndarray,
    current_policy_arm: np.ndarray,
    limits: Limits,
) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < ARM_JOINT_COUNT:
        raise RuntimeError(f"Invalid arm trajectory shape: {arr.shape}")

    safe_rows: list[np.ndarray] = []
    previous = np.asarray(current_policy_arm, dtype=np.float32)
    for row in arr:
        target_arm = row[:ARM_JOINT_COUNT].astype(np.float32)
        safe_row = _clip_delta(target_arm, previous[:ARM_JOINT_COUNT], limits.arm_delta).astype(np.float32)
        safe_rows.append(safe_row)
        previous = safe_row
    return np.stack(safe_rows, axis=0)


def _build_segment_trajectory(
    current_positions: np.ndarray,
    target_positions: np.ndarray,
) -> list[list[float]]:
    dof = ARM_JOINT_COUNT
    rk = ruckig.Ruckig(dof, MOVE_ARM_INTERVAL_S)
    rk_input = ruckig.InputParameter(dof)
    rk_output = ruckig.OutputParameter(dof)

    rk_input.current_position = list(np.asarray(current_positions, dtype=np.float64))
    rk_input.current_velocity = [0.0] * dof
    rk_input.current_acceleration = [0.0] * dof
    rk_input.target_position = list(np.asarray(target_positions, dtype=np.float64))
    rk_input.target_velocity = [0.0] * dof
    rk_input.target_acceleration = [0.0] * dof
    rk_input.max_velocity = [RUCKIG_MAX_VELOCITY] * dof
    rk_input.max_acceleration = [RUCKIG_MAX_ACCELERATION] * dof
    rk_input.max_jerk = [RUCKIG_MAX_JERK] * dof

    trajectory_points: list[list[float]] = []
    while rk.update(rk_input, rk_output) == ruckig.Result.Working:
        trajectory_points.append(list(rk_output.new_position))
        rk_output.pass_to_input(rk_input)

    target_list = list(np.asarray(target_positions, dtype=np.float64))
    if not trajectory_points:
        trajectory_points.append(target_list)
    elif any(abs(a - b) > 1e-6 for a, b in zip(trajectory_points[-1], target_list)):
        trajectory_points.append(target_list)
    return trajectory_points


def _build_move_arm_points(
    safe_policy_targets: np.ndarray,
    sdk_arm_order: ArmOrder,
) -> list[list[float]]:
    points: list[list[float]] = []
    prev_sdk = _policy_to_sdk_arm(safe_policy_targets[0], sdk_arm_order)
    points.append(prev_sdk.astype(np.float32).tolist())
    for row in safe_policy_targets[1:]:
        target_sdk = _policy_to_sdk_arm(row, sdk_arm_order)
        segment = _build_segment_trajectory(prev_sdk, target_sdk)
        if points and segment:
            segment = segment[1:]
        points.extend([list(np.asarray(p, dtype=np.float32)) for p in segment])
        prev_sdk = np.asarray(target_sdk, dtype=np.float32)
    return points


def _wait_arm_close(
    robot: Robot,
    target_policy_arm: np.ndarray,
    sdk_arm_order: ArmOrder,
    should_stop: Callable[[], bool],
) -> dict[str, np.ndarray | float | bool]:
    deadline = time.time() + MOVE_ARM_TIMEOUT_S
    last_policy_arm = _sdk_to_policy_arm(np.asarray(robot.arm_joint_states()[0], dtype=np.float32), sdk_arm_order)
    reached = False

    while time.time() < deadline:
        if should_stop():
            raise _StopRequested

        arm_after, _ = robot.arm_joint_states()
        last_policy_arm = _sdk_to_policy_arm(np.asarray(arm_after, dtype=np.float32), sdk_arm_order)
        arm_error = np.max(np.abs(last_policy_arm[:ARM_JOINT_COUNT] - target_policy_arm[:ARM_JOINT_COUNT]))
        if arm_error <= POSITION_TOLERANCE_RAD:
            reached = True
            break
        time.sleep(0.1)

    return {
        "arm_after_policy": last_policy_arm,
        "arm_error": float(np.max(np.abs(last_policy_arm[:ARM_JOINT_COUNT] - target_policy_arm[:ARM_JOINT_COUNT]))),
        "left_error": float(np.max(np.abs(last_policy_arm[:RIGHT_ARM_START] - target_policy_arm[:RIGHT_ARM_START]))),
        "right_error": float(np.max(np.abs(last_policy_arm[RIGHT_ARM_START:ARM_JOINT_COUNT] - target_policy_arm[RIGHT_ARM_START:ARM_JOINT_COUNT]))),
        "reached": reached,
    }


def _execute_arm_trajectory(
    robot: Robot,
    actions: np.ndarray,
    limits: Limits,
    sdk_arm_order: ArmOrder,
    should_stop: Callable[[], bool],
) -> dict[str, np.ndarray | int | float | bool]:
    arm_cur, _ = robot.arm_joint_states()
    arm_cur_sdk = np.asarray(arm_cur, dtype=np.float32)
    arm_cur_policy = _sdk_to_policy_arm(arm_cur_sdk, sdk_arm_order)

    safe_policy_targets = _build_safe_arm_targets(actions, arm_cur_policy, limits)
    move_arm_points = _build_move_arm_points(safe_policy_targets, sdk_arm_order)

    for point in move_arm_points:
        if should_stop():
            raise _StopRequested
        robot.move_arm(point)
        _sleep_with_stop(MOVE_ARM_INTERVAL_S, should_stop)

    wait_result = _wait_arm_close(
        robot=robot,
        target_policy_arm=safe_policy_targets[-1],
        sdk_arm_order=sdk_arm_order,
        should_stop=should_stop,
    )
    return {
        "arm_cur_sdk": arm_cur_sdk,
        "arm_cur_policy": arm_cur_policy,
        "first_arm_policy": safe_policy_targets[0],
        "last_arm_policy": safe_policy_targets[-1],
        "target_rows": np.asarray([safe_policy_targets.shape[0]], dtype=np.int32),
        "move_arm_points": np.asarray([len(move_arm_points)], dtype=np.int32),
        "arm_after_policy": wait_result["arm_after_policy"],
        "arm_error": wait_result["arm_error"],
        "left_error": wait_result["left_error"],
        "right_error": wait_result["right_error"],
        "reached": wait_result["reached"],
    }


def _apply_aux_actions(
    robot: Robot,
    parsed: dict[str, np.ndarray],
    gripper_command_policy: np.ndarray,
    limits: Limits,
    enable_gripper: bool,
) -> dict[str, np.ndarray]:
    head_cur, _ = robot.head_joint_states()
    waist_cur, _ = robot.waist_joint_states()

    head_cur = np.asarray(head_cur, dtype=np.float32)
    waist_cur = np.asarray(waist_cur, dtype=np.float32)
    if head_cur.shape[0] != 2:
        raise RuntimeError(f"head_joint_states length must be 2, got {head_cur.shape[0]}")
    if waist_cur.shape[0] != 2:
        raise RuntimeError(f"waist_joint_states length must be 2, got {waist_cur.shape[0]}")

    head_target = parsed["head"].astype(np.float32)
    safe_head = _clip_delta(head_target, head_cur, limits.head_delta)

    waist_target = parsed["waist"].astype(np.float32)
    safe_waist = np.asarray(
        [
            _clip_delta(waist_target[:1], waist_cur[:1], limits.waist_pitch_delta)[0],
            _clip_delta(waist_target[1:], waist_cur[1:], limits.waist_lift_delta)[0],
        ],
        dtype=np.float32,
    )

    safe_gripper_policy = np.clip(gripper_command_policy, 0.0, 1.0).astype(np.float32)
    safe_gripper_sdk = _policy_gripper_to_sdk(safe_gripper_policy)

    if enable_gripper:
        robot.move_gripper(safe_gripper_sdk.tolist())
    robot.move_head(safe_head.tolist())
    robot.move_waist(safe_waist.tolist())

    return {
        "gripper": safe_gripper_sdk,
        "gripper_policy": safe_gripper_policy,
        "head": safe_head,
        "waist": safe_waist,
        "wheel": parsed["wheel"].astype(np.float32),
    }


def _maybe_warn_sdk_order(sdk_arm_order: ArmOrder) -> None:
    if sdk_arm_order != "left_right":
        logging.warning(
            "Current validated baseline is arm_joint_states()[0:7]=left, [7:14]=right. "
            "Using sdk_arm_order=%s may reintroduce left/right confusion.",
            sdk_arm_order,
        )


def run_client(
    host: str,
    port: int,
    prompt: str,
    apply_actions: bool,
    enable_gripper: bool,
    limits: Limits,
    sdk_arm_order: ArmOrder,
    obs_flip_config: ObsFlipConfig,
    gripper_config: GripperConfig,
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

    # Warm up camera stream (first frame can be None)
    _wait_for_image(camera, "head")
    _wait_for_image(camera, "hand_left")
    _wait_for_image(camera, "hand_right")

    session_id = str(uuid.uuid4())
    step = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(".", "robot_json")
    os.makedirs(log_dir, exist_ok=True)
    action_log_path = os.path.join(log_dir, f"agibot_actions_{timestamp}.json")
    action_records: list[dict[str, object]] = []
    logging.info("Starting live client, session_id=%s", session_id)
    logging.info(
        "Client config: sdk_arm_order=%s apply_actions=%s enable_gripper=%s obs_flip_left=%s obs_flip_right=%s gripper_threshold=%.2f gripper_close_when_high=%s action_log_path=%s",
        sdk_arm_order,
        apply_actions,
        enable_gripper,
        list(obs_flip_config.left_joint_indices),
        list(obs_flip_config.right_joint_indices),
        gripper_config.close_threshold,
        gripper_config.close_when_high,
        action_log_path,
    )
    _maybe_warn_sdk_order(sdk_arm_order)

    try:
        with _KeyboardMonitor() as keyboard_monitor:
            logging.info("Continuous inference started. Press 'q' to stop the client.")
            while True:
                if keyboard_monitor.consume_stop_request():
                    logging.info("Stop hotkey received before inference. Exiting.")
                    break

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

                arm_pos_sdk = np.asarray(arm_pos, dtype=np.float32)
                arm_pos_policy = _sdk_to_policy_arm(arm_pos_sdk, sdk_arm_order)
                obs_arm_policy = _apply_obs_joint_sign_flips(arm_pos_policy, obs_flip_config)
                gripper_pos_sdk = np.asarray(gripper_pos, dtype=np.float32)
                obs_gripper_policy = _sdk_gripper_to_policy_obs(gripper_pos_sdk)

                obs = _build_obs(
                    head_img=head_img,
                    left_img=left_img,
                    right_img=right_img,
                    arm_pos=arm_pos_sdk,
                    head_pos=head_pos,
                    waist_pos=waist_pos,
                    gripper_pos=gripper_pos,
                    prompt=prompt,
                    session_id=session_id,
                    sdk_arm_order=sdk_arm_order,
                    obs_flip_config=obs_flip_config,
                )

                actions = client.infer(obs)
                actions = np.asarray(actions, dtype=np.float32)
                parsed = _parse_action_first(actions)
                gripper_plan = _select_gripper_command(actions, gripper_config)
                dt = time.time() - t0
                logging.info(
                    "Step %s | action shape=%s horizon=%s range=[%.4f, %.4f] dt=%.3fs",
                    step,
                    actions.shape,
                    int(parsed["horizon"][0]),
                    float(actions.min()),
                    float(actions.max()),
                    dt,
                )

                logging.info(
                    "Pred left[:3]=%s right[:3]=%s grip_first=%s grip_last=%s grip_exec=%s head=%s waist=%s wheel(pred only)=%s",
                    np.round(parsed["left_arm"][:3], 4).tolist(),
                    np.round(parsed["right_arm"][:3], 4).tolist(),
                    np.round(gripper_plan["first_policy"], 4).tolist(),
                    np.round(gripper_plan["last_policy"], 4).tolist(),
                    np.round(gripper_plan["command_policy"], 4).tolist(),
                    np.round(parsed["head"], 4).tolist(),
                    np.round(parsed["waist"], 4).tolist(),
                    np.round(parsed["wheel"], 4).tolist(),
                )
                logging.info(
                    "Arm state sdk_first7[:3]=%s sdk_last7[:3]=%s | raw_policy_left[:3]=%s raw_policy_right[:3]=%s | obs_policy_left[:3]=%s obs_policy_right[:3]=%s | gripper_sdk=%s gripper_obs=%s",
                    np.round(arm_pos_sdk[:7], 4).tolist(),
                    np.round(arm_pos_sdk[7:], 4).tolist(),
                    np.round(arm_pos_policy[:7], 4).tolist(),
                    np.round(arm_pos_policy[7:], 4).tolist(),
                    np.round(obs_arm_policy[:7], 4).tolist(),
                    np.round(obs_arm_policy[7:], 4).tolist(),
                    np.round(gripper_pos_sdk, 4).tolist(),
                    np.round(obs_gripper_policy, 4).tolist(),
                )

                if keyboard_monitor.consume_stop_request():
                    logging.info("Stop hotkey received after inference. Exiting before action execution.")
                    break

                action_record: dict[str, object] = {
                    "step": step,
                    "session_id": session_id,
                    "prompt": prompt,
                    "dt_seconds": round(dt, 6),
                    "shape": list(actions.shape),
                    "sdk_observation_arm_order": sdk_arm_order,
                    "sdk_execution_arm_order": sdk_arm_order,
                    "apply_actions": bool(apply_actions),
                    "obs_flip_left_joint_indices": list(obs_flip_config.left_joint_indices),
                    "obs_flip_right_joint_indices": list(obs_flip_config.right_joint_indices),
                    "raw_sdk_arm_joint_states": arm_pos_sdk.tolist(),
                    "raw_gripper_sdk_states": gripper_pos_sdk.tolist(),
                    "policy_left_arm_joint_position": arm_pos_policy[:7].tolist(),
                    "policy_right_arm_joint_position": arm_pos_policy[7:].tolist(),
                    "obs_left_arm_joint_position": obs_arm_policy[:7].tolist(),
                    "obs_right_arm_joint_position": obs_arm_policy[7:].tolist(),
                    "obs_gripper_policy_position": obs_gripper_policy.tolist(),
                    "predicted_first_left_arm_joint_position": parsed["left_arm"].tolist(),
                    "predicted_first_right_arm_joint_position": parsed["right_arm"].tolist(),
                    "predicted_first_gripper_position": gripper_plan["first_policy"].tolist(),
                    "predicted_last_gripper_position": gripper_plan["last_policy"].tolist(),
                    "executed_gripper_command_policy": gripper_plan["command_policy"].tolist(),
                    "actions": actions.tolist(),
                }

                if apply_actions:
                    try:
                        trajectory = _execute_arm_trajectory(
                            robot=robot,
                            actions=actions,
                            limits=limits,
                            sdk_arm_order=sdk_arm_order,
                            should_stop=keyboard_monitor.consume_stop_request,
                        )
                    except _StopRequested:
                        logging.info("Stop hotkey received during move_arm trajectory execution. Exiting immediately.")
                        break
                    aux = _apply_aux_actions(
                        robot=robot,
                        parsed=parsed,
                        gripper_command_policy=gripper_plan["command_policy"],
                        limits=limits,
                        enable_gripper=enable_gripper,
                    )
                    logging.info(
                        "Executed move_arm dual-arm trajectory | target_rows=%s move_arm_points=%s cur_left[:3]=%s cur_right[:3]=%s first_left[:3]=%s first_right[:3]=%s last_left[:3]=%s last_right[:3]=%s grip=%s head=%s waist=%s wheel(pred only)=%s",
                        int(trajectory["target_rows"][0]),
                        int(trajectory["move_arm_points"][0]),
                        np.round(trajectory["arm_cur_policy"][:3], 4).tolist(),
                        np.round(trajectory["arm_cur_policy"][7:10], 4).tolist(),
                        np.round(trajectory["first_arm_policy"][:3], 4).tolist(),
                        np.round(trajectory["first_arm_policy"][7:10], 4).tolist(),
                        np.round(trajectory["last_arm_policy"][:3], 4).tolist(),
                        np.round(trajectory["last_arm_policy"][7:10], 4).tolist(),
                        np.round(aux["gripper"], 4).tolist(),
                        np.round(aux["head"], 4).tolist(),
                        np.round(aux["waist"], 4).tolist(),
                        np.round(aux["wheel"], 4).tolist(),
                    )
                    logging.info(
                        "Post-trajectory arm state | left[:3]=%s right[:3]=%s delta_from_pre_left[:3]=%s delta_from_pre_right[:3]=%s arm_error_to_target=%.4f left_error=%.4f right_error=%.4f reached=%s",
                        np.round(trajectory["arm_after_policy"][:3], 4).tolist(),
                        np.round(trajectory["arm_after_policy"][7:10], 4).tolist(),
                        np.round((trajectory["arm_after_policy"] - trajectory["arm_cur_policy"])[:3], 4).tolist(),
                        np.round((trajectory["arm_after_policy"] - trajectory["arm_cur_policy"])[7:10], 4).tolist(),
                        float(trajectory["arm_error"]),
                        float(trajectory["left_error"]),
                        float(trajectory["right_error"]),
                        bool(trajectory["reached"]),
                    )
                    action_record.update(
                        {
                            "execution_mode": "move_arm_ruckig_dual_arm",
                            "trajectory_rows": int(trajectory["target_rows"][0]),
                            "move_arm_points": int(trajectory["move_arm_points"][0]),
                            "trajectory_first_left_arm_joint_position": trajectory["first_arm_policy"][:7].tolist(),
                            "trajectory_first_right_arm_joint_position": trajectory["first_arm_policy"][7:14].tolist(),
                            "trajectory_last_left_arm_joint_position": trajectory["last_arm_policy"][:7].tolist(),
                            "trajectory_last_right_arm_joint_position": trajectory["last_arm_policy"][7:14].tolist(),
                            "post_left_arm_joint_position": trajectory["arm_after_policy"][:7].tolist(),
                            "post_right_arm_joint_position": trajectory["arm_after_policy"][7:14].tolist(),
                            "post_arm_error_to_target": round(float(trajectory["arm_error"]), 6),
                            "post_left_arm_error_to_target": round(float(trajectory["left_error"]), 6),
                            "post_right_arm_error_to_target": round(float(trajectory["right_error"]), 6),
                            "post_arm_reached_target": bool(trajectory["reached"]),
                        }
                    )
                else:
                    action_record["execution_mode"] = "inference_only"
                    logging.info("apply-actions 未开启，本次仅推理不下发")

                action_records.append(action_record)
                with open(action_log_path, "w", encoding="utf-8") as action_log:
                    json.dump(action_records, action_log, ensure_ascii=False, indent=2)
                logging.info("Saved %s action record(s) to %s", len(action_records), action_log_path)

                step += 1
    finally:
        try:
            if hasattr(client, "_ws") and client._ws is not None:
                logging.info("Closing websocket connection...")
                client._ws.close()
                logging.info("Websocket connection closed.")
        except Exception as exc:
            logging.warning("Failed to close websocket cleanly: %s", exc)

        try:
            logging.info("Closing camera...")
            camera.close()
            logging.info("Camera closed.")
        except Exception as exc:
            logging.warning("Failed while closing camera: %s", exc)

        try:
            logging.info("Shutting down robot DDS...")
            robot.shutdown()
            logging.info("Robot DDS shutdown complete.")
        except Exception as exc:
            logging.warning("Failed during robot shutdown: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live AgiBot client using GDK Python APIs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument("--prompt", default="Pick up the object")
    parser.add_argument("--apply-actions", action="store_true", help="Enable sending actions to robot")
    parser.add_argument("--disable-gripper", action="store_true", help="Do not send gripper commands during execution")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gripper-close-when-low",
        action="store_true",
        help="Interpret low gripper logits/targets as close instead of high values as close.",
    )
    parser.add_argument("--arm-delta-limit", type=float, default=0.08)
    parser.add_argument("--head-delta-limit", type=float, default=0.15)
    parser.add_argument("--waist-pitch-delta-limit", type=float, default=0.10)
    parser.add_argument("--waist-lift-delta-limit", type=float, default=2.0)
    parser.add_argument(
        "--sdk-arm-order",
        choices=["left_right", "right_left"],
        default=DEFAULT_SDK_ARM_ORDER,
        help="SDK arm_joint_states()/move_arm() native order on this robot.",
    )
    parser.add_argument(
        "--obs-flip-left-joints",
        default="",
        help="Comma-separated 1-based left-arm joint indices to sign-flip only in observation before inference, e.g. '2,4'.",
    )
    parser.add_argument(
        "--obs-flip-right-joints",
        default="",
        help="Comma-separated 1-based right-arm joint indices to sign-flip only in observation before inference, e.g. '2'.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    limits = Limits(
        arm_delta=args.arm_delta_limit,
        head_delta=args.head_delta_limit,
        waist_pitch_delta=args.waist_pitch_delta_limit,
        waist_lift_delta=args.waist_lift_delta_limit,
    )

    run_client(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        apply_actions=args.apply_actions,
        enable_gripper=not args.disable_gripper,
        limits=limits,
        sdk_arm_order=args.sdk_arm_order,
        obs_flip_config=ObsFlipConfig(
            left_joint_indices=_parse_joint_index_list(args.obs_flip_left_joints),
            right_joint_indices=_parse_joint_index_list(args.obs_flip_right_joints),
        ),
        gripper_config=GripperConfig(
            close_threshold=args.gripper_close_threshold,
            close_when_high=not args.gripper_close_when_low,
        ),
    )


if __name__ == "__main__":
    main()
