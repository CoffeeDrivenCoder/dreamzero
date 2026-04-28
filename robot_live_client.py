#!/usr/bin/env python3
"""Live client for AgiBot policy server using GDK Python APIs (CosineCamera + RobotDds).

This client reads real-time images and robot states, maps them to the AgiBot
observation schema, and sends them to the websocket policy server.
"""

from __future__ import annotations

import argparse
from collections import deque
import datetime
import json
import logging
import os
import select
import sys
import termios
import threading
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
ArmExecutionMode = Literal["ruckig", "direct-48"]
ActionSmoothingMode = Literal["none", "savgol"]
DEFAULT_SDK_ARM_ORDER: ArmOrder = "left_right"
ARM_JOINT_COUNT = 14
RIGHT_ARM_START = 7
DEFAULT_MOVE_ARM_INTERVAL_S = 0.01
DEFAULT_MOVE_ARM_TIMEOUT_S = 5.0
POSITION_TOLERANCE_RAD = 5e-3
RUCKIG_MAX_VELOCITY = 2.0
RUCKIG_MAX_ACCELERATION = 1.0
RUCKIG_MAX_JERK = 5.0
GRIPPER_OBSERVATION_MAX = 120.0
DEGREE_LIKE_THRESHOLD = float(2.0 * np.pi)
WAIST_LIFT_M_TO_CM = 100.0


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


@dataclass(frozen=True)
class ObservationSample:
    head_img: np.ndarray
    left_img: np.ndarray
    right_img: np.ndarray
    head_ts: int
    left_ts: int
    right_ts: int
    sampled_at: float


@dataclass(frozen=True)
class PreparedObservation:
    step: int
    mode: str
    obs: dict[str, object]
    sequence: tuple[ObservationSample, ...]
    latest_sampled_at: float
    observation_sample_interval: float
    observation_history_span: float
    observation_consumed_interval: float
    arm_pos_sdk: np.ndarray
    arm_pos_policy: np.ndarray
    obs_arm_policy: np.ndarray
    head_pos: np.ndarray
    waist_pos: np.ndarray
    gripper_pos_sdk: np.ndarray
    obs_gripper_policy: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    prepared: PreparedObservation


@dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    actions: np.ndarray
    duration_s: float
    completed_at: float


@dataclass(frozen=True)
class SpeculativeConfig:
    enabled: bool = False
    prefetch_fraction: float = 0.7
    max_boundary_wait_s: float = 0.15
    max_state_age_s: float = 1.5
    arm_state_tolerance: float = 0.12
    waist_state_tolerance: float = 0.03


class _ObservationSampler:
    """Sample camera observations on a fixed wall-clock schedule in the background."""

    def __init__(
        self,
        camera: Camera,
        target_w: int | None,
        target_h: int | None,
        interval_s: float,
    ) -> None:
        self._camera = camera
        self._target_w = target_w
        self._target_h = target_h
        self._interval_s = max(interval_s, 1e-3)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_sample: ObservationSample | None = None
        self._samples: deque[ObservationSample] = deque()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="observation-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def wait_until_ready(self, timeout_s: float = 5.0) -> ObservationSample:
        if not self._ready_event.wait(timeout=timeout_s):
            raise RuntimeError(f"Observation sampler did not produce a frame within {timeout_s}s.")
        sample = self.get_latest_sample()
        if sample is None:
            raise RuntimeError("Observation sampler reported ready without a sample.")
        return sample

    def wait_until_history_ready(self, history_size: int, timeout_s: float = 5.0) -> list[ObservationSample]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            sequence = self.get_latest_sequence(history_size)
            if len(sequence) >= history_size:
                return sequence
            time.sleep(0.01)
        raise RuntimeError(
            f"Observation sampler did not accumulate {history_size} samples within {timeout_s}s."
        )

    def get_latest_sample(self) -> ObservationSample | None:
        with self._lock:
            return self._latest_sample

    def get_latest_sequence(self, history_size: int) -> list[ObservationSample]:
        if history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}")
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return []
        if len(samples) >= history_size:
            return samples[-history_size:]
        pad = [samples[0]] * (history_size - len(samples))
        return pad + samples

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if self._target_w is None or self._target_h is None:
            return image
        return cv2.resize(image, (self._target_w, self._target_h), interpolation=cv2.INTER_AREA)

    def _run(self) -> None:
        next_sample_time = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            sleep_s = next_sample_time - now
            if sleep_s > 0:
                if self._stop_event.wait(timeout=sleep_s):
                    break

            sample_wall_time = time.time()
            head_img, head_ts = self._camera.get_latest_image("head")
            left_img, left_ts = self._camera.get_latest_image("hand_left")
            right_img, right_ts = self._camera.get_latest_image("hand_right")

            if head_img is not None and left_img is not None and right_img is not None:
                sample = ObservationSample(
                    head_img=self._resize_if_needed(head_img),
                    left_img=self._resize_if_needed(left_img),
                    right_img=self._resize_if_needed(right_img),
                    head_ts=int(head_ts),
                    left_ts=int(left_ts),
                    right_ts=int(right_ts),
                    sampled_at=sample_wall_time,
                )
                with self._lock:
                    self._latest_sample = sample
                    self._samples.append(sample)
                    max_samples = 32
                    while len(self._samples) > max_samples:
                        self._samples.popleft()
                self._ready_event.set()
            else:
                logging.warning("Observation sampler got None frame(s); keeping previous sampled observation.")

            next_sample_time += self._interval_s
            now = time.monotonic()
            if next_sample_time < now - self._interval_s:
                next_sample_time = now + self._interval_s


class _InferenceWorker:
    """Single-owner background worker for websocket inference calls."""

    def __init__(self, client: WebsocketClientPolicy) -> None:
        self._client = client
        self._condition = threading.Condition()
        self._pending: InferenceRequest | None = None
        self._active_request_id: str | None = None
        self._results: dict[str, InferenceResult] = {}
        self._errors: dict[str, BaseException] = {}
        self._ignored_request_ids: set[str] = set()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="policy-inference-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def submit(self, prepared: PreparedObservation) -> InferenceRequest | None:
        request = InferenceRequest(request_id=str(uuid.uuid4()), prepared=prepared)
        with self._condition:
            if self._pending is not None or self._active_request_id is not None:
                return None
            self._pending = request
            self._condition.notify_all()
        return request

    def wait_for_result(self, request_id: str, timeout_s: float | None = None) -> InferenceResult | None:
        deadline = None if timeout_s is None else time.time() + max(timeout_s, 0.0)
        with self._condition:
            while True:
                if request_id in self._results:
                    return self._results.pop(request_id)
                if request_id in self._errors:
                    raise RuntimeError(
                        f"Inference worker failed for request {request_id}"
                    ) from self._errors.pop(request_id)
                if self._stop:
                    return None
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)

    def wait_until_idle(self, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.time() + max(timeout_s, 0.0)
        with self._condition:
            while self._pending is not None or self._active_request_id is not None:
                if self._stop:
                    return False
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)
            return True

    def forget_request(self, request_id: str) -> None:
        with self._condition:
            self._ignored_request_ids.add(request_id)
            self._results.pop(request_id, None)
            self._errors.pop(request_id, None)
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stop and self._pending is None:
                    self._condition.wait()
                if self._stop:
                    return
                request = self._pending
                self._pending = None
                assert request is not None
                self._active_request_id = request.request_id

            try:
                started_at = time.time()
                actions = self._client.infer(request.prepared.obs)
                actions = np.asarray(actions, dtype=np.float32)
                result = InferenceResult(
                    request=request,
                    actions=actions,
                    duration_s=time.time() - started_at,
                    completed_at=time.time(),
                )
                with self._condition:
                    if request.request_id in self._ignored_request_ids:
                        self._ignored_request_ids.remove(request.request_id)
                    else:
                        self._results[request.request_id] = result
                    self._active_request_id = None
                    self._condition.notify_all()
            except BaseException as exc:
                with self._condition:
                    if request.request_id in self._ignored_request_ids:
                        self._ignored_request_ids.remove(request.request_id)
                    else:
                        self._errors[request.request_id] = exc
                    self._active_request_id = None
                    self._condition.notify_all()


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


def _angles_to_policy_radians(values: list | np.ndarray) -> np.ndarray:
    """Normalize GDK joint angles to the training-time radian scale."""
    arr = np.asarray(values, dtype=np.float32)
    if np.any(np.abs(arr) > DEGREE_LIKE_THRESHOLD):
        arr = np.deg2rad(arr).astype(np.float32)
    return arr


def _policy_waist_to_sdk(values: list | np.ndarray) -> np.ndarray:
    """Convert policy waist [pitch(rad), lift(m)] to GDK move_waist [pitch(rad), lift(cm)]."""
    waist = np.asarray(values, dtype=np.float32).copy()
    if waist.shape[0] != 2:
        raise ValueError(f"waist command length must be 2, got {waist.shape[0]}")
    waist[1] *= WAIST_LIFT_M_TO_CM
    return waist


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
    """Map GDK gripper states to the training-time observation scale [0, 120].

    GDK gripper_states() already reports the physical gripper scale used by
    training logs: near 0 when open and near 120 when closed. Do not treat
    small open-state jitter such as 0.217 as a normalized [0, 1] value.
    """
    pair = _sdk_gripper_to_policy(values)
    return np.clip(pair, 0.0, GRIPPER_OBSERVATION_MAX).astype(np.float32)


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
    head_pos = _angles_to_policy_radians(head_pos)
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
    linear_velocity: float
    angular_velocity: float
    arm_trajectory_interval: float
    arm_close_timeout: float
    direct_control_hz: float


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


@dataclass(frozen=True)
class ActionSmoothingConfig:
    mode: ActionSmoothingMode = "none"
    window: int = 7
    polyorder: int = 2
    upsample: int = 2


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


def _clip_wheel_command(target: np.ndarray, limits: Limits) -> np.ndarray:
    wheel = np.asarray(target, dtype=np.float32)
    return np.asarray(
        [
            float(np.clip(wheel[0], -limits.linear_velocity, limits.linear_velocity)),
            float(np.clip(wheel[1], -limits.angular_velocity, limits.angular_velocity)),
        ],
        dtype=np.float32,
    )


def _is_finite_vector(values: np.ndarray | list[float]) -> bool:
    arr = np.asarray(values, dtype=np.float32)
    return bool(np.all(np.isfinite(arr)))


def _sleep_with_stop(duration_s: float, should_stop: Callable[[], bool]) -> None:
    deadline = time.time() + max(duration_s, 0.0)
    while time.time() < deadline:
        if should_stop():
            raise _StopRequested
        time.sleep(min(0.02, max(deadline - time.time(), 0.0)))


def _capture_prepared_observation(
    *,
    step: int,
    mode: str,
    sampler: _ObservationSampler,
    robot: Robot,
    observation_history: int,
    previous_latest_sampled_at: float | None,
    prompt: str,
    session_id: str,
    sdk_arm_order: ArmOrder,
    obs_flip_config: ObsFlipConfig,
) -> PreparedObservation:
    sequence = sampler.get_latest_sequence(observation_history)
    if len(sequence) != observation_history:
        raise RuntimeError(
            f"Observation sampler history is not ready yet: expected {observation_history}, got {len(sequence)}"
        )

    latest_sample = sequence[-1]
    observation_sample_interval = (
        sequence[-1].sampled_at - sequence[-2].sampled_at if len(sequence) >= 2 else 0.0
    )
    observation_history_span = (
        sequence[-1].sampled_at - sequence[0].sampled_at if len(sequence) >= 2 else 0.0
    )
    observation_consumed_interval = (
        latest_sample.sampled_at - previous_latest_sampled_at if previous_latest_sampled_at is not None else 0.0
    )

    head_imgs = np.stack([sample.head_img for sample in sequence], axis=0)
    left_imgs = np.stack([sample.left_img for sample in sequence], axis=0)
    right_imgs = np.stack([sample.right_img for sample in sequence], axis=0)

    arm_pos, _ = robot.arm_joint_states()
    head_pos, _ = robot.head_joint_states()
    waist_pos, _ = robot.waist_joint_states()
    gripper_pos, _ = robot.gripper_states()

    arm_pos_sdk = np.asarray(arm_pos, dtype=np.float32)
    arm_pos_policy = _sdk_to_policy_arm(arm_pos_sdk, sdk_arm_order)
    obs_arm_policy = _apply_obs_joint_sign_flips(arm_pos_policy, obs_flip_config)
    gripper_pos_sdk = np.asarray(gripper_pos, dtype=np.float32)
    obs_gripper_policy = _sdk_gripper_to_policy_obs(gripper_pos_sdk)
    head_pos_arr = np.asarray(head_pos, dtype=np.float32)
    waist_pos_arr = np.asarray(waist_pos, dtype=np.float32)

    obs = _build_obs(
        head_img=head_imgs,
        left_img=left_imgs,
        right_img=right_imgs,
        arm_pos=arm_pos_sdk,
        head_pos=head_pos_arr,
        waist_pos=waist_pos_arr,
        gripper_pos=gripper_pos_sdk,
        prompt=prompt,
        session_id=session_id,
        sdk_arm_order=sdk_arm_order,
        obs_flip_config=obs_flip_config,
    )

    return PreparedObservation(
        step=step,
        mode=mode,
        obs=obs,
        sequence=tuple(sequence),
        latest_sampled_at=latest_sample.sampled_at,
        observation_sample_interval=observation_sample_interval,
        observation_history_span=observation_history_span,
        observation_consumed_interval=observation_consumed_interval,
        arm_pos_sdk=arm_pos_sdk,
        arm_pos_policy=arm_pos_policy,
        obs_arm_policy=obs_arm_policy,
        head_pos=head_pos_arr,
        waist_pos=waist_pos_arr,
        gripper_pos_sdk=gripper_pos_sdk,
        obs_gripper_policy=obs_gripper_policy,
        captured_at=time.time(),
    )


def _read_policy_robot_state(robot: Robot, sdk_arm_order: ArmOrder) -> dict[str, np.ndarray]:
    arm_pos, _ = robot.arm_joint_states()
    waist_pos, _ = robot.waist_joint_states()
    head_pos, _ = robot.head_joint_states()
    gripper_pos, _ = robot.gripper_states()
    return {
        "arm_policy": _sdk_to_policy_arm(np.asarray(arm_pos, dtype=np.float32), sdk_arm_order),
        "waist": np.asarray(waist_pos, dtype=np.float32),
        "head": np.asarray(head_pos, dtype=np.float32),
        "gripper_sdk": np.asarray(gripper_pos, dtype=np.float32),
    }


def _evaluate_speculative_result(
    *,
    result: InferenceResult,
    robot: Robot,
    sdk_arm_order: ArmOrder,
    config: SpeculativeConfig,
) -> dict[str, object]:
    boundary_state = _read_policy_robot_state(robot, sdk_arm_order)
    prepared = result.request.prepared
    arm_delta_max = float(np.max(np.abs(boundary_state["arm_policy"] - prepared.arm_pos_policy)))
    waist_delta_max = float(np.max(np.abs(boundary_state["waist"] - prepared.waist_pos)))
    head_delta_max = float(np.max(np.abs(boundary_state["head"] - prepared.head_pos)))
    gripper_delta_max = float(np.max(np.abs(boundary_state["gripper_sdk"] - prepared.gripper_pos_sdk)))
    state_age_s = float(time.time() - prepared.captured_at)

    accepted = True
    reasons: list[str] = []
    if arm_delta_max > config.arm_state_tolerance:
        accepted = False
        reasons.append(f"arm_delta_max={arm_delta_max:.4f}>{config.arm_state_tolerance:.4f}")
    if waist_delta_max > config.waist_state_tolerance:
        accepted = False
        reasons.append(f"waist_delta_max={waist_delta_max:.4f}>{config.waist_state_tolerance:.4f}")
    if state_age_s > config.max_state_age_s:
        accepted = False
        reasons.append(f"state_age_s={state_age_s:.3f}>{config.max_state_age_s:.3f}")

    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "; ".join(reasons),
        "arm_delta_max": arm_delta_max,
        "waist_delta_max": waist_delta_max,
        "head_delta_max": head_delta_max,
        "gripper_delta_max": gripper_delta_max,
        "state_age_s": state_age_s,
        "boundary_arm_policy": boundary_state["arm_policy"],
        "boundary_waist": boundary_state["waist"],
        "boundary_head": boundary_state["head"],
        "boundary_gripper_sdk": boundary_state["gripper_sdk"],
    }


def _parse_action_row(row: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape[0] < 22:
        raise ValueError(f"Expected at least 22 dims, got {row.shape[0]}")
    return {
        "left_arm": row[0:7],
        "right_arm": row[7:14],
        "gripper": row[14:16],
        "head": row[16:18],
        "waist": row[18:20],
        "wheel": row[20:22],
        "horizon": np.asarray([horizon], dtype=np.int32),
    }


def _parse_action_first(actions: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected action shape (T, D), got {arr.shape}")
    if arr.shape[1] < 22:
        raise ValueError(f"Expected at least 22 dims, got {arr.shape}")
    return _parse_action_row(arr[0], arr.shape[0])


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


def _gripper_policy_to_command(policy_values: np.ndarray, gripper_config: GripperConfig) -> np.ndarray:
    pair = np.asarray(policy_values, dtype=np.float32).reshape(-1)
    if pair.shape[0] != 2:
        raise ValueError(f"Expected gripper pair with 2 dims, got {pair.shape[0]}")
    if gripper_config.close_when_high:
        close_mask = pair >= gripper_config.close_threshold
    else:
        close_mask = pair <= gripper_config.close_threshold
    return np.where(
        close_mask,
        gripper_config.closed_value,
        gripper_config.open_value,
    ).astype(np.float32)


def _savgol_coefficients(window: int, polyorder: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError(f"savgol window must be odd, got {window}")
    if window <= polyorder:
        raise ValueError(f"savgol window must be greater than polyorder, got window={window}, polyorder={polyorder}")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, polyorder + 1, increasing=True)
    return np.linalg.pinv(design)[0].astype(np.float64)


def _savgol_filter_1d(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return arr.astype(np.float32)
    window = min(window, arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1)
    if window <= polyorder:
        return arr.astype(np.float32)
    coeffs = _savgol_coefficients(window, polyorder)
    half = window // 2
    padded = np.pad(arr, (half, half), mode="edge")
    out = np.empty_like(arr)
    for index in range(arr.shape[0]):
        out[index] = float(np.dot(coeffs, padded[index:index + window]))
    return out.astype(np.float32)


def _smooth_actions_savgol(actions: np.ndarray, config: ActionSmoothingConfig) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < ARM_JOINT_COUNT:
        return arr.copy()

    upsample = max(int(config.upsample), 1)
    window = int(config.window)
    if window % 2 == 0:
        window += 1
    polyorder = max(int(config.polyorder), 0)

    smoothed = arr.copy()
    source_t = np.arange(arr.shape[0], dtype=np.float64)
    if upsample > 1:
        dense_t = np.linspace(0.0, float(arr.shape[0] - 1), arr.shape[0] * upsample, dtype=np.float64)
    else:
        dense_t = source_t

    for dim in range(ARM_JOINT_COUNT):
        dense_values = np.interp(dense_t, source_t, arr[:, dim].astype(np.float64))
        filtered_dense = _savgol_filter_1d(dense_values, window=window, polyorder=polyorder)
        smoothed[:, dim] = np.interp(source_t, dense_t, filtered_dense.astype(np.float64)).astype(np.float32)

    # Preserve chunk endpoints so smoothing does not drift the intended start/end targets.
    smoothed[0, :ARM_JOINT_COUNT] = arr[0, :ARM_JOINT_COUNT]
    smoothed[-1, :ARM_JOINT_COUNT] = arr[-1, :ARM_JOINT_COUNT]
    return smoothed


def _maybe_smooth_actions(actions: np.ndarray, config: ActionSmoothingConfig) -> tuple[np.ndarray, float]:
    started_at = time.time()
    if config.mode == "none":
        return np.asarray(actions, dtype=np.float32), 0.0
    if config.mode == "savgol":
        smoothed = _smooth_actions_savgol(actions, config)
        return smoothed, time.time() - started_at
    raise ValueError(f"Unsupported action smoothing mode: {config.mode}")


def _make_row_execution_record(
    *,
    row_index: int,
    total_rows: int,
    row_started_at: float,
    execution_started_at: float,
    safe_policy_target: np.ndarray,
    row_parts: dict[str, np.ndarray],
    safe_waist: np.ndarray | None,
    safe_wheel: np.ndarray | None,
    row_gripper_policy: np.ndarray,
    row_gripper_sdk: np.ndarray,
    arm_points_sent_this_row: int,
    waist_sent: bool,
    wheel_sent: bool,
    gripper_sent: bool,
) -> dict[str, object]:
    return {
        "row_index": int(row_index),
        "row_total": int(total_rows),
        "row_started_at_seconds": round(row_started_at, 6),
        "row_elapsed_from_execution_start_seconds": round(row_started_at - execution_started_at, 6),
        "arm_points_sent_this_row": int(arm_points_sent_this_row),
        "safe_left_arm_target": np.asarray(safe_policy_target[:7], dtype=np.float32).tolist(),
        "safe_right_arm_target": np.asarray(safe_policy_target[7:14], dtype=np.float32).tolist(),
        "raw_left_arm_target": np.asarray(row_parts["left_arm"], dtype=np.float32).tolist(),
        "raw_right_arm_target": np.asarray(row_parts["right_arm"], dtype=np.float32).tolist(),
        "raw_gripper_policy": np.asarray(row_parts["gripper"], dtype=np.float32).tolist(),
        "gripper_command_policy": np.asarray(row_gripper_policy, dtype=np.float32).tolist(),
        "gripper_command_sdk": np.asarray(row_gripper_sdk, dtype=np.float32).tolist(),
        "gripper_sent": bool(gripper_sent),
        "raw_waist_target": np.asarray(row_parts["waist"], dtype=np.float32).tolist(),
        "safe_waist_target_policy": [] if safe_waist is None else np.asarray(safe_waist, dtype=np.float32).tolist(),
        "safe_waist_target_sdk": [] if safe_waist is None else _policy_waist_to_sdk(safe_waist).tolist(),
        "waist_sent": bool(waist_sent),
        "raw_wheel_command": np.asarray(row_parts["wheel"], dtype=np.float32).tolist(),
        "safe_wheel_command": [] if safe_wheel is None else np.asarray(safe_wheel, dtype=np.float32).tolist(),
        "wheel_sent": bool(wheel_sent),
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
    interval_s: float,
) -> list[list[float]]:
    dof = ARM_JOINT_COUNT
    rk = ruckig.Ruckig(dof, interval_s)
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
    timeout_s: float,
    should_stop: Callable[[], bool],
) -> dict[str, np.ndarray | float | bool]:
    wait_started_at = time.time()
    deadline = time.time() + max(timeout_s, 0.0)
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

    wait_finished_at = time.time()
    return {
        "arm_after_policy": last_policy_arm,
        "arm_error": float(np.max(np.abs(last_policy_arm[:ARM_JOINT_COUNT] - target_policy_arm[:ARM_JOINT_COUNT]))),
        "left_error": float(np.max(np.abs(last_policy_arm[:RIGHT_ARM_START] - target_policy_arm[:RIGHT_ARM_START]))),
        "right_error": float(np.max(np.abs(last_policy_arm[RIGHT_ARM_START:ARM_JOINT_COUNT] - target_policy_arm[RIGHT_ARM_START:ARM_JOINT_COUNT]))),
        "reached": reached,
        "wait_started_at": wait_started_at,
        "wait_finished_at": wait_finished_at,
        "wait_duration_seconds": wait_finished_at - wait_started_at,
    }


def _execute_arm_trajectory(
    robot: Robot,
    actions: np.ndarray,
    limits: Limits,
    sdk_arm_order: ArmOrder,
    gripper_config: GripperConfig,
    enable_gripper: bool,
    enable_wheel: bool,
    should_stop: Callable[[], bool],
    on_row_complete: Callable[[int, int], None] | None = None,
    prefetch_fraction: float = 1.0,
    log_row_execution: bool = False,
) -> dict[str, np.ndarray | int | float | bool]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 22:
        raise RuntimeError(f"Invalid action shape: {arr.shape}")

    arm_cur, _ = robot.arm_joint_states()
    arm_cur_sdk = np.asarray(arm_cur, dtype=np.float32)
    arm_cur_policy = _sdk_to_policy_arm(arm_cur_sdk, sdk_arm_order)

    safe_policy_targets = _build_safe_arm_targets(actions, arm_cur_policy, limits)

    move_arm_points = 0
    waist_rows_sent = 0
    wheel_rows_sent = 0
    wheel_stop_sent = False
    gripper_updates = 0
    first_waist_sent: np.ndarray | None = None
    last_waist_sent: np.ndarray | None = None
    first_wheel_sent: np.ndarray | None = None
    last_wheel_sent: np.ndarray | None = None
    first_gripper_sent_policy: np.ndarray | None = None
    last_gripper_sent_policy: np.ndarray | None = None

    prev_sdk: np.ndarray | None = None
    last_gripper_sent_sdk: np.ndarray | None = None
    total_rows = safe_policy_targets.shape[0]
    trigger_row = min(total_rows, max(1, int(np.ceil(total_rows * min(max(prefetch_fraction, 0.0), 1.0)))))
    prefetch_triggered = False
    execution_started_at = time.time()
    first_command_at: float | None = None
    row_execution_log: list[dict[str, object]] = []

    def _mark_command_sent() -> None:
        nonlocal first_command_at
        if first_command_at is None:
            first_command_at = time.time()

    for row_index, safe_policy_target in enumerate(safe_policy_targets):
        row_started_at = time.time()
        target_sdk = _policy_to_sdk_arm(safe_policy_target, sdk_arm_order)
        arm_points_before_row = move_arm_points
        if prev_sdk is None:
            segment = [target_sdk.astype(np.float32).tolist()]
        else:
            segment = _build_segment_trajectory(prev_sdk, target_sdk, limits.arm_trajectory_interval)
            if segment:
                segment = segment[1:]

        for point in segment:
            if should_stop():
                raise _StopRequested
            robot.move_arm(point)
            _mark_command_sent()
            move_arm_points += 1
            _sleep_with_stop(limits.arm_trajectory_interval, should_stop)

        row_parts = _parse_action_row(arr[row_index], arr.shape[0])
        safe_waist: np.ndarray | None = None
        safe_wheel: np.ndarray | None = None
        waist_sent = False
        wheel_sent = False

        waist_target = row_parts["waist"].astype(np.float32)
        if _is_finite_vector(waist_target):
            waist_cur, _ = robot.waist_joint_states()
            waist_cur = np.asarray(waist_cur, dtype=np.float32)
            if waist_cur.shape[0] == 2:
                safe_waist = np.asarray(
                    [
                        _clip_delta(waist_target[:1], waist_cur[:1], limits.waist_pitch_delta)[0],
                        _clip_delta(waist_target[1:], waist_cur[1:], limits.waist_lift_delta)[0],
                    ],
                    dtype=np.float32,
                )
                robot.move_waist(_policy_waist_to_sdk(safe_waist).tolist())
                _mark_command_sent()
                waist_rows_sent += 1
                waist_sent = True
                if first_waist_sent is None:
                    first_waist_sent = safe_waist
                last_waist_sent = safe_waist

        wheel_target = row_parts["wheel"].astype(np.float32)
        if enable_wheel and _is_finite_vector(wheel_target):
            safe_wheel = _clip_wheel_command(wheel_target, limits)
            robot.move_wheel(float(safe_wheel[0]), float(safe_wheel[1]))
            _mark_command_sent()
            wheel_rows_sent += 1
            wheel_sent = True
            if first_wheel_sent is None:
                first_wheel_sent = safe_wheel
            last_wheel_sent = safe_wheel

        row_gripper_policy = _gripper_policy_to_command(row_parts["gripper"], gripper_config)
        row_gripper_sdk = _policy_gripper_to_sdk(row_gripper_policy)
        gripper_sent = False
        if enable_gripper and (
            last_gripper_sent_sdk is None
            or np.max(np.abs(row_gripper_sdk - last_gripper_sent_sdk)) > 1e-6
        ):
            robot.move_gripper(row_gripper_sdk.tolist())
            _mark_command_sent()
            gripper_updates += 1
            if first_gripper_sent_policy is None:
                first_gripper_sent_policy = row_gripper_policy
            last_gripper_sent_policy = row_gripper_policy
            last_gripper_sent_sdk = row_gripper_sdk
            gripper_sent = True

        if log_row_execution:
            row_execution_log.append(
                _make_row_execution_record(
                    row_index=row_index,
                    total_rows=total_rows,
                    row_started_at=row_started_at,
                    execution_started_at=execution_started_at,
                    safe_policy_target=safe_policy_target,
                    row_parts=row_parts,
                    safe_waist=safe_waist,
                    safe_wheel=safe_wheel,
                    row_gripper_policy=row_gripper_policy,
                    row_gripper_sdk=row_gripper_sdk,
                    arm_points_sent_this_row=move_arm_points - arm_points_before_row,
                    waist_sent=waist_sent,
                    wheel_sent=wheel_sent,
                    gripper_sent=gripper_sent,
                )
            )

        prev_sdk = np.asarray(target_sdk, dtype=np.float32)
        if on_row_complete is not None and not prefetch_triggered and (row_index + 1) >= trigger_row:
            on_row_complete(row_index + 1, total_rows)
            prefetch_triggered = True

    if enable_wheel and wheel_rows_sent > 0:
        robot.move_wheel(0.0, 0.0)
        wheel_stop_sent = True
        _mark_command_sent()

    stream_finished_at = time.time()
    wait_result = _wait_arm_close(
        robot=robot,
        target_policy_arm=safe_policy_targets[-1],
        sdk_arm_order=sdk_arm_order,
        timeout_s=limits.arm_close_timeout,
        should_stop=should_stop,
    )
    execution_finished_at = time.time()
    waist_after, _ = robot.waist_joint_states()
    return {
        "arm_cur_sdk": arm_cur_sdk,
        "arm_cur_policy": arm_cur_policy,
        "first_arm_policy": safe_policy_targets[0],
        "last_arm_policy": safe_policy_targets[-1],
        "target_rows": np.asarray([safe_policy_targets.shape[0]], dtype=np.int32),
        "move_arm_points": np.asarray([move_arm_points], dtype=np.int32),
        "waist_rows_sent": np.asarray([waist_rows_sent], dtype=np.int32),
        "wheel_rows_sent": np.asarray([wheel_rows_sent], dtype=np.int32),
        "wheel_stop_sent": np.asarray([int(wheel_stop_sent)], dtype=np.int32),
        "gripper_updates": np.asarray([gripper_updates], dtype=np.int32),
        "first_waist_sent": np.zeros(2, dtype=np.float32) if first_waist_sent is None else first_waist_sent,
        "last_waist_sent": np.zeros(2, dtype=np.float32) if last_waist_sent is None else last_waist_sent,
        "first_wheel_sent": np.zeros(2, dtype=np.float32) if first_wheel_sent is None else first_wheel_sent,
        "last_wheel_sent": np.zeros(2, dtype=np.float32) if last_wheel_sent is None else last_wheel_sent,
        "first_gripper_sent_policy": np.zeros(2, dtype=np.float32) if first_gripper_sent_policy is None else first_gripper_sent_policy,
        "last_gripper_sent_policy": np.zeros(2, dtype=np.float32) if last_gripper_sent_policy is None else last_gripper_sent_policy,
        "waist_after": np.asarray(waist_after, dtype=np.float32),
        "arm_after_policy": wait_result["arm_after_policy"],
        "arm_error": wait_result["arm_error"],
        "left_error": wait_result["left_error"],
        "right_error": wait_result["right_error"],
        "reached": wait_result["reached"],
        "execution_started_at": execution_started_at,
        "first_command_at": execution_started_at if first_command_at is None else first_command_at,
        "stream_finished_at": stream_finished_at,
        "wait_started_at": wait_result["wait_started_at"],
        "wait_finished_at": wait_result["wait_finished_at"],
        "execution_finished_at": execution_finished_at,
        "first_command_delay_seconds": (execution_started_at if first_command_at is None else first_command_at) - execution_started_at,
        "stream_duration_seconds": stream_finished_at - execution_started_at,
        "wait_duration_seconds": wait_result["wait_duration_seconds"],
        "execution_duration_seconds": execution_finished_at - execution_started_at,
        "row_execution_log": row_execution_log,
    }


def _execute_arm_direct48(
    robot: Robot,
    actions: np.ndarray,
    limits: Limits,
    sdk_arm_order: ArmOrder,
    gripper_config: GripperConfig,
    enable_gripper: bool,
    enable_wheel: bool,
    should_stop: Callable[[], bool],
    on_row_complete: Callable[[int, int], None] | None = None,
    prefetch_fraction: float = 1.0,
    log_row_execution: bool = False,
) -> dict[str, np.ndarray | int | float | bool]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 22:
        raise RuntimeError(f"Invalid action shape: {arr.shape}")

    arm_cur, _ = robot.arm_joint_states()
    arm_cur_sdk = np.asarray(arm_cur, dtype=np.float32)
    arm_cur_policy = _sdk_to_policy_arm(arm_cur_sdk, sdk_arm_order)
    safe_policy_targets = _build_safe_arm_targets(actions, arm_cur_policy, limits)

    move_arm_points = 0
    waist_rows_sent = 0
    wheel_rows_sent = 0
    wheel_stop_sent = False
    gripper_updates = 0
    first_waist_sent: np.ndarray | None = None
    last_waist_sent: np.ndarray | None = None
    first_wheel_sent: np.ndarray | None = None
    last_wheel_sent: np.ndarray | None = None
    first_gripper_sent_policy: np.ndarray | None = None
    last_gripper_sent_policy: np.ndarray | None = None
    last_gripper_sent_sdk: np.ndarray | None = None

    total_rows = safe_policy_targets.shape[0]
    trigger_row = min(total_rows, max(1, int(np.ceil(total_rows * min(max(prefetch_fraction, 0.0), 1.0)))))
    prefetch_triggered = False
    row_interval_s = 1.0 / max(limits.direct_control_hz, 1e-6)
    execution_started_at = time.time()
    first_command_at: float | None = None
    row_execution_log: list[dict[str, object]] = []

    def _mark_command_sent() -> None:
        nonlocal first_command_at
        if first_command_at is None:
            first_command_at = time.time()

    for row_index, safe_policy_target in enumerate(safe_policy_targets):
        if should_stop():
            raise _StopRequested

        row_started_at = time.time()
        arm_points_before_row = move_arm_points
        target_sdk = _policy_to_sdk_arm(safe_policy_target, sdk_arm_order)
        robot.move_arm(target_sdk.astype(np.float32).tolist())
        _mark_command_sent()
        move_arm_points += 1

        row_parts = _parse_action_row(arr[row_index], arr.shape[0])
        safe_waist: np.ndarray | None = None
        safe_wheel: np.ndarray | None = None
        waist_sent = False
        wheel_sent = False

        waist_target = row_parts["waist"].astype(np.float32)
        if _is_finite_vector(waist_target):
            waist_cur, _ = robot.waist_joint_states()
            waist_cur = np.asarray(waist_cur, dtype=np.float32)
            if waist_cur.shape[0] == 2:
                safe_waist = np.asarray(
                    [
                        _clip_delta(waist_target[:1], waist_cur[:1], limits.waist_pitch_delta)[0],
                        _clip_delta(waist_target[1:], waist_cur[1:], limits.waist_lift_delta)[0],
                    ],
                    dtype=np.float32,
                )
                robot.move_waist(_policy_waist_to_sdk(safe_waist).tolist())
                _mark_command_sent()
                waist_rows_sent += 1
                waist_sent = True
                if first_waist_sent is None:
                    first_waist_sent = safe_waist
                last_waist_sent = safe_waist

        wheel_target = row_parts["wheel"].astype(np.float32)
        if enable_wheel and _is_finite_vector(wheel_target):
            safe_wheel = _clip_wheel_command(wheel_target, limits)
            robot.move_wheel(float(safe_wheel[0]), float(safe_wheel[1]))
            _mark_command_sent()
            wheel_rows_sent += 1
            wheel_sent = True
            if first_wheel_sent is None:
                first_wheel_sent = safe_wheel
            last_wheel_sent = safe_wheel

        row_gripper_policy = _gripper_policy_to_command(row_parts["gripper"], gripper_config)
        row_gripper_sdk = _policy_gripper_to_sdk(row_gripper_policy)
        gripper_sent = False
        if enable_gripper and (
            last_gripper_sent_sdk is None
            or np.max(np.abs(row_gripper_sdk - last_gripper_sent_sdk)) > 1e-6
        ):
            robot.move_gripper(row_gripper_sdk.tolist())
            _mark_command_sent()
            gripper_updates += 1
            if first_gripper_sent_policy is None:
                first_gripper_sent_policy = row_gripper_policy
            last_gripper_sent_policy = row_gripper_policy
            last_gripper_sent_sdk = row_gripper_sdk
            gripper_sent = True

        if log_row_execution:
            row_execution_log.append(
                _make_row_execution_record(
                    row_index=row_index,
                    total_rows=total_rows,
                    row_started_at=row_started_at,
                    execution_started_at=execution_started_at,
                    safe_policy_target=safe_policy_target,
                    row_parts=row_parts,
                    safe_waist=safe_waist,
                    safe_wheel=safe_wheel,
                    row_gripper_policy=row_gripper_policy,
                    row_gripper_sdk=row_gripper_sdk,
                    arm_points_sent_this_row=move_arm_points - arm_points_before_row,
                    waist_sent=waist_sent,
                    wheel_sent=wheel_sent,
                    gripper_sent=gripper_sent,
                )
            )

        if on_row_complete is not None and not prefetch_triggered and (row_index + 1) >= trigger_row:
            on_row_complete(row_index + 1, total_rows)
            prefetch_triggered = True

        next_tick_at = execution_started_at + (row_index + 1) * row_interval_s
        _sleep_with_stop(next_tick_at - time.time(), should_stop)

    if enable_wheel and wheel_rows_sent > 0:
        robot.move_wheel(0.0, 0.0)
        wheel_stop_sent = True
        _mark_command_sent()

    stream_finished_at = time.time()
    wait_result = _wait_arm_close(
        robot=robot,
        target_policy_arm=safe_policy_targets[-1],
        sdk_arm_order=sdk_arm_order,
        timeout_s=limits.arm_close_timeout,
        should_stop=should_stop,
    )
    execution_finished_at = time.time()
    waist_after, _ = robot.waist_joint_states()
    return {
        "arm_cur_sdk": arm_cur_sdk,
        "arm_cur_policy": arm_cur_policy,
        "first_arm_policy": safe_policy_targets[0],
        "last_arm_policy": safe_policy_targets[-1],
        "target_rows": np.asarray([safe_policy_targets.shape[0]], dtype=np.int32),
        "move_arm_points": np.asarray([move_arm_points], dtype=np.int32),
        "waist_rows_sent": np.asarray([waist_rows_sent], dtype=np.int32),
        "wheel_rows_sent": np.asarray([wheel_rows_sent], dtype=np.int32),
        "wheel_stop_sent": np.asarray([int(wheel_stop_sent)], dtype=np.int32),
        "gripper_updates": np.asarray([gripper_updates], dtype=np.int32),
        "first_waist_sent": np.zeros(2, dtype=np.float32) if first_waist_sent is None else first_waist_sent,
        "last_waist_sent": np.zeros(2, dtype=np.float32) if last_waist_sent is None else last_waist_sent,
        "first_wheel_sent": np.zeros(2, dtype=np.float32) if first_wheel_sent is None else first_wheel_sent,
        "last_wheel_sent": np.zeros(2, dtype=np.float32) if last_wheel_sent is None else last_wheel_sent,
        "first_gripper_sent_policy": np.zeros(2, dtype=np.float32) if first_gripper_sent_policy is None else first_gripper_sent_policy,
        "last_gripper_sent_policy": np.zeros(2, dtype=np.float32) if last_gripper_sent_policy is None else last_gripper_sent_policy,
        "waist_after": np.asarray(waist_after, dtype=np.float32),
        "arm_after_policy": wait_result["arm_after_policy"],
        "arm_error": wait_result["arm_error"],
        "left_error": wait_result["left_error"],
        "right_error": wait_result["right_error"],
        "reached": wait_result["reached"],
        "execution_started_at": execution_started_at,
        "first_command_at": execution_started_at if first_command_at is None else first_command_at,
        "stream_finished_at": stream_finished_at,
        "wait_started_at": wait_result["wait_started_at"],
        "wait_finished_at": wait_result["wait_finished_at"],
        "execution_finished_at": execution_finished_at,
        "first_command_delay_seconds": (execution_started_at if first_command_at is None else first_command_at) - execution_started_at,
        "stream_duration_seconds": stream_finished_at - execution_started_at,
        "wait_duration_seconds": wait_result["wait_duration_seconds"],
        "execution_duration_seconds": execution_finished_at - execution_started_at,
        "row_execution_log": row_execution_log,
    }


def _apply_aux_actions(
    robot: Robot,
    parsed: dict[str, np.ndarray],
    limits: Limits,
) -> dict[str, np.ndarray]:
    head_cur, _ = robot.head_joint_states()

    head_cur = _angles_to_policy_radians(head_cur)
    if head_cur.shape[0] != 2:
        raise RuntimeError(f"head_joint_states length must be 2, got {head_cur.shape[0]}")

    head_target = parsed["head"].astype(np.float32)
    safe_head = _clip_delta(head_target, head_cur, limits.head_delta)
    robot.move_head(safe_head.tolist())

    return {
        "head": safe_head,
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
    enable_wheel: bool,
    limits: Limits,
    arm_execution_mode: ArmExecutionMode,
    sdk_arm_order: ArmOrder,
    obs_flip_config: ObsFlipConfig,
    gripper_config: GripperConfig,
    observation_fps: float,
    observation_history: int,
    speculative_config: SpeculativeConfig,
    smoothing_config: ActionSmoothingConfig,
    log_row_execution: bool,
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

    observation_interval_s = 1.0 / max(observation_fps, 1e-6)
    sampler = _ObservationSampler(
        camera=camera,
        target_w=target_w,
        target_h=target_h,
        interval_s=observation_interval_s,
    )
    sampler.start()
    sampler.wait_until_ready()
    initial_sequence = sampler.wait_until_history_ready(observation_history)
    worker = _InferenceWorker(client)
    worker.start()

    session_id = str(uuid.uuid4())
    step = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(".", "robot_json")
    os.makedirs(log_dir, exist_ok=True)
    action_log_path = os.path.join(log_dir, f"agibot_actions_{timestamp}.json")
    action_records: list[dict[str, object]] = []
    logging.info("Starting live client, session_id=%s", session_id)
    logging.info(
        "Client config: sdk_arm_order=%s arm_execution_mode=%s direct_control_hz=%.2f action_smoothing=%s savgol_window=%s savgol_polyorder=%s savgol_upsample=%s apply_actions=%s enable_gripper=%s enable_wheel=%s observation_fps=%.2f observation_interval=%.3fs arm_interval=%.4fs arm_close_timeout=%.2fs obs_flip_left=%s obs_flip_right=%s gripper_threshold=%.2f gripper_close_when_high=%s speculative_enabled=%s log_row_execution=%s action_log_path=%s",
        sdk_arm_order,
        arm_execution_mode,
        limits.direct_control_hz,
        smoothing_config.mode,
        smoothing_config.window,
        smoothing_config.polyorder,
        smoothing_config.upsample,
        apply_actions,
        enable_gripper,
        enable_wheel,
        observation_fps,
        observation_interval_s,
        limits.arm_trajectory_interval,
        limits.arm_close_timeout,
        list(obs_flip_config.left_joint_indices),
        list(obs_flip_config.right_joint_indices),
        gripper_config.close_threshold,
        gripper_config.close_when_high,
        speculative_config.enabled,
        log_row_execution,
        action_log_path,
    )
    _maybe_warn_sdk_order(sdk_arm_order)

    try:
        with _KeyboardMonitor() as keyboard_monitor:
            logging.info("Continuous inference started. Press 'q' to stop the client.")
            last_used_sequence_latest_sampled_at: float | None = initial_sequence[-1].sampled_at
            prefetched_result: InferenceResult | None = None
            while True:
                if keyboard_monitor.consume_stop_request():
                    logging.info("Stop hotkey received before inference. Exiting.")
                    break

                if prefetched_result is not None:
                    infer_result = prefetched_result
                    prepared = infer_result.request.prepared
                    inference_source = "speculative"
                    prefetched_result = None
                else:
                    prepared = _capture_prepared_observation(
                        step=step,
                        mode="sync",
                        sampler=sampler,
                        robot=robot,
                        observation_history=observation_history,
                        previous_latest_sampled_at=last_used_sequence_latest_sampled_at,
                        prompt=prompt,
                        session_id=session_id,
                        sdk_arm_order=sdk_arm_order,
                        obs_flip_config=obs_flip_config,
                    )
                    request = worker.submit(prepared)
                    if request is None:
                        if not worker.wait_until_idle(timeout_s=max(speculative_config.max_boundary_wait_s, 0.5)):
                            raise RuntimeError("Inference worker stayed busy before sync inference submission.")
                        request = worker.submit(prepared)
                    if request is None:
                        raise RuntimeError("Inference worker is unexpectedly busy before sync inference.")
                    infer_result = worker.wait_for_result(request.request_id, timeout_s=None)
                    if infer_result is None:
                        raise RuntimeError("Inference worker stopped before sync inference completed.")
                    inference_source = "sync"

                action_received_at = time.time()
                raw_actions = np.asarray(infer_result.actions, dtype=np.float32)
                actions, smoothing_duration_s = _maybe_smooth_actions(raw_actions, smoothing_config)
                parsed = _parse_action_first(actions)
                gripper_plan = _select_gripper_command(actions, gripper_config)
                last_used_sequence_latest_sampled_at = prepared.latest_sampled_at
                latest_sample = prepared.sequence[-1]
                logging.info(
                    "Step %s | source=%s action shape=%s horizon=%s range=[%.4f, %.4f] infer_dt=%.3fs obs_sample_interval=%.3fs obs_history_span=%.3fs obs_consumed_interval=%.3fs sampled_at=%.3f history_len=%s latest_camera_ts=[%s,%s,%s]",
                    step,
                    inference_source,
                    actions.shape,
                    int(parsed["horizon"][0]),
                    float(actions.min()),
                    float(actions.max()),
                    infer_result.duration_s,
                    prepared.observation_sample_interval,
                    prepared.observation_history_span,
                    prepared.observation_consumed_interval,
                    prepared.latest_sampled_at,
                    observation_history,
                    latest_sample.head_ts,
                    latest_sample.left_ts,
                    latest_sample.right_ts,
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
                    np.round(prepared.arm_pos_sdk[:7], 4).tolist(),
                    np.round(prepared.arm_pos_sdk[7:], 4).tolist(),
                    np.round(prepared.arm_pos_policy[:7], 4).tolist(),
                    np.round(prepared.arm_pos_policy[7:], 4).tolist(),
                    np.round(prepared.obs_arm_policy[:7], 4).tolist(),
                    np.round(prepared.obs_arm_policy[7:], 4).tolist(),
                    np.round(prepared.gripper_pos_sdk, 4).tolist(),
                    np.round(prepared.obs_gripper_policy, 4).tolist(),
                )

                if keyboard_monitor.consume_stop_request():
                    logging.info("Stop hotkey received after inference. Exiting before action execution.")
                    break

                speculative_request: InferenceRequest | None = None
                speculative_evaluation: dict[str, object] | None = None

                action_record: dict[str, object] = {
                    "step": step,
                    "session_id": session_id,
                    "prompt": prompt,
                    "inference_source": inference_source,
                    "dt_seconds": round(infer_result.duration_s, 6),
                    "inference_completed_at_seconds": round(infer_result.completed_at, 6),
                    "action_received_at_seconds": round(action_received_at, 6),
                    "action_receive_delay_after_inference_seconds": round(action_received_at - infer_result.completed_at, 6),
                    "observation_fps_target": float(observation_fps),
                    "observation_history_frames": int(observation_history),
                    "observation_interval_target_seconds": round(observation_interval_s, 6),
                    "observation_sampled_at_seconds": round(prepared.latest_sampled_at, 6),
                    "observation_interval_seconds": round(prepared.observation_sample_interval, 6),
                    "observation_history_span_seconds": round(prepared.observation_history_span, 6),
                    "observation_consumed_interval_seconds": round(prepared.observation_consumed_interval, 6),
                    "observation_head_camera_timestamp": latest_sample.head_ts,
                    "observation_left_camera_timestamp": latest_sample.left_ts,
                    "observation_right_camera_timestamp": latest_sample.right_ts,
                    "observation_history_sampled_at_seconds": [round(sample.sampled_at, 6) for sample in prepared.sequence],
                    "observation_history_head_camera_timestamps": [int(sample.head_ts) for sample in prepared.sequence],
                    "observation_history_left_camera_timestamps": [int(sample.left_ts) for sample in prepared.sequence],
                    "observation_history_right_camera_timestamps": [int(sample.right_ts) for sample in prepared.sequence],
                    "shape": list(actions.shape),
                    "sdk_observation_arm_order": sdk_arm_order,
                    "sdk_execution_arm_order": sdk_arm_order,
                    "arm_execution_mode": arm_execution_mode,
                    "direct_control_hz": float(limits.direct_control_hz),
                    "action_smoothing": smoothing_config.mode,
                    "action_smoothing_duration_seconds": round(smoothing_duration_s, 6),
                    "savgol_window": int(smoothing_config.window),
                    "savgol_polyorder": int(smoothing_config.polyorder),
                    "savgol_upsample": int(smoothing_config.upsample),
                    "apply_actions": bool(apply_actions),
                    "speculative_enabled": bool(speculative_config.enabled),
                    "log_row_execution": bool(log_row_execution),
                    "obs_flip_left_joint_indices": list(obs_flip_config.left_joint_indices),
                    "obs_flip_right_joint_indices": list(obs_flip_config.right_joint_indices),
                    "raw_sdk_arm_joint_states": prepared.arm_pos_sdk.tolist(),
                    "raw_waist_joint_states": prepared.waist_pos.tolist(),
                    "raw_head_joint_states": prepared.head_pos.tolist(),
                    "raw_gripper_sdk_states": prepared.gripper_pos_sdk.tolist(),
                    "policy_left_arm_joint_position": prepared.arm_pos_policy[:7].tolist(),
                    "policy_right_arm_joint_position": prepared.arm_pos_policy[7:].tolist(),
                    "obs_left_arm_joint_position": prepared.obs_arm_policy[:7].tolist(),
                    "obs_right_arm_joint_position": prepared.obs_arm_policy[7:].tolist(),
                    "obs_gripper_policy_position": prepared.obs_gripper_policy.tolist(),
                    "predicted_first_left_arm_joint_position": parsed["left_arm"].tolist(),
                    "predicted_first_right_arm_joint_position": parsed["right_arm"].tolist(),
                    "predicted_first_waist_position": parsed["waist"].tolist(),
                    "predicted_first_wheel_command": parsed["wheel"].tolist(),
                    "predicted_first_gripper_position": gripper_plan["first_policy"].tolist(),
                    "predicted_last_gripper_position": gripper_plan["last_policy"].tolist(),
                    "predicted_last_waist_position": np.asarray(actions[-1][18:20], dtype=np.float32).tolist(),
                    "predicted_last_wheel_command": np.asarray(actions[-1][20:22], dtype=np.float32).tolist(),
                    "executed_gripper_command_policy": [],
                    "speculative_prefetch_submitted": False,
                    "speculative_prefetch_result_ready_at_boundary": False,
                    "speculative_prefetch_result_ready_late": False,
                    "speculative_prefetch_accepted": False,
                    "speculative_prefetch_reject_reason": "",
                    "raw_actions": raw_actions.tolist(),
                    "actions": actions.tolist(),
                }

                if apply_actions:
                    def _maybe_submit_speculative(row_done: int, row_total: int) -> None:
                        nonlocal speculative_request
                        if not speculative_config.enabled or speculative_request is not None:
                            return
                        try:
                            speculative_prepared = _capture_prepared_observation(
                                step=step + 1,
                                mode="speculative",
                                sampler=sampler,
                                robot=robot,
                                observation_history=observation_history,
                                previous_latest_sampled_at=prepared.latest_sampled_at,
                                prompt=prompt,
                                session_id=session_id,
                                sdk_arm_order=sdk_arm_order,
                                obs_flip_config=obs_flip_config,
                            )
                        except Exception as exc:
                            logging.warning("Failed to capture speculative observation at row %s/%s: %s", row_done, row_total, exc)
                            return

                        speculative_request = worker.submit(speculative_prepared)
                        if speculative_request is None:
                            logging.info("Speculative prefetch skipped at row %s/%s because inference worker is busy.", row_done, row_total)
                            return

                        action_record["speculative_prefetch_submitted"] = True
                        action_record["speculative_prefetch_request_id"] = speculative_request.request_id
                        action_record["speculative_prefetch_row_done"] = row_done
                        action_record["speculative_prefetch_row_total"] = row_total
                        action_record["speculative_prefetch_sampled_at_seconds"] = round(
                            speculative_prepared.latest_sampled_at, 6
                        )
                        action_record["speculative_prefetch_capture_time_seconds"] = round(
                            speculative_prepared.captured_at, 6
                        )
                        action_record["speculative_prefetch_observation_interval_seconds"] = round(
                            speculative_prepared.observation_sample_interval, 6
                        )
                        action_record["speculative_prefetch_observation_history_span_seconds"] = round(
                            speculative_prepared.observation_history_span, 6
                        )
                        logging.info(
                            "Submitted speculative prefetch for step %s at row %s/%s sampled_at=%.3f",
                            step + 1,
                            row_done,
                            row_total,
                            speculative_prepared.latest_sampled_at,
                        )

                    execution_call_started_at = time.time()
                    try:
                        execute_fn = _execute_arm_direct48 if arm_execution_mode == "direct-48" else _execute_arm_trajectory
                        trajectory = execute_fn(
                            robot=robot,
                            actions=actions,
                            limits=limits,
                            sdk_arm_order=sdk_arm_order,
                            gripper_config=gripper_config,
                            enable_gripper=enable_gripper,
                            enable_wheel=enable_wheel,
                            should_stop=keyboard_monitor.consume_stop_request,
                            on_row_complete=_maybe_submit_speculative if speculative_config.enabled else None,
                            prefetch_fraction=speculative_config.prefetch_fraction,
                            log_row_execution=log_row_execution,
                        )
                    except _StopRequested:
                        if enable_wheel:
                            robot.move_wheel(0.0, 0.0)
                        logging.info("Stop hotkey received during move_arm trajectory execution. Exiting immediately.")
                        break
                    trajectory_returned_at = time.time()
                    aux_started_at = time.time()
                    aux = _apply_aux_actions(
                        robot=robot,
                        parsed=parsed,
                        limits=limits,
                    )
                    aux_finished_at = time.time()
                    action_to_execution_call_delay_s = execution_call_started_at - action_received_at
                    action_to_first_command_delay_s = float(trajectory["first_command_at"]) - action_received_at
                    logging.info(
                        "Executed move_arm dual-arm trajectory with row-synced gripper | target_rows=%s move_arm_points=%s waist_rows=%s wheel_rows=%s wheel_stop=%s gripper_updates=%s cur_left[:3]=%s cur_right[:3]=%s first_left[:3]=%s first_right[:3]=%s last_left[:3]=%s last_right[:3]=%s grip_first=%s grip_last=%s head=%s waist_first=%s waist_last=%s wheel_first=%s wheel_last=%s",
                        int(trajectory["target_rows"][0]),
                        int(trajectory["move_arm_points"][0]),
                        int(trajectory["waist_rows_sent"][0]),
                        int(trajectory["wheel_rows_sent"][0]),
                        bool(int(trajectory["wheel_stop_sent"][0])),
                        int(trajectory["gripper_updates"][0]),
                        np.round(trajectory["arm_cur_policy"][:3], 4).tolist(),
                        np.round(trajectory["arm_cur_policy"][7:10], 4).tolist(),
                        np.round(trajectory["first_arm_policy"][:3], 4).tolist(),
                        np.round(trajectory["first_arm_policy"][7:10], 4).tolist(),
                        np.round(trajectory["last_arm_policy"][:3], 4).tolist(),
                        np.round(trajectory["last_arm_policy"][7:10], 4).tolist(),
                        np.round(trajectory["first_gripper_sent_policy"], 4).tolist(),
                        np.round(trajectory["last_gripper_sent_policy"], 4).tolist(),
                        np.round(aux["head"], 4).tolist(),
                        np.round(trajectory["first_waist_sent"], 4).tolist(),
                        np.round(trajectory["last_waist_sent"], 4).tolist(),
                        np.round(trajectory["first_wheel_sent"], 4).tolist(),
                        np.round(trajectory["last_wheel_sent"], 4).tolist(),
                    )
                    logging.info(
                        "Timing | infer_dt=%.3fs action_to_exec_call=%.3fs action_to_first_cmd=%.3fs stream=%.3fs wait_close=%.3fs chunk_exec=%.3fs aux=%.3fs total_after_action=%.3fs",
                        infer_result.duration_s,
                        action_to_execution_call_delay_s,
                        action_to_first_command_delay_s,
                        float(trajectory["stream_duration_seconds"]),
                        float(trajectory["wait_duration_seconds"]),
                        float(trajectory["execution_duration_seconds"]),
                        aux_finished_at - aux_started_at,
                        aux_finished_at - action_received_at,
                    )
                    logging.info(
                        "Post-trajectory state | left[:3]=%s right[:3]=%s delta_from_pre_left[:3]=%s delta_from_pre_right[:3]=%s waist=%s arm_error_to_target=%.4f left_error=%.4f right_error=%.4f reached=%s",
                        np.round(trajectory["arm_after_policy"][:3], 4).tolist(),
                        np.round(trajectory["arm_after_policy"][7:10], 4).tolist(),
                        np.round((trajectory["arm_after_policy"] - trajectory["arm_cur_policy"])[:3], 4).tolist(),
                        np.round((trajectory["arm_after_policy"] - trajectory["arm_cur_policy"])[7:10], 4).tolist(),
                        np.round(trajectory["waist_after"], 4).tolist(),
                        float(trajectory["arm_error"]),
                        float(trajectory["left_error"]),
                        float(trajectory["right_error"]),
                        bool(trajectory["reached"]),
                    )

                    if speculative_request is not None:
                        def _record_speculative_evaluation(
                            speculative_result: InferenceResult,
                            *,
                            ready_at_boundary: bool,
                        ) -> None:
                            nonlocal prefetched_result
                            action_record["speculative_prefetch_result_ready_at_boundary"] = bool(ready_at_boundary)
                            action_record["speculative_prefetch_result_ready_late"] = not bool(ready_at_boundary)
                            action_record["speculative_prefetch_infer_dt_seconds"] = round(
                                speculative_result.duration_s, 6
                            )
                            speculative_evaluation = _evaluate_speculative_result(
                                result=speculative_result,
                                robot=robot,
                                sdk_arm_order=sdk_arm_order,
                                config=speculative_config,
                            )
                            action_record["speculative_prefetch_boundary_arm_delta_max"] = round(
                                float(speculative_evaluation["arm_delta_max"]), 6
                            )
                            action_record["speculative_prefetch_boundary_waist_delta_max"] = round(
                                float(speculative_evaluation["waist_delta_max"]), 6
                            )
                            action_record["speculative_prefetch_boundary_head_delta_max"] = round(
                                float(speculative_evaluation["head_delta_max"]), 6
                            )
                            action_record["speculative_prefetch_boundary_gripper_delta_max"] = round(
                                float(speculative_evaluation["gripper_delta_max"]), 6
                            )
                            action_record["speculative_prefetch_state_age_seconds"] = round(
                                float(speculative_evaluation["state_age_s"]), 6
                            )
                            action_record["speculative_prefetch_accepted"] = bool(
                                speculative_evaluation["accepted"]
                            )
                            action_record["speculative_prefetch_reject_reason"] = str(
                                speculative_evaluation["reason"]
                            )
                            if bool(speculative_evaluation["accepted"]):
                                prefetched_result = speculative_result
                                logging.info(
                                    "Accepted speculative result for next step | ready_at_boundary=%s arm_delta=%.4f waist_delta=%.4f age=%.3fs infer_dt=%.3fs",
                                    ready_at_boundary,
                                    float(speculative_evaluation["arm_delta_max"]),
                                    float(speculative_evaluation["waist_delta_max"]),
                                    float(speculative_evaluation["state_age_s"]),
                                    float(speculative_result.duration_s),
                                )
                            else:
                                logging.info(
                                    "Rejected speculative result | ready_at_boundary=%s reason=%s head_delta=%.4f gripper_delta=%.4f",
                                    ready_at_boundary,
                                    speculative_evaluation["reason"],
                                    float(speculative_evaluation["head_delta_max"]),
                                    float(speculative_evaluation["gripper_delta_max"]),
                                )

                        speculative_result = worker.wait_for_result(
                            speculative_request.request_id,
                            timeout_s=speculative_config.max_boundary_wait_s,
                        )
                        if speculative_result is not None:
                            _record_speculative_evaluation(speculative_result, ready_at_boundary=True)
                        else:
                            action_record["speculative_prefetch_reject_reason"] = "result_not_ready_at_boundary_waiting_late"
                            logging.info(
                                "Speculative result not ready at boundary; waiting for the in-flight result to avoid a busy worker."
                            )
                            late_result = worker.wait_for_result(speculative_request.request_id, timeout_s=None)
                            if late_result is None:
                                action_record["speculative_prefetch_reject_reason"] = "worker_stopped_before_late_result"
                            else:
                                _record_speculative_evaluation(late_result, ready_at_boundary=False)

                    action_record.update(
                        {
                            "execution_mode": f"move_arm_{arm_execution_mode}_with_row_synced_gripper",
                            "executed_gripper_command_policy": trajectory["last_gripper_sent_policy"].tolist(),
                            "trajectory_gripper_updates": int(trajectory["gripper_updates"][0]),
                            "trajectory_first_gripper_command_policy": trajectory["first_gripper_sent_policy"].tolist(),
                            "trajectory_last_gripper_command_policy": trajectory["last_gripper_sent_policy"].tolist(),
                            "execution_call_started_at_seconds": round(execution_call_started_at, 6),
                            "first_robot_command_at_seconds": round(float(trajectory["first_command_at"]), 6),
                            "trajectory_stream_finished_at_seconds": round(float(trajectory["stream_finished_at"]), 6),
                            "trajectory_wait_started_at_seconds": round(float(trajectory["wait_started_at"]), 6),
                            "trajectory_wait_finished_at_seconds": round(float(trajectory["wait_finished_at"]), 6),
                            "trajectory_returned_at_seconds": round(trajectory_returned_at, 6),
                            "aux_action_started_at_seconds": round(aux_started_at, 6),
                            "aux_action_finished_at_seconds": round(aux_finished_at, 6),
                            "action_to_execution_call_delay_seconds": round(action_to_execution_call_delay_s, 6),
                            "action_to_first_robot_command_delay_seconds": round(action_to_first_command_delay_s, 6),
                            "trajectory_first_command_internal_delay_seconds": round(float(trajectory["first_command_delay_seconds"]), 6),
                            "trajectory_stream_duration_seconds": round(float(trajectory["stream_duration_seconds"]), 6),
                            "trajectory_wait_close_duration_seconds": round(float(trajectory["wait_duration_seconds"]), 6),
                            "chunk_execution_duration_seconds": round(float(trajectory["execution_duration_seconds"]), 6),
                            "aux_action_duration_seconds": round(aux_finished_at - aux_started_at, 6),
                            "total_after_action_received_duration_seconds": round(aux_finished_at - action_received_at, 6),
                            "trajectory_rows": int(trajectory["target_rows"][0]),
                            "move_arm_points": int(trajectory["move_arm_points"][0]),
                            "trajectory_waist_rows_sent": int(trajectory["waist_rows_sent"][0]),
                            "trajectory_wheel_rows_sent": int(trajectory["wheel_rows_sent"][0]),
                            "trajectory_wheel_stop_sent": bool(int(trajectory["wheel_stop_sent"][0])),
                            "trajectory_first_left_arm_joint_position": trajectory["first_arm_policy"][:7].tolist(),
                            "trajectory_first_right_arm_joint_position": trajectory["first_arm_policy"][7:14].tolist(),
                            "trajectory_last_left_arm_joint_position": trajectory["last_arm_policy"][:7].tolist(),
                            "trajectory_last_right_arm_joint_position": trajectory["last_arm_policy"][7:14].tolist(),
                            "trajectory_first_waist_position": trajectory["first_waist_sent"].tolist(),
                            "trajectory_last_waist_position": trajectory["last_waist_sent"].tolist(),
                            "trajectory_first_wheel_command": trajectory["first_wheel_sent"].tolist(),
                            "trajectory_last_wheel_command": trajectory["last_wheel_sent"].tolist(),
                            "post_left_arm_joint_position": trajectory["arm_after_policy"][:7].tolist(),
                            "post_right_arm_joint_position": trajectory["arm_after_policy"][7:14].tolist(),
                            "post_waist_position": trajectory["waist_after"].tolist(),
                            "post_arm_error_to_target": round(float(trajectory["arm_error"]), 6),
                            "post_left_arm_error_to_target": round(float(trajectory["left_error"]), 6),
                            "post_right_arm_error_to_target": round(float(trajectory["right_error"]), 6),
                            "post_arm_reached_target": bool(trajectory["reached"]),
                            "row_execution_log": trajectory["row_execution_log"],
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
            logging.info("Stopping inference worker...")
            worker.stop()
            logging.info("Inference worker stopped.")
        except Exception as exc:
            logging.warning("Failed while stopping inference worker: %s", exc)

        try:
            logging.info("Stopping observation sampler...")
            sampler.stop()
            logging.info("Observation sampler stopped.")
        except Exception as exc:
            logging.warning("Failed while stopping observation sampler: %s", exc)

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
    parser.add_argument(
        "--observation-fps",
        type=float,
        default=5.0,
        help="Fixed camera observation sampling rate used for model inputs.",
    )
    parser.add_argument(
        "--observation-history",
        type=int,
        default=4,
        help="Number of most recent real camera samples to stack into each observation.",
    )
    parser.add_argument(
        "--enable-speculative-inference",
        action="store_true",
        help="Run next-step inference in the background during current chunk execution and only accept it at the chunk boundary if the robot state is still close.",
    )
    parser.add_argument(
        "--speculative-prefetch-fraction",
        type=float,
        default=0.7,
        help="Fraction of the current chunk completed before submitting speculative next-step inference.",
    )
    parser.add_argument(
        "--speculative-max-boundary-wait",
        type=float,
        default=0.15,
        help="How long to wait at the chunk boundary for a speculative inference result before falling back to sync inference.",
    )
    parser.add_argument(
        "--speculative-max-state-age",
        type=float,
        default=1.5,
        help="Reject speculative results whose observation snapshot is older than this many seconds at the chunk boundary.",
    )
    parser.add_argument(
        "--speculative-arm-state-tolerance",
        type=float,
        default=0.12,
        help="Max allowed arm-state drift between speculative snapshot and chunk boundary.",
    )
    parser.add_argument(
        "--speculative-waist-state-tolerance",
        type=float,
        default=0.03,
        help="Max allowed waist-state drift between speculative snapshot and chunk boundary.",
    )
    parser.add_argument("--apply-actions", action="store_true", help="Enable sending actions to robot")
    parser.add_argument("--disable-gripper", action="store_true", help="Do not send gripper commands during execution")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gripper-closed-value",
        type=float,
        default=1.0,
        help="Gripper command value sent when the model predicts close. Lower values make softer grasps.",
    )
    parser.add_argument(
        "--gripper-close-when-low",
        action="store_true",
        help="Interpret low gripper logits/targets as close instead of high values as close.",
    )
    parser.add_argument("--arm-delta-limit", type=float, default=0.08)
    parser.add_argument("--head-delta-limit", type=float, default=0.15)
    parser.add_argument("--waist-pitch-delta-limit", type=float, default=0.10)
    parser.add_argument("--waist-lift-delta-limit", type=float, default=2.0)
    parser.add_argument("--apply-wheel-actions", action="store_true", help="Enable streaming move_wheel commands from model output")
    parser.add_argument("--linear-velocity-limit", type=float, default=0.15)
    parser.add_argument("--angular-velocity-limit", type=float, default=0.40)
    parser.add_argument(
        "--arm-trajectory-interval",
        type=float,
        default=DEFAULT_MOVE_ARM_INTERVAL_S,
        help="Seconds between streamed move_arm trajectory points. Lower is faster but more aggressive.",
    )
    parser.add_argument(
        "--arm-execution-mode",
        choices=["ruckig", "direct-48"],
        default="ruckig",
        help="Use ruckig interpolation or send the model's 48 action rows directly at --direct-control-hz.",
    )
    parser.add_argument(
        "--direct-control-hz",
        type=float,
        default=15.0,
        help="Row frequency for --arm-execution-mode direct-48. 15 Hz is a conservative first test; 30 Hz targets the paper timing.",
    )
    parser.add_argument(
        "--arm-close-timeout",
        type=float,
        default=DEFAULT_MOVE_ARM_TIMEOUT_S,
        help="Max seconds to wait after sending the arm trajectory before starting next inference. Set lower to speed up.",
    )
    parser.add_argument(
        "--action-smoothing",
        choices=["none", "savgol"],
        default="none",
        help="Optional action smoothing before execution. Savgol only smooths the 14 arm joint dimensions.",
    )
    parser.add_argument("--savgol-window", type=int, default=7)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    parser.add_argument("--savgol-upsample", type=int, default=2)
    parser.add_argument(
        "--log-row-execution",
        action="store_true",
        help="Record per-row command targets and send timing in the action JSON log. Does not read robot state per row.",
    )
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
        linear_velocity=args.linear_velocity_limit,
        angular_velocity=args.angular_velocity_limit,
        arm_trajectory_interval=args.arm_trajectory_interval,
        arm_close_timeout=args.arm_close_timeout,
        direct_control_hz=args.direct_control_hz,
    )

    run_client(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        apply_actions=args.apply_actions,
        enable_gripper=not args.disable_gripper,
        enable_wheel=args.apply_wheel_actions,
        limits=limits,
        arm_execution_mode=args.arm_execution_mode,
        sdk_arm_order=args.sdk_arm_order,
        obs_flip_config=ObsFlipConfig(
            left_joint_indices=_parse_joint_index_list(args.obs_flip_left_joints),
            right_joint_indices=_parse_joint_index_list(args.obs_flip_right_joints),
        ),
        gripper_config=GripperConfig(
            close_threshold=args.gripper_close_threshold,
            close_when_high=not args.gripper_close_when_low,
            closed_value=args.gripper_closed_value,
        ),
        observation_fps=args.observation_fps,
        observation_history=args.observation_history,
        speculative_config=SpeculativeConfig(
            enabled=args.enable_speculative_inference,
            prefetch_fraction=args.speculative_prefetch_fraction,
            max_boundary_wait_s=args.speculative_max_boundary_wait,
            max_state_age_s=args.speculative_max_state_age,
            arm_state_tolerance=args.speculative_arm_state_tolerance,
            waist_state_tolerance=args.speculative_waist_state_tolerance,
        ),
        smoothing_config=ActionSmoothingConfig(
            mode=args.action_smoothing,
            window=args.savgol_window,
            polyorder=args.savgol_polyorder,
            upsample=args.savgol_upsample,
        ),
        log_row_execution=args.log_row_execution,
    )


if __name__ == "__main__":
    main()
