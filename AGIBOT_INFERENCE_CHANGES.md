# AgiBot Inference Changes

## What I changed

- Added `--embodiment-tag` to `socket_test_optimized_AR.py`.
- Kept `oxe_droid` as the default embodiment.
- Added an embodiment switch so the server now builds either the DROID wrapper or the AgiBot wrapper.
- Added an AgiBot-specific observation adapter.
- Added an AgiBot-specific action flattener.
- Updated the AgiBot `experiment_cfg/conf.yaml` to use local checkpoint paths under `/root/autodl-tmp/checkpoints`.
- Set `load_pretrained_det_decode_layer_path: null` for AgiBot, matching the working DROID setup.

## Server entrypoint changes

- New CLI flag: `--embodiment-tag`
- Supported values:
  - `oxe_droid`
  - `agibot`
- The server now selects both the wrapper and the websocket metadata based on the embodiment.

## AgiBot input schema expected by the new wrapper

### Video keys

- `observation/top_head` -> `video.top_head`
- `observation/hand_left` -> `video.hand_left`
- `observation/hand_right` -> `video.hand_right`

Accepted shapes per key:
- `(H, W, C)`
- `(T, H, W, C)`

Current behavior:
- Single frame input becomes `(1, H, W, C)`
- Multi-frame input keeps only the latest frame as `(1, H, W, C)`

### State keys

- `observation/left_arm_joint_position` -> `state.left_arm_joint_position`
- `observation/right_arm_joint_position` -> `state.right_arm_joint_position`
- `observation/left_effector_position` -> `state.left_effector_position`
- `observation/right_effector_position` -> `state.right_effector_position`
- `observation/head_position` -> `state.head_position`
- `observation/waist_pitch` -> `state.waist_pitch`
- `observation/waist_lift` -> `state.waist_lift`

Accepted shapes per key:
- scalar
- 1D
- 2D

The wrapper now raises a clear error if any required AgiBot key is missing.

## AgiBot action flattening

The wrapper now flattens AgiBot actions in this order:

1. `action.left_arm_joint_position`
2. `action.right_arm_joint_position`
3. `action.left_effector_position`
4. `action.right_effector_position`
5. `action.head_position`
6. `action.waist_pitch`
7. `action.waist_lift`
8. `action.robot_velocity`

Return shape:
- `(N, D_total)`

Where `D_total` is the concatenation of all AgiBot action dimensions.

## Local checkpoint path changes

The AgiBot config now points to:
- `/root/autodl-tmp/checkpoints/Wan2.1-I2V-14B-480P`
- `/root/autodl-tmp/checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth`
- `/root/autodl-tmp/checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- `/root/autodl-tmp/checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth`
- `/root/autodl-tmp/checkpoints/umt5-xxl`

## What is still not finished

- I did not add an AgiBot test client yet.
- I did not run a full AgiBot model load and infer pass yet.
- If your upstream caller still sends DROID-style keys, that caller must be updated to the AgiBot keys listed above.
