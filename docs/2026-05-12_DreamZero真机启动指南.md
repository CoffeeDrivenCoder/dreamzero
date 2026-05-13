# DreamZero 真机启动指南

**日期**：2026-05-12  
**状态**：`草稿`  
**关联文件**：`scripts/run_agibot_server_6006.sh`、`socket_test_optimized_AR.py`、`robot_live_client.py`、`eval_utils/policy_client.py`、`eval_utils/policy_server.py`

## 结论

服务器优先使用 `scripts/run_agibot_server_6006.sh` 启动 DreamZero AgiBot WebSocket 推理服务，客户端使用 `robot_live_client.py` 连接 `223.167.85.178:9443` 并按 `5 FPS + 4 帧 history + direct-48` 执行真机闭环。

## 适用范围

这份文档覆盖 DreamZero AgiBot 真机联调的两端启动流程：

- 服务器：加载 DreamZero-AgiBot checkpoint，启动 WebSocket policy server。
- 客户端：读取 GDK 相机与机器人状态，发送 observation 到服务器，并可选择把服务器返回的 action 下发到机器人。

截至 2026-05-12，代码中的 AgiBot server 配置为：

- WebSocket 监听地址：`0.0.0.0:<PORT>`
- 默认端口：`9443`
- 机器人 embodiment：`agibot`
- 图像分辨率元数据：`(640, 480)`
- 外部相机数量：`3`
- action space：`agibot_flattened`
- 分布式推理仅支持 `1` 或 `2` 张 GPU

## 服务器启动 DreamZero

### 服务器环境

在服务器机器执行，默认路径和环境如下：

- tmux 会话：`server`
- 代码目录：`/home/user/wangk/dreamzero`
- Python：`/home/user/miniconda3/envs/dreamzero/bin/python`
- 默认 checkpoint：`/data1/wangk/checkpoints/DreamZero-AgiBot`
- 默认 GPU：`CUDA_VISIBLE_DEVICES=0,1`
- 默认端口：`9443`

### 推荐启动方式

优先使用仓库里的脚本启动，因为脚本已经处理了代理变量、离线 Hugging Face、FlashAttention 后端和模型路径检查。

```bash
tmux attach -t server

cd /home/user/wangk/dreamzero

MODEL_PATH=/data1/wangk/checkpoints/DreamZero-AgiBot \
PORT=9443 \
NUM_GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
VIDEO_SAVE_MODE=full \
NUM_INFERENCE_TIMESTEPS=0 \
bash scripts/run_agibot_server_6006.sh
```

脚本实际会执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/user/miniconda3/envs/dreamzero/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 9443 \
  --model-path /data1/wangk/checkpoints/DreamZero-AgiBot \
  --embodiment-tag agibot \
  --video-save-mode full \
  --num-inference-timesteps 0
```

`NUM_INFERENCE_TIMESTEPS=0` 表示使用 checkpoint 自带的 diffusion inference step 配置；正整数会覆盖 checkpoint 配置。

### 手动展开启动方式

如果需要绕过脚本，可以直接执行完整命令：

```bash
tmux attach -t server

cd /home/user/wangk/dreamzero

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset ws_proxy wss_proxy WS_PROXY WSS_PROXY

export HF_HUB_OFFLINE=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
export ATTENTION_BACKEND=FA2

CUDA_VISIBLE_DEVICES=0,1 /home/user/miniconda3/envs/dreamzero/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 9443 \
  --model-path /data1/wangk/checkpoints/DreamZero-AgiBot \
  --embodiment-tag agibot \
  --video-save-mode full \
  --num-inference-timesteps 0
```

如果 `ATTENTION_BACKEND=FA2` 触发 attention 兼容问题，改用 torch 后端重新启动：

```bash
export ATTENTION_BACKEND=torch
```

### 服务端启动成功标志

日志中出现以下信息时，说明服务端进入可连接状态：

```text
Using roboarena policy server interface for agibot
Server config: PolicyServerConfig(...)
```

客户端连接后，服务端会打印连接来源；客户端关闭或 reset 时，服务端会按 `--video-save-mode` 保存 rollout 视频到：

```text
/home/user/wangk/dreamzero/video_rollout
```

## 客户端启动 DreamZero

### 客户端环境

在机器人客户端机器执行，默认路径如下：

- 代码目录：`/home/ke/dreamzero`
- 客户端入口：`robot_live_client.py`
- 服务器地址：`223.167.85.178`
- 服务器端口：`9443`

客户端依赖 GDK Python SDK。若启动时报 `Failed to import GDK python SDK`，需要先确认 `a2d_sdk` 已安装并且 GDK 的 `env.sh` 已 source。

### 客户端前置步骤

启动 `robot_live_client.py` 前，在机器人客户端机器按顺序执行下面步骤。

1. 激活客户端 conda 环境：

```bash
conda activate a2d
```

2. 进入 GDK 目录：

```bash
cd /home/ke/dreamzero/a2d_sdk
```

3. 加载 GDK 环境：

```bash
source env.sh
```

4. 启动机器人服务：

```bash
robot-service -s -c ./conf/copilot.pbtxt
```

保持 `robot-service` 运行后，再打开新的终端执行 DreamZero 客户端命令；新终端也需要先 `conda activate a2d` 并加载同一个 GDK 环境。

### 机器人复位

执行真机任务前，建议先在客户端机器完成手臂、腰部和夹爪复位。复位命令同样需要在 `a2d` 环境下执行，并先加载 `env.sh`。

1. 激活环境并加载 GDK：

```bash
conda activate a2d
cd /home/ke/dreamzero/a2d_sdk
source env.sh
```

2. 复位双臂到指定姿态：

```bash
python /home/ke/dreamzero/a2d_sdk/python_example/python/example/robot_control/arm_reset_to_pose.py \
  --target="-1.4933609364473674,0.7964081244626098,1.0012785508617679,-1.0159238772702275,0.5392512839139959,1.182603777155882,-0.004982494198811424,1.5749216974854596,-0.8573385721296016,-1.2055018800374875,0.9119414194334386,-0.44171195596984786,-1.1144042733809039,-0.018197123126293055" \
  --confirm
```

3. 启动交互式控制器：

```bash
robot-controller
```

4. 在 `robot-controller` 提示符中输入腰部复位命令：

```text
wa 50,45
```

5. 在 `robot-controller` 提示符中输入夹爪复位命令：

```text
gr 0,0
```

完成后退出或挂起 `robot-controller`，再回到 DreamZero 客户端启动流程。

### 推荐真机执行命令

下面命令会真实下发动作到机器人，因为包含 `--apply-actions`。

```bash
conda activate a2d

cd /home/ke/dreamzero/a2d_sdk
source env.sh

cd /home/ke/dreamzero

python robot_live_client.py \
  --host 223.167.85.178 \
  --port 9443 \
  --prompt "依次拿起桌上的裤子和杯子，放进中间的盒子中。" \
  --sdk-arm-order left_right \
  --apply-actions \
  --observation-fps 5 \
  --observation-history 4 \
  --image-transport jpeg \
  --image-jpeg-quality 80 \
  --arm-execution-mode direct-48 \
  --direct-control-hz 30 \
  --arm-close-timeout 0.15 \
  --action-smoothing savgol \
  --savgol-window 21 \
  --savgol-polyorder 3 \
  --savgol-upsample 2 \
  --gripper-closed-value 1.0
```

运行中按 `q` 可以请求客户端停止。客户端会在 `robot_json/` 下保存每轮 action 日志，例如：

```text
robot_json/agibot_actions_YYYYMMDD_HHMMSS.json
```

### 仅连接和推理验证

如果只验证相机、机器人状态读取、WebSocket 连接和模型推理，不希望机器人执行动作，去掉 `--apply-actions`：

```bash
conda activate a2d

cd /home/ke/dreamzero/a2d_sdk
source env.sh

cd /home/ke/dreamzero

python robot_live_client.py \
  --host 223.167.85.178 \
  --port 9443 \
  --prompt "依次拿起桌上的裤子和杯子，放进中间的盒子中。" \
  --sdk-arm-order left_right \
  --observation-fps 5 \
  --observation-history 4 \
  --image-transport jpeg \
  --image-jpeg-quality 80 \
  --arm-execution-mode direct-48 \
  --direct-control-hz 30 \
  --arm-close-timeout 0.15 \
  --action-smoothing savgol \
  --savgol-window 21 \
  --savgol-polyorder 3 \
  --savgol-upsample 2 \
  --gripper-closed-value 1.0
```

### 关键参数说明

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| `--host` | `223.167.85.178` | DreamZero server IP |
| `--port` | `9443` | DreamZero server WebSocket 端口 |
| `--prompt` | 任务自然语言 | 每次任务需要传给模型的语言指令 |
| `--sdk-arm-order` | `left_right` | SDK 手臂顺序；影响 observation 和 action 的左右臂映射 |
| `--apply-actions` | 开启 | 开启后客户端会真实执行模型输出动作 |
| `--observation-fps` | `5` | 后台相机采样频率 |
| `--observation-history` | `4` | 每次发送最近 4 帧 observation |
| `--image-transport` | `jpeg` | 用 JPEG 传图，降低 WebSocket payload |
| `--image-jpeg-quality` | `80` | JPEG 质量 |
| `--arm-execution-mode` | `direct-48` | 按模型 48 行 action 直接下发 |
| `--direct-control-hz` | `30` | `direct-48` 模式下 action row 下发频率 |
| `--arm-close-timeout` | `0.15` | chunk 结束后等待手臂接近目标的最长时间 |
| `--action-smoothing` | `savgol` | 对 14 维手臂关节 action 做 Savitzky-Golay 平滑 |
| `--gripper-closed-value` | `1.0` | 模型判定 close 时发送给夹爪的闭合值 |

## 启动顺序

1. 在服务器机器执行 `tmux attach -t server`。
2. 在 `server` tmux 会话中启动 DreamZero server。
3. 等待服务端日志显示 `Server config`。
4. 在机器人客户端机器执行 `conda activate a2d`。
5. 进入 `/home/ke/dreamzero/a2d_sdk` 并执行 `source env.sh`。
6. 启动 `robot-service -s -c ./conf/copilot.pbtxt` 并保持运行。
7. 打开新的复位终端，执行 `conda activate a2d`、`source /home/ke/dreamzero/a2d_sdk/env.sh`，按“机器人复位”小节完成手臂、腰部和夹爪复位。
8. 打开新的客户端终端，执行 `conda activate a2d`，并重新 `source /home/ke/dreamzero/a2d_sdk/env.sh`。
9. 在 `/home/ke/dreamzero` 启动 `robot_live_client.py`。
10. 客户端日志显示 `Server metadata` 后，确认 metadata 中包含 `image_resolution`、`n_external_cameras=3`、`action_space=agibot_flattened`。
11. 执行真机任务前确认机器人周围安全，再使用带 `--apply-actions` 的命令。

## 常见问题

### 客户端连不上服务器

优先检查三项：

- 服务端进程是否仍在运行。
- 客户端 `--host` 是否是服务器对机器人可访问的 IP。
- `--port` 是否和服务端启动端口一致。

客户端连接逻辑会先尝试 `ws://<host>:<port>`，失败后再尝试 `wss://<host>:<port>`。

### 服务端启动时报 checkpoint 不存在

`scripts/run_agibot_server_6006.sh` 会检查 `MODEL_PATH` 是否存在。若路径不同，用下面命令覆盖：

```bash
cd /home/user/wangk/dreamzero

MODEL_PATH=/data1/wangk/checkpoints/DreamZero-AgiBot bash scripts/run_agibot_server_6006.sh
```

### 服务端 GPU 数量报错

`socket_test_optimized_AR.py` 中的 DreamZero 推理路径只支持 `1` 或 `2` 张 GPU。若服务器只有单卡可用，使用：

```bash
cd /home/user/wangk/dreamzero

NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_agibot_server_6006.sh
```

### 客户端 GDK 导入失败

报错信息包含：

```text
Failed to import GDK python SDK. Make sure a2d_sdk is installed and env.sh is sourced.
```

处理方式是进入机器人客户端环境后，先加载 GDK 环境变量，再重新执行客户端命令。

### 客户端 robot-service 未启动

如果相机或 DDS 初始化失败，先确认机器人服务已在 `/home/ke/dreamzero/a2d_sdk` 下启动：

```bash
conda activate a2d
cd /home/ke/dreamzero/a2d_sdk
source env.sh
robot-service -s -c ./conf/copilot.pbtxt
```

## 验收标准

一次完整启动视为成功，需要同时满足：

- 服务端在 `server` tmux 会话中启动。
- 服务端加载 checkpoint 成功，并打印 `Using roboarena policy server interface for agibot`。
- 客户端机器上 `robot-service -s -c ./conf/copilot.pbtxt` 保持运行。
- 客户端机器已在 `a2d` 环境下完成手臂、腰部和夹爪复位。
- 客户端打印 `Server metadata`，其中 `action_space` 为 `agibot_flattened`。
- 客户端每轮推理打印 `Step ... action shape=...`。
- 带 `--apply-actions` 运行时，客户端打印 `Executed move_arm dual-arm trajectory with row-synced gripper`。
- 客户端在 `robot_json/` 生成本次运行的 action JSON。
- 服务端在 reset 或连接关闭后，按配置在 `video_rollout/` 保存视频。

## 更新日志

| 日期 | 修改内容 |
|------|---------|
| 2026-05-12 | 初始版本，整理服务器和客户端启动流程 |
