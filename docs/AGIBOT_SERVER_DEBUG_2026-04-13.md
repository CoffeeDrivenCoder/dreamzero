# AgiBot 服务排障记录

## 主要问题

- `DreamZero-AgiBot/config.json` 里的子权重路径是 `null`，导致服务虽然指定了本地模型目录，仍然会回退到 Hugging Face 下载 Wan/T5/VAE 权重。
- `experiment_cfg/conf.yaml` 里还残留了旧机器的绝对路径 `/mnt/aws-lfs-02/shared/ckpts/...`。
- 当前环境里的 PyTorch 版本不支持 `torch._dynamo.config.recompile_limit`。
- 默认 `ATTENTION_BACKEND=FA2` 时，真实推理阶段可能报错，并且会被 `torch.compile(..., fullgraph=True)` 包装成 `torch._dynamo.exc.Unsupported`。

## 已处理内容

- 本地模型路径统一改为：
  - `/home/user/wangk/checkpoints/DreamZero-AgiBot`
  - `/home/user/wangk/checkpoints/Wan2.1-I2V-14B-480P`
  - `/home/user/wangk/checkpoints/umt5-xxl`
- 已修复以下配置文件中的本地路径：
  - `/home/user/wangk/checkpoints/DreamZero-AgiBot/config.json`
  - `/home/user/wangk/checkpoints/DreamZero-AgiBot/experiment_cfg/conf.yaml`
- 已将 `load_pretrained_det_decode_layer_path` 设为 `null`。
- 当前建议先使用 `ATTENTION_BACKEND=torch`，避免 FlashAttention 兼容性问题影响排查。

## 当前启动命令

```bash
cd /home/user/wangk/dreamzero

export HF_HUB_OFFLINE=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
export ATTENTION_BACKEND=torch

CUDA_VISIBLE_DEVICES=0,1 /home/user/miniconda3/envs/dreamzero/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
  --port 9443 \
  --model-path /home/user/wangk/checkpoints/DreamZero-AgiBot \
  --embodiment-tag agibot \
  --enable-dit-cache
```

## 说明

- 如果后续确认 `flash-attn` 与 `torch 2.8` 匹配，可以再把 `ATTENTION_BACKEND` 从 `torch` 切回 `FA2`。
- 现在如果再出现新报错，通常已经不是本地模型路径或缺少 checkpoint 文件导致的。
