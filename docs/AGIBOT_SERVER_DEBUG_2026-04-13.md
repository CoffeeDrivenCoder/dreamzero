# AgiBot Server Debug Notes

## Main Issues

- `DreamZero-AgiBot/config.json` had null sub-weight paths, so the server tried to download Wan/T5/VAE files from Hugging Face even though local checkpoints already existed.
- `experiment_cfg/conf.yaml` still contained old absolute paths under `/mnt/aws-lfs-02/shared/ckpts/...`.
- The code used `torch._dynamo.config.recompile_limit`, but the local PyTorch version did not provide that field.
- The default `ATTENTION_BACKEND=FA2` path could fail during real inference, and the error was wrapped by `torch.compile(..., fullgraph=True)` as `torch._dynamo.exc.Unsupported`.

## Fix Summary

- Updated local checkpoint paths to:
  - `/home/user/wangk/checkpoints/DreamZero-AgiBot`
  - `/home/user/wangk/checkpoints/Wan2.1-I2V-14B-480P`
  - `/home/user/wangk/checkpoints/umt5-xxl`
- Fixed both:
  - `/home/user/wangk/checkpoints/DreamZero-AgiBot/config.json`
  - `/home/user/wangk/checkpoints/DreamZero-AgiBot/experiment_cfg/conf.yaml`
- Set `load_pretrained_det_decode_layer_path: null`.
- Recommended forcing `ATTENTION_BACKEND=torch` first to avoid FlashAttention-related runtime issues during validation.

## Current Start Command

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

## Notes

- If `flash-attn` is later verified to match `torch 2.8`, `ATTENTION_BACKEND` can be switched back from `torch` to `FA2`.
- If a new error appears now, it is no longer caused by missing local model files or wrong checkpoint paths.
