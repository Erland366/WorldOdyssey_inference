---
name: sglang-cuda-stack-diagnostics
description: Diagnose SGLang Diffusion CUDA, PyTorch, and kernel-wheel compatibility before running Wan/FastWan backend tests.
---

# SGLang CUDA Stack Diagnostics

## Purpose

Use this skill when installing, validating, or debugging SGLang Diffusion for video-generation tests on a machine with fixed NVIDIA driver and CUDA runtime constraints.

The goal is to prove the runtime stack is coherent before running any model. SGLang failures often come from mismatched host driver, PyTorch CUDA wheel, CUDA runtime libraries, and SGLang kernel wheels.

## Workflow

1. Keep SGLang isolated from the main FastVideo/Diffusers environment unless the dependency pins are already known to be compatible.
2. Check the host driver and maximum supported CUDA runtime first:
   - `nvidia-smi`
   - `ldconfig -p | grep -E 'libcuda|libcudnn'`
3. Select a PyTorch CUDA wheel that the host driver can execute.
4. Run a real CUDA/cuDNN probe before importing SGLang.
5. Inspect SGLang package pins and the installed kernel package:
   - `sgl-kernel` for older SGLang releases
   - `sglang-kernel` for newer SGLang releases
6. Run `sglang generate --help` as the CLI smoke test.
7. Only run slow backend tests after the CLI imports cleanly.

## CUDA Probe

Run this inside the isolated SGLang environment:

```bash
source .venv_sglangcuda12/bin/activate
python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
assert torch.cuda.is_available()

x = torch.randn(1, 4, 5, 8, 8, device="cuda", dtype=torch.bfloat16)
conv = torch.nn.Conv3d(4, 8, 3, padding=1).cuda().bfloat16()

with torch.no_grad():
    y = conv(x)

torch.cuda.synchronize()
print("conv ok", tuple(y.shape))
PY
```

If this probe fails, fix the PyTorch/driver/runtime stack before touching SGLang.

## Package Probe

Use Python package metadata instead of assuming what `uv pip install` selected:

```bash
source .venv_sglangcuda12/bin/activate
python - <<'PY'
from importlib import metadata

for name in ("torch", "torchvision", "sglang", "sgl-kernel", "sglang-kernel", "triton"):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, "not installed")
PY
```

Then run:

```bash
source .venv_sglangcuda12/bin/activate
sglang generate --help
```

Treat an import error here as a stack failure. Do not move on to model execution until this command works.

## Known WorldOdyssey Results

On the host with NVIDIA driver `550.127.08` and CUDA `12.4` reported by `nvidia-smi`:

- `torch==2.6.0+cu124` passed a CUDA bf16 `Conv3d` probe in `.venv_sglangcuda12`.
- Current `sglang[diffusion]` pulled CUDA-13 runtime/kernel expectations and failed on this driver with `CUDA driver version is insufficient for CUDA runtime version`.
- `sglang==0.5.5` pins `torch==2.8.0`.
- No `torch==2.8.0+cu124` wheel was available on the PyTorch cu124 index.
- Forcing `sglang==0.5.5` onto `torch==2.6.0+cu124` with cu124 `sgl-kernel` wheels failed before `sglang generate --help` with a torch C++ ABI symbol mismatch.
- A local `sgl-kernel` source build against `torch==2.6.0+cu124` needed compiler, CMake policy, CUDA, and NVTX path fixes. Treat that route as a production build attempt, not a quick foreground debug command.

## Decision Rules

- If the base CUDA probe fails, stop and fix PyTorch, CUDA libraries, or the host driver.
- If `sglang generate --help` fails, record package versions and resolve the kernel/runtime mismatch before running tests.
- Prefer an official SGLang container or a matched prebuilt wheel set over mixing SGLang, Torch, and kernel wheels across releases.
- Keep the main `.venv` stable for FastVideo/Diffusers unless the user explicitly wants to switch the project stack.

## Slow Backend Test

After the stack passes the probes, run the optional SGLang backend test:

```bash
source .venv_sglangcuda12/bin/activate
python -m pytest tests/backends/test_sglang_tiny_wan.py -m slow -s
```

If SGLang was intentionally installed into the main project environment, activate `.venv` instead.

Generated videos should remain under `artifacts/backend-videos/` for inspection.
