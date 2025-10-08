# DirectML ONNX Runtime Backend

This document describes how to use the optimised ONNX Runtime backend that targets Windows 11 systems equipped with AMD Radeon RX 7900 XTX GPUs (Ryzen 7 5800X3D / 32 GB RAM) via DirectML.

## Key capabilities

- DirectML and CPU execution providers with automatic device selection or explicit `--device-id` binding.
- Optional float16 execution for compatible GPUs while keeping CPU fallbacks in float32 for reproducibility.
- Spatial tiling and batching controls that reduce VRAM pressure when working with 4K videos.
- I/O binding with DirectML to avoid needless host/device copies and improve throughput.
- Simple pipelining helpers (`queue_depth`, `prefetch`) for background scheduling when driving the backend programmatically.

## Command-line flags

The backend accepts the following flags in both the GUI entry point (`FluidFrames.py`) and helper scripts such as `tools/run_benchmarks.py`:

| Flag | Description |
| --- | --- |
| `--provider {dml,cpu}` | Select DirectML or the CPU execution provider. |
| `--fp16` | Enable float16 execution (DirectML only). |
| `--tilesize [H W]` | Tile size used to split large frames before inference. Provide one value for square tiles or `H W` for rectangular tiles. |
| `--batch N` | Batch size for queued inference requests. |
| `--queue-depth N` | Number of in-flight requests when using the asynchronous pipeline helper. |
| `--prefetch N` | Prefetch depth for pipelined execution. |

Example:

```bash
python FluidFrames.py --provider dml --fp16 --tilesize 1080 1920 --batch 2 --queue-depth 4 --prefetch 4
```

When `FluidFrames.py` is launched without arguments it defaults to DirectML with automatic device selection. Any backend arguments are stripped from `sys.argv` so they do not interfere with the GUI initialisation.

## Exporting new ONNX models

A flexible exporter is provided in `tools/export_to_onnx.py`. The script expects a PyTorch checkpoint and a model class exposing a `forward` function that consumes tensors shaped `(1, 6, H, W)` (concatenated frame pairs used by RIFE). Typical usage:

```bash
python tools/export_to_onnx.py \
    path/to/rife_checkpoint.pth \
    AI-onnx/RIFE_fp32.onnx \
    --module inference_rife \
    --class-name Model \
    --height 1080 \
    --width 1920 \
    --opset 17
```

Add `--fp16` if you want to export a float16 variant. Ensure that `torch` is installed in your Python environment before running the exporter.

## Benchmarking targets

Use `tools/run_benchmarks.py` to validate performance and visual parity. The script generates deterministic synthetic data and compares DirectML outputs against the CPU provider. Metrics include FPS, estimated VRAM consumption, CPU load, and the PSNR/SSIM deltas relative to the CPU baseline.

```bash
python tools/run_benchmarks.py --provider dml --fp16 --tilesize 1080 1920 --batch 2
```

The script evaluates two default workloads:

- **1080p (1920×1080)** — passes when FPS ≥ 80.
- **4K (3840×2160)** — passes when FPS ≥ 28.

It also checks that the estimated VRAM footprint stays below 16 GB and that PSNR/SSIM deltas remain within the visual parity envelope (ΔPSNR ≥ 80 dB, ΔSSIM ≥ 0.998 compared to the CPU reference). Any unmet goal is highlighted at the end of the report.

## Integrating with custom pipelines

Developers can import the backend programmatically:

```python
from fluidframes_backend import BackendRuntimeConfig, OnnxRuntimeBackend

config = BackendRuntimeConfig(provider="dml", fp16=True, tile_size=(1080, 1920), batch_size=2)
backend = OnnxRuntimeBackend("AI-onnx/RIFE_fp32.onnx", config)

# Process a batched tensor shaped (N, 6, H, W)
output = backend.run(input_tensor)
```

For streaming workloads leverage `backend.pipeline(...)` which respects the `queue_depth` and `prefetch` settings to overlap compute and data preparation.
