"""Benchmark harness for the FluidFrames ONNX Runtime backend."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import psutil

from fluidframes_backend import BackendRuntimeConfig, OnnxRuntimeBackend, add_backend_cli_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DirectML/CPU benchmarks for FluidFrames.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("AI-onnx") / "RIFE_fp32.onnx",
        help="Path to the ONNX model to benchmark.",
    )
    parser.add_argument("--iterations", type=int, default=64, help="Number of timed iterations per resolution.")
    parser.add_argument("--warmup", type=int, default=4, help="Number of warm-up runs ignored from timing.")
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Explicit DirectML device id. Defaults to the backend selection logic.",
    )
    parser.add_argument(
        "--target-1080p",
        type=float,
        default=80.0,
        help="Minimum target FPS for 1920x1080 workloads.",
    )
    parser.add_argument(
        "--target-4k",
        type=float,
        default=28.0,
        help="Minimum target FPS for 3840x2160 workloads.",
    )
    add_backend_cli_args(parser)
    return parser.parse_args()


def compute_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((reference - test) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(reference: np.ndarray, test: np.ndarray) -> float:
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_x = reference.mean()
    mu_y = test.mean()
    sigma_x = reference.var()
    sigma_y = test.var()
    covariance = ((reference - mu_x) * (test - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))


def random_inputs(iterations: int, channels: int, height: int, width: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(42)
    for _ in range(iterations):
        yield rng.random((1, channels, height, width), dtype=np.float32)


def estimate_vram_usage(model_path: Path, tensor_shape: Tuple[int, int, int, int], dtype: np.dtype) -> float:
    dtype_size = np.dtype(dtype).itemsize
    batch, channels, height, width = tensor_shape
    activation_bytes = batch * channels * height * width * dtype_size
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    total_bytes = model_bytes + activation_bytes * 2  # account for input and output activations
    return total_bytes / (1024 ** 3)


def benchmark_resolution(
    backend: OnnxRuntimeBackend,
    cpu_backend: OnnxRuntimeBackend,
    height: int,
    width: int,
    iterations: int,
    warmup: int,
    model_path: Path,
) -> dict[str, float]:
    inputs = list(random_inputs(iterations + warmup, 6, height, width))

    # Warm-up both providers
    for sample in inputs[:warmup]:
        backend.run(sample)
        cpu_backend.run(sample)

    timed_inputs = inputs[warmup:]
    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    wall_before = time.perf_counter()

    for sample in timed_inputs:
        backend.run(sample)

    wall_after = time.perf_counter()
    cpu_times_after = process.cpu_times()

    elapsed = wall_after - wall_before
    fps = len(timed_inputs) / elapsed if elapsed else float("inf")

    user_cpu = cpu_times_after.user - cpu_times_before.user
    system_cpu = cpu_times_after.system - cpu_times_before.system
    cpu_seconds = user_cpu + system_cpu
    logical_cores = psutil.cpu_count(logical=True) or 1
    cpu_percent = (cpu_seconds / elapsed / logical_cores) * 100 if elapsed else 0.0

    reference_outputs = [cpu_backend.run(sample) for sample in timed_inputs[: min(3, len(timed_inputs))]]
    optimised_outputs = [backend.run(sample) for sample in timed_inputs[: min(3, len(timed_inputs))]]

    psnr_scores = [compute_psnr(ref, test) for ref, test in zip(reference_outputs, optimised_outputs)]
    ssim_scores = [compute_ssim(ref, test) for ref, test in zip(reference_outputs, optimised_outputs)]

    delta_psnr = statistics.fmean(psnr_scores) if psnr_scores else float("inf")
    delta_ssim = statistics.fmean(ssim_scores) if ssim_scores else 1.0

    estimated_vram = estimate_vram_usage(model_path, (1, 6, height, width), np.float16 if backend.session.get_inputs()[0].type == "tensor(float16)" else np.float32)

    return {
        "fps": fps,
        "cpu_percent": cpu_percent,
        "delta_psnr": delta_psnr,
        "delta_ssim": delta_ssim,
        "estimated_vram_gb": estimated_vram,
    }


def main() -> None:
    args = parse_args()
    config = BackendRuntimeConfig.from_cli_args(args)
    backend = OnnxRuntimeBackend(args.model.as_posix(), config, device_id=args.device_id)
    cpu_config = BackendRuntimeConfig(provider="cpu", fp16=False, tile_size=None, batch_size=config.batch_size)
    cpu_backend = OnnxRuntimeBackend(args.model.as_posix(), cpu_config)

    metrics_1080p = benchmark_resolution(backend, cpu_backend, 1080, 1920, args.iterations, args.warmup, args.model)
    metrics_4k = benchmark_resolution(backend, cpu_backend, 2160, 3840, args.iterations, args.warmup, args.model)

    print("=== Benchmark Summary ===")
    print(f"Provider: {config.provider.upper()} | FP16: {config.fp16} | Tile: {config.tile_size or 'full frame'}")
    print(f"1080p -> {metrics_1080p['fps']:.2f} FPS | CPU {metrics_1080p['cpu_percent']:.1f}% | ΔPSNR {metrics_1080p['delta_psnr']:.4f} | ΔSSIM {metrics_1080p['delta_ssim']:.6f} | VRAM ≈ {metrics_1080p['estimated_vram_gb']:.2f} GB")
    print(f"4K    -> {metrics_4k['fps']:.2f} FPS | CPU {metrics_4k['cpu_percent']:.1f}% | ΔPSNR {metrics_4k['delta_psnr']:.4f} | ΔSSIM {metrics_4k['delta_ssim']:.6f} | VRAM ≈ {metrics_4k['estimated_vram_gb']:.2f} GB")

    meets_1080p = metrics_1080p["fps"] >= args.target_1080p
    meets_4k = metrics_4k["fps"] >= args.target_4k
    meets_vram = max(metrics_1080p["estimated_vram_gb"], metrics_4k["estimated_vram_gb"]) <= 16
    meets_psnr = metrics_1080p["delta_psnr"] >= 80 and metrics_4k["delta_psnr"] >= 80
    meets_ssim = metrics_1080p["delta_ssim"] >= 0.998 and metrics_4k["delta_ssim"] >= 0.998

    if meets_1080p and meets_4k and meets_vram and meets_psnr and meets_ssim:
        print("All optimisation targets satisfied.")
    else:
        print("WARNING: One or more optimisation targets were not met.")
        if not meets_1080p:
            print(f" - 1080p FPS below target ({metrics_1080p['fps']:.2f} < {args.target_1080p})")
        if not meets_4k:
            print(f" - 4K FPS below target ({metrics_4k['fps']:.2f} < {args.target_4k})")
        if not meets_vram:
            print(" - Estimated VRAM usage exceeds 16 GB")
        if not meets_psnr:
            print(" - ΔPSNR target not satisfied (requires >= 80 dB compared to CPU baseline)")
        if not meets_ssim:
            print(" - ΔSSIM target not satisfied (requires >= 0.998 compared to CPU baseline)")


if __name__ == "__main__":  # pragma: no cover
    main()
