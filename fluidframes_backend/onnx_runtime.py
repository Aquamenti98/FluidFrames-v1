"""ONNX Runtime backend optimized for DirectML and CPU execution."""
from __future__ import annotations

import argparse
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort


@dataclass(slots=True)
class BackendRuntimeConfig:
    """Configuration parameters for the ONNX Runtime backend."""

    provider: str = "dml"
    fp16: bool = False
    tile_size: Optional[Tuple[int, int]] = None
    batch_size: int = 1
    queue_depth: int = 2
    prefetch: int = 2

    @staticmethod
    def _normalize_tile_size(value: Optional[Sequence[int] | str | int]) -> Optional[Tuple[int, int]]:
        if value in (None, "", 0):
            return None
        if isinstance(value, str):
            if "x" in value:
                height, width = value.lower().split("x", maxsplit=1)
                return int(height), int(width)
            value = int(value)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            seq = list(value)
            if not seq:
                return None
            if len(seq) == 1:
                size = int(seq[0])
                return size, size
            return int(seq[0]), int(seq[1])
        size = int(value)  # type: ignore[arg-type]
        return size, size

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BackendRuntimeConfig":
        tile = getattr(args, "tilesize", None)
        if isinstance(tile, str) and tile.startswith("["):
            # argparse can keep nargs sequences as strings when passed via env vars
            tile = tile.strip("[]").replace(" ", "")
            if tile:
                parts = tile.split(",")
                tile = tuple(int(part) for part in parts if part)
        return cls(
            provider=getattr(args, "provider", "dml").lower(),
            fp16=getattr(args, "fp16", False),
            tile_size=cls._normalize_tile_size(tile),
            batch_size=max(1, int(getattr(args, "batch", 1))),
            queue_depth=max(1, int(getattr(args, "queue_depth", 2))),
            prefetch=max(1, int(getattr(args, "prefetch", 2))),
        )


def add_backend_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register common backend command-line flags on *parser*."""

    parser.add_argument(
        "--provider",
        choices=["dml", "cpu"],
        default="dml",
        help="Execution provider to use (DirectML or CPU).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable float16 execution when supported by the provider.",
    )
    parser.add_argument(
        "--tilesize",
        nargs="*",
        type=int,
        help="Optional tiling size as a single value or HEIGHT WIDTH pair.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for inference requests.",
    )
    parser.add_argument(
        "--queue-depth",
        dest="queue_depth",
        type=int,
        default=2,
        help="Number of in-flight inference requests when pipelining.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Number of prefetched batches queued ahead of consumption.",
    )
    return parser


class OnnxRuntimeBackend:
    """Utility wrapper around :mod:`onnxruntime` with DirectML optimisations."""

    def __init__(
        self,
        model_path: str,
        config: Optional[BackendRuntimeConfig] = None,
        *,
        device_id: Optional[int | str] = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or BackendRuntimeConfig()
        self._device_id = int(device_id) if device_id not in (None, "auto", "Auto") else None

        self._provider = self.config.provider.lower()
        if self._provider not in ("dml", "cpu"):
            raise ValueError(f"Unsupported provider '{self.config.provider}'")

        self._session = self._create_session()
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Float16 is only supported on DirectML; gracefully downgrade if unavailable
        available_providers = {provider.lower() for provider in self._session.get_providers()}
        self._use_fp16 = self.config.fp16 and "dmlexecutionprovider" in available_providers
        if self.config.fp16 and not self._use_fp16:
            warnings.warn("FP16 requested but not supported by the selected provider; falling back to FP32.")

        self._input_dtype = np.float16 if self._use_fp16 else np.float32
        self._output_dtype = np.float16 if self._use_fp16 else np.float32
        self._device_type = "dml" if self._provider == "dml" else "cpu"

    @property
    def session(self) -> ort.InferenceSession:
        return self._session

    def _create_session(self) -> ort.InferenceSession:
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_cpu_mem_arena = True
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers, provider_options = self._resolve_providers()

        return ort.InferenceSession(
            path_or_bytes=self.model_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

    def _resolve_providers(self) -> Tuple[Sequence[str], Sequence[dict[str, str]]]:
        if self._provider == "cpu":
            return ["CPUExecutionProvider"], [{}]

        options: dict[str, str] = {}
        if self._device_id is not None:
            options["device_id"] = str(self._device_id)
        else:
            options["performance_preference"] = "high_performance"
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        return providers, [options, {}]

    def run(self, batch: np.ndarray) -> np.ndarray:
        """Run inference synchronously for *batch* (NCHW)."""

        if batch.ndim != 4:
            raise ValueError(f"Expected input with 4 dimensions (NCHW), received shape {batch.shape}.")

        if self.config.tile_size:
            return self._run_tiled(batch)

        if self.config.batch_size > 1 and batch.shape[0] > self.config.batch_size:
            outputs: list[np.ndarray] = []
            for start in range(0, batch.shape[0], self.config.batch_size):
                chunk = batch[start : start + self.config.batch_size]
                outputs.append(self._execute(chunk))
            return np.concatenate(outputs, axis=0)

        return self._execute(batch)

    def pipeline(self, batches: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        """Execute *batches* asynchronously, yielding results in order."""

        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=self.config.queue_depth)
        pending: deque = deque()

        def submit(batch: np.ndarray) -> None:
            pending.append(executor.submit(self.run, batch))

        iterator = iter(batches)
        for _ in range(self.config.prefetch):
            try:
                submit(next(iterator))
            except StopIteration:
                break

        for batch in iterator:
            submit(batch)
            yield pending.popleft().result()

        while pending:
            yield pending.popleft().result()

        executor.shutdown(wait=True)

    # Internal helpers -------------------------------------------------

    def _execute(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=self._input_dtype)

        if self._provider == "cpu":
            output = self._session.run([self._output_name], {self._input_name: batch})[0]
        else:
            io_binding = self._session.io_binding()
            ort_input = ort.OrtValue.ortvalue_from_numpy(
                batch,
                device_type=self._device_type,
                device_id=self._device_id or 0,
            )
            io_binding.bind_ortvalue_input(self._input_name, ort_input)
            io_binding.bind_output(self._output_name, self._device_type)
            self._session.run_with_iobinding(io_binding)
            output = io_binding.get_outputs()[0].numpy()

        return np.asarray(output, dtype=np.float32 if self._output_dtype == np.float16 else self._output_dtype)

    def _run_tiled(self, batch: np.ndarray) -> np.ndarray:
        if batch.shape[0] != 1:
            raise ValueError("Tiled execution currently supports batch size of 1.")

        tile_h, tile_w = self.config.tile_size or (batch.shape[2], batch.shape[3])
        _, _, height, width = batch.shape

        pad_h = (tile_h - height % tile_h) % tile_h
        pad_w = (tile_w - width % tile_w) % tile_w

        padded = np.pad(
            batch,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="edge",
        )
        padded_h, padded_w = padded.shape[2], padded.shape[3]

        tiles: list[Tuple[int, int, np.ndarray]] = []
        for y in range(0, padded_h, tile_h):
            for x in range(0, padded_w, tile_w):
                tiles.append((y, x, padded[:, :, y : y + tile_h, x : x + tile_w]))

        outputs: list[np.ndarray] = []
        for start in range(0, len(tiles), self.config.batch_size):
            chunk = tiles[start : start + self.config.batch_size]
            chunk_input = np.concatenate([tile for (_, _, tile) in chunk], axis=0)
            chunk_output = self._execute(chunk_input)
            outputs.extend(chunk_output)

        first_tile = outputs[0]
        channel_count = first_tile.shape[0] if first_tile.ndim == 3 else first_tile.shape[1]
        assembled = np.empty((1, channel_count, padded_h, padded_w), dtype=np.float32)

        for (y, x, _), tile_output in zip(tiles, outputs):
            if tile_output.ndim == 3:
                tile_output = np.expand_dims(tile_output, axis=0)
            assembled[:, :, y : y + tile_h, x : x + tile_w] = tile_output

        return assembled[:, :, :height, :width]


def create_backend_from_cli(
    model_path: str,
    args: argparse.Namespace,
    *,
    device_id: Optional[int | str] = None,
) -> OnnxRuntimeBackend:
    """Factory that parses CLI *args* and returns an initialised backend."""

    config = BackendRuntimeConfig.from_cli_args(args)
    return OnnxRuntimeBackend(model_path, config, device_id=device_id)
