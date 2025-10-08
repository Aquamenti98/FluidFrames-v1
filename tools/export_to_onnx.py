"""Utility script to export a PyTorch RIFE checkpoint to ONNX."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a RIFE model checkpoint to ONNX.")
    parser.add_argument("checkpoint", type=Path, help="Path to the PyTorch checkpoint file.")
    parser.add_argument("output", type=Path, help="Destination ONNX file path.")
    parser.add_argument(
        "--module",
        default="inference_rife",
        help="Module that exposes the model class (default: inference_rife).",
    )
    parser.add_argument(
        "--class-name",
        dest="class_name",
        default="Model",
        help="Class name implementing the network to export (default: Model).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Input tensor height used for tracing (default: 1080).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Input tensor width used for tracing (default: 1920).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version used for export (default: 17).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export the model weights using float16 precision.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Working directory appended to sys.path so custom modules can be discovered.",
    )
    return parser.parse_args()


def load_model(module_name: str, class_name: str, checkpoint_path: Path) -> torch.nn.Module:
    if str(module_name) not in ("", None):
        module = importlib.import_module(module_name)
    else:
        raise ValueError("A module name must be provided to resolve the model class.")

    try:
        model_class: type[torch.nn.Module] = getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Unable to resolve class '{class_name}' in module '{module_name}'.") from exc

    model = model_class()
    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    if args.workspace and args.workspace.exists():
        import sys

        sys.path.insert(0, str(args.workspace))

    model = load_model(args.module, args.class_name, args.checkpoint)

    dummy = torch.randn(1, 6, args.height, args.width)
    if args.fp16:
        model = model.half()
        dummy = dummy.half()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}

    torch.onnx.export(
        model,
        dummy,
        args.output.as_posix(),
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ONNX model saved to {args.output.as_posix()}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
