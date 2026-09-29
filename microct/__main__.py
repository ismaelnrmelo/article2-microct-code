"""Command-line entry points for the supplied final functions."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def require_cuda(runtime=None):
    if runtime is None:
        try:
            import torch as runtime
        except ImportError as exc:
            raise RuntimeError("Install PyTorch with CUDA support before neural inference.") from exc
    if not runtime.cuda.is_available():
        raise RuntimeError("CUDA is required for the accepted neural inference path; CPU inference is not supported.")
    if not runtime.cuda.is_bf16_supported():
        raise RuntimeError("The accepted neural inference path requires CUDA bfloat16 support.")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Final residual-texture estimation and TRSA-U-Net triplanar inference.")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer", help="Segment numbered TIFF slices using the fixed accepted settings.")
    infer.add_argument("--input-directory", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--checkpoint", type=Path, default=ROOT / "assets/accepted_weights.pt")
    infer.add_argument("--settings", type=Path, default=ROOT / "config/final_settings.json")
    noise = sub.add_parser("estimate-noise", help="Estimate residual spectrum from a supplied grayscale slab and matching labels.")
    noise.add_argument("--volume-npy", required=True, type=Path)
    noise.add_argument("--labels-npy", required=True, type=Path)
    noise.add_argument("--output", required=True, type=Path)
    noise.add_argument("--settings", type=Path, default=ROOT / "config/final_settings.json")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("The output already exists; choose a new output path.")
        cfg = json.loads(args.settings.read_text(encoding="utf-8"))
        if args.command == "infer":
            require_cuda()
            if not args.input_directory.is_dir():
                raise ValueError("The input directory does not exist.")
            from .inference import InferenceConfig, segment_tiff_volume
            segment_tiff_volume(args.input_directory, args.output, args.checkpoint,
                                config=InferenceConfig(**cfg["inference"]))
        else:
            import numpy as np
            from .noise import estimate_sqrt_spectrum
            volume = np.load(args.volume_npy, allow_pickle=False)
            labels = np.load(args.labels_npy, allow_pickle=False)
            spectrum = estimate_sqrt_spectrum(volume, labels,
                config=cfg["residual_estimator"],
                crop_origin=cfg["residual_crop_origin_zyx"],
                crop_shape=cfg["residual_crop_shape_zyx"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                np.save(stream, spectrum, allow_pickle=False)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()
