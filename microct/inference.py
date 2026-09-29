"""Fixed-setting triplanar segmentation with the accepted network weights.

The neural inference path requires CUDA and bfloat16 autocast. CPU numerical
equivalence and complete-volume reproduction have not been established.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile

from .noise import flatten_quadratic


ACCEPTED_CHECKPOINT_SHA256 = (
    "346652368d8fdf3398202a6a28b4e30dbad7c4e625bc0e4a52318e1986f5f0fc"
)


@dataclass(frozen=True)
class InferenceConfig:
    background: float = 45820.0
    contrast: float = 2887.0
    training_anchor: float = 0.30599761684099036
    patch: int = 96
    batch: int = 16
    slab_depth: int = 256
    margin: int = 32
    probability_threshold: float = 0.520098865032196
    nbc_threshold: float = 49111.34765625
    nbc_sigma: float = 1.4
    nbc_min_voxels: int = 8
    pore_sigma: float = 0.8
    pore_threshold: float = 38000.0
    pore_min_voxels: int = 3
    volume_depth: int = 1958
    crop_y: int = 372
    crop_x: int = 51
    crop_height: int = 480
    crop_width: int = 640


def normalize(image, config=InferenceConfig()):
    """Apply the input transform used by the accepted network."""
    return ((config.background - image) / config.contrast - config.training_anchor) * 2.0


def prepare_slab(raw):
    """Subtract the slice-wise polynomial fit and restore the raw block mean."""
    raw = np.asarray(raw, dtype=np.float32)
    return flatten_quadratic(raw, stride=8) + float(raw.mean())


def load_model(checkpoint, device="cuda"):
    """Check the accepted file digest and load its state dictionary strictly."""
    import torch
    from .model import UNet4

    checkpoint = Path(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ACCEPTED_CHECKPOINT_SHA256:
        raise ValueError("The checkpoint does not match the accepted file SHA-256.")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the accepted neural inference path.")
    network = UNet4(base=64).to(target)
    state = torch.load(checkpoint, map_location=target, weights_only=True)
    network.load_state_dict(state, strict=True)
    network.eval()
    return network


def infer_plane(network, image, config=InferenceConfig()):
    """Predict a plane with 50% tile overlap and a positive Hann window."""
    import torch

    device = next(network.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("CPU inference has not been implemented or verified for this path.")
    patch = config.patch
    x = normalize(image, config).astype(np.float32)
    if x.ndim != 2 or min(x.shape) < patch:
        raise ValueError("Each plane dimension must be at least the configured patch size.")
    height, width = x.shape

    def positions(length):
        indices = list(range(0, max(1, length - patch + 1), patch // 2))
        if indices[-1] != length - patch:
            indices.append(max(0, length - patch))
        return indices

    window_1d = np.hanning(patch + 2)[1:-1]
    window = (window_1d[:, None] * window_1d[None, :]).astype(np.float32) + 1e-4
    accumulated = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    tensor = torch.tensor(x, device=device)
    tiles = [(yy, xx) for yy in positions(height) for xx in positions(width)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(tiles), config.batch):
            batch = tiles[start:start + config.batch]
            patches = torch.stack(
                [tensor[yy:yy + patch, xx:xx + patch] for yy, xx in batch]
            )[:, None]
            probability = torch.sigmoid(
                network(patches, torch.zeros(len(batch), device=device)).float()
            )[:, 0].cpu().numpy()
            for (yy, xx), prediction in zip(batch, probability):
                accumulated[yy:yy + patch, xx:xx + patch] += prediction * window
                weight[yy:yy + patch, xx:xx + patch] += window
    return accumulated / np.maximum(weight, 1e-6)


def probabilities_triplanar(network, volume, config=InferenceConfig()):
    """Average probabilities from the three orthogonal plane orientations."""
    probability = np.zeros(volume.shape, np.float32)
    for axis in range(3):
        planes = np.moveaxis(volume, axis, 0)
        accumulated = np.empty(planes.shape, np.float32)
        for index in range(planes.shape[0]):
            accumulated[index] = infer_plane(
                network, np.ascontiguousarray(planes[index]), config
            )
        probability += np.moveaxis(accumulated, 0, axis)
    return probability / 3.0


def phase_masks(raw, flattened, config=InferenceConfig()):
    """Obtain NbC and pore masks with face-connected component filtering."""
    nbc = ndi.gaussian_filter(flattened, config.nbc_sigma) > config.nbc_threshold
    labels, count = ndi.label(nbc)
    if count:
        sizes = np.bincount(labels.ravel())
        nbc &= np.isin(labels, np.flatnonzero(sizes >= config.nbc_min_voxels))
    pores = ndi.gaussian_filter(raw, config.pore_sigma) < config.pore_threshold
    labels, count = ndi.label(pores)
    if count:
        sizes = np.bincount(labels.ravel())
        pores &= np.isin(labels, np.flatnonzero(sizes >= config.pore_min_voxels))
    return nbc, pores


def assign_labels(carbide, nbc, pores):
    """Assign matrix=2, chromium carbide=3, NbC=4 and pore=1, in that order."""
    labels = np.full(carbide.shape, 2, np.uint8)
    labels[carbide] = 3
    labels[nbc] = 4
    labels[pores] = 1
    return labels


def iter_segmented_slabs(network, read_block, depth, config=InferenceConfig()):
    """Yield non-overlapping output slabs after inference with context margins.

    read_block(start, stop) supplies the cropped raw volume in z, y, x order.
    Only one context-extended slab and its prediction arrays are materialized.
    """
    if depth <= 0 or config.slab_depth <= 0 or config.margin < 0:
        raise ValueError("Depth and slab size must be positive; margin must be non-negative.")
    for start in range(0, depth, config.slab_depth):
        read_start = max(0, start - config.margin)
        read_stop = min(depth, start + config.slab_depth + config.margin)
        raw = np.asarray(read_block(read_start, read_stop), dtype=np.float32)
        if raw.ndim != 3 or raw.shape[0] != read_stop - read_start:
            raise ValueError("The block reader returned an unexpected volume shape.")
        flattened = prepare_slab(raw)
        probability = probabilities_triplanar(network, flattened, config)
        carbide = probability > config.probability_threshold
        nbc, pores = phase_masks(raw, flattened, config)
        labels = assign_labels(carbide, nbc, pores)
        first = start - read_start
        last = first + min(config.slab_depth, depth - start)
        yield start, labels[first:last].copy()


def segment_tiff_volume(input_directory, output, checkpoint, *, config=InferenceConfig(), device="cuda"):
    """Write a new BigTIFF from numbered input TIFF slices using fixed settings."""
    input_directory = Path(input_directory)
    output = Path(output)
    if output.exists():
        raise FileExistsError("The output already exists; choose a new output path.")
    network = load_model(checkpoint, device=device)
    expected_shape = (config.crop_height, config.crop_width)

    def read_block(start, stop):
        planes = []
        for index in range(start, stop):
            image = tifffile.imread(input_directory / f"{index + 1:04d}.tif")
            image = image[
                config.crop_y:config.crop_y + config.crop_height,
                config.crop_x:config.crop_x + config.crop_width,
            ]
            if image.shape != expected_shape:
                raise ValueError(f"Input slice {index + 1:04d} does not contain the configured crop.")
            planes.append(image)
        return np.stack(planes).astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    mapped = tifffile.memmap(
        output,
        shape=(config.volume_depth, config.crop_height, config.crop_width),
        dtype=np.uint8,
        bigtiff=True,
    )
    try:
        for start, labels in iter_segmented_slabs(network, read_block, config.volume_depth, config):
            mapped[start:start + len(labels)] = labels
            mapped.flush()
    finally:
        mapped.flush()
        del mapped
    return output
