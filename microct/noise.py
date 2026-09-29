"""Residual texture estimation and mixed two-dimensional training noise.

Arrays use z, y, x order for volumes and batch, y, x order for batches.
The residual describes reconstructed-image texture, including segmentation
and image-model error; it does not isolate detector noise.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import scipy.ndimage as ndi


DEFAULT_CONFIG = {
    "fine_factor": 3,
    "psf_voxels": 0.854,
    "background": 45820.0,
    "contrast": 2887.0,
    "kernel_size": 64,
    "polynomial_stride": 8,
    "carbide_label": 3,
    "matrix_label": 2,
    "excluded_labels": (1, 4),
    "matrix_erosion": 3,
    "exclusion_dilation": 4,
    "max_lag": 8,
    "minimum_pairs": 500,
}


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _config(config):
    result = DEFAULT_CONFIG.copy()
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        result.update(config)
    for name in ("fine_factor", "kernel_size", "polynomial_stride", "matrix_erosion", "exclusion_dilation", "max_lag", "minimum_pairs"):
        result[name] = _positive_integer(result[name], name)
    for name in ("psf_voxels", "background", "contrast"):
        value = result[name]
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        result[name] = float(value)
    if result["psf_voxels"] <= 0 or result["contrast"] <= 0:
        raise ValueError("psf_voxels and contrast must be positive")
    for name in ("carbide_label", "matrix_label"):
        result[name] = _positive_integer(result[name], name)
    labels = tuple(result["excluded_labels"])
    if not labels:
        raise ValueError("excluded_labels must not be empty")
    result["excluded_labels"] = tuple(_positive_integer(v, "excluded label") for v in labels)
    return result


def _shape(shape, dimensions):
    if not isinstance(shape, Sequence) or len(shape) != dimensions:
        raise ValueError(f"shape must contain {dimensions} dimensions")
    return tuple(_positive_integer(s, "dimension") for s in shape)


def _finite_array(array, dimensions, name):
    array = np.asarray(array)
    if array.ndim != dimensions or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty {dimensions}-dimensional array")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    return array


def flatten_quadratic(volume, stride=8):
    """Subtract a least-squares quadratic intensity surface from each slice.

The fit samples every ``stride`` pixels in both in-plane directions. The
input must be floating point; its dtype and the slice dimensions are kept.
"""
    raw = _finite_array(volume, 3, "volume")
    if raw.dtype.kind != "f":
        raise ValueError("volume must have a floating-point dtype")
    passo = _positive_integer(stride, "stride")
    nz, ny, nx = raw.shape
    if len(range(0, ny, passo)) < 3 or len(range(0, nx, passo)) < 3:
        raise ValueError("each slice must provide at least three sampled positions per axis")
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    ys, xs = yy[::passo, ::passo].ravel(), xx[::passo, ::passo].ravel()
    A = np.stack([np.ones_like(ys), ys, xs, ys * ys, xs * xs, ys * xs], 1)
    Af = np.stack([np.ones(ny * nx, np.float32), yy.ravel(), xx.ravel(),
                   (yy * yy).ravel(), (xx * xx).ravel(), (yy * xx).ravel()], 1)
    out = np.empty_like(raw)
    for k in range(nz):
        c, *_ = np.linalg.lstsq(A, raw[k, ::passo, ::passo].ravel(), rcond=None)
        out[k] = raw[k] - (Af @ c).reshape(ny, nx)
    return out


def image_model_3d(occupancy, config=None):
    """Map carbide occupancy to gray levels with the separable blur model.

The one-dimensional kernel repeats the impulse on the fine grid, applies a
Gaussian filter with constant boundaries, and averages back to the voxel
grid. Three convolutions then use nearest-value boundaries.
"""
    cfg = _config(config)
    u_ct = _finite_array(occupancy, 3, "occupancy")
    if np.any((u_ct < 0) | (u_ct > 1)):
        raise ValueError("occupancy values must lie between zero and one")
    n = cfg["kernel_size"]
    factor = cfg["fine_factor"]
    imp = np.zeros(n, np.float64)
    imp[n // 2] = 1.0
    b = ndi.gaussian_filter1d(np.repeat(imp, factor), cfg["psf_voxels"] * factor,
                              mode="constant")
    kernel = (b.reshape(n, factor).mean(1)).astype(np.float32)
    x = np.asarray(u_ct, np.float32)
    for axis in range(3):
        x = ndi.convolve1d(x, kernel, axis=axis, mode="nearest")
    return cfg["background"] - cfg["contrast"] * x


def estimate_sqrt_spectrum(volume, labels, *, config=None,
                           crop_origin=(0, 0, 480), crop_shape=(160, 160, 160)):
    """Estimate the square-root spectrum from a preselected slab of slices.

Select the z slab before calling this function. Flattening, image formation
and morphology operate on the entire supplied slab before the spatial crop.
For the documented extraction, supply slices 1600 through 1759 of the
reconstructed array, with matching labels; the default crop uses all 160
slices, rows 0 through 159 and columns 480 through 639. Do not supply only
the already cropped cube, because that changes the fitted intensity surface
and the boundary conditions. Label meanings and numerical settings are
explicit in ``DEFAULT_CONFIG`` and may be supplied in ``config``.

The returned nonnegative array can be stored as an ordinary NumPy ``.npy``
file and passed to ``ResidualNoise`` without a dependency on source data.
"""
    cfg = _config(config)
    raw = _finite_array(volume, 3, "volume").astype(np.float32)
    seg = _finite_array(labels, 3, "labels")
    if seg.dtype.kind not in "iu" or seg.shape != raw.shape:
        raise ValueError("labels must be an integer array with the same shape as volume")
    shape = _shape(crop_shape, 3)
    if len(set(shape)) != 1:
        raise ValueError("crop_shape must be cubic")
    if not isinstance(crop_origin, Sequence) or len(crop_origin) != 3:
        raise ValueError("crop_origin must contain three nonnegative integers")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in crop_origin):
        raise ValueError("crop_origin must contain three nonnegative integers")
    if any(start + size > bound for start, size, bound in zip(crop_origin, shape, raw.shape)):
        raise ValueError("crop must lie inside the supplied slab")
    lag = cfg["max_lag"]
    if any(size < 2 * lag + 1 for size in shape):
        raise ValueError("crop dimensions must accommodate all covariance lags")

    carbide = (seg == cfg["carbide_label"]).astype(np.float32)
    modeled = image_model_3d(carbide, cfg)
    residual = (flatten_quadratic(raw, cfg["polynomial_stride"])
                - flatten_quadratic(modeled, cfg["polynomial_stride"]))
    structure = ndi.generate_binary_structure(3, 1)
    matrix = ndi.binary_erosion(seg == cfg["matrix_label"], structure,
                                iterations=cfg["matrix_erosion"])
    excluded = ndi.binary_dilation(np.isin(seg, cfg["excluded_labels"]), structure,
                                     iterations=cfg["exclusion_dilation"])
    mask = matrix & ~excluded
    crop = tuple(slice(start, start + size) for start, size in zip(crop_origin, shape))
    res_b, mask_b = residual[crop], mask[crop]
    if not mask_b.any():
        raise ValueError("the crop has no selected matrix voxels")

    d = res_b.astype(np.float64)
    d = d - d[mask_b].mean()
    dm = np.where(mask_b, d, 0.0)
    mk = mask_b.astype(np.float64)
    F_d, F_m = np.fft.fftn(dm), np.fft.fftn(mk)
    prod = np.real(np.fft.ifftn(F_d * np.conj(F_d)))
    cont = np.real(np.fft.ifftn(F_m * np.conj(F_m)))
    cov = np.zeros((2 * lag + 1,) * 3)
    for i, dz in enumerate(range(-lag, lag + 1)):
        for j, dy in enumerate(range(-lag, lag + 1)):
            for k, dx in enumerate(range(-lag, lag + 1)):
                count = cont[dz, dy, dx]
                cov[i, j, k] = prod[dz, dy, dx] / count if count > cfg["minimum_pairs"] else np.nan
    if not np.isfinite(cov).all():
        raise ValueError("a covariance lag has insufficient valid matrix pairs")
    r = np.sqrt(sum(np.subtract(np.indices(cov.shape)[axis], lag) ** 2 for axis in range(3)))
    taper = np.clip(0.5 * (1 + np.cos(np.pi * r / (lag + 1))), 0, None)
    kernel = cov * taper
    embedded = np.zeros(shape)
    for i, dz in enumerate(range(-lag, lag + 1)):
        for j, dy in enumerate(range(-lag, lag + 1)):
            for k, dx in enumerate(range(-lag, lag + 1)):
                embedded[dz % shape[0], dy % shape[1], dx % shape[2]] = kernel[i, j, k]
    spectrum = np.real(np.fft.fftn(embedded))
    return np.sqrt(np.clip(spectrum, 0, None))


class ResidualNoise:
    """Generate two-dimensional Gaussian texture from a residual spectrum."""

    def __init__(self, sqrt_spectrum):
        root = _finite_array(sqrt_spectrum, 3, "sqrt_spectrum")
        if len(set(root.shape)) != 1 or root.shape[0] < 4:
            raise ValueError("sqrt_spectrum must be cubic with a side length of at least four")
        if np.any(root < 0) or not np.any(root > 0):
            raise ValueError("sqrt_spectrum must be nonnegative and have positive energy")
        covariance = np.real(np.fft.ifftn(root.astype(np.complex128) ** 2))
        self._covariance = covariance / covariance[0, 0, 0]
        self._filters = {}

    def sample(self, shape, rng, std=1880.0):
        """Filter fresh white noise using the covariance plane at zero z lag."""
        shape = _shape(shape, 2)
        if min(shape) < 4:
            raise ValueError("sample dimensions must be at least four")
        if not np.isscalar(std) or not np.isfinite(std) or std <= 0:
            raise ValueError("std must be finite and positive")
        if shape not in self._filters:
            ac = self._covariance[0]
            side = ac.shape[0]
            embedded = np.zeros(shape, dtype=np.float64)
            lags = [min(side // 2 - 1, size // 2 - 1) for size in shape]
            source = tuple(np.r_[0:lag + 1, side - lag:side] for lag in lags)
            target = tuple(np.r_[0:lag + 1, size - lag:size] for lag, size in zip(lags, shape))
            embedded[np.ix_(*target)] = ac[np.ix_(*source)]
            spectrum = np.real(np.fft.fftn(embedded))
            spectrum[spectrum < 0] = 0.0
            self._filters[shape] = np.sqrt(spectrum)
        out = np.real(np.fft.ifftn(np.fft.fftn(rng.normal(0, 1, shape)) * self._filters[shape]))
        scale = out.std()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("the spectrum produced a texture with zero or invalid variance")
        return (out * (std / scale)).astype(np.float32)


def _range(values, name):
    if not isinstance(values, Sequence) or len(values) != 2:
        raise ValueError(f"{name} must contain two endpoints")
    low, high = values
    if not np.isfinite([low, high]).all() or low <= 0 or high < low:
        raise ValueError(f"{name} must have positive, ordered, finite endpoints")
    return low, high


def mixed_unit_noise(shape, rng, residual_noise, gaussian_width=(1.0, 1.4)):
    """Alternate residual-based and Gaussian-filtered texture in an even batch."""
    shape = _shape(shape, 3)
    if shape[0] % 2 or min(shape[1:]) < 4:
        raise ValueError("a mixed batch must have even size and image dimensions of at least four")
    low, high = _range(gaussian_width, "gaussian_width")
    n = np.stack([
        residual_noise.sample(shape[1:], rng, std=1.0) if i % 2 == 0
        else ndi.gaussian_filter(rng.normal(0, 1, shape[1:]).astype(np.float32),
                                  float(rng.uniform(low, high)), mode="wrap")
        for i in range(shape[0])
    ]).astype(np.float32)
    return n / (n.std(axis=(1, 2), keepdims=True) + 1e-6)


def mixed_noise(shape, rng, residual_noise, std_range=(1500.0, 2300.0),
                gaussian_width=(1.0, 1.4)):
    """Return additive mixed texture, drawing amplitudes before texture samples.

This is the noise-addition stage. Any subsequent image gain augmentation is
applied by the caller and also scales the added texture.
"""
    shape = _shape(shape, 3)
    low, high = _range(std_range, "std_range")
    levels = rng.uniform(low, high, shape[0]).astype(np.float32)
    return mixed_unit_noise(shape, rng, residual_noise, gaussian_width) * levels[:, None, None]
