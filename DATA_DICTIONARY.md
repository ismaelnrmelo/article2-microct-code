# Data dictionary

This dictionary describes the arrays and files consumed or produced by the functions distributed with TRSA-U-Net version 1.0.1. Array coordinates are zero-based. Volumes use `(z, y, x)` order; image batches use `(batch, y, x)` unless a channel axis is stated.

## Inputs and outputs

| Item | Format, shape and values | Meaning and units |
| --- | --- | --- |
| Reconstructed slices | Numbered two-dimensional TIFF files, `0001.tif` onward | Aligned grayscale reconstruction in its original intensity scale. Slice number equals zero-based z index plus one. Values are converted to float32 during processing. |
| Inference crop | Default `(1958, 480, 640)`, with in-plane origin `(372, 51)` | Crop of the numbered slices. Dimensions and offsets are in voxels. |
| Network input | float32 tensor `(batch, 1, 96, 96)` before CUDA autocast | Dimensionless intensity transform `2*((45820-image)/2887-0.30599761684099036)`. The supplied inference path passes a zero conditioning value per patch. |
| Network output | Tensor `(batch, 1, 96, 96)` | Chromium-carbide logits. Inference applies sigmoid, blends overlapping patches and averages three orthogonal orientations. |
| Carbide probability | float32 volume, same shape as the processed slab | Dimensionless value in `[0, 1]`. The accepted carbide decision uses `>0.520098865032196`. |
| Segmented output | BigTIFF, uint8, default `(1958, 480, 640)` | Phase labels defined below. The writer does not encode physical voxel spacing. |
| Residual-estimation volume | NumPy `.npy`, finite real array, `(160, height, width)` for the documented extraction | Reconstructed grayscale slab from zero-based z indices `[1600, 1760)`, retaining its full in-plane extent. Converted to float32 internally. |
| Residual-estimation labels | NumPy `.npy`, integer array with exactly the same shape and alignment as the volume | Segmented support with the phase labels below. The original support used for the stored spectrum is not included. |
| Stored or estimated square-root spectrum | NumPy `.npy`, float64, nonnegative, `(160, 160, 160)` by default | Square root of the clipped discrete spectrum of the masked residual covariance. Covariance has squared gray-level units; its square-root spectrum has gray-level units under the implemented FFT convention. It is not a calibrated continuous spectral density. |
| `ResidualNoise.sample` output | float32 array `(height, width)` | Synthetic additive texture in gray levels, rescaled to the requested standard deviation. The default is 1880 gray levels. |
| `mixed_unit_noise` output | float32 array `(even batch, height, width)` | Approximately unit-standard-deviation texture. Alternates residual-derived texture and Gaussian-filtered white noise. The default Gaussian width is drawn between 1.0 and 1.4 pixels. |
| `mixed_noise` output | float32 array `(even batch, height, width)` | Additive texture in gray levels. Per-image amplitude is drawn uniformly from 1500 to 2300 by default. Later image gain, if any, is the caller's responsibility. |

NumPy files are read with `allow_pickle=False`. Random sampling functions accept a caller-supplied NumPy random generator.

## Phase labels

| Value | Phase |
| --- | --- |
| 1 | Pore |
| 2 | Matrix |
| 3 | Chromium carbide |
| 4 | NbC |

Inference first assigns matrix, then chromium carbide, then NbC, then pore. A later assignment takes precedence when masks overlap. NbC and pore masks use intensity thresholds and connected-component filtering, not separate neural output classes.

## Residual settings

`config/final_settings.json` is the machine-readable source for all fixed settings. The residual estimator forms a grayscale image from the chromium-carbide support, detrends both volumes slice by slice, and subtracts the modeled image from the reconstruction. It measures masked covariance in matrix voxels after excluding pore and NbC neighborhoods.

- `psf_voxels` is the Gaussian blur width in reconstructed voxels; `fine_factor` is the internal integer refinement factor.
- `background`, `contrast`, and intensity thresholds are reconstruction gray levels, not physical attenuation coefficients.
- `matrix_erosion` and `exclusion_dilation` count face-connected morphology iterations.
- `max_lag` is a displacement in voxels; `minimum_pairs` is the minimum accepted count criterion for covariance pairs.
- `residual_crop_origin_zyx = [0, 0, 480]` and `residual_crop_shape_zyx = [160, 160, 160]` refer to the preselected slab. The spatial crop follows all full-slab preprocessing.

Lengths are expressed in pixels or voxels within this package. Physical distances require the acquisition calibration. The residual can include image-model and segmentation error as well as reconstruction texture.

## Accepted assets

`assets/accepted_weights.pt` is a PyTorch state dictionary with 175 tensors for `UNet4(base=64)`. The inference loader uses strict state-dictionary matching and verifies the file hash before loading. `assets/accepted_sqrt_spectrum.npy` is the stored residual spectrum. Exact hashes and file sizes are listed in `MANIFEST_SHA256.json`.

Raw reconstructions, optical images, segmented extraction support, training datasets and benchmark predictions are not distributed.
