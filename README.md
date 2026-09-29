# TRSA-U-Net micro-CT code

Version 1.0.1 accompanies Article 2. It contains the accepted network, fixed-setting triplanar inference, the residual-texture estimator and sampling functions, accepted weights, and the stored square-root residual spectrum. The scientific modules and numerical settings are unchanged from version 1.0.0.

## Contents

- `microct/model.py` defines TRSA-U-Net, with 4,737,729 trainable parameters.
- `microct/inference.py` implements preprocessing, overlapping tiles, three-orientation averaging, phase labels and slab-wise BigTIFF output.
- `microct/noise.py` estimates a segmentation-conditioned residual spectrum and samples additive two-dimensional texture.
- `config/final_settings.json` records the numerical settings and relative asset paths.
- `assets/accepted_weights.pt` contains the accepted checkpoint. Its SHA-256 is checked before loading.
- `assets/accepted_sqrt_spectrum.npy` contains the stored spectrum used by the noise sampler. Neural inference does not require it.
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md) defines arrays, labels, units and output formats.

The package contains no training program, optical benchmark program, raw images or original segmented extraction support. It does not reproduce training or the complete optical comparison.

## Installation

Use Python 3.10. Install the CPU dependencies for residual estimation, texture sampling and the supplied tests from the package directory.

```text
python -m pip install -r requirements.txt
```

Neural inference additionally requires PyTorch, a CUDA-capable GPU and bfloat16 support. `requirements-optional.txt` declares this dependency. Install a CUDA-compatible PyTorch build for the host following the [official installation instructions](https://pytorch.org/get-started/locally/), then install the optional requirements. The command checks CUDA and bfloat16 availability before inference. CPU neural inference is not supported by the distributed command.

```text
python -m pip install -r requirements-optional.txt
```

The pinned CPU dependencies reflect the bounded validation environment. No complete portable CUDA environment is certified by this release.

## Usage

Inspect the command-line options and run the bounded CPU tests.

```text
python -m microct --help
python -m microct infer --help
python -m microct estimate-noise --help
python -m unittest discover -s tests
```

Segment aligned grayscale TIFF slices named `0001.tif`, `0002.tif`, and so on, preserving the reconstruction intensity scale.

```text
python -m microct infer --input-directory path/to/reconstructed_slices --output outputs/labels.tif
```

The default output has 1,958 slices and a 480 by 640 crop starting at y=372, x=51. The input normalization, phase thresholds and slab settings are explicit in `config/final_settings.json`.

Estimate a residual spectrum from an already selected grayscale slab and its matching segmentation.

```text
python -m microct estimate-noise --volume-npy path/to/grayscale_slab.npy --labels-npy path/to/matching_labels.npy --output outputs/sqrt_spectrum.npy
```

For the documented extraction, select zero-based z slices 1600 through 1759 and retain the full in-plane slab. The configured 160-voxel cube is cropped only after detrending, image formation and morphology. Supplying the cube alone changes the computation. The original matching support is not distributed, so the stored spectrum cannot be regenerated from the supplied assets alone.

All output paths must be new. Interrupted inference can leave a partial BigTIFF; verify completion before using it. Changed inputs, supports or numerical settings define a different computation.

## Interpretation and verification

The residual contains reconstruction texture, segmentation error and image-model error; it does not isolate detector noise. The supplied sampling functions are the noise-addition stage, not a complete optical-image degradation or training pipeline.

`VALIDATION.json` preserves the version 1.0.0 implementation checks, which apply to the unchanged scientific modules and assets. They include strict loading of 175 checkpoint tensors, bounded CPU float32 comparisons, preprocessing, label precedence, deterministic noise, and runtime guards. A separate CPU adapter used for those comparisons is not part of this package. CUDA/bfloat16 equivalence, training, full-volume reproduction and regeneration of the stored spectrum from the original support were not demonstrated by those checks.

`MANIFEST_SHA256.json` gives the version 1.0.1 file inventory and hashes. [RIGHTS.md](RIGHTS.md) states the use and attribution terms.

## Versioned access

Use the [version 1.0.1 release snapshot](https://github.com/ismaelnrmelo/article2-microct-code/tree/v1.0.1) when citing or accessing the code associated with the submission. Version 1.0.0 remains available for traceability.
