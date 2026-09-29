# TRSA-U-Net final functions

Version 1.0.0. This package contains the final residual-texture estimator, the accepted network definition, fixed-setting triplanar inference, accepted weights and the stored square-root residual spectrum associated with Article 2. It contains no training implementation or optical benchmark program.

## Files and inputs

- `microct/model.py` defines the accepted network, with 4,737,729 trainable parameters and a 32-element frequency buffer.
- `microct/inference.py` provides input normalization, polynomial detrending, overlapping tiles, averaging of three orthogonal orientations, fixed NbC/pore channels and slab-wise BigTIFF output.
- `microct/noise.py` provides the segmentation-conditioned residual-spectrum estimator and mixed two-dimensional noise sampling functions.
- `config/final_settings.json` contains the explicit settings. Asset paths are relative to this package. Input/output paths are supplied by the caller.
- `assets/accepted_weights.pt` is the accepted checkpoint. Its hash is checked before loading.
- `assets/accepted_sqrt_spectrum.npy` is the accepted stored spectrum, not a raw tomogram or optical dataset. Inference does not need this spectrum; it is supplied for the noise functions.

Inference requires reconstructed, aligned grayscale slices named `0001.tif`, `0002.tif`, and so on. Default geometry is 1,958 slices, with a 480 by 640 crop beginning at y=372 and x=51. Slabs have depth 256 and 32-slice context margins. Input values must preserve the original reconstruction intensity scale. Labels are pore=1, matrix=2, chromium carbide=3, NbC=4. The model uses normalized inputs `2*((45820-image)/2887-0.30599761684099036)` and probability threshold `>0.520098865032196`.

Residual estimation requires the exact grayscale slab and the matching segmented support used in the accepted extraction. Supply zero-based slices 1600 through 1759 of the reconstructed array, keeping the full in-plane slab. The configured spatial crop is applied after detrending, image formation and morphology. Supplying only the final 160-cube changes boundary conditions. The original support is not included. The residual includes reconstruction texture, segmentation error and image-model error; it does not isolate detector noise. It is distinct from the mask-independent noise scenario used in the optical comparison.

## Environment and commands

Python 3.10 was used for the bounded CPU checks. `requirements.txt` records the observed NumPy, SciPy and tifffile versions. The PyTorch version is not locked; choose a build appropriate to the available CUDA runtime. Neural inference requires CUDA with bfloat16 support. CPU inference is deliberately rejected. A complete portable runtime environment has not been established.

From the package directory, inspect the commands without loading model weights.

```text
python -m microct --help
python -m microct infer --help
python -m microct estimate-noise --help
python -m unittest discover -s tests
```

Example inference, with paths supplied by the user.

```text
python -m microct infer --input-directory path/to/reconstructed_slices --output outputs/labels.tif
```

Example residual extraction from an already selected slab and its exact support.

```text
python -m microct estimate-noise --volume-npy path/to/grayscale_slab.npy --labels-npy path/to/matching_labels.npy --output outputs/sqrt_spectrum.npy
```

Outputs must be new paths. A failed inference can leave a partial BigTIFF; completion must be checked before using its contents. Changed inputs, supports or settings constitute a different computation.

## Verification and limits

The accepted checkpoint loaded strictly with all 175 state tensors. Using PyTorch 2.14.0+cpu in float32, the supplied and original models produced bitwise-identical outputs for a batch of two 96 by 96 patches. A separate CPU adapter also produced bitwise-identical tiled results for 96 by 96, 97 by 151 and 144 by 201 arrays. The adapter disabled the CUDA guard and autocast without changing the distributed modules; these checks do not establish CUDA/bfloat16 equivalence. Ten additional bounded comparisons cover preprocessing, labeling, tiling, orientation, slab margins and TIFF I/O with a deterministic predictor. The supplied CPU tests cover detrending, normalization, label precedence, deterministic noise and runtime guards. Verification scope is recorded in `VALIDATION.json`.

No training adapter, original training dataset, segmented extraction support, raw tomogram, projections or optical benchmark images are distributed. Regeneration of the training procedure, stored spectrum, accepted weights and complete tomographic result has not been demonstrated from this package. CUDA inference and full-volume reproduction have not been rerun for this release. Scientific accuracy and portability are separate from the bounded implementation checks.

`MANIFEST_SHA256.json` records the relative file inventory and SHA-256 hashes. See `RIGHTS.md` for use and attribution information.
