# DNNDenoiser — XPS Spectral Denoising with Deep Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22867628.svg)](https://doi.org/10.5281/zenodo.22867628)

Deep-learning denoising for X-ray photoelectron spectroscopy (XPS) spectra:
physics-based synthetic training data, eight 1-D network architectures,
Noise2Clean / Noise2Noise / self-supervised moving-average training, checkpoint
inference, and clean-referenced evaluation. Runs on CPU, NVIDIA CUDA, and Apple Silicon (MPS).

## Scope

This is a **standalone, generic denoiser for XPS spectra**. It contains network
architectures, training methods, physics-based synthetic data generation,
inference, evaluation, and a command-line interface.

Deliberately out of scope, and not present:

- depth profiling, peak fitting, and downstream quantification;
- measured-data loaders and instrument-specific calibration for particular
  instruments or beamlines — instrument constants reach the library by parameter
  injection from the calling application, and none are stored here;
- a bundled pretrained "universal" denoiser. What ships is a training workflow.
  "Turnkey" means turnkey *training*, not immediate denoising of arbitrary XPS
  data. See Limitations before applying a model outside the distribution it was
  trained on.

**Import boundary.** Importing this package pulls in nothing beyond its declared
dependencies. Library code references no other project of ours by import or by
path, so this repository can be installed and used on its own.
`tests/test_import_boundary.py` enforces that over the packaged tree, and is
part of the CI run.

## Installation

```bash
pip install -e .    # installs the `dnndenoiser` package and the `dnndenoiser` CLI
python -c "from dnndenoiser import DenoisingNetwork"
dnndenoiser --help
```

Requires Python 3.10+. From a source checkout without installing, use
`PYTHONPATH=src python3 -m dnndenoiser.cli`.

## Quick Start

```bash
# 1. Generate synthetic training data
dnndenoiser generate -o train.h5 -n 8192 --peak-set C1s_single

# 2. Train a model
dnndenoiser train -d train.h5 -o model.pt --arch bi-LSTM --epochs 30

# 3. Denoise a held-out set: a different seed, so no test spectrum was trained on
dnndenoiser generate -o test.h5 -n 512 --peak-set C1s_single --seed 7
dnndenoiser infer -d test.h5 -m model.pt -o denoised.h5

# 4. Evaluate against the clean reference
dnndenoiser evaluate -d denoised.h5
```

See [docs/QUICK_START.md](docs/QUICK_START.md) for options (peak-set presets,
noise models, position jitter, angle-/time-resolved generation) and a Python API
example.

## Training methods

| Method | CLI | Needs clean spectra | Notes |
|--------|-----|---------------------|-------|
| `noise2clean` | yes | yes | Supervised noisy→clean training. Recommended default. |
| `noise2noise` | yes | yes (to synthesize the pair) | The CLI creates a second, **independent synthetic noisy realization from the clean spectra**. It does not ingest measured noisy/noisy pairs. |
| noise2self | no (library only) | no | Masking-based self-supervision exists in `dnndenoiser.training.methods` as **experimental code**; it is not wired into the CLI because a correct masked loss (scoring only held-out coordinates) is not implemented there yet. |
| `moving-average` | yes | **no** | Leave-one-out self-supervision from a **stack of repeated acquisitions**: each frame's target is the mean of its `--window` temporally nearest *other* frames. Takes the [frame-stack schema](docs/QUICK_START.md), not `noisy`/`clean`. Training only — `infer` does not read that schema yet. Its optimiser, schedule, loss and architecture are fixed and flags that would change them are refused. |

## GPU support

Training and inference run on CPU, NVIDIA CUDA, and Apple Silicon (MPS);
the device is auto-detected, or select one with `--device {cpu,cuda,mps}`.
`pip install torch` ships CUDA support on Linux/Windows out of the box for
most GPUs; for recent architectures you may need the matching CUDA wheel
index, e.g. for Blackwell (sm_120):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"  # expect True
```

Verified configurations: NVIDIA RTX 5070 (CUDA 12.8, Windows), Apple M3 Max
(MPS, macOS), and CPU on Linux/macOS (CI). Device tests live in
`tests/test_network.py::TestDeviceCompatibility` (auto-skipped where the
backend is unavailable). Note: cuDNN runs RNN kernels (LSTM/GRU) in TF32 by
default, which relaxes CPU/CUDA agreement to ~1e-4 relative — set
`torch.backends.cudnn.allow_tf32 = False` for strict fp32 comparisons.

## Supported architectures

Eight architectures are available from the CLI and share one interface.
`DenoisingNetwork` also accepts two experimental variants, `bi-LSTM-seq` and
`Transformer-noPE`, which neither the test suite nor the reference measurement
covers. Sizes below are trainable parameter
counts at the default configuration (256 energy points, 100 hidden units,
encoder output 64) — a property of the model, reproducible with
`sum(p.numel() for p in DenoisingNetwork(...).parameters())`.

| Architecture | Parameters | Suggested LR | Input length |
|-------------|-----------|--------------|--------------|
| FCNN | 37K | 0.01 | any |
| ResNet-1DCNN | 92K | 0.001 | any |
| 1D-CNN | 152K | 0.01 | any |
| GRU | 194K | 0.01 | any |
| LSTM | 250K | 0.01 | any |
| bi-LSTM | 580K | 0.01 | any |
| ResNet-FCNN | 658K | 0.001 | any |
| Transformer | 862K | 0.001 | **must be divisible by 8** |

The Transformer reads the spectrum as patches of 8 energy points, so its input
length has to be a whole number of them; it refuses anything else at
construction, naming the nearest usable lengths. No other architecture here has
a length constraint.

The suggested learning rates are the per-architecture settings the reference
measurement used; they are starting points, not tuned optima. The CLI does
**not** apply them for you: `dnndenoiser train` defaults to `--lr 0.01
--epochs 30 --batch-size 32` for every architecture, so pass `--lr` explicitly.
**Denoising quality is deliberately not tabulated here.** Ranking architectures
by SNR gain requires stating the data, the split rule, the noise model, the
number of seeds, the metric's independence assumptions and the comparison
conditions; [`benchmarks/reference/report.md`](benchmarks/reference/report.md)
carries the per-architecture results together with those conditions and what
they do not support. See Limitations for what denoising quality does and does
not transfer to measured data.

## Limitations

- **Distribution shift is the dominant failure mode.** A trained model is only
  valid inside its training distribution (peak structure, peak positions, noise
  level, background). Applying a model outside it can *reduce* spectral quality,
  and position shifts beyond the trained jitter range degrade sharply. No claim
  of universal real-XPS validity is made.
- **Denoised output is a model estimate.** Like any learned denoiser, the
  network can oversmooth, suppress weak features, or hallucinate plausible
  structure; treat outputs as preprocessing, not measurement, and verify
  physically meaningful quantities (areas, positions, widths) downstream.
- **Evaluation needs a clean reference.** `dnndenoiser evaluate` computes
  truth-referenced SNR/MSE and therefore works on synthetic or
  high-statistics-referenced data only; there is no reference-free SNR for
  measured spectra.
- **Spectrum-wise processing.** Angle-/time-resolved arrays are generated and
  handled, but denoising flattens all non-energy axes and processes each 1-D
  spectrum independently — this is not a joint 3-D/4-D model.
- **Synthetic backgrounds** in the public generator are none/linear/Shirley-like.
- **Normalization**: spectra are normalized by default; intensity scale
  information is not preserved unless `--no-normalize` is used consistently.
- **No pretrained universal model is shipped.** The package provides the
  training workflow; "turnkey" means turnkey training, not immediate denoising
  of arbitrary XPS data.

## Related repositories (project split)

DNNDenoiser is the **generic XPS denoiser, standalone**. It contains no
depth-profiling or peak-fitting code. The AR-XPS/HAXPES application and the
published-paper reproductions were carved out into a sibling repo.

| Repo | Role | Depends on |
|------|------|-----------|
| **DNNDenoiser** (this) | Generic denoiser: networks, training (N2C/N2N), synthetic data | — (self-contained) |
| A separate depth-profiling project | Classical depth-profiling solver; consumes denoised spectra | this package |

The import boundary above is what keeps that split honest: a consumer may
depend on this package, and this package depends on no consumer.

## Directory structure

```
dnndenoiser/
├── pyproject.toml           # package definition (pip install -e .)
├── src/dnndenoiser/         # the installed package
│   ├── cli.py               #   CLI entry point (`dnndenoiser` command)
│   ├── models/network.py    #   network architectures
│   ├── data/                #   synthetic spectrum generation
│   └── training/            #   training methods (N2C/N2N) and LR schedules
├── tests/                   # pytest suite (import boundary, devices, methods)
├── benchmarks/reference/    # the reference measurement, its record, the renderer
├── docs/                    # documentation
└── paper/                   # JOSS paper sources
```

Everything the library offers is imported from the `dnndenoiser.*` namespace.

## Testing

```bash
pytest tests/ -v
```

## Documentation

See [docs/](docs/) for the quick-start guide, and
[benchmarks/reference/](benchmarks/reference/) for the reference measurement and
what its numbers do and do not support.

## Citation

If you use this software, please cite it via [CITATION.cff](CITATION.cff).

Archived on Zenodo:

- **All versions** (concept DOI; resolves to the latest):
  [10.5281/zenodo.22867628](https://doi.org/10.5281/zenodo.22867628)
- **v0.1.1**:
  [10.5281/zenodo.22870453](https://doi.org/10.5281/zenodo.22870453)
- **v0.1.0**:
  [10.5281/zenodo.22867629](https://doi.org/10.5281/zenodo.22867629)

When a result depends on this software, **cite the version DOI of the release
you used**, not the all-versions DOI. `generate` has already changed what it
produces for a given seed once (see [CHANGELOG.md](CHANGELOG.md)), so which
release produced a number is part of what makes it reproducible.

## License

MIT — see [LICENSE](LICENSE).
