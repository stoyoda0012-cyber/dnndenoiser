# Quick Start Guide

## Install

```bash
pip install -e .          # from the repository root; Python 3.10+
dnndenoiser --help
```

Every command below assumes the installed `dnndenoiser` console command. From a
source checkout without installing, substitute `PYTHONPATH=src python3 -m dnndenoiser.cli`.

## The four-step workflow

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

## `generate` — synthetic XPS spectra

```bash
# Peak-set presets: C1s_single, C1s_adventitious, C1s_polymer,
#                   O1s_oxide, Si2p_oxide, N1s_amine, test_doublet
dnndenoiser generate -o data.h5 -n 1000 --peak-set C1s_adventitious

# Noise models: poisson (default), gaussian, mixed, none
# `poisson` draws counts from the Poisson distribution in every bin. A
# Gaussian approximation is available in the library (`add_poisson_noise(..., use_gaussian_approx=True)`)
# and is applied only to bins whose expected count is at least 3, because below that
# its floor at zero biases the bin upward. It is not the default and the CLI does not reach it.
dnndenoiser generate -o data.h5 -n 500 --noise-type poisson --poisson-level 1000

# Sample-to-sample variation (important for generalization)
dnndenoiser generate -o data.h5 -n 8192 \
  --position-jitter 0.5 --width-var 0.2 --intensity-var 0.3

# Angle-resolved 3D data (energy x angle)
dnndenoiser generate -o arpes.h5 -n 100 --n-angles 16 --angle-model cosine

# Time-resolved 3D data (energy x time)
dnndenoiser generate -o time.h5 -n 100 --n-times 32 --time-model exponential_decay

# 4D data (energy x angle x time)
dnndenoiser generate -o 4d.h5 -n 50 --n-angles 16 --n-times 32
```

Backgrounds: `--background {none,linear,shirley}`. Peak shapes are
pseudo-Voigt by default (`--true-voigt` evaluates the Voigt profile exactly via
the Faddeeva function, slower; its width parameters are converted from FWHM and
mixing fraction by an approximation).

## `train` — model training

```bash
# Architectures: FCNN, ResNet-FCNN, 1D-CNN, ResNet-1DCNN, GRU, LSTM, bi-LSTM, Transformer
dnndenoiser train -d data.h5 -o model.pt --arch ResNet-FCNN --epochs 50 --lr 0.001

# Training methods (see README for exact semantics):
#   noise2clean (default) — supervised, needs 'clean' in the HDF5
#   noise2noise           — synthesizes a second independent noisy realization from 'clean'
#   moving-average        — self-supervised from a frame stack; needs no 'clean'
# noise2noise needs the Poisson level the data was generated with, so that the
# synthesized realization sits in the same noise regime as the input:
dnndenoiser train -d data.h5 -o model.pt --method noise2noise --noise-level 1000

# Device selection (auto-detected by default)
dnndenoiser train -d data.h5 -o model.pt --device mps
```

Rule-of-thumb learning rates: RNN/CNN/FCNN families 0.01; ResNet and
Transformer families 0.001. Use at least 8k training samples; 32k is better.

Multidimensional (angle/time) arrays are flattened and trained spectrum-wise.

## `infer` / `evaluate`

```bash
dnndenoiser infer -d noisy.h5 -m model.pt -o denoised.h5 --batch-size 256
dnndenoiser evaluate -d denoised.h5              # clean reference inside the file
dnndenoiser evaluate -d denoised.h5 --clean clean.h5 -o metrics.json
```

`evaluate` computes truth-referenced SNR/MSE, so it requires a clean reference
dataset; it cannot score measured data that has no reference.

## HDF5 schema

| Dataset | Shape | Notes |
|---------|-------|-------|
| `noisy` | (n, …, energy) | input spectra; extra axes (angle/time) allowed |
| `clean` | (n, …, energy) | reference; required for noise2clean/noise2noise and evaluate |
| `energy` | (energy,) | energy axis |
| `denoised` | (n, …, energy) | written by `infer` |
| `angles` / `times` | optional | written by `generate` for 3D/4D data |

### Frame stacks — `--method moving-average`

A different layout, for training from repeated acquisitions with **no clean
reference**. Each frame's target is the mean of its `--window` temporally
nearest *other* frames (leave-one-out).

| Dataset | Shape | Notes |
|---------|-------|-------|
| `frames` | (n_frames, energy) | the acquired frames, in any row order |
| `energy` | (energy,) | energy axis |
| `frame_index` | (n_frames,) | acquisition order; **integer and unique** |

`frame_index` is what "temporally nearest" is measured on, so a file that
omits it cannot be told from one whose frames were shuffled. **Duplicate
indices are rejected**: a frame is excluded from its own neighbourhood by
position, not by index value, so two frames sharing an index become distance-0
neighbours of each other and each leaks straight into the other's target.

```bash
dnndenoiser train -d stack.h5 -o model.pt --method moving-average \
    --window 5 --epochs 50
```

The optimiser, schedule, loss and architecture are fixed — they are part of the
method being reproduced — so `--arch`, `--lr`, `--lr-drop-period`,
`--lr-drop-factor`, `--scheduler`, `--warmup-epochs`, `--weight-decay`,
`--grad-clip`, `--hidden-units`, `--encoder-dim` and `--noise-level` are
**refused rather than ignored**, in whichever form they are written (`--lr 0.05`,
`--lr=0.05`, or an abbreviation argparse would accept). `--epochs`,
`--batch-size`, `--seed`, `--window` and `--device` still apply.

Stacks that are not 256 points are resampled.

**Evaluating this is not straightforward.** A measured stack has no clean
reference, and the obvious substitute — a mean over the same frames — is *not*
independent of targets built from subsets of those frames. An SNR computed
that way is not a held-out result and must not be reported as one. See
[the preregistration](preregistration/P1-selfsupervised-moving-average.md) for
what the port does and does not establish.

## Python API (minimal example)

```python
import torch
from dnndenoiser import DenoisingNetwork

model = DenoisingNetwork(num_features=256, num_hidden_units=100,
                         layer_type='ResNet-FCNN', encoder_output_dim=64)
ckpt = torch.load('model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with torch.no_grad():
    out = model(noisy_batch)          # (batch, 256) float32 tensor
    denoised = out[0] if isinstance(out, tuple) else out
```

## Troubleshooting

- `ModuleNotFoundError: dnndenoiser` — run `pip install -e .` from the
  repository root, or set `PYTHONPATH=src`.
- `command not found: dnndenoiser` — the console script is installed into the
  active environment; activate the venv you installed into.
- Error: method requires clean data — noise2clean/noise2noise need a `clean`
  dataset in the training HDF5 (the `generate` command always writes one).
- GPU check: `python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"`
