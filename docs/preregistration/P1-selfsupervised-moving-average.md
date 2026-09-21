# Preregistration — P1: the self-supervised moving-average training method

**Status: registered, not implemented.** Written 2026-09-21, before any of the
port exists. Nothing below may be revised to match a result; if a criterion
turns out to be wrong, it is changed *visibly*, with the reason and the date,
and the change is part of the record.

## Why this is registered before the code

The port's purpose is a **published claim** — that `dnndenoiser` provides the
training method used in the JVST and SIA papers. Under `AGENTS.md` §8 that
makes it an independent-audit item. A criterion written after seeing the output
is not a criterion, so the acceptance conditions, the fixture and the tolerances
are fixed here first.

## What is being ported

The papers do not train against a clean reference. For each acquired frame the
training target is the mean of its `W` temporally nearest **other** frames at
the same pixel — leave-one-out, so the target's noise is independent of the
input's. `W = 5` is canonical; larger `W` ≈ longer effective exposure.

`dnndenoiser` v0.1.0 cannot do this. Its methods are noise2clean and
Noise2Noise with the second realisation synthesised from clean spectra; neither
trains from measured frames alone.

## The reference, pinned

The reference is the **published** implementation, because it is citable and
cannot change. A local working copy is not the reference: the copies on this
machine differ from the deposit in six of fifteen files, including a `LICENSE`
that still carries a pre-release placeholder.

| | |
|---|---|
| Deposit | `10.5281/zenodo.22092109`, version 1.0.0 |
| Archive | `software_arhaxpes_denoise.zip`, 22,091 bytes, md5 `f8ac7f5ca4abafb30239ebe1d3d0b217` |
| Integrity | all 15 files verified against the deposit's own `SHA256SUMS.txt`; 0 mismatches |

The three files this port is measured against, by sha256:

| Path in the deposit | sha256 |
|---|---|
| `src/arhaxpes_denoise/selfsupervised.py` | `136f2e112430fbb42e72bc6d6c0f2b9b02d3a828f6774c11bd2392eb4a13cb09` |
| `src/arhaxpes_denoise/network.py` | `c2c7be4db79f6a2bbd5c8ffe607ad1fe71ba6049250708e303b016ef3907a3ed` |
| `examples/train_selfsupervised.py` | `deb2dd3d5751394bfec4699c2e78a7149445898c7a41e4249a293ae26eb13fe4` |

The reference is **not vendored into this repository**. It is re-obtained from
the DOI and checked against these digests. A digest that does not match voids
the comparison rather than being updated to fit.

## The fixture

Taken from the deposit's own `examples/train_selfsupervised.py`, so the fixture
is itself published and cannot drift. It is restated here in full, and the test
constructs it from these constants rather than importing the reference:

- grid: `energy = numpy.linspace(0, 1, 256)`; `TARGET_LENGTH = 256`, so no
  resampling is exercised on this path;
- clean spectrum: `300 * exp(-(energy - 0.5)**2 / (2 * 0.04**2)) + 20` — a
  single Gaussian core level on a flat background, in counts;
- noise: `rng = numpy.random.default_rng(1)`; **training pool** =
  `rng.poisson(tile(clean, (200, 1))).astype(float32)`, then **test frames** =
  `rng.poisson(tile(clean, (16, 1))).astype(float32)`, drawn from the same
  stream in that order. **The draw order and the `float32` cast are both part of
  the fixture**: a different order gives different frames from the same seed,
  and the cast happens before normalisation;
- normalisation: element-global min-max over the training pool; the same two
  constants are applied to the test frames and to the clean spectrum;
- targets: `moving_average_targets(frames_n, arange(200), W=5)`;
- training: `epochs=20`, `seed=0`, `batch_size=32`, `device="cpu"`;
- model: ResNet-FCNN, `num_features=256`, `num_hidden_units=100`,
  `encoder_output_dim=64`; Adam(`lr=1e-3`, `weight_decay=1e-9`),
  StepLR(`step_size=25`, `gamma=0.5`), HuberLoss(`delta=1.0`), gradient-norm
  clipping at 4.0.

**Leakage.** The test frames are drawn after the training pool and never enter
training, the targets, or the normalisation constants. That is the split, and
the unit is the frame.

**Environment.** Criterion C2 is a numerical-identity claim and is stated for
one environment: CPU, Python 3.12, `torch` 2.9.1, `numpy` 2.3.3. On any other
environment C2 is **reported, not required** — a mismatch there is evidence
about numerical reproducibility across builds, not evidence against the port.
C1, C3 and C4 are required everywhere the suite runs.

## Acceptance criteria

The claim under test is **"the port reproduces the published implementation"** —
not "the port denoises well". Every comparison below is port against reference
on the same fixture.

### C1 — targets are exactly equal

`moving_average_targets` is a deterministic computation with no floating-point
reduction order at issue beyond a mean over `W` rows. The port's targets must
equal the reference's **exactly** (`numpy.array_equal`), for `W ∈ {1, 2, 5, 10}`
— the values the paper sweeps — on the fixture's 200-frame pool.

Exact equality is demanded because there is no reason for it to fail: if it
does, the ported algorithm differs, and a tolerance would hide that.

### C2 — trained outputs agree at a fixed seed

Same seed, same environment (above), both implementations trained on the
fixture, both run over the 16 test frames:

> **relative L∞ < 1e-4**, where the statistic is
> `max|y_port − y_ref| / max|y_ref|` over all 16 × 256 output values.

Exact equality is **not** required and would be wrong to require: batch
shuffling, BLAS reduction order and thread count make bit-identity an unstable
target even within one machine.

### C3 — trained outputs agree across seeds, in what they achieve

Seeds `1, 2, 3, 4, 5` (five seeds, disjoint from C2's seed 0). For each seed,
both implementations are trained and evaluated on the fixture, and output SNR
is computed against the fixture's **known clean spectrum** — available because
the fixture is synthetic:

```
SNR_dB = 10 * log10( mean(clean_n**2) / mean((denoised - clean_n)**2) )
```

> **|SNR_port − SNR_ref| ≤ 0.5 dB for every one of the five seeds**, and the
> paired mean difference is reported with its spread.

The comparison is **paired** by seed, as `AGENTS.md` §6 requires. The absolute
SNR values are a property of this fixture and are **not** a performance claim
about the method; only the difference is under test.

### C4 — weights interchange unmodified

The SIA paper states publicly that the trained weights load unmodified into the
released implementation. The port must preserve that in both directions:

> a `state_dict` from the reference's `DenoisingNetwork` loads into
> `dnndenoiser`'s ResNet-FCNN with `strict=True` and no key renaming, and the
> reverse; and a model so loaded produces outputs equal to the originating
> model's within C2's tolerance on the fixture.

Already measured, 2026-09-21, before registration: both classes at
`num_features=256, num_hidden_units=100, encoder_output_dim=64` have the same
22 `state_dict` keys, the same shapes, and 658,177 parameters. C4 turns that
measurement into a guarded test.

### C5 — a documented frame-stack schema

The CLI path needs an HDF5 layout for frame stacks: the frames, the energy
axis, and the **acquisition order**, since order is what "temporally nearest"
means. The schema is documented in `docs/QUICK_START.md` alongside the existing
one, and a round-trip test reads back what it writes.

This is a completeness condition, not a numerical one; it passes or it does not.

## What this does not claim, and will not be written as claiming

- **Not a performance claim.** Nothing here licenses a statement of the form
  "dnndenoiser achieves N dB on measured data". The fixture is synthetic and
  its SNR values are properties of the fixture.
- **Not validated on measurement.** No measured frames are used. The port
  reproducing the reference on synthetic Poisson frames is evidence about the
  implementation, not about either one's behaviour on an instrument.
- **No reference-free SNR.** For a measured stack there is no clean reference.
  The only one constructible from the frames themselves — an all-frame mean —
  is **not independent of the training targets**, which are means of subsets of
  those same frames. Any number computed that way is reported as what it is,
  with the dependence stated, and never as a held-out result (`AGENTS.md` §5).
- **Distribution shift still governs.** A model trained by this method is valid
  inside the distribution of the frames it was trained on. Nothing here changes
  that, and the limits in `README.md` continue to apply.

## If a criterion fails

A failure is recorded and diagnosed; it is not silently renegotiated.

- **C1 fails** → the ported algorithm differs from the reference. The port is
  wrong until shown otherwise; no tolerance is introduced.
- **C2 fails in the pinned environment** → report the observed statistic, then
  determine whether the cause is the port or an environmental non-determinism
  the criterion did not anticipate. If the latter, the criterion is revised **in
  this file, visibly, with the evidence** — and the revision is an audit item.
- **C3 fails** → the port and the reference do not reach the same result. The
  claim "reproduces the method" is not made.
- **C4 fails** → the claim about weight interchange is withdrawn from the
  documentation, whatever else passes.

A partially met set does not become "reproduces the JVST/SIA method". It becomes
a statement of exactly what was and was not reproduced.

## Release and audit

The port targets **v0.1.1**. Before that release, and because the claim is a
published one, it gets the independent audit `AGENTS.md` §8 requires. The audit
sees this file, the implementation, and the test results together — and is asked
specifically whether the criteria were met as written, rather than as
interpreted afterwards.

## Record

| | |
|---|---|
| Registered | 2026-09-21 |
| Reference | `10.5281/zenodo.22092109` v1.0.0, digests above |
| Implementation | not started at registration |
| Result | to be recorded here, below this line, when the criteria have been run |
