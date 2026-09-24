# Preregistration — P1: the self-supervised moving-average training method

**Status: registered 2026-09-21; amended 2026-09-21 (Revision 1); implemented
and run 2026-09-21 — all eight criteria met. See Result, last section.
Independent audit under `AGENTS.md` §8 pending; no claim is published until it
has been done.** Nothing below may be revised to match a result. When a criterion
turns out to be wrong it is changed *visibly*, with the reason and the date, and
the change is part of the record — see **Revision log**, last section.

> **Revision 1 — after independent audit, before any implementation.** Two
> independent audits of the registered text found that, as written, the criteria
> could be **passed in full by a port that ignores acquisition order, uses the
> wrong loss function, and omits both the learning-rate scheduler and gradient
> clipping**, while a **correct independent reimplementation would likely fail**
> two criteria for reasons unrelated to correctness. Both are repaired here. The
> registered text is preserved in git at commit `100940d`; every change is
> itemised in the revision log with the finding that caused it.

## Why this is registered before the code

The port's purpose is a **published claim**. Under `AGENTS.md` §8 that makes it
an independent-audit item. A criterion written after seeing the output is not a
criterion, so the reference, the fixture and the tolerances are fixed here
first — and, as Revision 1 shows, audited here first.

## What is being ported, and what the claim is about

The method does not train against a clean reference. For each acquired frame the
training target is the mean of its `W` temporally nearest **other** frames at the
same pixel — leave-one-out, so that **for independent frames** the target's noise
is independent of the input's. `W = 5` is canonical.

On a measured stack that independence is an assumption, not a guarantee:
temporally adjacent frames subject to drift or charging are correlated, and
temporal nearness is not statistical independence. Nothing here tests that.

**The claim is about the deposit, not about the papers.** The deposited
implementation describes itself as *"Distilled from the paper's self-contained
training script"*. The chain is therefore **papers' script → deposit
(a distillation) → port**, and every criterion below pins only the second link.
A full pass licenses "reproduces the archived reference implementation". It does
**not** license "reproduces the JVST/SIA method", and no wording of that kind may
be published on the strength of this document.

**What v0.1.0 cannot do.** Its CLI methods are noise2clean and Noise2Noise with
the second realisation synthesised from clean spectra. The library also ships
`Noise2Self` (`src/dnndenoiser/training/methods.py`), which is not wired to the
CLI because its masked loss is incomplete. None of the three trains from a
measured frame stack.

## The reference, pinned

The reference is the **published** implementation, because it is citable and
cannot change. A local working copy is not the reference: the copies on this
machine differ from the deposit in six of fifteen files, one being a `LICENSE`
still carrying a pre-release placeholder.

| | |
|---|---|
| Deposit | `10.5281/zenodo.22092109`, version 1.0.0 |
| Archive | `software_arhaxpes_denoise.zip`, 22,091 bytes, md5 `f8ac7f5ca4abafb30239ebe1d3d0b217` |
| Integrity | all 15 files verified against the deposit's own `SHA256SUMS.txt`; 0 mismatches |

| Path in the deposit | sha256 |
|---|---|
| `src/arhaxpes_denoise/selfsupervised.py` | `136f2e112430fbb42e72bc6d6c0f2b9b02d3a828f6774c11bd2392eb4a13cb09` |
| `src/arhaxpes_denoise/network.py` | `c2c7be4db79f6a2bbd5c8ffe607ad1fe71ba6049250708e303b016ef3907a3ed` |
| `examples/train_selfsupervised.py` | `deb2dd3d5751394bfec4699c2e78a7149445898c7a41e4249a293ae26eb13fe4` |

The reference is **not vendored into this repository**; it is re-obtained from
the DOI and checked against these digests. A digest that does not match voids the
comparison rather than being updated to fit.

**What "reproduces" means here.** A reimplementation inside `dnndenoiser` that
does not import or vendor the deposit. `AGENTS.md` §3's import boundary names
`deppro` and `toyomacro` only; `arhaxpes_denoise` is added to the prohibition
for the duration of this work — the port may read it to be tested against.

*Amended after the implementation audit:* the registered text said "an
**independent** reimplementation", and what was written is closer to a faithful
transcription — `_interpolation_matrix` is the deposit's `_interp_matrix` with
renamed locals, and the training loop follows it closely. The guard enforces
the absence of an *import*; it cannot see transcription, so the prohibition is
narrower than the word "independent" implied. The word is withdrawn rather than
the practice defended: nothing here establishes that an unrelated
implementation would agree.

## The fixture

Taken from the deposit's own `examples/train_selfsupervised.py`, so the fixture
is itself published and cannot drift. It is restated in full and the test
constructs it from these constants rather than importing the reference.

- grid: `energy = numpy.linspace(0, 1, 256)`; `TARGET_LENGTH = 256`;
- clean spectrum: `300 * exp(-(energy - 0.5)**2 / (2 * 0.04**2)) + 20` — a single
  Gaussian core level on a flat background, in counts;
- noise: `rng = numpy.random.default_rng(1)`; **training pool** =
  `rng.poisson(tile(clean, (200, 1))).astype(float32)`, then **test frames** =
  `rng.poisson(tile(clean, (16, 1))).astype(float32)`, drawn from the same stream
  in that order. **The draw order and the `float32` cast are both part of the
  fixture**: a different order gives different frames from the same seed, and the
  cast happens before normalisation;
- noise model: **pure Poisson on counts. No detector or read-noise term**, and no
  Gaussian approximation — unlike `dnndenoiser`'s own generator, which offers
  both;
- normalisation: element-global min-max over the training pool
  (`g_min = 5.0`, `g_max = 387.0`); the same two constants are applied to the test
  frames and to the clean spectrum;
- targets: `moving_average_targets(frames_n, arange(200), W=5)`. Note the
  function upcasts internally to `float64`;
- training: `epochs=20`, `seed=0`, `batch_size=32`, `device="cpu"`,
  `shuffle=True`, no `drop_last`;
- **model, in full** — ResNet-FCNN with **four residual blocks**, each
  `Linear → ReLU → Dropout(p=0.1) → Linear` with a post-add ReLU; two output
  heads, and **the loss is taken on the first output only**; `global_skip=False`.
  `num_features=256`. `num_hidden_units=100` and `encoder_output_dim=64` are
  passed and are **inert on this path** — the deposit's `network.py` accepts them
  "for call-site compatibility";
- optimiser: Adam(`lr=1e-3`, `weight_decay=1e-9`); StepLR(`step_size=25`,
  `gamma=0.5`); `HuberLoss(delta=1.0)`; gradient-norm clipping at 4.0.

### What the fixture does not exercise — measured, not assumed

Three named components of the method are **inert on this fixture**. They are
recorded here so that a pass is not mistaken for evidence about them:

| Component | Measurement on the fixture | Consequence |
|---|---|---|
| `HuberLoss(delta=1.0)` | 140 optimiser steps, max\|residual\| = 0.9463 < δ; **0/140** steps leave the quadratic region | Huber ≡ `0.5·MSE` identically. Substituting `MSELoss` passes C3 |
| `StepLR(step_size=25)` | `epochs=20`, so the decay never fires; learning rate constant at 1e-3 throughout | Omitting the scheduler changes nothing |
| clip at 4.0 | max gradient norm 0.0464, **0/140** steps bound (86× headroom) | Omitting clipping changes nothing |

C7 adds one case that makes the scheduler fire, which is enough to verify it.
Measured after implementation: **only the clip is genuinely unverified** — a
port with no clipping at all reproduces the reference to the bit. The loss is
verified up to the equivalence class that agrees with `0.5 · MSE` inside
`|r| < 1`. See the corrected bullets under "What this does not claim".

### Split, leakage, and the regime

**Split.** The 16 test frames are drawn after the training pool and enter
neither training, the targets, nor the normalisation constants. The unit is the
frame.

**What the split does not prevent.** The test frames are Poisson draws from the
**same single clean spectrum** the model saw 200 draws of. This is an
in-distribution, single-signal evaluation; the absolute SNR values support no
generalisation statement whatever.

**Regime.** Training and inference operate at the same *input* signal-to-noise
regime — training and test frames come from the same clean spectrum at the same
Poisson rate (input SNR ≈ 22.2 dB). The training **target** sits at roughly `W`×
that exposure by construction, so the training pair is signal-to-noise asymmetric
by design. `C2`–`C6` exercise `W = 5` only; **no claim about W-dependence or
effective-exposure scaling is made or supported.**

### Environment

C0, C1, C2, C3, **C6's exactness** and C7 are numerical-identity or
tight-tolerance claims and are required **only** in one environment: CPU,
Python 3.12.11, `torch` 2.9.1, `numpy` 2.3.3. Elsewhere they are **reported,
not required**. C4, C5 and C6's *tolerance* form are required wherever the suite
runs.

*The platform was added to this environment 2026-09-24 (Revision 5): macOS on
Apple silicon (`Darwin-arm64`), the development machine the fixture and the
Result were produced on; the Result section did not record it. The versions compare
on their release number, without a wheel's build label.*

*C7's scope was amended 2026-09-21 (Revision 2). The registered text put it with
C4–C6, which was an oversight: C7 **is** C2's statistic at 30 epochs, so it
carries exactly C2's environment dependence and cannot be required where C2 is
not. The implementation had already scoped it this way; the document is what was
wrong.*

This scoping is not caution for its own sake. `benchmarks/reference/report.md`
records, for this project's own measurements, that reduction order differs
between CPU, MPS and CUDA and that re-running "may move them by a few tenths of
a dB" — from one implementation. Two implementations differencing on such a
backend cannot be held to 0.5 dB.

## Acceptance criteria

The claim under test is **"the port reproduces the archived reference
implementation"** — not "the port denoises well". Every comparison is port
against reference on the fixture.

### C0 — RNG-stream alignment (precondition for C2 and C3)

> After `torch.manual_seed(s)` and construction, the port's model parameters are
> **bit-identical** to the reference's, for `s ∈ {0, 1, 2, 3, 4, 5}`.

C2 and C3 compare trained outputs pairwise by seed. That pairing cancels
variance **only if both implementations consume the same random stream**. A port
that is algorithmically identical but draws one extra value before constructing
the model — which is all it takes to build a DataLoader before a model — produces
a different trained model, and C3 then differences two effectively independent
draws.

C0 makes that presupposition explicit and testable instead of hidden. **If C0
fails, C2 and C3 are not evaluated**: they are reported as inapplicable, and
equivalence must be argued on C1, C4–C7 and code review. A C0 failure is not by
itself evidence that the port is wrong.

**What C0 does not catch, measured 2026-09-21 (Revision 2).** C0 tests
construction, and the example above is a draw *inside the training function*
before construction. Inserting `torch.randn(1)` there leaves C0 passing at all
six seeds while C2, C3 and C7 fail — the misattribution C0 exists to prevent,
undetected. C0 is therefore extended: as well as freshly constructed parameters,
the parameters returned by `train_selfsupervised(..., epochs=0, seed=s)` must be
bit-identical to the reference's, which places the check inside the function
where the draw would happen.

### C1 — targets are exactly equal

`moving_average_targets` is deterministic given a tie-break rule. The port's
targets must equal the reference's **exactly** (`numpy.array_equal`) in every
case below.

| Case | `frame_indices` | `W` | Why |
|---|---|---|---|
| a | `arange(200)` | 1, 2, 5, 10 | the paper's sweep |
| b | **`default_rng(7).permutation(200)`** | 5 | **acquisition order must actually be used** |
| c | `arange(3)` | 10 | the `min(W, n-1)` clamp fires |
| d | `arange(1)` / `arange(200)` with `W = 0` | — | `ValueError` is raised in both |
| e | `arange(200)` with `frame_indices[1] = frame_indices[0]` | 5 | duplicate indices; the reference's output is pinned |

**Case b is the one that matters most.** On `arange(200)` the temporal distance
`|t_i − t_j|` equals the row distance `|i − j|`, so a port that **ignores
`frame_indices` entirely** and windows by row position is `array_equal` to the
reference for all four values of case a, and then passes C2–C4 because the
targets are identical. Acquisition order is the method's central ingredient and
the registered fixture never exercised it. On a shuffled order the two
implementations differ by up to 0.1026 — about 12% of full scale.

**Tie-break.** Distances from `arange(n)` come in exact symmetric pairs
(`1,1,2,2,3,3,…`), so for **odd `W`** the outermost neighbour slot is a genuine
tie. The reference resolves it with `np.argsort`'s default (`quicksort`, not
stable). Measured on the fixture: switching to `kind='stable'` changes the
neighbour set for **86/200 rows at W=1 and 107/200 rows at W=5** — and changes
target *values* on 145/200 rows by up to **0.0471** on the normalised scale.
W=2 and W=10 are unaffected.

> The port must reproduce the reference's neighbour selection **including
> `numpy.argsort`'s default tie-breaking**. Because that is an implementation
> detail numpy does not contract, C1 is pinned to the environment above, and a
> demonstrated tie-break difference is diagnosed as such — **not** charged to the
> port.

*(The registered text asserted "there is no reason for it to fail". That was
false; see the revision log.)*

### C2 — trained outputs agree at a fixed seed

Given C0, same seed 0, pinned environment, both trained on the fixture, both run
over the 16 test frames:

> **relative L∞ < 1e-4**, where the statistic is
> `max|y_port − y_ref| / max|y_ref|` over all 16 × 256 outputs.
> (`max|y_ref| = 0.8529`, so the bound is ≈ 8.5e-5 absolute.)

**What this tolerance is and is not.** The registered text justified 1e-4 by
batch shuffling, BLAS reduction order and thread count. Measured, none of those
populate the band on this fixture: shuffling is seeded and deterministic, and
training at 1 vs 12 threads gives relative L∞ of **exactly 0.0**. Observed
differences are either exactly 0 or ≥ 1.5e-3 — fifteen times the threshold. On
the training path C2 is therefore, in practice, an exact-equality test with a
margin, and it is kept as a margin rather than tightened.

**Revision of C2 requires a demonstration, not an argument.** If C2 fails and
environmental non-determinism is proposed as the cause, that cause must be
**reproduced between two runs of the unmodified reference** before C2 may be
changed. Without that demonstration the failure belongs to the port. Note also
that the environment pin fixes Python, `torch` and `numpy` but **not** the CPU
microarchitecture or the BLAS kernel selected; C2 across different CPUs is
reported, not required.

### C3 — trained outputs agree across seeds, in what they achieve

Given C0, seeds `1, 2, 3, 4, 5`, pinned environment. Output SNR against the
fixture's **generative** clean spectrum — not an estimate from the frames, which
is what makes this truth-referenced:

```
SNR_dB = 10 * log10( mean(clean_n**2) / mean((denoised - clean_n)**2) )
```

> **|SNR_port − SNR_ref| ≤ 0.5 dB for every one of the five seeds**, paired by
> seed, with the mean difference and its spread reported.

**Calibration, stated before any result exists.** The reference's own SNR across
these five seeds is 35.78 / 36.08 / 34.41 / 33.88 / 32.55 dB — **sd 1.442 dB**,
range 3.53 dB. (Reproduced independently on the fixture, single-threaded CPU,
together with `g_min = 5.0`, `g_max = 387.0`, `max|y_ref| = 0.8529` and an input
SNR of 22.24 dB. The fixture is deterministic as specified.) The 0.5 dB bound is 3.5× tighter than that spread, and is
defensible **only** under C0: pairing removes the seed variance when the streams
align. Without C0 the difference of two effectively independent draws has
sd ≈ 2 dB and five-for-five agreement would be a roughly 1-in-3700 event. This
is why C0 gates C3 rather than sitting beside it.

**What C3 does not do.** It is not a sensitive test of the method's ingredients.
Training with `targets = the frame itself` — leave-one-out removed entirely,
the method's defining property gone — moves the mean ΔSNR by only about 0.24 dB.
Conversely `weight_decay=0` passes C3 at 0.076 dB while failing C2. C3 detects
gross divergence and RNG misalignment; it does not certify the algorithm. That
work is C1's and code review's.

The estimand is the **port-minus-reference difference**. The absolute SNR values
are properties of this fixture and are not a performance claim.

### C4 — the two ResNet-FCNN definitions interchange

> A `state_dict` from the reference's `DenoisingNetwork` loads into
> `dnndenoiser`'s ResNet-FCNN with `strict=True` and no key renaming, and the
> reverse; and a model so loaded produces eval-mode outputs equal to the
> originating model's on the fixture.

Measured 2026-09-21, before registration: both classes at
`num_features=256, num_hidden_units=100, encoder_output_dim=64` expose the same
22 `state_dict` keys with the same shapes and 658,177 parameters, and eval-mode
outputs differ by exactly 0.0.

**Scope, corrected.** This tests **structural compatibility between two class
definitions at one configuration**, using untrained weights. It does **not**
establish that any paper's trained weights load into `dnndenoiser`: those were
never deposited, and the configuration they were trained at is not fixed here.
C4 also cannot see the dropout rate — `state_dict` keys, shapes and parameter
count are identical for `p=0.1` and `p=0.5`. The permissible sentence is "these
two ResNet-FCNN definitions are `state_dict`-compatible at `(256, 100, 64)`".

*(The registered text justified C4 by an SIA statement about trained weights.
The criterion does not test what that sentence described; see the revision log.
The paper has since been published — `10.1002/sia.70123`, 2026-09-21 — which
makes the statement citable but does not make C4 test it.)*

### C5 — a documented frame-stack schema

An HDF5 layout for frame stacks: the frames, the energy axis, and the
**acquisition order**, since order is what "temporally nearest" means.
Documented in `docs/QUICK_START.md` beside the existing schema, with a
round-trip test.

> The schema **requires acquisition indices to be unique**, and the reader
> rejects duplicates. `numpy.fill_diagonal` excludes a frame from its own
> neighbourhood **by position, not by index value**, so two frames sharing an
> index make each a distance-0 "other" frame of the other — a target/input
> dependence that defeats leave-one-out silently, and that a user-written file
> can produce.

### C6 — the resample path

Every real frame stack whose length is not 256 passes through `resample`.

> `resample(x, 128)` and `resample(x, 512)` on the fixture's test frames equal
> the reference exactly.

The reference builds its interpolation matrix in `float32`, clips the source
index at `n_old − 2`, and returns the input **by identity** when
`n_old == n_new`. A port that returns a copy, or interpolates in `float64`, is
not equivalent.

*Amended 2026-09-21 (Revision 4).* `resample` is a `float32` matrix multiply,
and which order the BLAS sums in is the build's choice — the same mathematics
written three ways (`@`, `einsum`, per-row) differs by one ULP on a single
machine. **Exactness is therefore a claim about one build.** C6 is split: the
result must match the reference **to `atol = 1e-6` on any build**, and
**exactly on the pinned one**. Shape, dtype, the `n_old − 2` clip and the
identity return are checked everywhere, since no BLAS has a say in them.

### C7 — one case where the scheduler fires

> The fixture at `epochs=30`, seed 0: C2's statistic, same bound.

At 30 epochs `StepLR(step_size=25)` fires once and the learning rate halves, so
this case exercises a component the 20-epoch fixture leaves inert. The loss
function and the clip threshold remain unexercised; see the fixture's table.

## What this does not claim, and will not be written as claiming

- **Not a claim about the papers.** The deposit is the papers' authors' own
  distillation of a training script. Nothing here connects the port to the
  papers' published results.
- **No claim about denoising performance**, on measured or synthetic data,
  beyond the port-versus-reference *difference* C3 measures. (That difference
  itself is supported and should not be disclaimed away.)
- **Not validated on measurement.** No measured frames are used.
- **A model estimate, not a measurement.** `AGENTS.md` §5: the network can
  oversmooth, suppress weak features and hallucinate plausible structure.
  Nothing here bounds any of those.
- **No reference-free SNR.** For a measured stack there is no clean reference.
  The only one constructible from the frames — an all-frame mean — is **not
  independent of the training targets**, which are means of subsets of those
  same frames. Any number computed that way is reported as what it is, with the
  dependence stated, never as a held-out result.
- **Denoising is preprocessing.** Nothing here shows that peak areas, positions
  or widths survive it. Physically meaningful quantities must be verified
  downstream, not assumed preserved.
- **The CLI workflow is training only, and is barely covered by the criteria.**
  `infer` does not read the frame-stack schema — pointed at one it fails with an
  unhandled `KeyError` — and does not apply the normalisation constants the
  checkpoint records, so the model's output cannot be returned to counts through
  the CLI. The element-global min-max the training command applies, its
  `--window` clamping, and its default device (`auto`, which selects MPS on
  Apple silicon, outside the environment C0–C3 and C7 are pinned to) are
  methodological choices **no criterion pins and none compares to the
  reference**.
- **One `W`, one fixture, one peak.** `W = 5` only, a single Gaussian core level,
  a single clean spectrum. No statement about W-dependence, effective exposure,
  other line shapes, multi-peak spectra or real backgrounds.
- **The gradient-clip threshold is not verified by the criteria.** Removing
  clipping entirely, or setting it to 0.5, reproduces the reference **to the
  bit** on this fixture — C2, C3 and C7 all pass. It is checked by code review
  and by nothing else.
- **The scheduler *is* verified**, by C7: dropping `schedule.step()` leaves C2
  untouched at 20 epochs and fails C7 at 30.
- **The loss is verified only up to an equivalence class.** Anything identical
  to `0.5 · MSE` inside `|r| < 1` passes, because the fixture's residuals never
  reach `δ = 1.0`; a wrong `δ` or a wrong scale fails C2 and C7. Substituting
  plain `MSELoss` fails C2 at 1.25e-3.

## If a criterion fails

A failure is recorded and diagnosed, never silently renegotiated.

- **C0 fails** → C2 and C3 are not evaluated. Not by itself evidence against the
  port.
- **C1 fails** → the ported algorithm differs, *unless* the difference is a
  demonstrated `numpy.argsort` tie-break difference, which is diagnosed as such.
  Otherwise the port is wrong until shown otherwise; no tolerance is introduced.
- **C2 fails** → the port owns the failure unless the proposed environmental
  cause is **reproduced between two runs of the unmodified reference**.
- **C3 fails, C0 passing** → the implementations do not reach the same result.
  The claim is not made.
- **C4 fails** → the compatibility statement is withdrawn from the documentation.
- **C5/C6/C7 fail** → the corresponding capability is not documented as present.

A partially met set does not become "reproduces the reference implementation". It
becomes a statement of exactly what was and was not reproduced, **published in
this file**, and that statement is itself an audit item.

**Any revision to any criterion is an audit item under `AGENTS.md` §8** — not
only C2's.

## Release and audit

The port targets **v0.1.1**. Because the claim is a published one it gets the
independent audit §8 requires, before release.

**The audited object is the claim, not only the evidence.** The audit approves
the exact wording as it will appear in `README.md`, `paper/paper.md`,
`CHANGELOG.md` and the release note, and is asked whether the criteria were met
as written rather than as interpreted afterwards.

### Documents the port falsifies on release

Listed now so the release cannot quietly leave them stale:

*Corrected after the implementation audit: two rows were wrong and three
statements were missing.*

| Document | Statement | Status |
|---|---|---|
| `README.md` header | "Noise2Clean / Noise2Noise training" | incomplete → amend |
| `README.md` training-methods table | three rows, none of them this method | incomplete → amend |
| `docs/QUICK_START.md` train section | the method enumeration in the worked example | incomplete → amend |
| `paper/paper.md` | "The generic self-supervised methods it implements, Noise2Noise and Noise2Self" | incomplete → amend |
| `paper/paper.md` | "`training` holds the noise2clean / Noise2Noise methods" | incomplete → amend |
| `paper/paper.md` | "Some capabilities are intentionally narrower than a reader might assume" | **still true, but misleading by omission**: the package now does ingest measured frame stacks. Needs a sentence, not a correction |
| ~~`docs/QUICK_START.md` troubleshooting~~ | "noise2clean/noise2noise need a `clean` dataset" | **not falsified** — already scoped to those two methods |
| ~~`paper/paper.md`~~ | "it does not ingest measured noisy/noisy pairs" | **not falsified** — scoped to Noise2Noise, and remains true of it |

`paper/paper.md` is this repository's JOSS draft and is **not** submitted, so
amending it is ordinary repository work. The JVST and SIA papers are published
and are not in this repository; they are not touched.

## Revision log

### Revision 1 — 2026-09-21, after two independent audits, before implementation

Both audits ran against the registered text (git `100940d`) without access to
this project's private records. Measurements marked **[verified here]** were
reproduced independently before this revision was written; the rest are the
auditors' and are marked as such. Every auditor number that was cheap to re-run
reproduced **exactly** — the fixture constants, the input SNR, `max|y_ref|`, and
the five-seed SNR spread — which is why the remainder are carried with
attribution rather than re-derived.

| # | Finding | Change |
|---|---|---|
| 1 | **A port ignoring `frame_indices` passes every criterion.** On `arange(200)` temporal and row distance coincide; a positional port is `array_equal` for W = 1, 2, 5, 10 and then passes C2–C4 on identical targets. *(auditor's measurement; differs by 0.1026 on a permuted order)* | C1 case **b** added |
| 2 | **C1's justification was false.** "No reason for it to fail" — but `np.argsort`'s unstable tie-break decides the outermost neighbour for odd `W`. **[verified here]** 86/200 rows differ at W=1, 107/200 at W=5; target values differ on 145/200 rows by up to 0.0471 | Tie-break pinned; C1 moved under the environment pin; failure rule amended |
| 3 | **C3 presupposed RNG-stream alignment without stating it.** A port drawing one extra value before model construction fails 3 of 5 seeds *(auditor's measurement)*; the reference's own seed spread is **sd 1.442 dB** — 3.5× the bound **[verified here]** | **C0 added** as an explicit precondition; C3's calibration written out |
| 4 | **The fixture was not reproducible from the document.** `Dropout(0.1)` is load-bearing and was omitted, while the two parameters that *were* stated are inert. **[verified here]** `nn.Dropout` at `network.py:28`; the docstring calls the others "accepted for call-site compatibility"; the word "dropout" appeared 0 times in the registered text | Model spec restated in full: four blocks, dropout, two heads, loss on the first, `global_skip=False` |
| 5 | **Three named components are inert on the fixture.** **[verified here]** Huber: 0/140 steps leave the quadratic region (max residual 0.9463 < δ=1.0), so Huber ≡ 0.5·MSE. StepLR: never fires at 20 epochs; LR constant 1e-3. Clip: max grad norm 0.0464, 0/140 bound | Measured table added; **C7** added so the scheduler fires; loss and clip declared unverified by the criteria |
| 6 | **C2's stated rationale was empirically false.** Shuffling is seeded; 1 vs 12 threads gives exactly 0.0 *(auditor's measurement)*. The band (0, 1e-4) is unpopulated on the training path | Rationale corrected; revision of C2 now requires a **demonstration** against the unmodified reference; CPU/BLAS declared unpinned |
| 7 | **The purpose claimed the papers; the criteria pin the deposit.** The deposit's own docstring says it is *"Distilled from the paper's ... script"* **[verified here]** | Purpose and failure text restated as the deposit; a "not about the papers" disclaimer added |
| 8 | **§6's signal-to-noise regime element was absent** | Regime paragraph added: same input regime, target at ≈W× exposure, W = 5 only |
| 9 | **The metric's independence assumptions were unstated** | C3 states the reference is generative, and the split section states this is an in-distribution single-signal evaluation |
| 10 | **Edge cases uncovered**: the `min(W, n-1)` clamp, the `ValueError` trigger (n=1 and W=0, not n=2), and duplicate acquisition indices making a distance-0 "other" frame | C1 cases **c, d, e**; C5 requires unique indices |
| 11 | **The `resample` path was noted as unexercised but not recorded as a gap** | **C6** added |
| 12 | **A vendored copy would pass every criterion** | "What reproduces means" added; `arhaxpes_denoise` added to the import prohibition for this work |
| 13 | **C4's justification described the papers' trained weights; C4 tests two untrained class definitions.** The SIA paper is also in press | C4's scope corrected; the dropout blind spot recorded |
| 14 | **The negative section was missing four disclaimers** (§5 preprocessing, §5 model-estimate, single-`W`, deposit-as-distillation) **and over-disclaimed one thing** C3 does support | All four added; the performance bullet narrowed to preserve C3's supported difference |
| 15 | **§8 process gaps**: the audit was scoped to the criteria, not the published sentence; only C2's revision was an audit item | Audit scoped to the claim wording; all criterion revisions are audit items |
| 16 | **"v0.1.0's methods are noise2clean and Noise2Noise" was false.** **[verified here]** `Noise2Self` ships at `src/dnndenoiser/training/methods.py:283` | Corrected to the CLI methods, with `Noise2Self`'s status named |
| 17 | **C3 was required on every device**, contradicting `benchmarks/reference/report.md`'s own record of device non-determinism | C0–C3 scoped to the pinned environment, reported elsewhere |
| 18 | **Leave-one-out independence was stated unconditionally** | Qualified: it holds *for independent frames*; drift and charging break it on measured stacks |

**Audited and found sound, unchanged:** C4's scope as a compatibility test; the
split statement; the draw-order and `float32` specification (an auditor rebuilt
the fixture from the document's constants alone and obtained the deposit's
values); C3's pairing design; C2's refusal of bit-identity; C1's refusal of a
tolerance; the reference-free-SNR disclaimer; the non-vendoring of the reference;
and the register-before-implementation discipline itself.

## Record

| | |
|---|---|
| Registered | 2026-09-21 (git `100940d`) |
| Amended | 2026-09-21, Revision 1, before implementation |
| Reference | `10.5281/zenodo.22092109` v1.0.0, digests above |
| Implementation | `6df2bfb`, `5515dfd`, `471ce84` |
| Result | below |
| Audit | **not yet done.** §8 requires it before the claim is published |

## Result — 2026-09-21

Run in the pinned environment: CPU, Python 3.12.11, `torch` 2.9.1, `numpy`
2.3.3, `torch.set_num_threads(1)` for the training criteria.

**All eight criteria are met.** 52 tests across five files; 1 skipped (the live
cross-package load, which runs only where the reference is present, and did pass
there).

| Criterion | Outcome | Measured |
|---|---|---|
| **C0** RNG-stream alignment | **met** | parameters bit-identical at all six seeds |
| **C1** targets exactly equal | **met** | all five cases; `numpy.array_equal` against the pinned reference digests |
| **C2** trained output, seed 0 | **met** | relative L∞ = **0.000e+00** (bound 1e-4) |
| **C3** five seeds, paired | **met** | **0.0000 dB** on every seed; paired mean +0.0000 ± 0.0000 dB (bound 0.5) |
| **C4** weights interchange | **met** | 22 keys, identical shapes, 658,177 parameters; cross-loaded outputs differ by 0.0 |
| **C5** frame-stack schema | **met** | documented, round-trips, rejects duplicate acquisition indices |
| **C6** resample path | **met** | exact at 128 and 512 points |
| **C7** scheduler fires | **met** | relative L∞ = **0.000e+00** at 30 epochs |

### Why C2, C3 and C7 are exact rather than merely inside tolerance

They agree to the bit. That is a fact about this pair of implementations, not
evidence about ports in general: `dnndenoiser`'s ResNet-FCNN and the deposit's
`DenoisingNetwork` share a code lineage — the deposit vendored its copy **from
this project** — so the two consume the random stream identically and the whole
computation is the same arithmetic in the same order. **C0 is what makes that a
stated, tested precondition rather than a coincidence**, and it is why the
criteria gate C2 and C3 behind it. An unrelated reimplementation would have no
such guarantee, and the tolerances exist for that case.

### What the implementation found that the criteria did not

Recorded because a preregistration that only records its own score is worth less
than one that records what running it cost:

- **The refusal rule in the CLI was wrong on the first attempt.** It keyed on
  "differs from the parser default", which lets `--arch` through in silence: the
  default is `FCNN` while this method is always ResNet-FCNN, so the value a user
  never touches is *already* the wrong one. It now keys on the flag being
  present. Two tests hold the distinction.
- **The pinned `.npz` fixture was matched by a blanket `*.npz` ignore rule** and
  would have shipped as a test CI cannot run. It has an explicit exception now.
- **The import guard failed the moment the C4 test imported the reference** —
  correctly. That test is a legitimate reader under "may read it to be tested
  against", so it is named in the allowlist rather than the rule being loosened.
  The guard parses rather than greps, because a substring search flagged the
  guard's own token list and the port's docstring citation.
- **A transcription error in the duplicate-index hazard.** It was recorded as
  "target[0] is exactly frames[1]", which is true at `W = 1` and false at
  `W = 5`, where the duplicate is one of five averaged neighbours. The test
  asserts the sharp `W = 1` form.
- **`--seed` did not exist on the train parser**, though the ported API takes
  one and construction consumes the random stream.

### The claim this licenses, and nothing stronger

*Audited wording, 2026-09-21.* The registered version of this sentence was
reviewed clause by clause; three clauses understated what was measured, and one
omission was structural — the sentence travels into `README.md`, `paper/paper.md`
and the changelog **without** the subsection above it, and read alone it would
sound like an independent-reimplementation result. The lineage disclosure now
travels inside the sentence.

> `dnndenoiser` implements the leave-one-out moving-average self-supervised
> training target and the ResNet-FCNN training loop of the archived
> `arhaxpes_denoise` reference implementation (Zenodo
> `10.5281/zenodo.22092109` v1.0.0). On that deposit's own synthetic fixture —
> 200 Poisson frames of a single Gaussian core level on a flat background, 16
> held-out frames, CPU — the port's targets are exactly equal to the
> reference's for `W ∈ {1, 2, 5, 10}` on contiguous acquisition order, for
> `W = 5` on a permuted one, and on the window-clamp, error and duplicate-index
> cases. At `W = 5` its trained outputs are bit-identical to the reference's at
> seed 0 (relative L∞ 0.0, against a registered bound of 1e-4), and the two
> implementations' clean-referenced output SNR differ by 0.000 dB on each of
> five further seeds (registered bound 0.5 dB). ResNet-FCNN `state_dict`s
> interchange between the two packages without renaming at
> `num_features=256, num_hidden_units=100, encoder_output_dim=64`. **The two
> agree to the bit because they share a code lineage — the deposit vendored its
> network from this project — not because an unrelated reimplementation would;
> the registered tolerances exist for that case.** The evidence does not pin the
> gradient-clip threshold: removing clipping entirely reproduces the reference
> exactly on this fixture. No measured data was used, and nothing here measures
> how well either implementation denoises.

Everything under "What this does not claim" still holds and is not relaxed by
the result. In particular this is **not** a claim about the JVST or SIA papers:
the deposit is those authors' own distillation of a training script, and only
the link from the deposit to this port was tested.

## Revision 2 — 2026-09-21, after the implementation audit

The audit re-derived all 31 pinned values from the deposit without using this
repository's generator: **31/31 match**, as do the measured constants and the
inert-component table. It found no criterion met by a weaker test than the one
registered, and no overstatement in the recorded Result. It found three
statements that were false as shipped, and blocked publication until they were
fixed.

| # | Finding | Change |
|---|---|---|
| 1 | **`QUICK_START` claimed flags were "refused rather than ignored"; five of six forms went through silently.** The rule matched literal `sys.argv` tokens, so `--lr=0.05`, `--arch=FCNN` and argparse's abbreviation `--weight-deca` all bypassed it, and `--grad-clip` and `--noise-level` were never in the list — `--grad-clip` names a component this method fixes at 4.0 | Tokens normalised and abbreviations resolved against the real option set; both flags added; `build_parser()` extracted so the option set is knowable; five bypass forms pinned as tests; the sentence made specific |
| 2 | **The sdist shipped the tests without their fixtures**, so collection aborted and *no* test ran from the released archive — a §10 defect twice over, since it also denies a third party the ability to reproduce the claim | `MANIFEST.in`; the golden generator ships too, since regenerating the pinned values is part of what makes the claim checkable. Rebuilt: 161 passed, 36 skipped, distribution boundary still clean |
| 3 | **"Loss, scheduler and clip are not verified" was false** and contradicted this document's own C7 paragraph. Measured: dropping the scheduler fails C7; `MSELoss` fails C2 at 1.25e-3; **removing gradient clipping entirely passes every criterion to the bit** | Bullet replaced with what is actually true of each of the three |
| 4 | **C0 passed while the failure it exists to catch went undetected.** An extra draw *inside* `train_selfsupervised` leaves construction identical; C2, C3 and C7 then fail and would be charged to the port | C0 extended to the parameters returned by `train_selfsupervised(..., epochs=0)`. Verified: the mutation now fails C0 itself |
| 5 | **C7's environment scope was narrowed in code without a log entry** | Document corrected — C7 *is* C2's statistic and carries its environment dependence. The code was right; the registered text was wrong |
| 6 | **A meta-guard was vacuous.** Changing the parser's `--arch` default left all eleven CLI tests passing, which would have made the refusal rule's rationale untestable while the Result claimed two tests held it | The guard reads the default out of the parser |
| 7 | **Nothing asserted the goldens' provenance.** The only check lived in the script regenerated in the same act as the file it certifies | The digest, DOI and version are restated as literals from this document |
| 8 | **"An independent reimplementation" overstated what was written** — closer to a faithful transcription, and the import guard cannot see transcription | The word withdrawn rather than the practice defended |
| 9 | **The falsification table had two wrong rows and three missing statements** | Corrected, with the not-falsified rows kept and marked |
| 10 | **The CLI creates claim surface no criterion touches** — `infer` cannot read the schema, normalisation constants are written but never applied, the default device is outside the pinned environment | Added to "What this does not claim" |

**Verdict recorded:** the criteria were met as written, with C7's scope the single
exception and that one an error in the document rather than in the code. The
claim may be published in the audited wording above, once findings 1–3 are
fixed — which they now are.

## Revision 3 — 2026-09-21, from CI

**The tie-break dependence is confirmed across builds, and C1's gate was
missing.** Revision 1 recorded, as *PLAUSIBLE and unverified* — "I have one
numpy build" — that the reference's own targets for odd `W` might not be
reproducible across numpy versions. Linux CI settled it: on `ubuntu-latest` at
Python 3.10 and 3.12, C1 failed **exactly** the tie-ambiguous cases and passed
**exactly** the tie-free ones.

| Case | Tie-ambiguous rows | CI |
|---|---:|---|
| `a`, `W = 1` | 86 | **failed** |
| `a`, `W = 2` | 0 | passed |
| `a`, `W = 5` | 107 | **failed** |
| `a`, `W = 10` | 0 | passed |
| `b`, permuted, `W = 5` | 90 | **failed** |
| `c`, clamp | 0 | passed |
| `e`, duplicate index | 85 | **failed** |

The correlation is exact, and it promotes that finding from PLAUSIBLE to
**CONFIRMED**.

This was a defect in the tests, not a revision of a criterion: the Environment
section already required C1 only in the pinned environment, and the test file
simply never applied the gate. The document was right and the code was wrong.

The gate is now **computed rather than listed** — `tests/p1_environment.py`
works out for each `(frame_indices, W)` whether the sort's tie-breaking decides
any row, and skips only those cases off the pinned build. Ten of C1's fourteen
tests still run everywhere, including `W = 2`, `W = 10`, the clamp, the error
cases, the two-frame boundary and both meta-guards; four skip with a reason that
names the row count. A hand-maintained list of "the odd-W ones" would have gone
stale the first time a case was added.

## Revision 4 — 2026-09-21, from CI again

Two failures on the release candidate, both defects in what had just been
written rather than in the port.

| # | Finding | Change |
|---|---|---|
| 1 | **C6's exactness was registered as required on every build, and is not.** `resample` is a `float32` `sgemm`; the reduction order is the BLAS implementation's choice, so a pinned digest is a statement about one build. CI failed both `n_new` values on `ubuntu-latest` while the port is exact against the reference locally — the same shape of error as C7's scope in Revision 2, found the same way | C6 split into a tolerance form required everywhere (`atol = 1e-6`, against the reference's array, now pinned alongside its digest) and an exactness form required on the pinned build. A structural test covers what no BLAS decides: shape, dtype, the `n_old − 2` clip, and that each output point is a convex combination |
| 2 | **The new version-consistency test used `tomllib`**, which arrived in Python 3.11, while `pyproject.toml` declares `requires-python = ">=3.10"` and CI runs 3.10. A test that cannot run on a supported interpreter is not a test of that interpreter | Read by regex instead; no new dependency |

**The pattern worth naming.** Three times now — C7 in Revision 2, C1 in
Revision 3, C6 here — a criterion turned out to be environment-dependent in a
way the registered text did not say. Each time the port was correct and the
*scoping* was wrong. Writing "required wherever the suite runs" is easy; knowing
which claims survive a different BLAS is not something a single machine can
tell you. The general rule this settles: **any criterion whose statistic is an
exact comparison of floating-point results is a claim about one build**, and
belongs behind the environment pin unless it has a tolerance form as well.

## Revision 5 — 2026-09-24, from a Windows run

The suite was run on Windows 11 (x86-64, Ryzen 9 8940HX) at the pinned
versions, `torch` 2.9.1 in its `+cpu` and `+cu128` builds. Two findings, both
about the gate, not the port.

| # | Finding | Change |
|---|---|---|
| 1 | **The gate never admitted any Windows wheel.** It compared `torch.__version__` verbatim, and Windows wheels carry a build label (`2.9.1+cpu`, `2.9.1+cu128`). C0, C2, C3, C6's exactness, C7 and C1's tie-ambiguous cases skipped there for a reason that was not the real one | Versions compare on the release number alone |
| 2 | **With the label mismatch removed, the gate would have required these criteria on x86-64, where they fail.** The Environment section already said the pin fixes Python, `torch` and `numpy` "but **not** the CPU microarchitecture", and makes results on another CPU reported, not required; the gate did not encode that. Removing the label check alone would have turned the section's "reported" into "required and failing" | The platform (`platform.system()-platform.machine()`) is part of the pin: `Darwin-arm64`, recorded in the fixture's `_environment`. `tests/test_p1_environment_gate.py` shows the gate refusing `Windows-AMD64` and `Linux-x86_64` and admitting the pin with or without a build label |

**No criterion's statistic, tolerance or fixture value changed.** C7 keeps
`1e-4`; a separate x86-64 tolerance would be a revision of a criterion, and
"Revision of C2 requires a demonstration, not an argument" applies to it
equally.

**Reported, not required — x86-64, Windows 11, CPU**, as reported by the
Windows run and not reproduced here:

- C1 fails in the four tie-ambiguous cases, C2, C3 and C7 fail.
- With numpy's SIMD dispatch turned off (`NPY_DISABLE_CPU_FEATURES`), C1's
  targets match the pinned sha256 bit for bit, and C2 and C3 pass. That
  attributes C1's difference to x86 SIMD `argsort` resolving ties differently —
  the dependence Revision 3 confirmed across builds, now seen across
  instruction sets — and C2's and C3's to the targets they train on.
  **PLAUSIBLE**: one machine, one run. *(Upgraded to CONFIRMED for that
  machine below, from the run's own output.)*
- C7 still fails with SIMD off: relative L∞ `1.8e-4` against `1e-4`. The cause
  proposed, a difference between `torch`'s CPU kernels on x86-64 and arm64, is
  **unverified**.

### The Windows run's output — received 2026-09-24, after Revision 5

The findings above were written from the Windows run's summary. Its output
followed and is recorded here. It is still the output of one machine, received
as text, and not reproduced on it by anyone else; what was checked here is
stated where it was.

**Machine and build.** Windows 11, AMD Ryzen 9 8940HX (Zen 4, with AVX-512),
Python 3.12.14, `torch` 2.9.1+cpu, `numpy` 2.3.3. To run the gated tests the
run replaced `platform.system` and `platform.machine` after importing `torch`;
nothing else in the gate or the tests was changed.

**SIMD off** means
`NPY_DISABLE_CPU_FEATURES="AVX F16C FMA3 AVX2 AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"`:
every AVX-family feature numpy dispatched to on that CPU, plus F16C and FMA3,
leaving SSSE3, SSE41, POPCNT and SSE42. Which one of them decides the ties was
not narrowed down.

**C1.** Targets computed from the fixture's cases directly, outside pytest.
"Tie rows" counts the rows whose neighbour set differs between the default and
a stable `argsort`, on that build.

| Case | Tie rows, SIMD on → off | sha256 equals the pinned value, SIMD on / off |
|---|---|---|
| `a_arange_W1` | 98 → 86 | no / **yes** |
| `a_arange_W2` | 0 → 0 | yes / yes |
| `a_arange_W5` | 98 → 107 | no / **yes** |
| `a_arange_W10` | 0 → 0 | yes / yes |
| `b_permuted_W5` | 98 → 90 | no / **yes** |
| `c_clamp_n3_W10` | 0 → 0 | yes / yes |
| `e_duplicate_index_W5` | 97 → 85 | no / **yes** |

*Checked here:* the seven SIMD-off digests the run reported were compared with
`cases/*/sha256` in `tests/fixtures/p1_reference_targets.json` and all seven are
equal; the four SIMD-on digests equal none. The SIMD-off tie counts equal
Revision 3's table, which was computed on the pinned build.

**C2, C3, C7**, at `1bcbbf7`, with the platform replaced as above:

| | SIMD on | SIMD off |
|---|---|---|
| C1 | 4 failed (the four tie-ambiguous cases) | passed |
| C2, relative L∞ (limit `1e-4`) | `4.066e-3`, failed | passed |
| C3 | failed: seed 5, port 33.245 dB vs reference 32.549 dB, \|Δ\| 0.696 dB > 0.5 | passed |
| C7, relative L∞ (limit `1e-4`) | `3.351e-3`, failed | `1.825e-4`, failed |

**Reading.**

- **C1's difference is the SIMD `argsort` tie-break: CONFIRMED on this machine.**
  Turning numpy's SIMD dispatch off, and changing nothing else, moves every
  tie-ambiguous case from a wrong digest to the pinned one, bit for bit, and the
  tie counts to the pinned build's. That is an intervention with an exact
  outcome, not a correlation. It is confirmed for one CPU; which instruction set
  is responsible, and whether other x86-64 CPUs behave the same, is not known.
- **C2's and C3's failures with SIMD on follow from C1's targets.** With the
  targets restored, both pass on this machine. This is the attribution above,
  now observed rather than inferred.
- **C7's remaining `1.825e-4` is not explained.** A difference between
  `torch`'s CPU kernels on x86-64 and arm64 is the proposed cause and remains
  **unverified**. On the same targets C2 is inside the limit at 20 epochs and C7
  is not at 30, where the scheduler fires; the run did not report C2's value, so
  how far the two are apart is not known. That C7's excess is small is not a
  reason to widen its tolerance, and none is widened.

