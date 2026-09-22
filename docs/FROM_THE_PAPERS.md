# From the papers to your own data

For a reader of the JVST A or Surface and Interface Analysis paper who wants to
apply the denoising method to their own measurements.

- **JVST A** — denoising strategies for three-dimensional visualisation of
  AR-HAXPES depth profiles: [10.1116/6.0005585](https://doi.org/10.1116/6.0005585)
- **SIA** — cross-exposure transferability and the failure boundaries of
  self-supervised DNN denoising: [10.1002/sia.70123](https://doi.org/10.1002/sia.70123)
- **The papers' deposit** — figure data, the figure-generating code, and a
  standalone implementation of the denoising front-end they evaluate:
  [10.5281/zenodo.22092108](https://doi.org/10.5281/zenodo.22092108)

This page says what each piece is for, and — more usefully — what is *not*
available and what this repository will not do for you.

## What this repository is, in this context

`dnndenoiser` is the maintained, general home of the denoising method. It
carries a port of the papers' self-supervised training target, so you can train
on **your own frames** rather than reuse a model trained on someone else's
sample.

It is not the papers' pipeline. It does no depth profiling, no peak fitting and
no quantification, and it reads no vendor file formats — those are deliberate
non-goals ([`AGENTS.md`](../AGENTS.md) §1), not gaps waiting to be filled here.

**What the port establishes, and what it does not**, is recorded in full in
[the preregistration](preregistration/P1-selfsupervised-moving-average.md): its
acceptance criteria were fixed before it was written and audited twice. The
short version is that it reproduces the deposited reference implementation on
that deposit's own synthetic fixture, exactly — and that nothing in that
exercise measures how well either implementation denoises, or says anything
about behaviour on measured data.

## The route

### 1. Read your files — with `toyomacro`, not here

[`toyomacro`](https://github.com/stoyoda0012-cyber/toyomacro) has the readers
(`toyomacro.io`: PXT, VAMAS, NPL, two-column text). `dnndenoiser` takes NumPy
arrays and HDF5, and that is the boundary: an instrument loader here would be a
non-goal, and a per-instrument calibration constant in library code would be
worse.

### 2. Put the frames into the frame-stack layout

The method trains from **repeated acquisitions of the same spectrum**, not from
clean/noisy pairs. The layout is three datasets — `frames`, `energy`,
`frame_index` — and it is described in [QUICK_START](QUICK_START.md).

`frame_index` is the part that matters and the part easiest to get wrong. It is
the acquisition order, and "temporally nearest" is measured on it. If your
frames are not stored in the order they were acquired, the index must say so;
a file that omits it cannot be told from one whose frames were shuffled.
Duplicate indices are rejected, because two frames sharing one become
distance-zero neighbours of each other and each leaks straight into the other's
training target.

### 3. Train on your own frames

```bash
dnndenoiser train -d your_stack.h5 -o model.pt \
    --method moving-average --window 5 --epochs 50
```

`W = 5` is the papers' canonical window. Larger `W` is a longer effective
exposure in the target. The optimiser, schedule, loss and architecture are
fixed — they are part of the method — so flags that would change them are
refused rather than ignored.

**Training only.** Inference from a frame stack is not wired through the CLI
yet. `infer` reads the `noisy`/`energy` layout, and a moving-average checkpoint
used with it will have its recorded normalisation applied and inverted — but
the constants are your *training stack's*, so if the spectra you infer on sit
on a different scale, you are feeding the model data outside the distribution
it saw.

### 4. Evaluating it is the hard part, and not for the reason people expect

There is **no reference-free SNR for a measured spectrum**. The obvious
substitute — a mean over all your frames — is *not independent of the training
targets*, which are means of subsets of those same frames. A number computed
that way is not a held-out result and must not be reported as one.

This is not a limitation of this implementation. It is why the papers evaluate
against reconstructed depth profiles and pseudo-ground-truth rather than against
a spectrum-level SNR, and why `dnndenoiser evaluate` requires a clean reference
and therefore synthetic data.

Denoising is also preprocessing: peak areas, positions and widths have to be
checked downstream, not assumed to have survived
([`AGENTS.md`](../AGENTS.md) §5).

### 5. Depth-profile metrics — in the deposit, not here

Depth RMSE, Δthickness and the Wasserstein-1 distance, along with bin-pool
resampling, are in the papers' deposit
([10.5281/zenodo.22092108](https://doi.org/10.5281/zenodo.22092108)). They stay
out of `dnndenoiser` because depth profiling is out of scope. Use them there.

The two DOIs you will see for that deposit are not a contradiction: the paper
cites the **concept** DOI, which resolves to the latest version, while this
repository's preregistration pins the **version** DOI of v1.0.0, because a
reference measured against has to be immutable.

## What is not available

Stated plainly, because the paper states it plainly and a reader should not have
to discover it by trying. Per the SIA Data Availability Statement, these were
**not deposited**:

- the L1-regularised depth-profile inversion solver;
- the trained network weights;
- the raw single-frame HAXPES measurement data (~60 GB).

They are available from the corresponding author on reasonable request. So
**you cannot reproduce the papers' figures end to end from public artifacts
alone**, and this repository does not change that. What you can do is train the
same method on your own measurements.

The paper also states that those weights load unmodified into the released
implementation. `dnndenoiser`'s ResNet-FCNN is `state_dict`-compatible with that
implementation at the default shape, which is tested — but that is a statement
about two class definitions, not a route to weights nobody has published.

## If your sample is not like theirs

Read [`AGENTS.md`](../AGENTS.md) §5 before trusting any output. Distribution
shift is the dominant failure mode: a model is valid inside the distribution of
the frames it was trained on — peak structure, position range, noise regime,
background, normalisation, instrument response — and outside it, the output can
be worse than the input. Training on your own frames is what keeps you inside
it, which is the reason this route exists rather than a pretrained model.

The SIA paper is itself about where that boundary is. It is worth reading for
the failure modes, not only for the method.
