# P2-A — the position-shift boundary

A reproducible record of **where this package's denoiser stops working** when the
spectrum it is given sits at a different binding energy from the one it was
trained on, and of **what training across a range of positions does to that
boundary**.

It exists because the reference benchmark's own README says, in as many words,
that nothing in it probes distribution shift — its training and test spectra are
independent draws from one distribution — while `AGENTS.md` §5 says distribution
shift is the dominant failure mode of a denoiser. So the package shipped a
measurement of in-distribution performance and no measurement of where that
stops being true. This is the first of the P2 series that closes that gap.

Nothing in this directory is distributed in the sdist or wheel (`AGENTS.md` §3).

## Files

| File | What it is |
|---|---|
| `position_shift_boundary.py` | The measurement. Run it to produce a record. |
| `results/position_shift_boundary.json` | **The primary record.** The only place numbers live. |
| `render_report.py` | Regenerates the Markdown report *from* a record. |
| `report.md` | Generated output of `render_report.py --write`. Do not edit by hand. |

`../../../docs/preregistration/P2A-position-shift-boundary.md` is the registered
design. It was fixed, audited and revised **before** any of this was implemented,
and it is the document to read first: it says what was predicted, what the
decision rules were, and what a failed prediction means.

The rule that keeps this honest is the reference benchmark's: **numbers are never
typed by hand**, and nor is anything else in `report.md`. `render_report.py`
recomputes, from the per-run raw numbers, everything it is about to render — the
aggregate means, the across-seed standard deviations, the degradation series, the
bias-corrected displacements, the boundary medians and censored counts, and every
prediction's sign count and one-sided binomial *p* — and refuses to render if any
of them disagrees. It reports all disagreements together rather than stopping at
the first.

## What it measures

A **rigid energy shift** Δ of every peak at once, with the energy grid pinned.
In XPS that is the shape of a sample-charging offset or a binding-energy
calibration error: the whole envelope moves, the structure within it does not. A
chemical shift is **not** this — it moves components relative to one another —
and is out of scope.

Four models, all `ResNet-FCNN` at the reference benchmark's recipe, differing
only in the position distribution they were trained on:

| Arm | N | Training shift |
|---|---|---|
| **A** narrow | 2304 | none |
| **B** augmented | 2304 | Δ ~ U(−1.5, +1.5) eV, rigid, drawn per training spectrum |
| **C** density control | 461 | none |
| **D** density control | 144 | none |

All four are tested over the same Δ ∈ ±4.0 eV sweep, which reaches **beyond**
arm B's training range on purpose. The registered question is not "does
augmentation fix it" but **"does augmentation remove the boundary or move it"**,
and a test range that stopped at the augmentation range could not tell those
apart.

**Why C and D exist.** Arms A and B differ in *two* things, not one: whether
they were augmented, and how densely they sample the position axis. At equal N,
arm B has one fifth of arm A's per-peak marginal density and one sixteenth in the
three-peak joint configuration space. This repository's own training-set-size
record shows that a density change of that size is worth several dB on its own,
so without C and D any deficit arm B shows at Δ = 0 would be unattributable — and
the "augmentation costs something" prediction would pass for entirely the wrong
reason. An independent audit of the preregistration caught this before any
compute was spent.

## Re-running it

```bash
pip install -e ".[dev]"

# The full measurement, as recorded. About half an hour on an Apple-silicon
# machine; device auto-selects cuda > mps > cpu.
python benchmarks/boundaries/position_shift/position_shift_boundary.py

# Regenerate the report from the record.
python benchmarks/boundaries/position_shift/render_report.py --write

# Fast smoke test of the whole pipeline (seconds). Give it an output directory:
# the default is results/, and a quick run there overwrites the published record.
# Its numbers are NOT meaningful -- two seeds cannot satisfy a 15-of-20 sign rule,
# so every prediction reports FAIL by construction.
python benchmarks/boundaries/position_shift/position_shift_boundary.py \
    --quick --output-dir "$(mktemp -d)"

# Lint, matching the project's configured rules. CI lints src/ and tests/ only,
# so this directory is not linted automatically.
ruff check benchmarks/boundaries/position_shift/
```

Useful flags: `--device {auto,cpu,cuda,mps}`, `--seeds`, `--n-test-per-level`,
`--epochs-cap`, `--output-dir`, `--output-name`, `--skip-determinism-check`.

**If you have to cut, cut test-set size, not seeds.** The seed is the replicate
unit and the only thing the error bars mean. Note one asymmetry with the
reference benchmark, though: here the test-set size is doing real work of its
own, because the boundary statistic is a *first crossing* whose downward bias is
governed by the noise in each seed's own gain curve and is one-sided, so no
amount of averaging across seeds removes it. Self-check 6's tolerance is
calibrated at the registered test size and is scaled by √(512/n) below it, so a
smaller run is not voided for a reason that has nothing to do with the
manipulation.

## Self-checks that void the record

Twelve, each with a stated failure condition. The record is not written if one
fails; a broken measurement is not reinterpreted as a finding.

1. **Parameter count** — read from the reference record, not typed here.
2. **Grid invariance** — the energy axis is bit-identical at every shift.
3. **Test-sweep rigidity** — every peak centre equals *the literal* nominal value
   plus Δ. The baseline is a literal because `get_peak_set` returns the shared
   module-level object: a check that re-read its baseline through it would
   compare a mutated registry against a result produced from the same mutation.
4. **Training-pool rigidity** — the same, on a 5 % sample of *every* arm's pool,
   plus an end-of-run comparison of `PEAK_SETS` against the literals. This is the
   check that fails on a pool built with per-peak jitter instead of a rigid
   shift — the manipulation this study puts out of scope, and the one that an
   audit showed would otherwise have satisfied every prediction.
5. **Truncation, total and per peak** — the pseudo-Voigt tails put about 1.6 % of
   the nominal peak area outside this window at *every* shift, zero included.
   What the cap on |Δ| controls is the *change*. Per-peak figures are recorded
   separately because truncation is asymmetric between directions, in the same
   direction an asymmetric boundary would be: **a claimed directional asymmetry
   has to clear the asymmetry recorded here.**
6. **Input-SNR invariance** — the noisy input's own SNR must not vary with Δ, or
   the gain curve is confounded at the source. Level 10000 has a wider allowance
   because its noise is not paired across shifts (see below).
7. **Translation equivariance** — checks 3 and 4 inspect peak *centres* and would
   pass on a spectrum that is not a translate at all. This bounds the departure
   from a true translate, which is non-zero because `linear_background` returns
   `level + slope * (x - x[0])` — a ramp pinned to the **window**, so it does
   not travel with the peaks and a peak moving along it sits on a different
   background level. The comparison is against `np.roll` at integer grid
   offsets, because re-evaluating the generator on a displaced grid computes the
   identical expression on both sides and cannot fail; the first implementation
   did exactly that and is recorded as a tautology in Revision 3.
8. **Pairing integrity**, and **8b replay faithfulness** — that the arms and the
   shifts really do share their per-peak draws, and that the replay used to check
   it reproduces the generator bit-for-bit. Without 8b, check 8 would be comparing
   one model of the generator against itself.
9. **Argmax well-posedness and reference identity** — the array inspected is
   asserted to be the same object the SNR metric divides by, which is what
   establishes that the metric scores the Δ-matched truth rather than the
   unshifted one.
10. **Augmentation actually happened.**
11. **Noise-model identity** — field by field against the literals. The
    Gaussian-approximation boolean is `False` at level 10000, so the three levels
    do not share a code path, and an implementation that set it everywhere would
    change the noise model with no other check noticing.
12. **Leakage** — byte-identity and stream-base disjointness. Stated limitation:
    byte-identity can only detect a collision in the Δ = 0 column, because at any
    other shift a collided spectrum is shifted and no longer identical.

**Diagnostics are recorded, never asserted** — run-to-run determinism, the
nearest-duplicate statistic, each arm's realised shift distribution, and the
per-shift values from checks 5, 6 and 7. They are listed separately so the count
of actual gates is not overstated.

## What the record says, in one paragraph

Read `report.md` for the numbers; it is generated and they are not repeated here
(`AGENTS.md` §6). In outline: the boundary for a model trained at one
calibration sits **well under one electronvolt** — a shift of the size a
practitioner would not think twice about is enough to make the denoiser worse
than doing nothing. Training across a ±1.5 eV range moves the boundary out by
roughly a factor of four and does **not** remove it: that model has a flat
plateau over the range it saw and its own cliff just outside it. Seven of the
eight registered predictions held. The one that failed, R5b, failed because its
premise did not survive contact with the data — see the preregistration's
Record section, which says what that costs the interpretation.

## Things about this record that are easy to misread

- **A positive SNR gain does not mean the peak is in the right place.** The
  record contains a cell where the gain is comfortably positive *and* the
  denoised peak sits about four electronvolts from the truth, at the noisiest
  level, where the input is so poor that a smooth wrong answer still scores well.
  This is `AGENTS.md` §5's "model estimate, not a measurement" as a measured
  number. It is also **outside every registered prediction** and carries no
  inferential claim here — it is a caution and a candidate for a new
  preregistration.
- **The half-electronvolt figure is not a property of XPS.** It is a property of
  the training distribution. The same architecture trained across a wider
  position range has a boundary several times further out, in this same record.
  What generalises is the *shape* — a model is valid just past the position
  range it was trained on, and not beyond — not the number.
- **R5b's failure is not evidence that augmentation is free.** The registered
  fallback is an undecided verdict, and the reason is that the density controls
  turned out to be *N* controls: cutting the training-set size removes
  information about noise, intensity and width as well as position, so it is a
  harsher handicap than spreading a fixed N over a wider range. Attributing
  augmentation's cost needs a design this one does not have.

- **The boundary is a property of the training distribution, not of XPS.**
  Whatever |Δ|\* the record reports is a property of *this* peak set, *this*
  ±0.3 eV per-peak jitter, *this* architecture, *this* training-set size and
  *this* noise model. One point was measured in each of those spaces.
- **Augmentation "working" is not augmentation being free, and not augmentation
  being right.** The design measures its price (arms C and D) and looks for its
  own edge (the sweep runs past its training range). Calibrating the instrument
  is a different kind of answer to a calibration error and this record does not
  compare the two.
- **Beyond |Δ| = 1.5 eV, degradation is not separable from window-edge effects.**
  Inside that range arm B *is* an edge-proximity control, because it saw those
  edge distances in training. Beyond it, no arm did.
- **Level 10000 carries more draw noise than the other two levels**, and not for
  a reason about denoising: the pinned noise reconstruction takes the `rng.poisson`
  branch there, whose bit consumption is rate-dependent, so the noise realisation
  is effectively independent across shifts (measured correlation ≈ 0.18) where at
  the other two levels it is a shared standard-normal vector rescaled
  (≈ 0.98). Degradation curves at that level are therefore noisier by
  construction.
- **Only the cells named in the registered predictions carry an inferential
  claim.** The record holds roughly six hundred cells across four arms, three
  levels, twenty-five shifts and two metrics. A dip or an asymmetry in a cell no
  prediction named is a candidate for a *new* preregistration, not a finding of
  this one.
- **M3 is a grid argmax, not a peak fit.** It is quantised to the energy step,
  describes the dominant peak only, and supports no claim about fitted binding
  energies. It is reported bias-corrected because a learning-free smoother
  produces a shift-*independent* offset of up to three grid steps, which would
  otherwise help in one direction and hurt in the other.
- **A denoised spectrum is a model estimate, not a measurement.** A high SNR gain
  says the output agrees with a known synthetic reference. It does not establish
  that structure in the output is real.

## If you change something

- Knobs on the command line (`--seeds`, `--n-test-per-level`, `--device`) are
  written into each record, so two records differing only in those are comparable
  *as long as you say which is which*.
- Changing a fixed condition in the source — the peak set, the generator config,
  the noise model, the shift grid, the arms, the recipe, or the meaning of a
  metric — makes the new record incomparable with the old one. Bump
  `RECORD_VERSION` and say what changed.
- **Changing anything the preregistration fixed means revising the
  preregistration first**, visibly, with the reason and the date. Two such
  revisions already exist and both were forced by implementation, not by results.
- Changing the noise model, the generator, or the meaning of a metric is an
  independent-audit item (`AGENTS.md` §8). So is quoting any number from this
  record anywhere outward-facing.
- Do not delete a superseded record to make a new one look tidier, and do not
  soften a failed prediction. Negative and undecided results stay
  (`AGENTS.md` §6).
