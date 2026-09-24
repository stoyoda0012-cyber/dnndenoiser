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
| `record_citations.py` | What every number quoted in the preregistration's Record section means and how it is derived from the record. Checked by `tests/test_p2a_record_citations.py`. |

`../../../docs/preregistration/P2A-position-shift-boundary.md` is the registered
design. Its predictions and decision rules were fixed and audited **before** any of
this was implemented (Revision 1); the later revisions, all logged, changed self-checks,
provenance, rendering, scope wording and the record's structure — never a prediction.
It is the document to read first: it says what was predicted, what the decision rules
were, and what a failed prediction means.

**Units.** Shifts and boundaries are measured in eV. `report.md` also gives each
boundary in bins of this record's energy grid and as a multiple of the dominant
peak's nominal FWHM. Those are conversions of the same number for this record's one
setup; whether any of the three units carries over to another grid or line width
was not tested.

The rule that keeps this honest is the reference benchmark's: **numbers are never
typed by hand**, and nor is anything else in `report.md`. `render_report.py`
rebuilds, from the record's `runs` array alone, every aggregate field and the
whole `boundaries` and `predictions` trees — including each verdict, sign count,
test statistic and Holm-adjusted *p* — and refuses to render if any of it
disagrees. It reports all disagreements together rather than stopping at the
first.

Two things about that guard, stated because its first version claimed more than
it did. The aggregate half is an independent recomputation; the trees are a
consistency check against the measurement module's own functions, so it catches an
edited or stale record but not an error inside those functions. The self-check
figures, the environment and the consistency anchor are **not** derivable from
`runs` and are not verified — the report says so where it prints them.

The claim is under test, not asserted: `tests/test_position_shift_boundary_record.py`
tampers with a copy of the record one field at a time and requires the guard to
reject each one. It exists because an audit tamper-tested the first guard and it
accepted 25 of 30 edits, including flipping a prediction's verdict.

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
record shows that a change in N of that size is worth several dB on its own, and
the design assumed that thinning position density by the same factor would cost
about as much — an assumption R5b's result calls into question (see below). Without
C and D,
any deficit arm B shows at Δ = 0 would be unattributable — and
the "augmentation costs something" prediction would pass for entirely the wrong
reason. An independent audit of the preregistration caught this before any
compute was spent.

## Re-running it

```bash
pip install -e ".[dev]"

# The full measurement, as recorded. Between about half an hour and nearly three
# hours on the Apple-silicon machine that made the record; why it varied was not
# investigated. The record's own total_wall_clock_seconds is authoritative. Device auto-selects
# cuda > mps > cpu.
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

Thirteen, each with a stated failure condition. Every one except check 6 is also
**shown to fail**: `tests/test_position_shift_boundary_gates.py` gives each a correct
input it must accept and a named wrong input it must reject, with the rejection
pinned to the reason the gate exists for. Check 6 is written inline in `run()` and is
exercised only by full runs. Rejecting one constructed failure is not proof that a
gate catches every failure; each test names the one it covers. The record is not written if one
fails; a broken measurement is not reinterpreted as a finding.

1. **Parameter count** — read from the reference record, not typed here.
2. **Grid invariance** — the energy axis is bit-identical at every shift.
3. **Test-sweep rigidity** — every peak centre equals *the literal* nominal value
   plus Δ. The baseline is a literal because `get_peak_set` returns the shared
   module-level object: a check that re-read its baseline through it would
   compare a mutated registry against a result produced from the same mutation.
4. **Training-pool rigidity** — each sampled training spectrum, in *every* arm, is
   reconstructed from the literal peaks, the shift the pool recorded for it, and
   the per-peak draws replayed from the generator's own RNG at the pinned jitter,
   and must be **bit-identical to the spectrum in the pool**. Plus an end-of-run
   comparison of `PEAK_SETS` against the literals. The first version of this check
   re-derived a peak set from the recorded shift and compared it against the same
   expression, never touching the pool; an audit built the wrong pool it was
   written to catch and it passed. The repaired check is verified to fail on that
   pool, on a shift recorded but never applied, and on a shift applied at the
   wrong jitter width.
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
8. **Pairing integrity** — that the arms share their per-sample generator seeds,
   keyed on `(level, sample)` because the arms hold different numbers of samples
   per level. It establishes seeding; check 4 establishes that the data follows.
   **8b test-family rigidity** — test spectra are reconstructed from the
   shift-independent family seed and required bit-identical, so it fails if a
   family was drawn from a different seed, if the sweep is not rigid, or if the
   jitter width moved. It replaces a loop whose body never used the shift variable.
9. **Argmax well-posedness and reference identity** — the clean references peak at
   `284.8 + Δ`, and the array inspected is asserted to be the one the SNR metric
   actually received, which `snr_db` records. The check runs *after* the metric.
   The first version asserted a variable against itself at the call site and could
   not fail, while the record advertised that it had.
10. **Augmentation actually happened** — the narrow arms drew no shift; the
    augmented arm's realised SD matches the uniform SD and all ten deciles of its
    range are occupied. Min, max and mean alone are satisfied by an arm that was
    never augmented at all, which an audit demonstrated.
11. **Noise-model identity** — field by field against the literals. The
    Gaussian-approximation boolean is `False` at level 10000, so the three levels
    do not share a code path, and an implementation that set it everywhere would
    change the noise model with no other check noticing.
12. **Leakage** — byte-identity across **all four arms** and stream-base
    disjointness. Stated limitation:
    byte-identity can only detect a collision in the Δ = 0 column, because at any
    other shift a collided spectrum is shifted and no longer identical.

**Diagnostics are recorded, never asserted** — run-to-run determinism, the
nearest-duplicate statistic, each arm's realised shift distribution, and the
per-shift values from checks 5, 6 and 7. They are listed separately so the count
of actual gates is not overstated.

## Status

A record is published here, from the **sixth** full run, made in the environment
`uv.lock` pins, with every self-check as registered, carrying its own provenance, and
holding only what was measured, what is derived from it, and what was fixed before
the results — the result-dependent cautions are in the preregistration's Record
section and in `report.md`. The first two runs were discarded:
the first because a self-check was inert, the second because two independent audits
found the apparatus did not verify the study's independent variable — nothing
inspected the training data of the augmented arm or the density controls, and the
checks meant to were comparing expressions against themselves. The third to
fifth were superseded — the third and fourth because their self-check 4 sampled 12
spectra per arm instead of the registered 5 %, the fifth because its `claim_scope`
still held result-dependent interpretation. The preregistration's revision log documents all of it,
and the gate tests now reproduce the wrong inputs that got through.

The second to sixth runs' records are all in git history, and every per-run
value in them is identical — across a change of operating-system version and a move
from a shared conda base to the pinned environment. The numbers were never what was
wrong; the claim "all self-checks passed" was.

**Re-running it** uses the pinned environment: `uv sync --extra dev`, then
`.venv/bin/python benchmarks/boundaries/position_shift/position_shift_boundary.py`
from a clean working tree — a full run from a tree with uncommitted or untracked
changes is refused, because the record would name a commit that is not what ran.

**Not yet cleared for outward-facing quotation.** Revision 11 of the preregistration
proposes a limited clearance for one page,
[`docs/WHEN_TO_TRUST.md`](../../../docs/WHEN_TO_TRUST.md); it takes effect only when the
owner confirms it in Revision 11, after an independent review of the page and of the
Record-section claims it quotes. Until then, and for everything
the proposal does not list, no number from here goes into the package README, the
documentation or a release note.

## Things about a record from this design that are easy to misread

Most of these follow from the design; where one rests on the record's result, it says
so. The result itself, with its conditions, is in the preregistration's Record section.

- **The boundary is a property of the training distribution, not of XPS.**
  Whatever |Δ|\* the record reports is a property of *this* peak set, *this*
  ±0.3 eV per-peak jitter, *this* architecture, *this* training-set size and
  *this* noise model. One point was measured in each of those spaces.
- **Augmentation "working" is not augmentation being free, and not augmentation
  being right.** The sweep runs past the augmented arm's training range so that
  its own edge is visible. Its *price* is a different question, and arms C and D
  do **not** answer it: they were designed to, and R5b failed 0/20 in the
  opposite direction. Either cutting N does not stand in for thinning position
  density, or augmentation has an effect at Δ = 0 the design did not anticipate;
  no arm separates the two, so nothing in this design bounds augmentation's cost
  in either direction. Calibrating the instrument is a different kind of answer
  to a calibration error, and nothing here compares the two.
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
  preregistration first**, visibly, with the reason and the date. The
  preregistration's revision log shows how many such revisions there have been and
  what forced each.
- Changing the noise model, the generator, or the meaning of a metric is an
  independent-audit item (`AGENTS.md` §8). So is quoting any number from this
  record anywhere outward-facing.
- Do not delete a superseded record to make a new one look tidier, and do not
  soften a failed prediction. Negative and undecided results stay
  (`AGENTS.md` §6).
