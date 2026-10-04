# Design: training several seeds, and combining their outputs

**Status: adopted 2026-10-04 (revision 2, after two independent audits); implemented (see
the CHANGELOG).** Step 6, the last, of the improvement plan that follows a reproducibility
assessment of a published study. It adds a way to train one model per seed in one command,
and a way to apply several such models to the same input and keep every output with an
ensemble mean and a between-run spread. Under `AGENTS.md` §8 it is audited before code is
written: the ensemble mean is a new estimator and the spread a new reported quantity, and
both could be misread as uncertainty. It uses the vocabulary of
`docs/design/PROVENANCE_MANIFEST.md` (step 2) and `docs/design/REPRODUCIBILITY.md` (step 4).

## 1. The problem

In the assessed study, conclusions changed with the seed — which architecture did best, and
whether a model failed — and a single training run was all that stood behind several of them.
Today `train --seed` makes one model; seeing seed dependence means scripting several runs by
hand and keeping track of which output came from which.

The plan audit (2026-09-30) set the conditions this step must meet, in substance:

- several seeds show variation between runs; they do not test whether a comparative
  conclusion would reverse, and three seeds are a cheap exploratory start, not evidence of
  stability. More seeds on one measured stack do not add independent specimens;
- the ensemble mean is a new estimator: it can hide an individual failure, and models can
  agree closely while sharing a bias or a distribution-shift failure; the spread between runs
  is **not** measurement uncertainty, a confidence interval or a bound on error, and must not be
  divided by √K and shown as uncertainty of the spectrum;
- models must be checked compatible before they are combined, each applied with its own
  normalisation; different training conditions must not be summarised as "seed spread", and
  duplicate models must not count as independent members;
- every member's prediction is kept, failures included; diagnostics are per member;
- comparisons between conditions are paired per seed, on the quantities behind each
  conclusion; that downstream analysis belongs to the consuming project.

## 2. Training several seeds: `train --seeds`

`train --seeds 0 1 2 -o model.pt` (mutually exclusive with `--seed`; at least two seeds, since
one seed is `--seed`) runs the existing single-seed training once per seed, **in the order given, in one process**, and writes one
ordinary checkpoint per seed.

**Names.** `-o` must end in `.pt`; member `s` is written to `<stem>.seed<s>.pt` beside it
(`model.seed0.pt`, …). `-o` itself is not written. Before the first member trains, every
member path is computed and **an existing file at any of them is refused**, as are seeds that
are repeated or negative.

**Each member is the single run.** For member `s`, `args.seed` is set to `s` and the run goes
through exactly the code path of `train --seed s`; its manifest records `seed: s`,
`seeds.torch: s`, and `flags_passed` as if `--seed s` had been typed in place of `--seeds …` (the seal takes the flag set as a
parameter instead of reading `sys.argv`).
`--seeds` itself is not recorded (it joins `EXCLUDED_ARGUMENTS`), so nothing in a member says
it was one of a batch, and the manifest schema does not change (open question 3 of
revision 0, as proposed there and recommended by the audit). **Member `s` has the same `model_body_digest` as
`train --seed s` run alone** with the same other arguments under step 4's Tier 1 conditions,
and its manifest differs only in `created_utc`. This holds because each run calls
`torch.manual_seed(s)` before building the model, which resets the global generator that the
initialisation, the batch order and dropout draw from; noise2noise builds fresh NumPy
generators from `s`; no path draws from NumPy's global generator. It was checked by the audit
on one machine (all architectures, all methods, 1 and 4 threads) and is tested (§6).

**Before the first member trains**, everything that does not depend on the seed is checked:
the data file and its readability as training data, the method's requirements (noise2noise's
`--noise-level`, noise2clean's `clean`), moving-average's refused flags, `--epochs` and
`--threads` at least 1, every recorded argument recordable (a non-finite `--lr` is refused
here, not at the first member's seal), and `--device` (resolved once; `auto` resolving to MPS
or CUDA makes every member Tier 2, said in the printed output). `--threads` is applied once,
before the first member. The training data is read once per member, as the single run reads
it.

**The code is recorded per member**, when each is sealed; a commit or an edit of the tree
during a long batch makes its members differ in `code`, and `infer` would then refuse to
combine them (rule 3). The help and QUICK_START say not to change the checkout while a batch
runs.

**What a seed varies**, stated in the help and QUICK_START:

| method | the seed sets |
|---|---|
| noise2clean | weight initialisation, batch order, dropout masks (ResNet-FCNN, ResNet-1DCNN, Transformer) |
| noise2noise | the same, **and the synthesised training target**: the target noise is drawn from a generator seeded `s + 1000`; the input is the file's own `noisy` |
| moving-average | weight initialisation, batch order, dropout masks (its fixed architecture, ResNet-FCNN, has dropout); the targets are built from the frames and do not depend on the seed |

For noise2noise a seed batch therefore varies the training target as well as the
optimisation, and its spread is not only optimisation variability. (Revision 0 proposed
refusing seed sets containing `s` and `s + 1000`; that rested on a misreading — the stream
seeded `s` is drawn and discarded, never used as a target — and is dropped.)

**Cost and failure.** K seeds cost K trainings, one after another, with one model in memory
at a time; the help says so. A failure in one member stops the batch with that member named;
members already written stay, and the message lists them.

## 3. Applying several models: `infer` with repeated `-m`

`infer -d data.h5 -m a.pt -m b.pt -m c.pt -o out.h5`. `-m` becomes repeatable; with one `-m`,
`infer` is unchanged and its output byte-identical to before. With K ≥ 2 models:

**Compatibility, checked before any model is applied, in this order** (the first rule that
fails refuses, by name, with the field and both values; each rule is reached only by models
that passed the earlier ones):

1. **every checkpoint has a verified manifest** (open question 4 of revision 0, as proposed
   and recommended by the audit: a checkpoint without one would present an unknown training
   condition as seed spread);
2. **schema `dnd-provenance-3` for every member**: versions 1 and 2 do not record the thread
   count in force, so members trained at different counts — execution variability — would pass
   as seed spread. (Versions 1 and 2 were never in a release.)
3. **the same manifest outside a fixed exclusion list.** The excluded fields are exactly:
   `created_utc`; `command.arguments.seed`; `command.seeds`; `command.flags_passed`;
   `command.arguments.device` and `command.arguments.threads` (as typed — their resolved
   values, `software.device` and `software.torch_threads`, are compared); `result.final_loss`;
   `statuses`. Everything else is compared, including `software` (versions and resolved
   device), `code`, `reproducibility`, the whole `training_data` (so units, acquisition id,
   reference declaration, angles and order basis as well as the digest), `targets`,
   `preprocessing`, `command.effective` and every other recorded argument. Arguments recorded
   but not applying to the method (for example `window` for noise2clean) are equal in a
   batch, and must be equal across members given by hand too: the rule does not interpret
   which apply. The list is a named constant, and a test fails if a manifest field is added
   without being classified;
4. **distinct, integer `seeds.torch`**: a null or repeated seed is refused (unseeded
   noise2noise models share their default target noise; a repeated seed is a duplicate);
5. **distinct `model_body_digest`**: a duplicate is not an independent member. (`-m a.pt
   -m a.pt` is refused already by rule 4, the repeated seed; rule 5 is reached by a copy whose
   recorded seed differs but whose weights are the same.)

The network length needs no rule of its own: `command.effective` and `training_data.layout`,
compared by rule 3, fix it, so all outputs are on one grid.

**A tree that is not clean** (owner's decision, 2026-10-04): when `code.tree_clean` is `false`
for the members (it is then equal for all, by rule 3, with an equal commit), the members are
combined and a note says that the code had uncommitted changes, so the records cannot show
that it was the same code. `code: "unknown"` (an installed release, or a run outside a
checkout) is compared through `software.dnndenoiser` and gets the same kind of note: the
records show the version string, not that the installed code was the same.

**Failed members** (owner's decision, 2026-10-04): a member whose output contains a non-finite
value is refused, named, and nothing is written; the message says that if the user drops it and
combines the rest, the excluded member and the reason should be reported with the result, since
the new file cannot show the exclusion. A member whose manifest records a status (a
non-finite final loss) is **kept**, and named in a note and in the file: the plan audit asks
that failures be kept, not dropped, and dropping one silently would make the ensemble look
better than its members.

Each member is applied exactly as single-model `infer` applies a model — the same
resampling, its own normalisation and inverse, `model.eval()`, the same batches — on one
device.

**Output file.** The input and coordinates exactly as single-model `infer` writes them,
with their attributes, and:

- `denoised_members`: shape `(K, *input_shape)`, float32, every member's output, in the
  order given;
- `ensemble_mean_estimate`: shape `input_shape`, float32, the mean over members;
- `between_run_std_fixed_input`: shape `input_shape`, float32, the standard deviation over
  members with ddof 1;
- all three carry the input's `intensity_units` and `acquisition_id` when the input has them
  (each member's inverse normalisation returns the input's units), as single-model `infer`
  carries them onto `denoised`; `noisy` carries `input_array_digest` as there;
- a group `members` with one subgroup per member, named `0`, `1`, … in the order given, each
  holding the member's manifest as the scalar string dataset `model_provenance` and, as
  attributes of the subgroup, `model_digest`, `model_body_digest` (the same canonical JSON as
  on a single output's `denoised`) and `seed` (integer). The reader of output records is
  generalised to take the node holding the manifest and the node holding the two digest
  attributes; a single output passes the file and `denoised`, a member its subgroup twice.
  Each member's records therefore verify exactly as a single output's do;
- attributes on the file root: `ensemble_k` (integer K), `members_seeds` (integer array, the
  seeds in member order), `inference_device` (string), and `ensemble_notes` (a JSON list of
  strings, `[]` when no note applies);
- nothing else: no top-level `model_provenance`, no digest attributes outside `members`, no
  `denoised`. The exact list of datasets and attributes is pinned by a test.

**There is no `denoised` dataset**, so `evaluate` refuses the file (open question 5 of
revision 0, as proposed and recommended by the audit); its message gives the command: run `infer` with one member's
checkpoint, evaluate that output, and compare members paired by seed. Evaluating the ensemble
mean would treat a new estimator as if it were one of the members.

**What is printed and documented, fixed in wording:**

- the mean is an *ensemble mean estimate*: a model estimate like any member's output
  (`AGENTS.md` §5). It can hide one member's failure; the members' outputs are in the file. It
  is not shown to be closer to the signal than any member, and no comparison with the members
  is made or implied;
- the spread is the *between-run standard deviation on fixed input*: how much these K runs
  differ from each other on this input. It is not measurement uncertainty, a confidence
  interval or a bound on error; it is not divided by √K; it does not reflect a bias the
  models share, and members can agree closely and all be wrong, inside or outside the training
  distribution; it includes no variation from other training data or other acquisitions; with
  K of 2 or 3 the spread is itself a poor estimate of the between-run spread; it includes
  execution variability when inference runs on a device other than the CPU or when members
  were trained in Tier 2 (MPS, CUDA), and the printed output says so when either applies;
- K, the seeds and the device are printed. **No scalar summary of the spread is printed or
  stored** (no mean or maximum over points), so it cannot be quoted as a single figure.

**The `-m` option.** In `infer`, `-m` becomes repeatable (`args.model` a list); `diagnose -m`
stays single. Because `--seeds` is added, `--see` no longer abbreviates `--seed`; the CHANGELOG
says so. Inference holds one model at a time and the K outputs in memory (K × the input).

## 4. What this does not do

- No comparison between conditions, ranking, or "most stable architecture"; no stability
  claim from any number of seeds.
- No evaluation of the ensemble mean, no ensemble `diagnose` (each member can be applied and
  diagnosed alone).
- No seed defaults: `--seeds` must be given; `0 1 2` appears in QUICK_START only as a cheap
  exploratory example.
- No change to single-seed training, single-model inference, the manifest schema, or any
  committed record.

## 5. Arithmetic

The mean and the spread are computed in float64 over the members' float32 outputs, then
stored as float32. ddof 1 needs K ≥ 2, which every ensemble has (K = 1 is single-model
`infer`). The members are on one grid by rule 3 (same training data, same network
length, the same resampling of the input).

## 6. Tests

1. **Batch equals single runs**, in separate processes on both sides (step 4's harness): for
   noise2clean, noise2noise and moving-average (2-D and 3-D), each member of `--seeds 3 5`
   and of `--seeds 5 3` has the `model_body_digest` of `train --seed 3` / `--seed 5`, and its
   manifest equals that run's except `created_utc`; also the Transformer at `--threads 2`.
2. **Negative controls:** two members of one batch differ; a member differs from a single run
   with another seed.
3. **Refusals before any training** (a spy shows no member trained), each with a positive
   counterpart: `--seed` with `--seeds`; a single seed; a repeated seed; a negative seed; `-o`
   not ending in `.pt`; an existing member file; moving-average's refused flags with
   `--seeds`; noise2noise without `--noise-level`; noise2clean without `clean`; `--epochs 0`;
   `--threads 0`; a non-finite `--lr`. And a member failing mid-batch: the earlier members are
   kept and listed.
4. **What a seed varies:** noise2noise members' target seeds are `s + 1000` and their targets
   differ between `s` and `s + 1000` members (the case revision 0 would have refused);
   moving-average members' targets are equal (spy on the target construction).
5. **Compatibility**, each with checkpoints really trained (or re-sealed through `seal`, never
   an edited manifest that fails verification instead) and each asserting which rule refused:
   another training file, another units attribute only, another `--lr`, another architecture,
   another resolved device or thread count, another commit, another software version, a clean
   tree against a dirty one, unknown code against a commit, noise2noise's `--noise-level`,
   moving-average's `--window`, a version-2 member, an unseeded member, a repeated seed
   (`-m a.pt -m a.pt`), a re-sealed copy with another recorded seed and the same weights
   (rule 5), another network length (refused by rule 3), a checkpoint without a manifest;
   accepted: a real batch, two separate `train --seed` runs, members typed with
   `--device auto` and `--device cpu` resolving alike, a member with a recorded status (kept
   and named). A classifying test fails when a manifest field is neither compared nor
   excluded.
6. **Arithmetic against an independent computation:** stub networks with known outputs give
   the expected mean and ddof-1 spread, with values where float32 accumulation differs from
   float64, and a K = 2 case where ddof 0 and ddof 1 differ; below the compatibility check
   (stubs through the network seam), members with different normalisations each applied
   with their own (swapping them changes the result).
7. **Output layout:** member order preserved; shapes for 2-D, 3-D and resampled inputs; no
   `denoised`; `evaluate` refuses with its message; each member's records in `members/i`
   verify as single-output records do; the tree and failed-member notes present exactly
   when they apply; a non-finite member output refused and nothing written; the exact list of
   datasets and attributes, with a planted `between_run_std_max` caught; each member's records
   tampered (an edited manifest, swapped digests between members) detected; single-model
   `infer` output unchanged: a content digest of every dataset and attribute, through the
   network stub seam (so independent of platform arithmetic), equal to a golden committed
   before the change.
8. **Words:** no dataset name, attribute name or printed label contains a forbidden word of
   step 1 or *uncertainty, confidence, interval, error, precision, stable, stability, robust*,
   outside the fixed sentences that say what the spread and the mean are not, which are named
   constants and excluded by exact equality (scoped to names and labels, not `Error:`
   messages); planted names are caught.
9. **The single-seed path is unchanged:** a `train --seed s` manifest equals one written
   before the change except `created_utc`, `code` and `software.dnndenoiser` (`--seeds` added
   to `EXCLUDED_ARGUMENTS` changes nothing recorded).

## Revision log

- **Revision 0 (2026-10-03).** First draft, for the independent design audit.
- **Revision 1 (2026-10-04), after the first independent audit** (verdict: revise and
  re-audit; two blocking). The noise2noise row corrected (only the `s + 1000` stream is a
  target) and the overlap refusal dropped; compatibility defined as equality of the whole
  manifest outside a fixed, tested exclusion list, with resolved device and threads, code and
  software compared; members recorded exactly as single runs (`args.seed`, `flags_passed`,
  `--seeds` excluded); null and repeated seeds refused; dropout added to what a seed sets; the
  members group holds each member's full records; units and attributes stated; wording
  covers hidden failures, shared out-of-distribution failure, small K and no scalar summary;
  non-finite outputs and diverged members handled; names, early validation, cost and device
  stated; tests extended (separate processes, reversed order, real planted differences,
  ddof and float32 cases, golden digest, words scoped). Owner's decisions (2026-10-04): a tree
  that is not clean is combined with a note; a non-finite output is refused, a diverged member
  kept and named. Open questions 1, 3, 4 and 5 resolved as proposed and recommended by the
  audit (for the owner's adoption of this revision); 2 dropped by the correction above.
- **Revision 2 (2026-10-04), after the second independent audit** (verdict: adopt with
  named changes G1–G14; both blockers of revision 0 closed). Members' digests as subgroup
  attributes with a generalised reader, and tamper tests; the wording on a shared bias
  corrected (it is not reflected, rather than zero) and extended to the mean (not shown closer
  to the signal) and to Tier 2 training; ensembles require schema 3; unknown code gets a note
  like a dirty tree; the order of the rules, with the length rule folded into rule 3 and the
  duplicate case assigned to the right rule; feasible goldens (content digest through the stub
  seam) and a feasible single-seed test; the output's exact contents, encodings and carried
  attributes; early refusals extended; at least two seeds; the `-m`, `--see` and memory
  consequences; the message after a refused member asks that an exclusion be reported.

## Confirmation

**2026-10-04 — the owner adopted this design, revision 2, as a whole**, after two
independent audits (the second returning "adopt with named changes", all applied in this
revision), including the resolution of open questions 1, 3, 4 and 5 as proposed and the
author's choices made in applying the second audit (schema 3 for ensembles, at least two
seeds, member records in subgroups with a generalised reader). The status line says
"implemented" only in the change that merges the implementation.
