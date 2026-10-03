# Design: what `--seed` promises, where, and the tests that can fail

**Status: adopted 2026-10-03 (revision 2, after two independent audits); not yet
implemented.** Step 4 of the improvement plan that follows a reproducibility assessment of a
published study. It states a reproducibility contract for training, records the conditions
it depends on, adds a `--threads` option, and adds tests that would fail if the contract
broke. It changes no metric and no training arithmetic; it is audited under `AGENTS.md` §8
because it adds a claim (what reproduces) and revises recorded conditions (the provenance
manifest). It uses the vocabulary of `docs/design/PROVENANCE_MANIFEST.md` (step 2) and
`docs/design/FRAME_STACK_CHANNELS.md` (step 3).

## 1. The problem

`train --seed` "seeds every method … It does not make training reproducible on every
device." Nothing says on which devices, under which conditions, it *does*. The tests of
`--seed` (`tests/test_cli_seed.py`) train twice **in one process** on the CPU; the P1
criteria require bit identity in one pinned environment only (`tests/p1_environment.py`).
A user who trains twice and gets different weights cannot tell whether that is expected.

**Measured on one machine**, the development Mac (macOS arm64, Python 3.12.11, torch 2.9.1,
NumPy 2.3.3), comparing a hash of the trained weights of separate `dnndenoiser train`
processes. The author used `generate -n 64 --n-energy 128 --seed 0` (default Poisson level),
`--epochs 2 --batch-size 8 --seed 3`, and a 40-frame stack for moving-average; the design
audit used its own data (Poisson level 500, batch 16) and found the same. These are
observations on that machine, not a law:

- same seed, CPU, default thread count: equal for all eight architectures and for
  moving-average;
- **no `--seed`**: different for every method tried;
- thread count changed (`OMP_NUM_THREADS`, `MKL_NUM_THREADS` — which overrides it — or
  `torch.set_num_threads`, which overrides both): the **Transformer** changed with every
  count tried; **bi-LSTM** gave one result at 1 and 2 threads and another at 4 and 12; the
  other architectures and moving-average did not change; at a fixed count of 2 both
  reproduced across processes;
- MPS: two runs equal; CPU and MPS differ;
- interop threads, oneDNN, NumPy's BLAS threads, the input path: no effect.

Recorded elsewhere: on x86-64, NumPy's SIMD `argsort` breaks ties differently and torch's CPU
kernels reduce in another order (`tests/p1_environment.py`, found on Windows); on x86-64 CI
runners a test that assumed a float mean of one value equals the value failed (step 3,
fixed in the test); cuDNN runs RNNs in TF32 by default (README). PyTorch documents that
CUDA results are not guaranteed reproducible unless deterministic algorithms are enforced;
its documentation does not cover MPS.

A checkpoint trained on MPS stores its state-dict tensors on the MPS device. `infer` loads on
the CPU, so it works, but the file is not device-neutral.

## 2. The contract

Two tiers, stated in QUICK_START and in the `--seed` help, and recorded in every manifest
(§3).

- **Tier 1 — promised, bit for bit: equal `model_body_digest`** (the weights and the
  configuration `infer` applies; `model_digest` and the file always differ, because the
  manifest records `created_utc`). The conditions, all of them:
  1. the same machine;
  2. the same installed environment: the same Python, and the same builds of torch, NumPy
     and h5py — not only the same version strings (a pip and a conda torch of one version,
     or NumPy linked to different BLAS libraries, are different builds);
  3. the same code: the same commit with a clean tree, or the same installed release;
  4. device `cpu`;
  5. an integer seed;
  6. the same thread count;
  7. the same input bytes and the same recorded arguments.

  Then two training processes give the same `model_body_digest`, for every method and
  architecture the CLI offers.

  **What a manifest can show.** Conditions 4–7 and the version strings of 2 are recorded
  (`software.device`, `command.seeds.torch`, `software.torch_threads`,
  `training_data.digest`, `command.arguments`, `software`), and so is the commit of 3 when
  the run is in a git checkout. **The machine, the builds behind a version string, and the
  identity of an installed release are not recorded** — a manifest names no host by design
  (step 2 §2). So two manifests can show that two runs were *not* both in Tier 1; they
  cannot show that the runs shared a machine or an environment. Whoever compares two runs
  has to know that.
- **Tier 2 — not promised.** Any other difference: another machine or environment, device
  (MPS, CUDA), thread count, code, or no seed. Two such runs may agree — on the development
  machine several did — or differ; nothing is claimed in either direction and nothing is
  tested to agree or to differ. Known sources: thread count (measured for the Transformer
  and bi-LSTM), `argsort` tie-breaking on x86, BLAS and kernel choice by CPU, GPU kernels,
  TF32.

`--device auto` selects MPS or CUDA where available, so a default run on a Mac or a GPU
host is a Tier 2 run; QUICK_START says to pass `--device cpu` for Tier 1.

Inference and evaluation are outside the contract; this design makes no claim about them.

## 3. What is recorded, `--threads`, and the label

**`--threads N`.** `train` gains `--threads N`, an integer ≥ 1 (otherwise refused with a
message). Its first statement — before the moving-average dispatch, before any data is read
or any tensor or model is built — calls `torch.set_num_threads(N)`. Without it the thread
count is torch's default, as today (which reflects `OMP_NUM_THREADS` and
`MKL_NUM_THREADS`). Every method accepts it, moving-average included (it is not one of
moving-average's fixed flags), and QUICK_START lists it with the flags that still apply.
It is a recorded argument of type integer or null.

**The manifest becomes `dnd-provenance-3`.** It adds:

- `software.torch_threads`: `torch.get_num_threads()` read when training starts (the
  effective count, from `--threads` or the environment);
- a top-level object `reproducibility`: `{"tier": …}`, where `tier` is the contract label
  **computed from the recorded fields**: `"tier-1-eligible"` when `software.device` is
  `cpu`, `command.seeds.torch` is an integer and `code.tree_clean` is not `false`;
  `"tier-2"` otherwise (owner's decision, 2026-10-03: a dirty tree is not "the same code";
  `code: "unknown"`, as for an installed release, stays eligible, since the release's
  identity is one of the conditions the manifest does not show). The label says which
  promise applies to the run; it is not a comparison result. The validator recomputes it
  and **refuses a manifest whose stored label disagrees** with its fields.
- the recorded argument `threads`.

`deterministic_algorithms` is not recorded: the CLI never enables it.

**Version rules** (steps 2 and 3). Each version has its own exact key sets — for
`software`, `command.arguments`, `training_data` and the top level — and a manifest is
validated with exactly its version's sets; stored records are never changed. The accepted
schemas are listed explicitly (versions 1, 2, 3). Where a field exists from version 2 on
(`training_data.angles`, `frame_index_basis`), code tests membership in the set of versions
that have it, never equality with the current schema, so `evaluate` keeps reporting a
version-2 model's declarations. Every comparison of a schema in `provenance.py` is reviewed
for this.

**Reported by `evaluate`** (owner's decision, 2026-10-03): `evaluation_context.model` gains
`torch_threads` and `tier` for a version-3 manifest, `"not recorded"` for earlier ones.

**Device-neutral checkpoints.** `train` moves the state dict to the CPU before saving. Values
and digests are unchanged (the body digest is computed on the CPU); a checkpoint trained on
any device stores CPU tensors.

## 4. Tests

The cross-process harness runs `dnndenoiser` in a child process (`python -c "from
dnndenoiser.cli import main; …"` with `sys.argv` set, since the package has no `__main__`),
with `MKL_NUM_THREADS` and `OMP_NUM_THREADS` removed from the child's environment unless a
test sets them, the rest inherited (on Windows `SYSTEMROOT` stays). Each run asserts exit
status 0 and a newly written checkpoint, and reads `model_body_digest` from it. The tests
run under plain `pytest` (no marker deselects them), so CI runs them.

1. **Tier 1 across processes.** For noise2clean on every architecture, noise2noise on one,
   and moving-average on a 2-D and a 3-D stack: two runs with the same seed at `--threads
   1` give the same `model_body_digest`; the Transformer also at `--threads 2`, and bi-LSTM
   at `--threads 4` (its result changes between 2 and 4 on the development machine). About
   30 small one-epoch child runs per CI job; four jobs.
2. **Negative controls** (one configuration, FCNN noise2clean): another seed; one byte of
   the trained `noisy` array changed; `--lr` changed — each gives a different
   `model_body_digest`.
3. **The recorded thread count** (child processes, so no in-process test changes the
   thread count of later tests): with `--threads 1` and `--threads 2` the record equals
   the count; with no `--threads` and `OMP_NUM_THREADS=3` it records 3; `--threads 1`
   overrides an inherited `MKL_NUM_THREADS=4`; on both the supervised and the
   moving-average path. A spy on `train_selfsupervised` / the training loop reads
   `torch.get_num_threads()` when training starts and finds the requested count, so a
   record that merely echoed the argument, or a call placed after the model was built,
   fails. `--threads 0` and `--threads -1` are refused by name.
4. **The label.** Computed `"tier-1-eligible"` for cpu with a seed and a clean or unknown
   tree; `"tier-2"` without a seed, with a dirty tree, and for device `mps` or `cuda` (by
   constructing the manifest); a stored label that disagrees with its fields (for example
   `mps` labelled `"tier-1-eligible"`) is refused.
5. **Device-neutral checkpoints.** In process, on MPS, local only (skipped where
   unavailable, which is all of CI): the weights are captured before saving, the saved
   tensors are on the CPU, and an independent hash of their bytes equals one of the
   captured weights moved to the CPU.
6. **The manifest.** Version-3 exact key sets and types; versions 1 and 2 still read, still
   verify in `infer`, and `evaluate` still reports a version-2 model's declared angles and
   basis; `evaluate` reports `torch_threads` and `tier`, and `"not recorded"` for
   versions 1 and 2.

## 5. What this does not do

- It does not make GPU training reproducible and adds no `--deterministic` option (owner's
  decision, 2026-10-03: no GPU runner could check it, and PyTorch does not document MPS).
- It does not promise agreement across machines or environments, and a manifest cannot
  show that two runs shared one.
- It does not test that Tier 2 runs differ.
- It does not cover inference or evaluation.
- Evaluation-level negative controls (a model returning its input or a constant, to check
  that a metric separates them) belong to step 5 with the output-contraction diagnostic
  (owner's decision, 2026-10-03).
- It changes no training or evaluation arithmetic, and no committed record.

## Revision log

- **Revision 0 (2026-10-03).** First draft.
- **Revision 1 (2026-10-03), after the first independent audit** (two blocking, nine should
  fix, seven minor). Tier 1 requires an integer seed and the same machine; a contract label
  computed from recorded conditions; bi-LSTM among the thread-dependent architectures;
  "the same code"; `--threads`; the negative controls and the harness specified; the MPS
  test made independent and local; schema comparisons reviewed; the claim about evaluation
  dropped. Owner's decisions: the same machine; no `--deterministic`; `--threads` added;
  evaluation-level negative controls go to step 5.
- **Revision 2 (2026-10-03), after the second independent audit** (verdict: adopt with
  named changes N1–N11). The manifest is said to show only that two runs were *not* in
  Tier 1 — the machine, the builds and an installed release's identity are not recorded;
  "the same installed environment" replaces version strings; the label requires a tree that
  is not dirty, lives in a top-level `reproducibility` object and is recomputed by the
  validator; per-version key sets for every object, and explicit version sets in place of
  "≥ 2"; `--threads` validated, placed first in `train`, accepted by moving-average and
  documented; test 3 checks the count actually in force with a spy and inherited
  environments, in child processes; bi-LSTM tested at 4 threads; the CI cost corrected; the
  MPS test in process; §1's attribution corrected. Owner's decisions (2026-10-03): a dirty
  tree makes a run Tier 2, an unknown commit does not; `evaluate` reports `torch_threads`
  and the tier.

## Confirmation

**2026-10-03 — the owner adopted this design, revision 2, as a whole**, after two
independent audits (the second returning "adopt with named changes", all applied in this
revision). The status line says "implemented" only in the change that merges the
implementation.
