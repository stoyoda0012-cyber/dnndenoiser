# AGENTS.md — shared instructions for AI agents working in this repository

This file is the tool-agnostic source of shared working instructions. Tool-specific
entry points (for example `CLAUDE.md`) should point here rather than restate it.

It holds working rules only. It is not a description of current behaviour, a
changelog, a roadmap, or a results record; see §11 for where those live.

## 1. Scope and non-goals

DNNDenoiser is a **standalone, generic denoiser for XPS (X-ray photoelectron
spectroscopy) spectra**: network architectures, training methods, physics-based
synthetic data generation, inference, evaluation, and a command-line interface.

Out of scope for this repository:

- depth profiling, peak fitting, and downstream quantification;
- measured-data loaders and instrument-specific calibration for particular
  instruments or beamlines;
- a bundled pretrained "universal" denoiser. The project ships a training workflow;
  "turnkey" means turnkey training, not immediate denoising of arbitrary XPS data.

Do not add these back. Applications that consume denoised spectra live in sibling
projects.

## 2. Repository map and the code that ships

- `src/dnndenoiser/` — **the package, and the source of truth for library code.**
  New code goes here, and it is the only importable library surface: everything
  is reached through the `dnndenoiser.*` namespace.
- `tests/`, `benchmarks/`, `docs/`, `paper/` — supporting trees outside the
  installed package.

The installed console command and the package entry point are declared in
`pyproject.toml`. Use those rather than inventing script paths.

## 3. Boundaries that must not regress

**Import boundary.** Library code must not import sibling projects (`deppro`,
`toyomacro`) and must not hardcode sibling-project paths. Instrument-specific
constants belong in the calling application's own configuration and reach the
library by parameter injection; this repository holds none.
`tests/test_import_boundary.py` enforces this over the packaged tree, and must
keep passing.

**Distribution boundary.** Research, application, and agent-control layers must never
appear in a built sdist or wheel: `benchmarks/`, `docs/`, and this file together
with any tool-specific pointer file. The `build` job in
`.github/workflows/ci.yml` asserts this against the real archives. Its deny list is
deliberately wider than the trees present here, so a research tree reintroduced by
accident is refused rather than shipped.

A change that moves either boundary must update the enforcing test or CI assertion in
the same change, and is an independent-audit item (§8).

## 4. Authority: normative versus factual

Keep these two questions apart. They have different sources of truth.

### 4.1 Normative authority — what you are allowed and required to do

1. Explicit human instructions and the current task brief.
2. **This file.**
3. Accepted decisions recorded in this repository: the changelog, and the
   conditions recorded alongside any published measurement (for example
   `benchmarks/reference/results/`).

### 4.2 Factual authority — how the code currently behaves and is configured

- the implementation under `src/dnndenoiser/`;
- the test suite;
- `pyproject.toml`;
- the CI workflow;
- the actual build artifacts.

Prose documentation — `README.md`, `docs/`, past experiment reports — records intent
and history. It is **not** the arbiter of current behaviour.

### 4.3 When documentation and reality disagree

Do not silently trust the older document, and do not silently change the code to
match it. **Stop and report the conflict**, naming the document, the observed
reality, and how you verified it. A human decides which side changes.

## 5. Scientific boundaries specific to a denoiser

- Denoised output is a **model estimate, not a measurement**. The network can
  oversmooth, suppress weak features, and hallucinate plausible structure.
- **Distribution shift is the dominant failure mode.** A model is valid only inside
  the distribution it was trained on — peak structure, position range, noise regime,
  background, normalization, instrument response. Outside it, output quality can be
  worse than the input.
- Truth-referenced evaluation requires a clean reference. There is no reference-free
  SNR for measured spectra; do not present one.
- Denoising is preprocessing. Physically meaningful quantities (areas, positions,
  widths) must be verified downstream, not assumed preserved.

Do not write claims into code, docstrings, or documentation that contradict these
limits.

## 6. Discipline for performance and scientific claims

Any added or revised claim must state, together, what makes it checkable:

- the data and its provenance;
- the split rule, including the unit that prevents leakage;
- the noise model;
- the number of seeds;
- the metric **and the independence assumptions it relies on**;
- the comparison conditions, including whether training and inference operate at the
  same signal-to-noise regime.

Further rules:

- A single run is not evidence for a comparison. Report dispersion.
- Prefer a paired comparison over comparing separately reported aggregates.
- Do not restate a numeric result in a second place; link to the record that owns it.
- Retractions, "undecided" verdicts, and negative results stay in the record. Do not
  delete or quietly soften them.
- Do not promote an empirical regularity to a law.

## 7. Routine commands

`pyproject.toml` is authoritative for the supported Python range and dependencies;
use the environment it declares.

```bash
pip install -e ".[dev]"        # core + test/lint tooling
pytest                         # test suite
ruff check src/ tests/         # lint, same scope as CI
python -m build                # sdist + wheel; then check archive contents (§3)
```


## 8. Changes that require independent audit

Normal work — user-visible changes, ordinary public API changes, bug fixes — needs
appropriate tests and the implementer's own review. That is sufficient.

Request an independent audit when the task brief asks for one, or when a change:

- alters the meaning of the scientific model, the noise model, or the synthetic data
  generation process;
- alters the meaning of an evaluation metric, its units, its estimand, or its
  comparison conditions;
- adds or revises a published performance, accuracy, or scientific claim;
- moves a public/private, package/import, or distribution-artifact boundary;
- finalizes a release candidate;
- revises the acceptance criteria or the recorded conditions of a published
  measurement.

## 9. Git and local state

- **Stage explicit paths.** Do not use `git add -A`, `git add .`, or `git commit -a`.
  Correctness must not depend on ignore rules catching everything.
- One logical change per commit.
- Never stage local agent state: `CLAUDE.local.md`, and under `.claude/` the local
  settings override, notes, archives, and worktrees. Only `.claude/settings.json`,
  `.claude/commands/`, and `.claude/hooks/` are shareable.
- Do not copy the contents of local private notes into tracked files, commit
  messages, or reports.
- Commit and push only when asked. Never push as a side effect of another task.

## 10. Preparing distributions and public artifacts

These rules apply whenever a distribution, release, or outward-facing artifact is
being prepared. They do not assert that any particular publication has been decided.

- Enforce the distribution boundary (§3) against the real archive, not by reading
  the configuration.
- Tracked files must not contain developer-specific absolute paths, private data
  locations, or personal or affiliation details beyond what the project has already
  chosen to publish.
- Claims in shipped documentation must satisfy §6 and stay inside §5.

## 11. What this file deliberately does not contain

| Looking for | Go to |
|---|---|
| Installation, supported surface, limitations | `README.md` |
| CLI workflow and data format details | `docs/QUICK_START.md` |
| Contribution process and style | `CONTRIBUTING.md` |
| Dependencies, Python range, entry points, packaging | `pyproject.toml` |
| Enforced checks | `.github/workflows/ci.yml`, `tests/` |
| Reference measurement, its conditions and how to re-run it | `benchmarks/reference/` |

Benchmark numbers, dates, a current "best architecture", session logs, and progress
notes are intentionally absent: they go stale, and this file has no way to notice.
