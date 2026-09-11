# Contributing to dnndenoiser

Thank you for your interest in contributing! This project welcomes bug
reports, feature suggestions, documentation improvements, and pull
requests.

## Reporting issues

- Search [existing issues](../../issues) before opening a new one.
- Include: a minimal reproducer, expected vs actual behavior, Python
  version, OS, PyTorch version, and relevant package versions
  (`pip list` excerpt).
- For denoising / training issues, attach the input array shape and
  dtype, the architecture and training method used, and a small
  synthetic dataset (or the `dnndenoiser generate` command that produces one)
  so the problem can be reproduced without your private data.

## Development setup

```bash
git clone https://github.com/<your-fork>/dnndenoiser.git
cd dnndenoiser
pip install -e ".[dev]"      # core + pytest
pytest                        # run the test suite
```

The denoiser is deliberately **self-contained**: it depends only on
`numpy`, `scipy`, `torch` and `h5py`. It contains **no** depth-profiling or
peak-fitting code — those live in separate companion projects that *consume*
the denoised spectra. Importing the package must never pull in a companion
project; this boundary is enforced by `tests/test_import_boundary.py` and must
keep passing.

`benchmarks/reference/` holds the reference measurement: the script, its
records and the renderer. It is not part of the installable library API and is
excluded from the built package.

## Pull requests

1. Fork the repository and create a topic branch from `main`.
2. Make focused commits — one logical change per commit, present-tense
   imperative subject line (e.g. `Fix ResNet-FCNN skip-connection shape
   mismatch`).
3. Add or update tests for the change. Existing tests must keep passing.
4. Run `pytest` before pushing.
5. Open a PR describing what changed and why; link any related issue.
6. Be patient — review is best-effort, not real-time.

## Style

- Python 3.10+. PEP 8; type hints on public functions.
- Docstrings in English (comments may be Japanese, per project
  convention). Public APIs need a docstring and at least one test.
- Prefer small, composable functions. Avoid speculative abstraction.

## Code of Conduct

This project adheres to the [Contributor Covenant
v2.1](CODE_OF_CONDUCT.md). By participating, you agree to uphold its
terms.

## License

By contributing, you agree that your contributions will be licensed
under the [MIT License](LICENSE).
