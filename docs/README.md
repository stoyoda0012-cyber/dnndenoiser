# DNNDenoiser documentation

Deep-learning denoising for XPS spectra. Start with the top-level
[README](../README.md) for installation, the supported surface, and
limitations.

## Contents

| Document | What it covers |
|----------|----------------|
| [WHEN_TO_TRUST.md](WHEN_TO_TRUST.md) | What has been measured about where a trained model stops being trustworthy, with the conditions and the record each answer comes from |
| [QUICK_START.md](QUICK_START.md) | Installed-CLI workflow: generate → train → infer → evaluate, HDF5 schema, Python API example |
| [../benchmarks/reference/README.md](../benchmarks/reference/README.md) | The reference measurement: what it measures, how to re-run it, and what its numbers do not support |
| [FROM_THE_PAPERS.md](FROM_THE_PAPERS.md) | For a reader of the JVST or SIA paper: how to apply the method to your own measurements, and what is not available |
| [preregistration/](preregistration/) | Acceptance criteria fixed **before** the work they judge. Registered first because a criterion written after seeing the result is not a criterion |

## The supported surface

The supported surface is the `dnndenoiser` package under `src/` and its console
command, and that is the whole of the library: everything is imported from the
`dnndenoiser.*` namespace.

Research scripts and their records are not part of this repository. The one
exception is `benchmarks/reference/`, the reference measurement, which runs from
a clean checkout and carries its own record.

## History

The physics models and the original implementation were developed in MATLAB
and migrated to PyTorch in 2026. Development history is tracked in Git and in
the changelog once releases begin.
