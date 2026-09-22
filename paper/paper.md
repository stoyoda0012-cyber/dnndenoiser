---
title: 'dnndenoiser: deep-learning denoising of X-ray photoelectron spectroscopy spectra'
tags:
  - Python
  - PyTorch
  - X-ray photoelectron spectroscopy
  - XPS
  - denoising
  - deep learning
authors:
  - name: Satoshi Toyoda
    orcid: 0009-0007-1091-1579
    affiliation: 1
affiliations:
  - name: "Advanced Equipment Division, Vacuum Products Corporation, Tokyo, Japan"
    index: 1
date: 17 July 2026
bibliography: paper.bib
---

# Summary

X-ray photoelectron spectroscopy (XPS) measures the energy distribution of
photoemitted electrons to determine the chemical composition and bonding state of
a surface. The recorded spectra are corrupted by Poisson shot noise, which becomes
severe under the short per-frame exposures required for spatially- and
time-resolved ("4D-XPS") acquisition, for beam-sensitive samples, and for
high-throughput mapping.

`dnndenoiser` is a PyTorch library for training and applying neural-network
denoisers to XPS spectra. It provides eight interchangeable architectures behind a
single interface — fully-connected, 1D convolutional, residual variants of both,
GRU, LSTM, bidirectional LSTM, and a Transformer — together with supervised
(noise2clean) and Noise2Noise [@lehtinen2018] training; a masking-based Noise2Self
[@batson2019] implementation is included as experimental library code. A
physics-based generator synthesises training spectra from Voigt / pseudo-Voigt
peaks on flat, linear, or Shirley-like [@shirley1972] backgrounds, with Poisson and
Gaussian detector noise and configurable variation of peak position, width, and
intensity. A command-line interface covers the full
`generate → train → infer → evaluate` workflow. Because a denoiser can be trained
on synthetic spectra whose peak structure and noise regime are chosen to match a
given instrument, the package supports building a model for a specific measurement
setting without dedicated calibration measurements.

# Statement of need

Denoising allows a short, low-dose measurement to approach the quality of a long
acquisition, which is valuable wherever per-spectrum counts are low: synchrotron
and laboratory hard-XPS, beam-sensitive materials, and spatially- or
time-resolved mapping where thousands to millions of spectra must each be analysed.
Most denoising used in practice is either generic (Savitzky–Golay smoothing,
principal-component filtering) or implemented ad hoc for a single dataset and
network, making methods hard to reuse or compare.

`dnndenoiser` provides a reusable, architecture-agnostic framework with a
reproducible synthetic-data generator, so that users can train, apply, and
benchmark denoisers under controlled noise. It is deliberately self-contained — it
depends only on PyTorch [@paszke2019], NumPy, SciPy [@virtanen2020], and h5py, and
contains no depth-profiling or peak-fitting code — and is complementary to
peak-fitting and depth-reconstruction tools that consume the denoised spectra. The
package is aimed at surface-analysis researchers who want a turnkey XPS denoising
workflow, and at method developers benchmarking architectures and training schemes
for one-dimensional spectral denoising. It is the denoising component of a
published angle-resolved hard-XPS depth-profiling study [@toyoda2026jvst].

Two limits are stated deliberately rather than left implicit. First, no pretrained
universal denoiser is distributed: the package ships a training workflow, and
"turnkey" refers to that workflow, not to immediate denoising of arbitrary XPS
data. Second, denoised output is a model estimate rather than a measurement — the
network can oversmooth, suppress weak features, or reconstruct plausible structure
that the data do not support — so physically meaningful quantities such as peak
areas, positions, and widths must be verified downstream.

# State of the field

Learned denoisers for scientific spectra have advanced quickly, but the closest
comparable study is instructive about what actually determines performance.
@oppliger2024 denoised X-ray diffraction data with a deep network and found that
training on experimentally paired low-count/high-count measurements outperformed
training on artificially added noise for quantitatively faithful reconstruction,
concluding that the training data and protocol mattered more than the architectural
details. That conclusion motivates the design here: `dnndenoiser` treats the
architecture as a swappable argument and puts the configurable surface in the data
generator and the training method, so that data-side questions can be asked
directly.

Within the Python spectroscopy ecosystem, `spectrapepper` [@grauluque2021] provides
general spectroscopic preprocessing and analysis, `Fitspy` [@quemere2024] performs
spectral decomposition by fitting, and `WrightTools` [@thompson2019] handles
multidimensional spectroscopic datasets. These are adjacent rather than
substitutable: none of them trains a neural denoiser, and none embeds an XPS noise
and lineshape model. Conversely `dnndenoiser` does not fit peaks or quantify
composition. The generic self-supervised methods it implements, Noise2Noise
[@lehtinen2018], Noise2Self [@batson2019] and a leave-one-out moving-average
target over repeated acquisitions, are domain-agnostic by construction;
what this package adds is an XPS-specific physical generator, a common interface
across architectures, and an evaluation path that is explicit about requiring a
clean reference.

# Software design

The distributed package is a `src/`-layout Python package, `dnndenoiser`, exposing
one console command of the same name. Its importable surface is deliberately
small:

- `data.synthetic_generator` builds spectra from peak shapes, backgrounds, and
  detector noise, with instrument-dependent quantities supplied by parameter
  injection rather than hardcoded, and writes them as HDF5.
- `models` holds the architectures behind the common `DenoisingNetwork`
  interface, and `training` holds the noise2clean / Noise2Noise / moving-average
  methods and the
  learning-rate schedules.
- The `dnndenoiser` command drives the `generate → train → infer → evaluate`
  workflow. The training loop and the clean-referenced metrics live there rather
  than in importable modules of their own, so the generator, the architectures
  and the training methods are the pieces that can be reused on their own.

Two boundaries are enforced by automated checks rather than convention. The package
never imports the author's depth-profiling or macro toolchains, which keeps the
denoiser usable on its own; and the research, benchmark, and documentation trees
are excluded from the built distributions. The import boundary is checked by a
test over the shipped package's source; the distribution boundary is asserted in
continuous integration against the real archives, not against configuration files.

Some capabilities are intentionally narrower than a reader might assume. Noise2Noise
in the command line synthesises a second, independent noisy realisation from clean
spectra; it does not ingest measured noisy/noisy pairs. The moving-average method
does train from measured frames and requires no clean reference, but it is exposed
for training only: inference from a frame stack is not yet wired through the command
line, and evaluating it on measured data has no independent reference, since a mean
over the same frames is not independent of targets built from subsets of them. Noise2Self is present as
library code but is not exposed through the command line, because a correct masked
loss that scores only held-out coordinates is not yet implemented there.
Multidimensional (angle- and time-resolved) arrays can be generated and processed,
but denoising flattens the non-energy axes and treats each one-dimensional spectrum
independently; this is not a joint multidimensional model. Evaluation computes
truth-referenced metrics and therefore requires synthetic data or a
high-statistics reference — there is no reference-free signal-to-noise estimate for
measured spectra. Training and inference run on CPU, NVIDIA CUDA, and Apple
Silicon (MPS), with device-compatibility tests that skip where a backend is
unavailable.

# Research impact statement

The package originated as the denoising component of hard-XPS depth-profiling work.
It underpins a published measurement-methodology paper on denoising strategies for
angle-resolved hard-XPS [@toyoda2026jvst], and a companion study of cross-exposure
transferability and the failure boundaries of self-supervised denoising
[@toyoda2026sia]. Its research value rests on
making that kind of study repeatable by others: because the noise model, the peak
structure, the position/width/intensity jitter, and the training method are all explicit
parameters, questions about *why* a denoiser generalises — or fails to — can be
posed as controlled comparisons rather than as anecdotes about a single trained
network. The project's own internal audits have used it this way, including to
retract earlier conclusions that did not survive re-examination under corrected
statistical assumptions. For the
surface-analysis community the practical impact is a shorter path from "we have a
noisy dataset and a hypothesis about its noise" to a trained, evaluated denoiser
whose limitations are stated.

# AI usage disclosure

Generative AI was used in developing this software and manuscript.
**Tools:** Claude Code (Anthropic), using Claude Opus 4-family and Claude 5-family
models; and ChatGPT and Codex (OpenAI), using GPT-4-family and GPT-5-family models.
**Where used:** implementation, refactoring, tests, documentation, manuscript
drafting and editing, and internal metric and architecture audits. **Nature and
scope:** the assistants drafted code and text
against research questions, preregistered acceptance criteria, and numerical targets
specified by the author. The public repository begins with a single import commit,
which carries a `Co-Authored-By` trailer; later commits carry one wherever an
assistant contributed. For selected high-risk changes,
implementation and audit were assigned to separate AI contexts, and findings were
closed against code, tests, and primary sources. This AI-assisted audit was a
development control, not independent human peer review. In the project's
development records, comparisons that failed their preregistered criteria are kept
as failures, and earlier conclusions overturned by re-examination as retractions,
rather than being removed.
The author made all scientific, algorithmic, and publication decisions; reviewed and
validated all AI-assisted output; and retains full responsibility for accuracy,
originality, licensing, and ethical and legal compliance.

# Acknowledgements

This software was developed during the author's work at Vacuum Products Corporation,
which approved its release under the MIT License with the author as copyright holder.
No external funding was received. The author thanks collaborators on the hard-XPS
measurement campaigns for discussions that shaped its requirements.

# References
