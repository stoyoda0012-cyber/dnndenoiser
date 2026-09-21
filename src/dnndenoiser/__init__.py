"""DNNDenoiser — generic deep-learning denoiser for XPS spectra.

This is the **stable public API** for external consumers (e.g. the
``depthprofiler`` HAXPES pipeline). It re-exports the small surface that other
projects depend on:

    from dnndenoiser import DenoisingNetwork

The repo deliberately contains *no* depth-profiling / peak-fitting code — those
live in sibling repos (``depthprofiler``, ``toyomacro``). Importing this
package must never pull in ``deppro`` or ``toyomacro`` (enforced by
``tests/test_import_boundary.py``).

All library code is namespaced under ``dnndenoiser.*`` (``dnndenoiser.models``,
``dnndenoiser.data``, …), and that namespace is the whole of it: there is no
second import surface.
"""
from __future__ import annotations

from dnndenoiser.models.network import DenoisingNetwork, build_denoising_network
from dnndenoiser.data.synthetic_generator import (
    SyntheticGenerator,
    GeneratorConfig,
    NoiseConfig,
    PEAK_SETS,
)

__version__ = "0.1.1"

__all__ = [
    "DenoisingNetwork",
    "build_denoising_network",
    "SyntheticGenerator",
    "GeneratorConfig",
    "NoiseConfig",
    "PEAK_SETS",
]
