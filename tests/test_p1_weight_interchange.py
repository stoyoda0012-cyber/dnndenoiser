"""Criterion C4 of the P1 preregistration: the two definitions interchange.

The claim is narrow and the preregistration says so: this establishes that
``dnndenoiser``'s ResNet-FCNN and the archived reference's ``DenoisingNetwork``
are ``state_dict``-compatible **at one configuration, using untrained weights**.
It does *not* establish that any paper's trained weights load into
``dnndenoiser`` — those were never deposited, and the configuration they were
trained at is not fixed anywhere.

The reference's weights are 2.6 MB and are not committed. What CI checks is the
pinned key/shape manifest, which is what compatibility actually consists of; the
live cross-package load is checked by :func:`test_c4_real_cross_load`, which
runs only where the reference is available, and by the generator script, whose
recorded result is asserted below.

**C4 cannot see the dropout rate.** Keys, shapes and parameter count are
identical for ``p=0.1`` and ``p=0.5``, so a port that got dropout wrong would
pass this and fail C2/C3 instead. That is why the fixture spells dropout out.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from dnndenoiser.models.network import DenoisingNetwork

MANIFEST = json.loads(
    (Path(__file__).parent / "fixtures" / "p1_reference_targets.json").read_text(encoding="utf-8")
)["c4_state_dict_manifest"]

NET_KW = dict(
    num_features=256, num_hidden_units=100, layer_type="ResNet-FCNN", encoder_output_dim=64
)


def test_c4_keys_and_shapes_match_the_reference():
    """Same keys, same order, same shapes — no renaming."""
    model = DenoisingNetwork(**NET_KW)
    ours = [[k, list(v.shape)] for k, v in model.state_dict().items()]
    assert ours == MANIFEST["keys"]


def test_c4_parameter_count_matches_the_reference():
    model = DenoisingNetwork(**NET_KW)
    assert sum(p.numel() for p in model.parameters()) == MANIFEST["parameter_count"]


def test_c4_a_state_dict_of_that_shape_loads_strictly():
    """A ``state_dict`` built to the reference's manifest loads with ``strict=True``."""
    donor = {
        k: torch.randn(*shape) if shape else torch.randn(())
        for k, shape in (tuple(e) for e in MANIFEST["keys"])
    }
    model = DenoisingNetwork(**NET_KW)
    result = model.load_state_dict(donor, strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_c4_cross_load_was_verified_when_the_manifest_was_generated():
    """The generator performed the real cross-package load; assert its result.

    This is a recorded measurement, not a live one. It is asserted here so that
    regenerating the manifest against a reference that *fails* to cross-load
    cannot quietly produce a green suite.
    """
    assert MANIFEST["cross_loaded_outputs_max_abs_diff"] == 0.0


@pytest.mark.skipif(
    not os.environ.get("DNND_ARHAXPES_REF"),
    reason="DNND_ARHAXPES_REF is not set; the reference is not available here",
)
def test_c4_real_cross_load():
    """The live criterion, where the reference is present.

    Both directions, ``strict=True``, then eval-mode outputs equal.
    """
    sys.path.insert(0, os.environ["DNND_ARHAXPES_REF"])
    from arhaxpes_denoise.network import DenoisingNetwork as ReferenceNetwork

    torch.manual_seed(0)
    reference = ReferenceNetwork(**NET_KW)
    port = DenoisingNetwork(**NET_KW)

    port.load_state_dict(reference.state_dict(), strict=True)
    reference.load_state_dict(port.state_dict(), strict=True)

    reference.eval()
    port.eval()
    probe = torch.randn(4, 256)
    with torch.no_grad():
        a, _ = reference(probe)
        b, _ = port(probe)
    assert torch.equal(a, b)
