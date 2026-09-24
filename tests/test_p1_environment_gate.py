"""The P1 environment gate must admit the pin and refuse what is not the pin.

A gate that has only ever admitted the author's machine has not been shown to
refuse anything. Before 2026-09-24 it compared ``torch.__version__`` verbatim,
so every Windows wheel (``2.9.1+cpu``, ``2.9.1+cu128``) was refused for its
build label, and nothing said that Windows x86-64 is also a different
*platform*. Found by running the suite on Windows 11.
"""
from __future__ import annotations

import pytest

from tests.p1_environment import ENV, environment_mismatches

PIN = (ENV["platform"], ENV["python"], ENV["torch"], ENV["numpy"])


def test_the_pin_itself_is_admitted():
    assert environment_mismatches(*PIN) == []


def test_a_build_label_alone_does_not_refuse():
    platform_id, python, torch_version, numpy_version = PIN
    assert environment_mismatches(platform_id, python, torch_version + "+cpu",
                                  numpy_version) == []


@pytest.mark.parametrize("field,value", [
    ("platform", "Windows-AMD64"),
    ("platform", "Linux-x86_64"),
    ("python", "3.13.7"),
    ("torch", "2.11.0"),
    ("numpy", "2.2.6"),
])
def test_anything_else_is_refused_and_named(field, value):
    fields = dict(zip(("platform", "python", "torch", "numpy"), PIN))
    fields[field] = value
    got = environment_mismatches(fields["platform"], fields["python"],
                                 fields["torch"], fields["numpy"])
    assert len(got) == 1 and got[0].startswith(f"{field}: have ")


def test_the_windows_wheels_that_ran_the_suite_are_refused_for_the_platform():
    for label in ("+cpu", "+cu128"):
        got = environment_mismatches("Windows-AMD64", ENV["python"],
                                     ENV["torch"] + label, ENV["numpy"])
        assert got == [f"platform: have Windows-AMD64, pinned {ENV['platform']}"]
