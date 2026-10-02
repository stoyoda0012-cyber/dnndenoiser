"""Content digests of what ``evaluate`` compared: ``dnd-digest-1``.

Implements §6.6 of ``docs/design/EVALUATION_REFERENCE_CONTRACT.md`` (phase 2). A digest
is SHA-256 over a sequence of named components, each framed as the name (UTF-8), ``0x00``,
an 8-byte little-endian length, then the payload. An absent component is framed with its
name and length 0 and is listed as absent.

- An array's payload is the ASCII header ``<dtype name>|<shape as a comma list>``, ``0x00``,
  then the array's bytes in little-endian byte order and C order. The dtype name is NumPy's
  (``float32``, ``int64``, …), which carries no byte order.
- A JSON object's payload is its serialisation with sorted keys, the separators ``,`` and
  ``:``, non-ASCII characters kept, in UTF-8.

A digest identifies content; it does not establish provenance.
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Optional, Sequence

import numpy as np

FORMAT = "dnd-digest-1"
REFERENCE_COMPONENTS = ("reference", "energy", "angles", "times", "frame_index",
                        "declaration_stored")
EVALUATED_COMPONENTS = ("noisy", "denoised", "energy", "angles", "times", "frame_index")


def array_payload(value) -> bytes:
    arr = np.asarray(value)
    if not arr.flags.c_contiguous:
        # Not np.ascontiguousarray: it turns a 0-d array into shape (1,), which would
        # change the header the format defines.
        arr = arr.copy(order="C")
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"an array of dtype {arr.dtype} has no digest")
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    header = f"{arr.dtype.name}|{','.join(str(n) for n in arr.shape)}".encode("ascii")
    return header + b"\x00" + arr.tobytes(order="C")


def json_payload(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def frame(name: str, payload: Optional[bytes]) -> bytes:
    body = b"" if payload is None else payload
    return name.encode("utf-8") + b"\x00" + len(body).to_bytes(8, "little") + body


def digest(components: Sequence[tuple]) -> dict:
    """``{"sha256": hex, "absent": [names]}`` over ``(name, payload or None)`` pairs."""
    sha = hashlib.sha256()
    absent = []
    for name, payload in components:
        sha.update(frame(name, payload))
        if payload is None:
            absent.append(name)
    return {"sha256": sha.hexdigest(), "absent": absent}


def _array_or_none(value) -> Optional[bytes]:
    return None if value is None else array_payload(value)


def reference_digest(reference, energy=None, angles=None, times=None, frame_index=None,
                     declaration_stored: Optional[dict] = None) -> dict:
    """Over the reference array, the coordinates carried with it, and the stored bundle
    (``reference_origin`` and ``reference_schema_version`` as a JSON object, or absent).
    The effective command-line declaration is not part of it."""
    return digest([
        ("reference", array_payload(reference)),
        ("energy", _array_or_none(energy)),
        ("angles", _array_or_none(angles)),
        ("times", _array_or_none(times)),
        ("frame_index", _array_or_none(frame_index)),
        ("declaration_stored",
         None if declaration_stored is None else json_payload(declaration_stored)),
    ])


def evaluated_digest(noisy, denoised, energy=None, angles=None, times=None,
                     frame_index=None) -> dict:
    return digest([
        ("noisy", array_payload(noisy)),
        ("denoised", array_payload(denoised)),
        ("energy", _array_or_none(energy)),
        ("angles", _array_or_none(angles)),
        ("times", _array_or_none(times)),
        ("frame_index", _array_or_none(frame_index)),
    ])
