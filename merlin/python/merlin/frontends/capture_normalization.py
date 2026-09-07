"""Canonical, target-neutral normalization of captured frontend programs.

Capture cleanliness is a property of the program handed to every backend, not of any one target.
Frontend importers may nevertheless leave a small operation as an opaque call when its semantics can
be recovered exactly from the captured program and its provenance.  This module owns the ordered,
fail-closed normalization boundary between those two facts:

* every transform is semantic and shared by all targets;
* the importer's raw opaque census is checked structurally before a transform runs;
* the raw and normalized program digests and both opaque censuses remain in a receipt; and
* an unrecognized call remains opaque, so the caller's zero-opaque gate still refuses it.

Do not add target names, model names, capsule ids, or shape-specific rewrites here.  A transform
belongs in this registry only when its semantics are derived from the IR/provenance and independently
tested against the originating framework.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.torchao_affine import lower_torchao_affine_quant
from merlin.xdsl_dialects._common import text as module_text


class CaptureNormalizationError(RuntimeError):
    """The raw capture or a canonical normalization could not be proved well formed."""


@dataclass(frozen=True)
class _Normalizer:
    identity: str
    apply: Callable[[object], int]


# Order is part of the normalization contract.  Future entries must be framework-semantic and
# target-neutral; placement, scheduling and target lowering never belong here.
_NORMALIZERS = (
    _Normalizer(
        "merlin.frontends.capture_normalization.torchao_affine/v1",
        lower_torchao_affine_quant,
    ),
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _opaque_calls(module) -> dict[str, int]:
    """Structurally census the same placeholders model2MLIR calls opaque.

    The upstream worker deliberately regards every ``func.call`` as opaque at this boundary.  Use the
    parsed operation and its callee rather than reimplementing its textual regular expression.
    """
    names = Counter(
        op.callee.string_value()
        for op in module.walk()
        if op.name == "func.call"
    )
    return dict(sorted(names.items()))


def normalize_capture_mlir(
    raw_text: str,
    *,
    reported_opaque: Mapping[str, int] | None = None,
) -> tuple[str, dict]:
    """Normalize one captured module and return ``(text, content-addressed receipt)``.

    ``reported_opaque`` is the importer worker's census.  When supplied it must agree exactly with a
    structural re-read of the raw module; normalization must not launder a corrupt or stale report.
    Unknown calls are returned in ``remaining_opaque_detail`` rather than approximated.
    """
    try:
        module = parse_mlir_text(raw_text)
        raw_opaque = _opaque_calls(module)
    except Exception as exc:  # noqa: BLE001 - normalized into a stable boundary error
        raise CaptureNormalizationError(
            f"captured MLIR could not be parsed for normalization: {type(exc).__name__}: {exc}"
        ) from exc

    if reported_opaque is not None:
        try:
            reported = {
                str(name): int(count)
                for name, count in reported_opaque.items()
                if int(count) != 0
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise CaptureNormalizationError("capture worker opaque census is malformed") from exc
        reported = dict(sorted(reported.items()))
        if reported != raw_opaque:
            raise CaptureNormalizationError(
                "capture worker opaque census disagrees with the parsed raw program: "
                f"reported={reported}, observed={raw_opaque}"
            )

    applied = []
    try:
        for normalizer in _NORMALIZERS:
            rewrites = int(normalizer.apply(module))
            if rewrites:
                applied.append({"identity": normalizer.identity, "rewrites": rewrites})
        module.verify()
        normalized = module_text(module)
        # Reparse the serialized artifact: this is what the cache and every backend will consume.
        serialized = parse_mlir_text(normalized)
        serialized.verify()
        remaining = _opaque_calls(serialized)
    except CaptureNormalizationError:
        raise
    except Exception as exc:  # noqa: BLE001 - name the semantic boundary that failed
        raise CaptureNormalizationError(
            f"captured MLIR normalization failed: {type(exc).__name__}: {exc}"
        ) from exc

    receipt = {
        "schema": "merlin.capture-normalization.v1",
        "input_sha256": _sha(raw_text),
        "raw_opaque_detail": raw_opaque,
        "normalizers": applied,
        "output_sha256": _sha(normalized),
        "remaining_opaque_detail": remaining,
    }
    return normalized, receipt
