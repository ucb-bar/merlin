"""Load a whole-model capsule and its independent numerical reference."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class WholeModelBuildError(RuntimeError):
    """The whole model cannot be built as one program, and the message says which part stopped it."""


# --------------------------------------------------------------------------------------------- capsule


@dataclasses.dataclass(frozen=True)
class ModelCapsule:
    """A model capsule's files and the numbers its golden was computed on."""

    directory: Path
    name: str
    interface: Path
    weights: Path
    weights_manifest: Path
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]


def load_model_capsule(path: str | Path) -> ModelCapsule:
    """Read a model capsule the way the capsule runner does, refusing one that is not whole.

    The weights and the golden are gitignored beside the tracked capsule, so a checkout can hold the
    interface without them; that is named here rather than discovered as a missing file mid-build.
    """
    import numpy as np

    from merlin.development.targetgen import capsule_common as CC
    from merlin.development.targetgen import capsule_inputs as CI

    capsule = CC.load_capsule(path)
    directory = Path(capsule["__dir__"])
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    interface = directory / str(capsule.get("interface_mlir") or "capsule.interface.mlir")
    if not attrs.get("weights") or not attrs.get("weights_manifest"):
        raise WholeModelBuildError(f"capsule {directory.name!r} declares no weights, so it is not a model capsule")
    weights, manifest = directory / str(attrs["weights"]), directory / str(attrs["weights_manifest"])
    missing = [str(p) for p in (interface, weights, manifest) if not p.is_file()]
    if missing:
        raise WholeModelBuildError(f"capsule {directory.name!r} is missing {missing}")
    declared = CI.canonical_input_values(capsule, directory)
    if not declared:
        raise WholeModelBuildError(f"capsule {directory.name!r} records no decoded input in its golden")
    inputs = {
        name: np.asarray(spec["values"], dtype=np.float32).reshape([int(e) for e in spec["shape"]])
        for name, spec in declared.items()
    }
    outputs = {name: np.asarray(value, dtype=np.float32) for name, value in _model_golden(capsule, directory).items()}
    return ModelCapsule(directory, directory.name, interface, weights, manifest, inputs, outputs)


def _model_golden(capsule: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """A MODEL capsule's captured outputs: its independent golden, read as declared.

    A whole model's expectation is never recomputed here -- the integer Tensor engine cannot evaluate
    an ``op: model`` graph -- so a capsule whose golden declares the recompute source, or declares no
    outputs, is refused by name rather than graded against something invented.
    """
    from merlin.development.targetgen import golden_provenance as GP
    from merlin.development.targetgen.golden_store import load_golden

    if GP.golden_source(dict(capsule), directory) == "merlin_tensor_int":
        raise WholeModelBuildError(f"capsule {directory.name!r} declares no independent golden for its outputs")

    outputs = (load_golden(directory) or {}).get("outputs")
    if not outputs:
        raise WholeModelBuildError(f"capsule {directory.name!r} golden records no outputs")
    return dict(outputs)
