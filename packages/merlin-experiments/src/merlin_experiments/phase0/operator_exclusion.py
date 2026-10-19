"""Fixed operator-only exclusion of public source and contraction signatures.

Only the operator supplies this content-pinned hash selection. No protected
program or shape file is discovered or opened. The selection is never a corpus
field, receipt or qualification issuer; callers retain its access privately.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen import component_program

SCHEMA = "merlin.phase0.operator_private_exclusion.v1"
SIGNATURE_SCHEMA = "merlin.phase0.contraction_exclusion_signature.v1"
REFUSAL = "public corpus publication eligibility is not established"
_ISSUER = object()


class PublicationRefusal(ValueError):
    """One generic refusal, never an excluded source or matched-member detail."""

    def __init__(self):
        super().__init__(REFUSAL)


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def contraction_signature_hashes(capsule, *, directory=None):
    """Hash each actual typed contraction, independent of node or output names.

    Shapes and ordered storage types come from the rebuilt typed program. The
    arithmetic policy retains its semantic choices, removing only its per-source
    bundle binding. These exact signatures grant no numerical or device support.
    """
    try:
        if (
            capsule.get("public_boundary")
            or (capsule.get("materialized_capture") or {}).get("schema") in {
                "merlin.materialized_public_capture.v2", "merlin.materialized_public_capture.v3"
            }
        ):
            from .public_programs import original_contractions, verify_written

            if capsule.get("public_boundary"):
                from .public_boundaries import verify_written

            if directory is None:
                raise PublicationRefusal()
            verify_written(capsule, directory)
            trace = json.loads((Path(directory) / "frontend-trace.json").read_bytes())
            source_mode = (capsule.get("materialized_capture") or {}).get("source_mode")
            return tuple(
                _digest({"schema": SIGNATURE_SCHEMA, **row, "numeric_policy": (
                    {"compare": "exact_int", "dtype": row["ordered_results"][0]["dtype"]}
                    if source_mode == "native_integer_v1" else capsule["numeric_policy"]
                )})
                for row in original_contractions(
                    trace, source_mode=source_mode
                )
            )
        saved = capsule["component_program"]
        original = capsule["operation"]
        if original["op"] != "component_program":
            raise PublicationRefusal()
        program = component_program.analyze(
            original["attributes"]["program"],
            operand_dtype=saved["selected_storage"]["operand"],
            accumulator_dtype=saved["selected_storage"]["accumulator"],
        )
        if program != saved:
            raise PublicationRefusal()
        numeric = copy.deepcopy(capsule["numerical_semantics"])
        if not isinstance(numeric, dict) or not numeric:
            raise PublicationRefusal()
        if isinstance(numeric.get("model"), dict):
            numeric["model"].pop("source_bundle_sha256", None)
        values = {row["name"]: row for row in program["inputs"] + program["nodes"]}
        return tuple(
            _digest(
                {
                    "schema": SIGNATURE_SCHEMA,
                    "operation": "matmul",
                    "ordered_inputs": [
                        {key: values[name][key] for key in ("shape", "dtype")} for name in node["actual_inputs"]
                    ],
                    "result": {key: node[key] for key in ("shape", "dtype")},
                    "numerical_semantics": numeric,
                    "semantics": "rank-two row-major signed integer contraction with selected result projection",
                }
            )
            for node in program["nodes"]
            if node["op"] == "matmul"
        )
    except (KeyError, TypeError, ValueError):
        raise PublicationRefusal() from None


class OperatorPrivateExclusion:
    """Opaque live pin: no serialization or public projection of exclusions."""

    __slots__ = ("_path", "_sha256")

    def __init__(self, path, sha256, *, _issuer):
        if _issuer is not _ISSUER:
            raise PublicationRefusal()
        object.__setattr__(self, "_path", Path(path))
        object.__setattr__(self, "_sha256", sha256)
        self._hashes()

    def __setattr__(self, _name, _value):
        raise AttributeError("fixed operator private selection")

    def __repr__(self):
        return "<OperatorPrivateExclusion>"

    def _hashes(self):
        try:
            path = self._path
            if (
                not path.is_absolute()
                or ".." in path.parts
                or any(p.is_symlink() for p in (path, *path.parents))
                or not path.is_file()
                or path.stat().st_mode & 0o077
                or path.stat().st_size > 1024 * 1024
                or not _sha256(self._sha256)
            ):
                raise PublicationRefusal()
            raw = path.read_bytes()
            if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != self._sha256:
                raise PublicationRefusal()
            data = json.loads(raw)
            keys = ("excluded_source_sha256", "excluded_contraction_sha256")
            if not isinstance(data, dict) or set(data) != {"schema", *keys} or data["schema"] != SCHEMA:
                raise PublicationRefusal()
            for key in keys:
                rows = data[key]
                if (
                    not isinstance(rows, list)
                    or len(rows) > 8192
                    or any(not _sha256(row) for row in rows)
                    or len(set(rows)) != len(rows)
                ):
                    raise PublicationRefusal()
            if not any(data[key] for key in keys):
                raise PublicationRefusal()
            return tuple(frozenset(data[key]) for key in keys)
        except (OSError, TypeError, ValueError):
            raise PublicationRefusal() from None

    def outside(self, root):
        """Keep the selected private file out of every publication tree."""
        self._hashes()
        if self._path.is_relative_to(Path(root).absolute().resolve()):
            raise PublicationRefusal() from None

    def check_member(self, directory):
        """Reopen complete ordinary source before promotion or corpus sealing."""
        try:
            sources, contractions = self._hashes()
            directory = Path(directory)
            self.outside(directory)
            if any(path.is_symlink() for path in (directory, *directory.parents, *directory.rglob("*"))):
                raise PublicationRefusal()
            cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            if not isinstance(cap, dict) or cap.get("label") not in {"public", "dev", "hidden"}:
                raise PublicationRefusal()
            if cap["label"] == "hidden":
                return
            signatures = contraction_signature_hashes(cap, directory=directory)
            source = (directory / "capsule.interface.mlir").read_bytes()
            if "component_program" in cap:
                program = cap["component_program"]
                _, expected = component_program.render(
                    cap["operation"]["attributes"]["program"],
                    operand_dtype=program["selected_storage"]["operand"],
                    accumulator_dtype=program["selected_storage"]["accumulator"],
                )
                if source != expected.encode():
                    raise PublicationRefusal()
            if hashlib.sha256(source).hexdigest() in sources:
                raise PublicationRefusal()
            if set(signatures) & contractions:
                raise PublicationRefusal()
            self._hashes()
        except (OSError, KeyError, TypeError, ValueError):
            raise PublicationRefusal() from None


def select(path, sha256):
    """Select an exact private ordinary JSON file; drift never inherits a pass."""
    return OperatorPrivateExclusion(path, sha256, _issuer=_ISSUER)


def require_selection(selection, *publication_roots):
    """Only the fixed operator selection can add this publication safeguard."""
    if selection is None:
        return
    if type(selection) is not OperatorPrivateExclusion:
        raise PublicationRefusal() from None
    for root in publication_roots:
        if root is not None:
            selection.outside(root)


def check_publication(selection, directory):
    """New integer publication requires the actual private selector at each seal.

    Legacy callers retain their optional selection. No saved capsule field can
    supply a private selection or establish that protected work was excluded.
    """
    if selection is not None:
        require_selection(selection, directory)
        selection.check_member(directory)
        return
    try:
        document = yaml.safe_load((Path(directory) / "capsule.yaml").read_bytes())
        capture = document.get("materialized_capture") if isinstance(document, dict) else None
        if isinstance(capture, dict) and capture.get("schema") == "merlin.materialized_public_capture.v3":
            raise PublicationRefusal()
    except (OSError, TypeError, ValueError):
        raise PublicationRefusal() from None
