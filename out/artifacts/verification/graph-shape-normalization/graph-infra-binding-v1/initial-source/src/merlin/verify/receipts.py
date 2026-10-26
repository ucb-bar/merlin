"""Reproducible evidence for one supported compiler transformation.

The receipt identifies both *actual* artifacts, the verifier implementation, translator binary,
solver version, SMT query, and outcome. A qualification gate reruns the obligation. Neither an IR
syntax check nor a copied ``unsat`` string is accepted as semantic evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from .smt_export import SmtUnavailable, bounded_tool_output, module_text
from .smt_semantics import UnsupportedSemantics

Boundary = Literal["linalg_to_interface", "interface_to_command_buffer"]
_BOUNDARIES = frozenset(("linalg_to_interface", "interface_to_command_buffer"))
_SEMANTIC_FILES = (
    "cb_semantics.py",
    "linalg_semantics.py",
    "merlin_iface_semantics.py",
    "receipts.py",
    "refine.py",
    "smt_export.py",
    "smt_ops.py",
    "smt_semantics.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(value: Any, *, command_buffer: bool = False) -> dict[str, str]:
    if command_buffer:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        kind = "canonical_json"
    elif isinstance(value, str):
        data = value.encode("utf-8")
        kind = "merlin_iface_text_utf8"
    else:
        data = module_text(value).encode("utf-8")
        kind = "xdsl_generic_ir"
    return {"encoding": kind, "sha256": _sha(data)}


def _ir_signature(module: Any) -> dict[str, Any]:
    if isinstance(module, str):
        from .merlin_iface_semantics import merlin_iface_signature

        try:
            return merlin_iface_signature(module)
        except UnsupportedSemantics as exc:
            return {"grammar": "merlin_iface", "parse_error": str(exc)}
    funcs = [op for op in module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        return {"function_count": len(funcs)}
    block = funcs[0].body.block
    returns = [op for op in block.ops if op.name == "func.return"]
    return {
        "function_count": 1,
        "inputs": [str(arg.type) for arg in block.args],
        "return_count": len(returns),
        "outputs": [str(value.type) for value in returns[0].operands] if len(returns) == 1 else [],
    }


def _cb_signature(cb: dict) -> dict[str, Any]:
    return {
        "tensors": {
            name: {key: tensor.get(key) for key in ("dtype", "shape", "role")}
            for name, tensor in sorted((cb.get("tensors") or {}).items())
        },
        "outputs": list(cb.get("outputs") or ()),
    }


def _verifier_digest() -> str:
    root = Path(__file__).parent
    data = b"".join(name.encode() + b"\0" + (root / name).read_bytes() + b"\0" for name in _SEMANTIC_FILES)
    bridge = root.parent / "targetgen" / "contract" / "interface_emit.py"
    data += b"targetgen/contract/interface_emit.py\0" + bridge.read_bytes() + b"\0"
    return _sha(data)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _toolchain(translator: str | Path) -> dict[str, str | None]:
    path = Path(translator).resolve(strict=True)
    if not path.is_file():
        raise SmtUnavailable(f"translator {path} is not a regular file")
    returncode, stdout, stderr = bounded_tool_output(
        [str(path), "--version"], timeout_s=10, max_output_bytes=1024 * 1024
    )
    if returncode != 0:
        raise SmtUnavailable(f"cannot identify translator {path}: exit {returncode}: {stderr[:1000]}")
    reported = stdout.strip()
    try:
        import z3

        solver_version = z3.get_version_string()
    except ImportError:
        solver_version = None
    return {
        "mlir_translate_path": str(path),
        "mlir_translate_sha256": _sha(path.read_bytes()),
        "mlir_translate_version": reported,
        "z3_version": solver_version,
        "xdsl_version": _package_version("xdsl"),
        "verifier_sha256": _verifier_digest(),
    }


@dataclass(frozen=True)
class TransformReceipt:
    schema: str
    method: str
    boundary: str
    source: dict[str, str]
    target: dict[str, str]
    typed_signatures: dict[str, Any]
    toolchain: dict[str, str | None]
    acc_width: int
    timeout_ms: int
    assumptions: tuple[str, ...]
    status: str
    reason: str | None
    smt2_sha256: str | None
    counterexample: dict[str, Any] | None

    @property
    def verified(self) -> bool:
        return self.method == "smt_translation_validation" and self.status == "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> TransformReceipt:
        """Load a JSON round trip; qualification still replays the obligation."""
        if record.get("schema") != "merlin.transform_verification.v1":
            raise ValueError(f"unrecognized transform receipt schema {record.get('schema')!r}")
        return cls(**{**record, "assumptions": tuple(record["assumptions"])})


def verify_transformation(
    boundary: Boundary,
    source: Any,
    target: Any,
    *,
    translator: str | Path,
    expected_translator_sha256: str | None = None,
    acc_width: int = 32,
    timeout_ms: int = 60_000,
) -> TransformReceipt:
    """Prove one concrete, supported source/target pair or record why no proof was obtained.

    A recorded ``verified`` means QF_BV unsatisfiability for every bit pattern of the typed input
    tensors at the concrete IR shape, under the explicitly listed semantics assumptions. It does
    not claim all shapes, arbitrary models, floating point, or equivalence to real hardware.
    """
    if boundary not in _BOUNDARIES:
        raise ValueError(f"unknown transformation boundary {boundary!r}")
    if acc_width <= 0 or timeout_ms <= 0:
        raise ValueError("acc_width and timeout_ms must be positive")

    is_cb = boundary == "interface_to_command_buffer"
    src_id = _identity(source)
    dst_id = _identity(target, command_buffer=is_cb)
    typed_signatures = {
        "source": _ir_signature(source),
        "target": _cb_signature(target) if is_cb else _ir_signature(target),
    }
    assumptions = (
        "rank-2 positive concrete integer tensor extents; signed bitvectors; modular accumulation",
        "input tensors bound by command-buffer role and order"
        if is_cb
        else (
            "merlin_iface leaves named argN bound to source argument N"
            if isinstance(target, str)
            else "input tensors bound by block-argument position"
        ),
        "contraction init must have defined contents; a bare tensor.empty init abstains"
        if not is_cb
        else "interface.resident_pack preserves values; physical layout is outside this value proof",
        *(("merlin_iface.resident_pack preserves values; physical layout is outside this value proof",)
          if isinstance(target, str) and not is_cb else ()),
    )
    tool: dict[str, str | None] = {}
    status = "unavailable"
    reason: str | None = None
    smt_hash: str | None = None
    counterexample: dict[str, Any] | None = None
    try:
        tool = _toolchain(translator)
        actual_sha = tool["mlir_translate_sha256"]
        if expected_translator_sha256 is not None and actual_sha != expected_translator_sha256:
            raise SmtUnavailable(
                f"translator SHA-256 mismatch: expected {expected_translator_sha256}, found {actual_sha}"
            )
        from .refine import validate_compilation, validate_pass

        verdict = (
            validate_compilation(source, target, acc_width=acc_width, timeout_ms=timeout_ms, translator=translator)
            if is_cb
            else validate_pass(source, target, acc_width=acc_width, timeout_ms=timeout_ms, translator=translator)
        )
        # Reject a replacement of the selected binary between the initial hash/version read and
        # export. The receipt's tool identity must describe the executable that produced the query.
        after_hash = _sha(Path(translator).resolve(strict=True).read_bytes())
        if after_hash != actual_sha:
            raise SmtUnavailable("translator bytes changed during the verification attempt")
        smt_hash = _sha(verdict.smt2.encode("utf-8"))
        status = {"unsat": "verified", "sat": "refuted", "unknown": "unknown"}[verdict.status]
        reason = verdict.reason
        if status == "refuted":
            counterexample = {"model": verdict.model, "inputs": verdict.model_values}
    except UnsupportedSemantics as exc:
        status, reason = "unsupported", str(exc)
    except SmtUnavailable as exc:
        status, reason = "unavailable", str(exc)
    except Exception as exc:
        # A checker error is visible and cannot be promoted into a proof. The exception class is
        # included to make an unexpected translator or encoder failure diagnosable from the receipt.
        status, reason = "error", f"{type(exc).__name__}: {exc}"

    return TransformReceipt(
        schema="merlin.transform_verification.v1",
        method="smt_translation_validation",
        boundary=boundary,
        source=src_id,
        target=dst_id,
        typed_signatures=typed_signatures,
        toolchain=tool,
        acc_width=acc_width,
        timeout_ms=timeout_ms,
        assumptions=assumptions,
        status=status,
        reason=reason,
        smt2_sha256=smt_hash,
        counterexample=counterexample,
    )


def qualify_receipt(
    receipt: TransformReceipt,
    source: Any,
    target: Any,
    *,
    translator: str | Path,
) -> bool:
    """Rerun and require the *same* semantic proof for the exact current artifacts/toolchain.

    A syntax verifier may be useful separately, but it cannot produce this receipt's method. Old,
    fabricated, edited, unsupported, refuted, unknown, and tool-mismatched receipts cannot qualify.
    """
    if not receipt.verified or receipt.boundary not in _BOUNDARIES:
        return False
    replay = verify_transformation(
        receipt.boundary,
        source,
        target,
        translator=translator,
        expected_translator_sha256=receipt.toolchain.get("mlir_translate_sha256"),
        acc_width=receipt.acc_width,
        timeout_ms=receipt.timeout_ms,
    )
    return replay.verified and replay == receipt
