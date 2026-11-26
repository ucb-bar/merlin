"""Authenticate producer policy inputs before numerical family selection.

These content identities bind an export, its saved program, and a materialized
capture. They establish no numerical authorization or source equivalence.
Producer schema identities and the saved-program metadata path are caller data.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from merlin.common.digest import is_sha256


@dataclass(frozen=True)
class AuthenticatedPolicyInputs:
    export_policy_json: str
    export_manifest_sha256: str
    export_policy_sha256: str
    capture_receipt_sha256: str
    quantization_manifest_sha256: str
    quantization_manifest_json: str
    fx_manifest_sha256: str
    native_program_sha256: str


def _fail(field, reason):
    raise ValueError(f"{field}: {reason}")


def _directory(value, field):
    root = Path(value).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents) or not root.is_dir():
        _fail(field, "requires a real directory without symlink parents")
    return root


def _object(raw, field):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    try:
        value = json.loads(raw, parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as error:
        _fail(field, str(error))
    if not isinstance(value, dict):
        _fail(field, "requires a JSON object")
    return value


def _canonical(value, field):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except ValueError as error:
        _fail(field, str(error))


def authenticate_policy_inputs(
    export_dir, fx_dir, capture_dir, *, export_manifest_schema, export_policy_schema,
    fx_manifest_schema, adapter_id, native_program_sha256_path: tuple[str, ...],
    program_member="model.pt2",
):
    """Join checked export policy bytes with receipt-bound capture provenance.

    The returned receipt identity must still equal the receipt used during
    compiler staging. The canonical quantization identity must still be bound
    by the staged IR. Those later checks belong to the staging caller.
    """
    for field, value in (("manifest.schema", export_manifest_schema),
                         ("policy.schema", export_policy_schema),
                         ("fx.schema", fx_manifest_schema), ("capture.provenance.adapter_id", adapter_id)):
        if type(value) is not str or not value.strip():
            _fail(field, "requires an explicit schema or adapter identity")
    if (type(native_program_sha256_path) is not tuple or not native_program_sha256_path
            or any(type(key) is not str or not key for key in native_program_sha256_path)):
        _fail("capture.provenance.saved_native_program.sha256", "requires an immutable metadata path")
    if type(program_member) is not str or not program_member or Path(program_member).name != program_member:
        _fail("fx.files", "requires a single saved-program member name")

    # Use the same byte checks as staging without loading IR or runtime dependencies.
    from merlin.compiler.frontend.inputs import _checked_member, _read_member

    def read(root, name, field, receipt=None):
        try:
            return (_read_member(root, name) if receipt is None
                    else _checked_member(root, receipt, name))
        except (ValueError, OSError) as error:
            _fail(field, str(error))

    export_root = _directory(export_dir, "manifest")
    manifest_raw = read(export_root, "manifest.json", "manifest")
    manifest = _object(manifest_raw, "manifest")
    if manifest.get("schema") != export_manifest_schema:
        _fail("manifest.schema", f"requires {export_manifest_schema}")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != 1:
        _fail("manifest.format_version", "requires version 1")
    policy_raw = read(export_root, "policy.json", "policy")
    policy_sha = sha256(policy_raw).hexdigest()
    files = manifest.get("files")
    entry = files.get("policy.json") if isinstance(files, dict) else None
    if (not isinstance(entry, dict) or entry.get("sha256") != policy_sha
            or manifest.get("policy_sha256") != policy_sha
            or type(entry.get("bytes")) is not int or entry["bytes"] != len(policy_raw)):
        _fail("manifest.policy_sha256", "policy bytes or SHA-256 differ")
    policy = _object(policy_raw, "policy")
    if policy.get("schema") != export_policy_schema:
        _fail("policy.schema", f"requires {export_policy_schema}")
    if type(policy.get("format_version")) is not int or policy["format_version"] != 1:
        _fail("policy.format_version", "requires version 1")
    manifest_sha = sha256(manifest_raw).hexdigest()

    capture_root = _directory(capture_dir, "capture.provenance")
    receipt_raw = read(capture_root, "capture_receipt.json", "capture.provenance")
    receipt = _object(receipt_raw, "capture.provenance")
    if receipt.get("schema") != "m2m.capture-receipt.v1" or not isinstance(receipt.get("artifacts"), dict):
        _fail("capture.provenance", "requires the capture producer's receipt")
    quantization = _object(read(capture_root, "quantization-manifest.json", "capture.provenance", receipt),
                           "capture.provenance")
    if (quantization.get("schema") not in ("m2m.quantization_manifest.v1", "m2m.quantization_manifest.v2")
            or quantization.get("adapter_id") != adapter_id):
        _fail("capture.provenance", "requires the declared adapter's captured quantization manifest")
    for field, expected in (("contract_sha256", manifest_sha), ("policy_sha256", policy_sha)):
        if quantization.get(field) != expected:
            _fail(f"capture.provenance.{field}", "differs from the supplied export")

    fx_root = _directory(fx_dir, "capture.provenance")
    fx_raw = read(fx_root, "fx_manifest.json", "capture.provenance")
    fx = _object(fx_raw, "fx")
    if fx.get("schema") != fx_manifest_schema:
        _fail("fx.schema", f"requires {fx_manifest_schema}")
    provenance = fx.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("export_manifest_sha256") != manifest_sha:
        _fail("fx.provenance.export_manifest_sha256", "differs from the supplied export")
    native_sha = quantization
    for key in native_program_sha256_path:
        native_sha = native_sha.get(key) if isinstance(native_sha, dict) else None
    files = fx.get("files")
    program = files.get(program_member) if isinstance(files, dict) else None
    if not is_sha256(native_sha) or not isinstance(program, dict) or program.get("sha256") != native_sha:
        _fail("capture.provenance.saved_native_program.sha256", "missing or differs from the supplied FX manifest")

    quantization_json = _canonical(quantization, "capture.provenance")
    return AuthenticatedPolicyInputs(
        export_policy_json=_canonical(policy, "policy"),
        export_manifest_sha256=manifest_sha,
        export_policy_sha256=policy_sha,
        capture_receipt_sha256=sha256(receipt_raw).hexdigest(),
        quantization_manifest_sha256=sha256(quantization_json.encode()).hexdigest(),
        quantization_manifest_json=quantization_json,
        fx_manifest_sha256=sha256(fx_raw).hexdigest(),
        native_program_sha256=native_sha,
    )


@contextmanager
def stage_policy_inputs(bound: AuthenticatedPolicyInputs, capture_dir):
    """Stage the same authenticated capture and check its IR provenance join.

    Yield the temporary compiler directory and its checked inputs so source
    translation can load authenticated lifted constants before cleanup. Mutable
    buffers remain buffers; this boundary grants no immutable ancestry or target
    numerical contract. No generated policy is published before these checks.
    """
    from tempfile import TemporaryDirectory

    from xdsl.dialects.builtin import StringAttr

    from merlin.compiler.frontend.inputs import load_compile_inputs, stage_compile_inputs
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if type(bound) is not AuthenticatedPolicyInputs:
        raise ValueError("policy staging requires authenticated policy inputs")
    capture = _directory(capture_dir, "capture.provenance")
    with TemporaryDirectory(prefix="merlin-policy-") as temporary:
        directory = Path(temporary) / "staged"
        manifest = stage_compile_inputs(capture, directory)
        if manifest["source"]["capture_receipt_sha256"] != bound.capture_receipt_sha256:
            _fail("capture.provenance", "capture receipt changed during staging")
        staged = load_compile_inputs(directory)
        module = parse_mlir_text(staged.program)
        if module.attributes.get("prov.quantization_manifest_sha256") != StringAttr(
            bound.quantization_manifest_sha256
        ):
            _fail("capture.provenance", "captured IR does not bind the quantization manifest")
        yield directory, staged
