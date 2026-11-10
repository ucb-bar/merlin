"""Capture artifacts, exact input provenance and receipt-bound materialized source admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

_MODEL_CAPTURE_ABI_VERSION = 6


_CORRECTNESS_ONLY_NOTE = (
    "This capsule grades COMPILER CORRECTNESS: whether the compiled program reproduces, on these exact "
    "inputs, the reference the same loader produced. It is not a statement about the model's accuracy "
    "on real data; only a capture whose inputs are real and attributed can support that."
)


class M2MUnavailable(RuntimeError):
    """The m2m venv (torch) or the model2MLIR repo could not be located/run — fail closed, never fake."""


@dataclass
class CapsuleArtifacts:
    """Everything a grounded capsule needs. ``pytorch_src`` + ``linalg_mlir`` are agent-visible realistic
    lowering context. ``golden`` is the host-eager output and ``weights_path`` is the private model instance
    from which that output derives; both are operator-side answer surfaces masked from the agent. The weights
    still ship in the complete capsule bundle because the grader needs the exact external compile/runtime input
    named by the linalg. Only ``kind == model`` capsules ever carry a ``.safetensors``; op capsules write none."""

    op: str
    dtype: str
    pytorch_src: str
    linalg_mlir: str
    inputs: list
    golden: list
    weights_path: str
    meta: dict = field(default_factory=dict)
    #: The untransformed model's outputs on the same inputs (``float_reference.json``), when the capture
    #: recorded them; ``None`` for an already-quantized loader or a capture that predates the record.
    float_reference: object = None


def _float_reference_of(workdir: Path):
    """The worker's ``float_reference.json`` in ``workdir``, or ``None`` when it wrote none."""
    path = Path(workdir) / "float_reference.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _portable_weights_reference(mlir: str, original_weight: str | None, *, sidecar: bool) -> tuple[str, dict]:
    """Make one captured weight-reference attribute portable, recording the exact edit.

    Whole-model capsules copy the referenced weights and retain the attribute with a
    relative filename. Op capsules have explicit tensor operands and no weight sidecar,
    so only their non-load-bearing module reference is removed. Other MLIR bytes are
    left intact; an ambiguous or unexpected reference refuses instead of redacting.
    """
    key = 'prov.weights_file = "'
    start = mlir.find(key)
    if start < 0:
        return mlir, {"kind": "identity", "edit_count": 0}
    if mlir.find(key, start + len(key)) >= 0:
        raise M2MUnavailable("captured MLIR has multiple weights-file references")
    end_quote = mlir.find('"', start + len(key))
    if end_quote < 0:
        raise M2MUnavailable("captured MLIR has an unterminated weights-file reference")
    observed_reference = mlir[start + len(key) : end_quote]
    if sidecar and (not original_weight or observed_reference != str(original_weight)):
        raise M2MUnavailable("captured MLIR weight reference differs from selected weights")
    if not Path(observed_reference).is_absolute():
        raise M2MUnavailable("captured MLIR weight reference is not an absolute source path")
    end = end_quote + 1
    reference_hash = hashlib.sha256(observed_reference.encode()).hexdigest()
    if sidecar:
        portable = mlir[:start] + 'prov.weights_file = "capsule.weights.safetensors"' + mlir[end:]
        return portable, {
            "kind": "weights_reference_relocation",
            "edit_count": 1,
            "from_reference_sha256": reference_hash,
            "to": "capsule.weights.safetensors",
        }
    cut_start, cut_end = start, end
    if mlir[end : end + 2] == ", ":
        cut_end += 2
    elif mlir[start - 2 : start] == ", ":
        cut_start -= 2
    portable = mlir[:cut_start] + mlir[cut_end:]
    return portable, {
        "kind": "weights_reference_omission",
        "edit_count": 1,
        "from_reference_sha256": reference_hash,
        "to": None,
        "scope": "module provenance only; op tensors are explicit operands and no weights sidecar is shipped",
    }


def model_input_abi(art: CapsuleArtifacts) -> list[dict]:
    """Return the captured model's runtime-input ABI, cross-checked at both sides of capture.

    The worker records torch tensor leaf dtypes *before* JSON conversion. The linalg interface records the
    compiler-visible ABI after capture. Externalized parameters precede runtime leaves in model2MLIR's
    ``@forward`` signature, so its trailing leaves must agree exactly. Refuse generation if either record is
    absent or drifts; guessing from decoded values is invalid (an f32 tensor may contain only integral values).
    """
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    loader_abi = (art.meta or {}).get("input_abi")
    if not isinstance(loader_abi, list) or len(loader_abi) != len(art.inputs):
        raise M2MUnavailable(
            "model capture has no complete input_abi from the loader; refusing to infer dtypes from JSON "
            f"values (abi={loader_abi!r}, decoded_inputs={len(art.inputs)})"
        )
    try:
        iface_args = parse_linalg_mlir(art.linalg_mlir)["args"]
    except Exception as e:  # noqa: BLE001 - turn any malformed captured ABI into a closed generation error
        raise M2MUnavailable(f"could not read captured model @forward ABI: {e}") from e
    if len(iface_args) < len(loader_abi):
        raise M2MUnavailable(f"captured @forward has {len(iface_args)} args for {len(loader_abi)} loader input leaves")
    runtime_args = iface_args[-len(loader_abi) :] if loader_abi else []
    out: list[dict] = []
    for i, (decl, iface, decoded) in enumerate(zip(loader_abi, runtime_args, art.inputs, strict=True)):
        loader_shape = [int(x) for x in (decl.get("shape") or [])]
        loader_dtype = str(decl.get("dtype") or "")
        decoded_shape = _shape_of(decoded)
        iface_shape = [int(x) for x in (iface.get("shape") or [])]
        iface_dtype = str(iface.get("dtype") or "")
        if not loader_dtype or loader_shape != decoded_shape:
            raise M2MUnavailable(
                f"loader input leaf I{i} ABI {loader_shape}/{loader_dtype!r} disagrees with decoded "
                f"input shape {decoded_shape}"
            )
        if (loader_shape, loader_dtype) != (iface_shape, iface_dtype):
            raise M2MUnavailable(
                f"loader input leaf I{i} ABI {loader_shape}/{loader_dtype} disagrees with captured "
                f"@forward runtime arg {iface_shape}/{iface_dtype}"
            )
        out.append({"shape": loader_shape, "dtype": loader_dtype})
    return out


def input_provenance_record(
    workload: str | None, applied_env: dict, meta: dict, capture_declaration: dict | None = None
) -> dict:
    """What the capsule records about WHERE ITS INPUTS CAME FROM, from the loader's own declaration.

    A whole-model capsule can be captured on real, attributed dataset samples or on a seeded synthetic
    stream, and the compiled-vs-reference comparison is equally valid either way -- both sides come from
    the same loader on the same operands. What differs is what the PASS may then be quoted as: only real
    attributed inputs over a trained checkpoint can back an accuracy claim, and a capsule that does not
    say which it ran on invites the stronger reading of the weaker fact.

    FAIL CLOSED, in the direction that matters for each field separately:

    * ``synthetic_inputs`` is tri-state and stays ``"unknown"`` when the loader declared nothing -- an
      undeclared capture is not thereby a real-data one, and it is not thereby a synthetic one either.
    * ``accuracy_claim_supported`` is False unless the loader positively declares real inputs AND
      certifies the capture (``paper_ready``). An unknown withholds the claim and says so in
      ``accuracy_claim_withheld_because``, which is a different sentence from "the inputs were
      synthetic" so the two never read as one.
    """
    declared = dict(meta.get("loader_provenance") or {})
    status = str(meta.get("loader_provenance_status") or "unknown")
    raw = declared.get("synthetic_inputs")
    synthetic = raw if isinstance(raw, bool) else "unknown"
    ready = meta.get("loader_paper_ready")
    paper_ready = ready if isinstance(ready, bool) else "unknown"
    supported = (synthetic is False) and (paper_ready is True)
    if supported:
        why = ""
    elif status != "declared":
        why = (
            f"the loader declares no input provenance (status {status!r}), so whether these inputs "
            f"are real dataset samples is UNKNOWN -- an unknown may not be quoted as either"
        )
    elif synthetic is True:
        why = (
            "the inputs are a seeded SYNTHETIC stream, not real dataset samples; the reference is "
            "self-consistent with them, which proves the compiler and says nothing about accuracy"
        )
    elif synthetic == "unknown":
        why = "the loader's declaration does not state whether its inputs are real dataset samples"
    else:
        why = (
            "the loader did not certify this capture (paper_ready is "
            f"{paper_ready!r}); real inputs alone do not make the run an accuracy measurement"
        )
    return {
        "workload": workload or "",
        "status": status,
        "synthetic_inputs": synthetic,
        "paper_ready": paper_ready,
        "accuracy_claim_supported": bool(supported),
        **({"accuracy_claim_withheld_because": why} if why else {}),
        "grades": "compiler_correctness",
        "note": _CORRECTNESS_ONLY_NOTE,
        # WHAT MERLIN APPLIED, next to WHAT THE LOADER SAID. Two independent records: the first says how
        # the capture was invoked, the second what the loader made of it, and a disagreement between
        # them is visible rather than averaged away.
        "loader_env": {str(k): str(v) for k, v in sorted((applied_env or {}).items())},
        **({"capture_declaration": capture_declaration} if capture_declaration is not None else {}),
        "declared": declared,
        **({"declaration_error": meta["loader_provenance_error"]} if meta.get("loader_provenance_error") else {}),
    }


def materialized_model_artifacts(selection: dict) -> CapsuleArtifacts:
    """Read a receipt-bound model bundle without importing or rerunning its loader.

    Materialization proves byte identity only. The original source/tool closure,
    input attribution and host reference retain their recorded qualification.
    """
    import io

    import numpy as np

    from merlin.targetgen import materialized_integer_capture as native_capture
    from merlin.targetgen.application_inventory import verify_capture_receipt

    native_integer = native_capture.selected(selection)
    public_performance = (
        selection.get("schema") == "merlin.materialized_public_capture.v2" or native_integer
    ) and selection.get("workload_role") == "performance_scale"
    if (selection.get("workload_role") != "iteration" and not public_performance) or selection.get(
        "coverage_scope"
    ) != "full_capture":
        raise M2MUnavailable(
            "materialized derivation requires an explicit full iteration capture, never held-out validation"
        )
    path = Path(selection.get("path", "")).absolute()
    if path.name != "model.mlir" or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise M2MUnavailable("materialized capture must select a nonsymlink model.mlir bundle")
    if (path.parent / "capture_receipt.json").is_symlink():
        raise M2MUnavailable("materialized capture receipt cannot be symlinked")
    observed = verify_capture_receipt(path)
    if observed["status"] != "verified_materialized" or observed["receipt_sha256"] != selection.get("receipt_sha256"):
        raise M2MUnavailable(f"materialized capture receipt is absent, changed or invalid: {observed['errors']}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selection.get("capture_sha256"):
        raise M2MUnavailable("materialized model bytes differ from the selected source inventory")
    directory = path.parent
    receipt_raw = (directory / "capture_receipt.json").read_bytes()
    if hashlib.sha256(receipt_raw).hexdigest() != selection.get("receipt_sha256"):
        raise M2MUnavailable("materialized capture receipt changed during selection")
    receipt = json.loads(receipt_raw)
    from merlin.targetgen.claim_models import is_claim_bundle

    declared_source = receipt.get("source") or {}
    identities = [
        selection.get("workload_id"),
        selection.get("source_workload_id"),
        declared_source.get("workload_id"),
        declared_source.get("model"),
        *Path(str(declared_source.get("path") or "")).parts,
    ]
    if declared_source.get("workload_role") == "validation" or any(
        isinstance(identity, str) and is_claim_bundle(identity) for identity in identities if identity
    ):
        raise M2MUnavailable(
            "materialized source is declared held-out validation even if its selected bundle was renamed"
        )
    reference = native_capture.REFERENCE if native_integer else "golden.npy"
    required = {"meta.json", "inputs.npz", "input_order.json", reference, "frontend-trace.json"}
    if not required <= set(receipt["artifacts"]):
        raise M2MUnavailable("materialized bundle lacks receipt-bound metadata, inputs, golden or source trace")
    if receipt.get("lifted_constants"):
        raise M2MUnavailable("materialized model has lifted runtime constants without a capsule input binding")

    def read_bound(name):
        selected = directory / name
        if selected.is_symlink() or not selected.is_file():
            raise M2MUnavailable(f"materialized input disappeared or became symlinked: {name}")
        value = selected.read_bytes()
        declared = receipt["artifacts"].get(name) or {}
        if declared.get("bytes") != len(value) or declared.get("sha256") != hashlib.sha256(value).hexdigest():
            raise M2MUnavailable(f"materialized input changed during selection: {name}")
        return value

    meta = json.loads(read_bound("meta.json"))
    if not meta.get("ok") or meta.get("opaque", -1) != 0:
        raise M2MUnavailable("materialized model metadata does not declare a clean capture")
    if public_performance and meta.get("extra"):
        raise M2MUnavailable("public performance capture needs a complete portable extra-constant binding")
    output_abi = meta.get("output_abi") or []
    # The existing bundle format records only the first eager tensor as golden.npy.
    # fp16/BF16 values lift losslessly into f32; integer/f64 references do not.
    # Never silently certify a multi-result or integer-result ABI with that file.
    complete_reference = not native_integer and len(output_abi) > 1
    if complete_reference:
        from merlin.targetgen.application_inventory import verified_static_integerization
        from merlin.targetgen.materialized_reference import complete_float_outputs

        if not verified_static_integerization(
            observed.get("capture_integerization"), receipt_sha256=observed["receipt_sha256"]
        ):
            raise M2MUnavailable("multiple materialized outputs require the complete independent integer reference")
        try:
            golden_values = complete_float_outputs(meta, read_bound("integer-reference.json"))
        except (ValueError, TypeError, KeyError) as error:
            raise M2MUnavailable(str(error)) from error
    if (
        not native_integer
        and not complete_reference
        and (len(output_abi) != 1 or output_abi[0].get("dtype") not in {"f32", "f16", "bf16"})
    ):
        raise M2MUnavailable(
            "materialized golden.npy needs one losslessly represented floating result; "
            "other ABIs need complete references"
        )
    count = len(meta.get("input_abi") or [])
    if native_integer:
        inputs = native_capture.inputs(meta, read_bound("inputs.npz"))
        golden_values = native_capture.outputs(meta, read_bound(reference))
    else:
        with np.load(io.BytesIO(read_bound("inputs.npz")), allow_pickle=False) as archive:
            names = [f"in{i}" for i in range(count)]
            if set(archive.files) != set(names):
                raise M2MUnavailable("materialized input leaves differ from the declared input ABI")
            inputs = [archive[name].tolist() for name in names]
    order = json.loads(read_bound("input_order.json"))
    if not isinstance(order, dict) or sorted(order.values()) != list(range(count)):
        raise M2MUnavailable("materialized input-order mapping is not a complete positional ABI")
    if not native_integer and not complete_reference:
        golden_array = np.load(io.BytesIO(read_bound("golden.npy")), allow_pickle=False)
        if list(golden_array.shape) != output_abi[0].get("shape") or str(golden_array.dtype) != "float32":
            raise M2MUnavailable("materialized golden shape/dtype differs from the output ABI")
        if output_abi[0]["dtype"] == "f16" and not np.array_equal(
            golden_array, golden_array.astype(np.float16).astype(np.float32), equal_nan=True
        ):
            raise M2MUnavailable("materialized golden is not an exact fp16 value lifted to f32")
        if output_abi[0]["dtype"] == "bf16" and np.any(
            (golden_array.view(np.uint32) & 0xFFFF != 0) & np.isfinite(golden_array)
        ):
            raise M2MUnavailable("materialized golden is not an exact BF16 value lifted to f32")
        golden_values = golden_array.tolist()
    trace = dict(meta.get("frontend_trace") or {})
    trace["path"] = str(directory / "frontend-trace.json")
    meta["frontend_trace"] = trace
    catalog = dict(meta.get("framework_catalog") or {})
    if catalog:
        catalog["path"] = str(directory / "pytorch-opset.json")
        catalog_path = Path(catalog["path"])
        if (
            catalog_path.is_symlink()
            or not catalog_path.is_file()
            or hashlib.sha256(catalog_path.read_bytes()).hexdigest() != catalog.get("sha256")
        ):
            raise M2MUnavailable("materialized same-process framework catalog is absent or changed")
        meta["framework_catalog"] = catalog
    meta["frontend_raw_mlir_path"] = str(path)
    weights = directory / "weights.safetensors"
    meta["weights"] = str(weights)
    meta["weights_manifest"] = str(directory / "weights.safetensors.manifest.json")
    meta["materialized_capture"] = {
        **selection,
        "path": str(path),
        "source_closure_verified": observed["source_closure_verified"],
        "scope": "full saved source program and host-eager reference; target execution unverified",
    }
    if bool(selection.get("loader_path")) != bool(selection.get("loader_sha256")):
        raise M2MUnavailable("materialized loader needs both its selected path and exact digest")
    if selection.get("loader_path"):
        loader = Path(selection["loader_path"]).absolute()
        if loader.name != "loader.py" or loader.parent != directory or loader.is_symlink() or not loader.is_file():
            raise M2MUnavailable("materialized loader must be an ordinary member beside its selected capture")
        if hashlib.sha256(loader.read_bytes()).hexdigest() != selection["loader_sha256"]:
            raise M2MUnavailable("materialized loader differs from its preselected workload snapshot")
    if selection.get("loader_path"):
        profile_sha = (meta.get("loader_provenance") or {}).get("profile_sha256")
        if profile_sha is not None or selection.get("profile_path"):
            profile = Path(selection.get("profile_path", ""))
            if (
                profile.parent != directory
                or profile.name != "profile.json"
                or profile.is_symlink()
                or not profile.is_file()
                or hashlib.sha256(profile.read_bytes()).hexdigest() != profile_sha
                or profile_sha != selection.get("profile_sha256")
            ):
                raise M2MUnavailable("materialized loader profile differs from its preselected capture")
    meta["materialized_artifact_identities"] = receipt["artifacts"]
    meta["materialized_receipt_bytes"] = receipt_raw
    # Retain original source bytes. The writer records the one known relocation.
    source_weights = json.loads(read_bound("meta.json")).get("weights")
    program = raw.decode("utf-8")
    if not isinstance(source_weights, str) or f'prov.weights_file = "{source_weights}"' not in program:
        raise M2MUnavailable("materialized model lacks its exact recorded weights-location attribute")
    meta["materialized_weights_reference"] = source_weights
    return CapsuleArtifacts("model", meta.get("dtype", "fp32"), "", program, inputs, golden_values, str(weights), meta)


def _shape_of(nested) -> list[int]:
    s = []
    x = nested
    while isinstance(x, list):
        s.append(len(x))
        x = x[0] if x else None
    return s
