"""Actual Q/DQ compiler boundaries, relative to the proposed source semantics.

Source capture qualification is optional and reported separately. Selecting it
requires fresh reconstruction; saved receipt statuses never authorize a check.
Neither component approves the source specification or establishes IEEE fidelity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import bf16_policy_gate as bindings
from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

SELECTOR = "lower_quant_ext"
CHECK = "compiler_qdq_refinement"
TEXTUAL = "preprocess_text_textual.dequant_to_generic"
ENV_CAPTURE = "MERLIN_QDQ_SOURCE_CAPTURE"
SOURCE_NAMES = ("quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor", "quant_ext.dequantize_per_channel")
_SOURCE_BINDING_PATHS = (
    "src/merlin/targetgen/qdq_capture.py",
    "src/merlin/targetgen/qdq_capture_identity.py",
    "src/merlin/targetgen/_m2m_capture_worker.py",
    "src/merlin/targetgen/_m2m_capture_artifacts.py",
    "src/merlin/targetgen/__init__.py",
    "src/merlin/integrations/model2mlir.py",
    "src/merlin/integrations/__init__.py",
    "src/merlin/common/provenance.py",
    "src/merlin/common/provenance_lost.py",
    "build_tools/toolchains/lean-torch-requirements.txt",
    "merlin/contract/software_pins.yaml",
)


def _source_refs(root):
    """Bind canonical capture/pin observers; foreign dependency closure stays trusted."""
    return [bindings._ref(root / path) for path in _SOURCE_BINDING_PATHS]


def source_sites(source):
    from merlin.llvmlower.passes_xdsl import parse_mlir_text

    return [
        op.op_name.data
        for op in parse_mlir_text(source).walk()
        if getattr(op, "op_name", None) is not None and op.op_name.data in SOURCE_NAMES
    ]


@dataclass(frozen=True)
class SourceCapture:
    """Explicit operator selections, not a proof completion or an approval record."""

    directory: Path
    python: Path
    producer: Path
    torch_wheel: Path
    dependencies: Path | None = None
    prove_prefix: bool = False

    def __post_init__(self):
        if type(self.prove_prefix) is not bool:
            raise ValueError("Q/DQ prefix proof requires an explicit boolean")

    def describe(self):
        return {
            **{
                name: str(value.absolute()) if value is not None else None
                for name, value in vars(self).items()
                if name != "prove_prefix"
            },
            "prove_prefix": self.prove_prefix,
        }

    @classmethod
    def from_environment(cls):
        directory = os.environ.get(ENV_CAPTURE)
        if not directory:
            return None
        names = ("MERLIN_QDQ_CAPTURE_PYTHON", "MERLIN_QDQ_PRODUCER", "MERLIN_QDQ_TORCH_WHEEL")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise ValueError("Q/DQ source qualification requires explicit " + ", ".join(missing))
        dependencies = os.environ.get("MERLIN_QDQ_CAPTURE_DEPS")
        prefix = os.environ.get("MERLIN_QDQ_PROVE_PREFIX", "0")
        if prefix not in ("0", "1"):
            raise ValueError("MERLIN_QDQ_PROVE_PREFIX requires 0 or 1")
        return cls(
            Path(directory),
            *(Path(os.environ[name]) for name in names),
            Path(dependencies) if dependencies else None,
            prove_prefix=prefix == "1",
        )


def observe_textual(source, transform):
    """Observe the actual complete textual preprocessing result, including wrappers.

    This route also performs legacy slice repairs and ABI annotation. Its exact
    complete endpoints must pass the checker; those edits receive no exemption.
    The inner xDSL call is included in this one boundary rather than submitted twice.
    """
    from . import pass_validation

    if not pass_validation.selected(SELECTOR):
        return transform(source)
    names = source_sites(source)
    if not names:
        return transform(source)
    token = pass_validation._OBSERVING.set((*pass_validation._OBSERVING.get(), (SELECTOR, None)))
    try:
        result = transform(source)
    finally:
        pass_validation._OBSERVING.reset(token)
    label = TEXTUAL if "quant_ext.dequantize_per_channel" in names else SELECTOR
    pass_validation.submit(SELECTOR, source, result[0], label=label)
    return result


def _capture(selection, before, directory):
    from merlin.common import owned_process
    from merlin.llvmlower.passes_xdsl import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    required = {
        "directory",
        "python",
        "producer",
        "torch_wheel",
        "dependencies",
    }
    if type(selection) is not dict or set(selection) not in (required, required | {"prove_prefix"}):
        raise GateUnsupported("Q/DQ source qualification requires exact explicit capture/tool selections")
    if type(selection.get("prove_prefix", False)) is not bool:
        raise GateUnsupported("Q/DQ prefix proof requires an explicit boolean")
    if any(
        type(selection[key]) is not str or not selection[key]
        for key in ("directory", "python", "producer", "torch_wheel")
    ):
        raise GateUnsupported("Q/DQ source qualification has missing selections")
    if selection["dependencies"] is not None and type(selection["dependencies"]) is not str:
        raise GateUnsupported("invalid Q/DQ dependency selection")
    root = bindings.installed_root()
    capture = Path(selection["directory"])
    refs = [
        bindings._ref(capture / name)
        for name in (
            "original.pt2",
            "qdq-source.mlir",
            "qdq-frontend.json",
            "qdq-producer.json",
            "qdq-source-check.json",
        )
    ]
    exact_input = text(parse_mlir_text(before), generic=True) == text(
        parse_mlir_text((capture / "qdq-source.mlir").read_text()), generic=True
    )
    if not exact_input and not selection.get("prove_prefix", False):
        raise GateRejected("actual Q/DQ boundary input differs from the selected source capture")
    refs.append(bindings._ref(Path(selection["python"])))
    script = (
        "import json,sys; "
        "request=json.loads(sys.argv[1]); "
        "sys.path[:0]=request['imports']; "
        "from merlin.verify.compiler_qdq_source import reconstruct; "
        "print(json.dumps(reconstruct(request['selection']),sort_keys=True))"
    )
    imports = [str(root / "src"), selection["producer"]]
    if selection["dependencies"] is not None:
        imports.append(selection["dependencies"])
    request = {"imports": imports, "selection": selection}
    environment = {**os.environ, "MERLIN_REPO_ROOT": str(root), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    process = owned_process.run(
        [selection["python"], "-I", "-B", "-c", script, json.dumps(request)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    for suffix, contents in (("stdout", process.stdout), ("stderr", process.stderr)):
        path = directory / ("source." + suffix)
        path.write_text(contents)
        refs.append(bindings._ref(path))
    cleanup = getattr(process, "owned_process_cleanup", None)
    (directory / "source.cleanup.json").write_text(json.dumps(cleanup, sort_keys=True) + "\n")
    refs.append(bindings._ref(directory / "source.cleanup.json"))
    if type(cleanup) is not dict or cleanup.get("complete") is not True:
        raise GateUnavailable("Q/DQ source reconstruction child cleanup is unknown")
    if process.returncode:
        raise GateRejected("fresh Q/DQ original-source reconstruction refused: " + process.stderr[-1200:])
    result = json.loads(process.stdout.strip().splitlines()[-1])
    if result.get("status") != "matched" or result.get("original_program_bound") is not True or not result.get("sites"):
        raise GateRejected("Q/DQ source reconstruction did not establish its complete scoped claim")
    for ref in refs:
        bindings._bound(ref)
    return {**result, "actual_input_matches_capture": exact_input}, refs


def _check_prefix(captured, actual, directory, params):
    """Fresh whole-module refinement; an earlier receipt supplies no authority.

    This connects the captured Q/DQ IR to a later compiler input under the same
    proposed model. It is not direct original-graph correspondence for that
    later syntax, nor proof of a particular prefix producer's implementation.
    """
    from .gate_checks import tensor_refinement

    directory.mkdir(parents=True, exist_ok=False)
    before, after = directory / "before.mlir", directory / "after.mlir"
    before.write_text(captured)
    after.write_text(actual)
    detail = tensor_refinement(
        {
            "directory": str(directory),
            "label": "qdq_capture_to_boundary",
            "before": str(before),
            "after": str(after),
            "params": {name: params[name] for name in ("sdk_bin", "compiler_python", "index_bits")},
        }
    )
    refs = [bindings._ref(path) for path in sorted(directory.rglob("*")) if path.is_file()]
    return {
        "status": "verified",
        "claim": detail["claim"] + "; relative to the proposed Q/DQ source semantics",
        "relation": "strict_typed_refinement",
        "source_spec_approved": False,
        "before": bindings._ref(before),
        "after": bindings._ref(after),
        "receipt": detail["receipt"],
        "chains": detail["chains"],
        "theorems": list(detail["theorems"]),
        "strengthenings": detail["strengthenings"],
    }, refs


def _compose_prefix(prefix, detail, directory, params):
    """Replay and compose both actual stages on the captured input domain."""
    from .lean import check as checker

    if set(prefix["chains"]) != set(detail["functions"]):
        raise GateRejected("Q/DQ prefix and lowering function rosters differ")
    first = Path(prefix["receipt"]).parent
    final = Path(detail["receipt"]).parent
    chains = {}
    for function in detail["functions"]:
        path = directory / f"capture-chain-{function}.json"
        outcome = checker.compose_chain(
            [first, final],
            first.parent / f"support-{function}" / "receipt.json",
            function,
            path,
            m2m_python=Path(params["compiler_python"]),
        )
        if outcome.get("refinement_verdict") != "verified":
            raise GateRejected(f"captured Q/DQ refinement did not compose for @{function}")
        chains[function] = bindings._ref(path)
    return {"status": "verified", "relation": "strict_typed_refinement", "chains": chains}, list(chains.values())


def check(job):
    """Canonical worker: fresh tensor refinement, with a distinct proposed-model claim."""
    from .gate_checks import tensor_refinement

    params = job["params"]
    required = {"sdk_bin", "compiler_python", "index_bits"}
    if type(params) is not dict or set(params) not in (required, required | {"source_capture"}):
        raise GateUnsupported("Q/DQ gate accepts explicit tools/index and optional source capture only")
    if type(params["index_bits"]) is not int or params["index_bits"] != 64:
        raise GateUnsupported("Q/DQ compiler refinement supports index64 only")
    root = bindings.installed_root()
    sources = bindings.source_bindings(root)
    extra = _source_refs(root)
    directory = Path(job["directory"])
    endpoints = [bindings._ref(Path(job[name])) for name in ("before", "after")]
    sites = source_sites(Path(job["before"]).read_text())
    if not sites:
        raise GateUnsupported("no supported source Q/DQ sites; no positive lowering coverage")
    source = {"status": "not_requested", "original_program_bound": False}
    selection = params.get("source_capture")
    prefix = None
    if selection is not None:
        first = directory / "source-before-proof"
        first.mkdir()
        source, refs = _capture(selection, Path(job["before"]).read_text(), first)
        extra.extend(refs)
        if not source["actual_input_matches_capture"]:
            captured = (Path(selection["directory"]) / "qdq-source.mlir").read_text()
            if bindings._digest(captured.encode()) != source["mlir_sha256"]:
                raise GateRejected("Q/DQ capture changed before prefix checking")
            prefix, refs = _check_prefix(
                captured, Path(job["before"]).read_text(), directory / "capture-prefix", params
            )
            extra.extend(refs)
    detail = tensor_refinement(job)
    if selection is not None:
        last = directory / "source-after-proof"
        last.mkdir()
        repeated, refs = _capture(selection, Path(job["before"]).read_text(), last)
        if source != repeated:
            raise GateRejected("Q/DQ source/tool identity changed during the compiler proof")
        extra.extend(refs)
        composed = None
        if prefix is not None:
            composed, refs = _compose_prefix(prefix, detail, directory, params)
            extra.extend(refs)
        source = {
            **source,
            "capture_to_boundary": prefix or {"status": "exact_generic_ir"},
            "capture_to_output": composed,
            "correspondence_scope": "original graph to captured IR; optional model-relative prefix to actual input",
        }
    for ref in [*endpoints, *extra]:
        bindings._bound(ref)
    if bindings.source_bindings(root) != sources:
        raise GateRejected("Q/DQ checker sources changed during the compiler proof")
    return {
        **detail,
        "theorems": list(detail["theorems"]),
        "status": "checked_proposed_qdq_refinement",
        "claim_type": "proposed_qdq_model_refinement",
        "refinement_verdict": "verified",
        "source_model_status": "proposed",
        "source_spec_approved": False,
        "source_sites": sites,
        "source_correspondence": source,
        "bindings": {"endpoints": endpoints, "project_sources": sources, "source_artifacts": extra},
        "source_capture_selection": selection,
        "source_closure_verified": False,
        "framework_equivalence": False,
        "native_fidelity": False,
        "claim": detail["claim"]
        + "; relative to the proposed Q/DQ source semantics; source specification approval is not established",
    }
