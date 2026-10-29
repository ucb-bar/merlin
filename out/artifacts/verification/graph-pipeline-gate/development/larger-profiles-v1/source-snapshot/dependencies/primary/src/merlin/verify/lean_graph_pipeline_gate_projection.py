"""Fresh actual-buffer projection for the whole-program compiler gate.

Only Python parsing/projection and source bytes are inspected here. This is
trusted mapping code, not a proof. The unchanged scalar checker proves the
generated operational relation when an authorized Lean context checks it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


class GateUnsupported(ValueError):
    """A specifically recognized boundary outside this route's domain."""


class GateUnavailable(RuntimeError):
    """Host, implementation or dependency failure; never a semantic refusal."""


class GateNotEstablished(Exception):
    """An authorized kernel ran but did not establish the generated obligation."""


DOMAIN = "complete_scalar_roster_equal_outcome_v1"
RELATION = "all_defined_entries_equal_returns_errors_terminal_world"
KINDS = frozenset({"scalar_redundancy_elimination", "scalar_loop_invariant_motion"})
PREMISES = (
    "trusted xDSL parser and complete typed SSA/module projection",
    "one common deterministic total state-free pure primitive interpretation",
    "one common opaque descriptor interpretation on explicit operands and entire world",
    "local opaque successful result arity contract; errors and terminal world observable",
    "common finite abstract SCF iterations and index interpretation; no divergence claim",
    "private SSA cells are not observations; raw MLIR and physical implementation correspondence reviewed separately",
)

# Only exact, inspected domain messages from project() are semantic refusals.
# The old source-identity/runtime helpers also raise ValueError: they must never
# be included here or caught by a generic ValueError -> unsupported adapter.
_PROJECTION_DOMAIN = frozenset({
    "complete actual builtin.module pair required",
    "complete surrounding module metadata changed",
    "unsupported complete module block geometry",
    "complete nonempty function/declaration roster required",
    "unsupported module-level operation; no missing surrounding context",
    "duplicate actual function symbols",
    "missing/unsupported actual terminator",
    "actual operand lacks lexical availability/dominance",
    "complete graph exceeds explicit scalar intake budget",
    "unsupported actual CFG; no unreachable block can be dropped",
    "changed/unsupported terminator metadata or control",
    "unsupported loop metadata or block geometry",
    "supported actual SCF control needs a checked positive constant stride",
    "actual induction/carry argument slots differ",
    "actual loop result slots differ from carries",
    "unsupported region semantics; entire boundary refused",
    "internal-call/fuel composition bridge unavailable; external contracts only",
    "complete function interface/attributes changed",
    "unsupported/dropped function CFG blocks",
    "actual function argument slots differ from interface",
})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    size: int


def source_ref(path):
    path = Path(path).resolve()
    if path.suffix not in {".py", ".lean"}:
        raise GateUnavailable("gate source identity must not inspect runtime objects")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields) or len(data) != after.st_size:
        raise GateUnavailable("host source changed during identity observation")
    return Source(str(path), sha(data), len(data))


def project_lean_sources(roots):
    """Project .lean source closure only; roots belong to trusted emitters."""
    paths = set()
    root = Path(__file__).parent / "lean"
    todo = list(roots)
    seen = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        if not name.startswith("Merlin.") or any(not part.isidentifier() for part in name.split(".")):
            raise GateUnavailable("invalid project source import")
        seen.add(name)
        path = root / (name.replace(".", "/") + ".lean")
        paths.add(path)
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "import":
                todo += [item for item in fields[1:] if item.startswith("Merlin.")]
    return tuple(source_ref(path) for path in sorted(paths))


def source_closure():
    """Executed project/parser Python plus exact transitive project Lean sources.

    No SDK objects are read. Their matching import/object identity is a separate
    native-check prerequisite, and is currently unavailable under the hold.
    """
    # Load the selected SOURCE-ONLY provider before taking the closure, so the
    # first check and fresh postcheck bind identical frozen class/source imports.
    # This never constructs a context or opens a runtime image.
    from .lean_graph_pipeline_gate_provider import selected
    selected()
    paths = {Path(__file__), Path(__file__).with_name("lean_graph_pipeline_gate.py"),
             Path(__file__).with_name("lean_graph_pipeline_gate_native.py"),
             Path(__file__).with_name("lean_graph_pipeline_gate_provider.py"),
             Path(__file__).with_name("lean_graph_pipeline_gate_emit.py")}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("merlin.", "xdsl.")):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).suffix == ".py":
                paths.add(Path(filename))
    lean = project_lean_sources(("Merlin.GraphStateScalar", "Merlin.GraphStateGateEntries"))
    return tuple(source_ref(path) for path in sorted(paths)) + lean


@dataclass(frozen=True)
class Prepared:
    schema: str
    kind: str
    source: str
    target: str
    change: str
    domain: str
    relation: str
    premises: tuple[str, ...]
    python_version: str
    query_contract: str
    projection: str
    query: str
    roots: tuple[str, ...]
    sources: tuple[Source, ...]

    @property
    def identity(self):
        return sha(canonical(asdict(self)))


def complete_query(projection):
    from .lean_graph_pipeline_gate_emit import emit_query

    text, roots = emit_query(projection)
    # Imports precede all commands; add no new axioms or semantic definitions.
    text = "import Merlin.GraphStateGateEntries\n" + text
    text += "namespace Merlin.GraphPipelineGateInstance\n"
    text += "open Merlin.GraphState.Scalar Merlin.GraphState.GateEntries\n"
    text += "abbrev registry := Merlin.GraphStateScalarInstance.registry\n"
    defined = [i for i, row in enumerate(projection["functions"]) if not row["declaration"]]
    for side in ("before", "after"):
        entries = []
        for index in defined:
            base = f"Merlin.GraphStateScalarInstance.F{index}."
            entries.append("{inputs := " + base + "inputs, outputs := " + base + "outputs, body := " + base + side + "}")
        text += f"def {side}Entries : List Entry := [" + ",".join(entries) + "]\n"
    proof = "entries_nil model"
    for index in reversed(defined):
        proof = "entries_cons model rfl rfl (Merlin.GraphStateScalarInstance.F" + str(index) + ".actual Value World Error model) (" + proof + ")"
    text += "theorem complete {Value World Error : Type} (model : Model Value World Error registry) :\n"
    text += "    PreservesEntries model beforeEntries afterEntries :=\n  " + proof + "\n"
    text += "end Merlin.GraphPipelineGateInstance\n"
    # One exact public root. A native check must audit this root transitively,
    # not merely report that some imported local theorem compiled.
    text += "theorem merlin_transform {Value World Error : Type}\n"
    text += "    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphPipelineGateInstance.registry) :\n"
    text += "    Merlin.GraphState.GateEntries.PreservesEntries model\n"
    text += "      Merlin.GraphPipelineGateInstance.beforeEntries Merlin.GraphPipelineGateInstance.afterEntries :=\n"
    text += "  Merlin.GraphPipelineGateInstance.complete model\n"
    roots += ["Merlin.GraphState.GateEntries.entries_cons", "Merlin.GraphState.GateEntries.entries_nil",
              "Merlin.GraphPipelineGateInstance.complete", "merlin_transform"]
    # NEW query contract: compile pure declarations with empty stdout. The
    # common Context owns separate, bound exact-root axiom/type audit commands.
    # Historical generated queries with in-source reports remain unchanged.
    return text, tuple(roots)


def prepare(change):
    """No caller-supplied projection, query, model, domain or status is consumed."""
    from xdsl.utils.exceptions import ParseError, VerifyException
    from merlin.transforms.certificates import ChangeCertificate, TRANSFORM_SCOPES, TransformationChange, parse_ir
    from .lean_graph_input_admission import InputAdmissionError, admit_text, admit_module
    from .lean_graph_state_scalar_capture import project

    if type(change) is not TransformationChange or change.kind not in KINDS:
        raise GateUnsupported("no whole-program route for this transformation kind")
    if type(change.certificate) is not ChangeCertificate or change.certificate.schema != "merlin.shared_transform.v2":
        raise GateUnsupported("exact registered v2 transformation certificate required")
    if type(change.certificate.scope) is not tuple or change.certificate.scope != TRANSFORM_SCOPES[change.kind]:
        raise GateUnsupported("incorrect registered transformation scope")
    if type(change.before_ir) is not str or type(change.after_ir) is not str:
        raise GateUnsupported("exact complete UTF-8 text buffers required")
    if max(len(change.before_ir.encode()), len(change.after_ir.encode())) > 4 * 1024 * 1024:
        raise GateUnsupported("complete boundary exceeds explicit text budget")
    if (change.certificate.kind != change.kind or change.certificate.before_ir_sha256 != sha(change.before_ir)
            or change.certificate.after_ir_sha256 != sha(change.after_ir)):
        raise GateUnsupported("stale/wrong-program transformation certificate")
    try:
        # Both texts are admitted BEFORE either parser touches global resources.
        admit_text(change.before_ir)
        admit_text(change.after_ir)
        before, after = parse_ir(change.before_ir), parse_ir(change.after_ir)
        admit_module(before, calls="external-only")
        admit_module(after, calls="external-only")
    except InputAdmissionError as error:
        raise GateUnsupported("input admission: " + str(error)) from error
    except (ParseError, VerifyException) as error:
        raise GateUnsupported("malformed actual IR: " + str(error)) from error
    try:
        projection = project(before, after)
    except InputAdmissionError as error:
        raise GateUnsupported("input admission: " + str(error)) from error
    except ValueError as error:
        if str(error) in _PROJECTION_DOMAIN:
            raise GateUnsupported(str(error)) from error
        raise GateUnavailable("unclassified projection error: " + str(error)) from error
    try:
        query, roots = complete_query(projection)
    except ValueError as error:
        # Author defects stay unavailable. Two explicit work bounds and absence
        # of a defined entry are the only non-author-error cases here.
        if str(error) in {"complete bounded nonempty supported block required",
                          "supported scalar symbolic expression budget exceeded",
                          "no defined complete function available for a graph proof"}:
            raise GateUnsupported(str(error)) from error
        raise GateUnavailable("unclassified author error: " + str(error)) from error
    if len(query.encode()) > 1024 * 1024:
        raise GateUnsupported("generated complete proof exceeds explicit query budget")
    return Prepared("merlin.graph_pipeline_prepared.v3", change.kind,
                    sha(change.before_ir), sha(change.after_ir), sha(canonical(asdict(change))),
                    DOMAIN, RELATION, PREMISES, sys.version,
                    "pure_declarations_separate_root_audit_v1", canonical(projection), query, roots, source_closure())
