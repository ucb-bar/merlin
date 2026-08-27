"""LoweringTrace — the structured, LLM-digestible thread from *flattened graph → transformation steps →
code → asm*, per kernel/region.

This is the "missing middle" of the CCA pipeline: we already have a structured GRAPH view (model2MLIR
MLIR + ``prov.*`` via ``frontends.linalg_mlir``) and a deep ASM view (``kernels.decode`` +
``kernels.features``), but nothing recorded the *sequence of transformations* that carried a graph op down
to the emitted asm. A ``LoweringTrace`` ties those three levels into one object so (a) the CCA can be
DERIVED from the derivation path (not hand-picked) and (b) the compiler's own pipeline is describable in an
LLM-digestible form to compare compiler-CCA vs expert-CCA.

Everything here is DETERMINISTIC — assembled by reading the compiler's own pass catalog + pipeline + the
decoded stream. No LLM composes a trace; an LLM may later read ``to_markdown()`` to reason about it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The compilation PHASE each pipeline stage belongs to (the stable top level of the C3 region taxonomy
# in kernels.regions.PHASES). Phase is the clean join between the pipeline description and the region
# registry; the fine per-concern region breakdown lives in kernels.regions (queried separately).
_STAGE_PHASE = {
    "normalize": "global",
    "quant": "global",
    "outline": "dispatch",
    "vectorize": "kernel-codegen",
    "transform_schedule": "kernel-codegen",
    "bufferize": "memory",
    "runtime": "dispatch",
    "edge": "emission",
    "llvm": "emission",
    "contract": "cross-cutting",
    "schedule": "kernel-codegen",
    "interface": "runtime",
    "target": "emission",
}


@dataclass(frozen=True)
class GraphRegion:
    """A region of the flattened exported graph (from the model2MLIR MLIR + ``prov.*`` attributes)."""
    region_id: str
    op: str                              # the FINE op (aten-level: matmul / addmm / softmax / ...)
    family: str | None = None
    module: str | None = None
    shape: dict[str, int] | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    # The REGION-level identity this op belongs to, recognized structurally by
    # ``dse_guidance.attribution.recognize_regions`` (attention / linear / mlp / conv / norm / softmax).
    # Distinct from ``op``: a `matmul` op inside an attention block has op="matmul", recognized_op=
    # "attention". None when the region was not recognized (e.g. a bare MatmulRecord with no capture).
    recognized_op: str | None = None


@dataclass(frozen=True)
class TransformStep:
    """One transformation applied during lowering — a compiler pass or a transform-schedule step."""
    name: str
    plane: str              # "dialect" | "llvm" | "transform_schedule"
    stage: str              # the pipeline stage (normalize/outline/vectorize/bufferize/llvm/...)
    summary: str = ""
    entry: str | None = None        # dotted path of the implementing callable = the edit point
    phase: str | None = None        # the compilation phase (kernels.regions.PHASES) this step belongs to
    modifiable_by: str | None = None  # the seam/feature that can change it


@dataclass(frozen=True)
class AsmRegion:
    """A region of emitted asm (a decoded ``InsnStream`` span) that a CCA facet is lifted from."""
    label: str
    span: tuple[int, int] | None = None   # (lo, hi) loop address span, or None for straight-line
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoweringTrace:
    """graph → [transform steps] → asm for one kernel/region. Serializes to an LLM-digestible view."""
    kernel: str
    target: str
    source: str                                  # "ours" | an expert kernel id
    graph: GraphRegion | None = None
    steps: list[TransformStep] = field(default_factory=list)
    asm: AsmRegion | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kernel": self.kernel, "target": self.target, "source": self.source,
            "graph": asdict(self.graph) if self.graph else None,
            "steps": [asdict(s) for s in self.steps],
            "asm": asdict(self.asm) if self.asm else None,
            "provenance": self.provenance,
        }

    def to_markdown(self) -> str:
        """A compact, LLM-digestible narrative of the lowering (what ran, on what, to what)."""
        out = [f"# LoweringTrace: {self.kernel} ({self.source} @ {self.target})", ""]
        if self.graph:
            g = self.graph
            shp = f" {g.shape}" if g.shape else ""
            out += [f"**Graph region** `{g.region_id}` — op=`{g.op}` family=`{g.family}`{shp}", ""]
        out += ["**Transformation steps** (graph → asm):", ""]
        for i, s in enumerate(self.steps):
            seam = f" — edit: `{s.modifiable_by or s.entry or '?'}`"
            out.append(f"{i + 1}. [{s.plane}/{s.stage}] **{s.name}** ({s.phase or '?'}){seam}")
            if s.summary:
                out.append(f"   - {s.summary}")
        out.append("")
        if self.asm:
            a = self.asm
            out += [f"**Asm region** `{a.label}` span={a.span}",
                    *(f"   - {k}: {v}" for k, v in a.facts.items())]
        return "\n".join(out)


# ---- deterministic assembly of the compiler pipeline as ordered TransformSteps ------

def _split_pass_list(pipeline_str: str) -> list[str]:
    """Split a comma-joined MLIR pass-list into individual pass strings, respecting ``{...}`` option
    groups (which are space-separated inside braces, never comma-separated) — structured, no regex."""
    out, depth, cur = [], 0, []
    for ch in pipeline_str:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            tok = "".join(cur).strip()
            if tok:
                out.append(tok)
            cur = []
        else:
            cur.append(ch)
    tok = "".join(cur).strip()
    if tok:
        out.append(tok)
    return out


def _pass_stage(name: str) -> str:
    """Best-effort phase for a raw upstream pass name (structured prefix check)."""
    n = name.split("{", 1)[0]
    if "bufferize" in n or "buffer-" in n:
        return "bufferize"
    if "vectorize" in n or "transform-interpreter" in n or "vector" in n:
        return "vectorize"
    if "to-llvm" in n or "llvm" in n or "reconcile" in n:
        return "llvm"
    if n in ("canonicalize", "cse"):
        return "normalize"
    return "normalize"


def _transform_schedule_steps(schedule_text: str) -> list[TransformStep]:
    """Extract the transform-dialect schedule's ops as ordered steps (structural line scan for the
    ``transform.structured.*`` / ``transform.apply_patterns.*`` tokens — not semantic regex)."""
    steps: list[TransformStep] = []
    for line in schedule_text.splitlines():
        s = line.strip()
        for prefix in ("transform.structured.", "transform.apply_patterns.vector.",
                       "transform.apply_patterns.linalg."):
            if s.startswith(prefix) or (prefix in s and "= transform.structured." in s):
                op = s.split(prefix, 1)[1].split()[0].split("(")[0] if prefix in s else s
                steps.append(TransformStep(
                    name=op, plane="transform_schedule", stage="transform_schedule",
                    summary=s[:120], phase=_STAGE_PHASE["transform_schedule"]))
                break
    return steps


def _catalog_steps() -> list[TransformStep]:
    """The Merlin-authored dialect passes from ``passes.CATALOG`` as TransformSteps (static metadata —
    no xDSL runtime needed; CATALOG is a plain tuple of PassInfo)."""
    from ..xdsl_dialects.lowering.passes import CATALOG
    return [TransformStep(name=p.name, plane="dialect", stage=p.stage, summary=p.summary,
                          entry=p.entry, phase=_STAGE_PHASE.get(p.stage)) for p in CATALOG]


def _llvm_steps() -> list[TransformStep]:
    """The LLVM-plane mechanical-descent passes from ``pipeline.build_rvv_pipeline`` as TransformSteps.
    Calls build_rvv_pipeline with a placeholder schedule path (it only embeds the path in a
    transform-interpreter pass string) and splits the comma-joined pass list, respecting {options}."""
    from ..llvmlower import pipeline as P
    raw = P.build_rvv_pipeline("<schedule.mlir>")
    steps = []
    for name in _split_pass_list(raw):
        stage = _pass_stage(name)
        steps.append(TransformStep(name=name.split("{", 1)[0], plane="llvm", stage=stage,
                                   summary=name, phase=_STAGE_PHASE.get(stage)))
    return steps


def pipeline_steps(target: str = "rvv") -> list[TransformStep]:
    """The compiler's transformation sequence as ordered TransformSteps: the Merlin-authored dialect
    passes (``passes.CATALOG``) → the transform-dialect schedule (tiling/vectorize/lower) → the
    LLVM-plane mechanical descent. Deterministic and dependency-light: reads the static pass catalog +
    the pipeline/schedule strings; does not run the compiler. ``target`` currently instantiates RVV."""
    if target != "rvv":
        raise ValueError(f"pipeline_steps: only 'rvv' instantiated today, got {target!r}")
    from ..llvmlower import pipeline as P
    return _catalog_steps() + _transform_schedule_steps(P.RVV_TRANSFORM_SCHEDULE) + _llvm_steps()


# ---- linkage: graph region <- prov ; asm region <- decoded InsnStream --------------

def graph_region_from_record(rec, recognized_op: str | None = None) -> GraphRegion:
    """A GraphRegion from a ``frontends.linalg_mlir.MatmulRecord`` (its ``prov.*`` + m/n/k/kind).
    This is the flattened-graph end of the trace, read structurally from the model2MLIR provenance.
    ``recognized_op`` (optional) is the region-level identity from ``recognize_regions`` — the caller
    passes it when a capture is available, so the trace records WHAT the region is, not just the op."""
    prov = dict(rec.prov or {})
    shape = {k: v for k, v in (("M", rec.m), ("N", rec.n), ("K", rec.k)) if v is not None}
    return GraphRegion(
        region_id=prov.get("prov.region_id") or prov.get("prov.module") or rec.kind,
        op=prov.get("prov.op") or rec.kind,
        family=prov.get("prov.family"),
        module=prov.get("prov.module"),
        shape=shape or None,
        provenance=prov,
        recognized_op=recognized_op)


def op_agree(graph_region: GraphRegion, asm_cca) -> "object":
    """Cross-check the graph's FINE op against the op the asm CCA was lifted as, reusing the CCA
    validity gate (``cca.cca_agree``). Disagreement means the asm region was decoded under an op
    inconsistent with the graph — the trace is then quarantined (the op flowed unverified before).
    Returns a ``cca.AgreementReport``."""
    from .cca import CCA, ComputeFacet, cca_agree
    g = CCA(op=graph_region.op, backend=list(getattr(asm_cca, "backend", []) or []),
            compute=ComputeFacet(op=graph_region.op), provenance={"level": "graph"})
    return cca_agree(g, asm_cca)


def asm_region_from_stream(stream, *, op: str, source: str = "asm", label: str = "kernel") -> AsmRegion:
    """An AsmRegion from a decoded ``decode.rvv.InsnStream``: the innermost (K-reduction) loop span +
    the CCA-relevant facts, reusing ``cca.lift_asm`` so the trace's asm end and the CCA agree by
    construction (deterministic — read from the stream, never guessed)."""
    from dataclasses import asdict as _asdict

    from .cca import lift_asm
    c = lift_asm(stream, op=op, source=source)
    facts: dict[str, Any] = {}
    for facet in (c.compute, c.vector):
        if facet is not None:
            facts.update({k: v for k, v in _asdict(facet).items() if v is not None})
    return AsmRegion(label=label, span=stream.innermost_loop(), facts=facts)


def asm_region_from_roles(decoded, endpoint, *, op: str, source: str = "asm", label: str = "kernel",
                          geometry: dict | None = None, loop_spans=None) -> AsmRegion:
    """An AsmRegion from a ROLE-tagged stream — the non-RVV sibling of :func:`asm_region_from_stream`.

    That function reuses ``cca.lift_asm``, which is the RVV lifter: it reads a vtype history and counts
    vector mnemonics. Handing it an accelerator's command stream does not fail loudly, it produces an
    EMPTY vector facet and an asm region whose facts are silently about nothing. So a trace whose asm
    end is an endpoint command stream lifts through the role lifter instead, and the region's facts come
    from whichever facets that endpoint's engine actually licenses.
    """
    from dataclasses import asdict as _asdict

    from .cca import lift_asm_roles
    c = lift_asm_roles(decoded, endpoint, op=op, source=source, geometry=geometry,
                       loop_spans=loop_spans)
    facts: dict[str, Any] = {}
    for facet in (c.compute, c.spatial, c.simt, c.vector, c.dispatch):
        if facet is not None:
            facts.update({k: v for k, v in _asdict(facet).items() if v is not None})
    facts["role_counts"] = c.provenance.get("role_counts", {})
    facts["confidence"] = c.provenance.get("confidence")
    spans = list(loop_spans or ())
    return AsmRegion(label=label, span=(spans[0] if spans else None), facts=facts)


# ---- expert-side trace (matched at the contract + asm level; graph deferred by design) ----

# The declared-transformation sections of a framework contract, in lowering order, mapped to the step
# stage + coarse compiler region. (Experts are matched at the asm/source + contract level we already
# extract; per-framework internal-IR tracing is deferred — see the plan.)
_CONTRACT_STEP_SPECS = (
    ("layout", "operand-layout", "layout", "memory"),
    ("operand_prepack", "operand-prepack", "prepack", "memory"),
    ("accumulator", "accumulator-block", "accumulate", "kernel-codegen"),
    ("calling_convention", "epilogue", "epilogue", "kernel-codegen"),
)
_CONTRACT_SUMMARY_KEYS = ("implication", "packed_layout", "epilogue", "width", "init", "signature")


def _contract_summary(section: Any) -> str:
    """A compact one-line summary of a contract section (prefer the most informative subfields)."""
    if isinstance(section, dict):
        for k in _CONTRACT_SUMMARY_KEYS:
            if section.get(k):
                return f"{k}: {section[k]}"[:200]
        return "; ".join(f"{k}={v}" for k, v in section.items())[:200]
    return str(section)[:200]


def expert_steps_from_contract(framework: str) -> list[TransformStep]:
    """The expert framework's DECLARED transformation sequence as TransformSteps, read deterministically
    from the hand-authored ``framework_contracts/<framework>.yaml`` (reusing ``load_contract``). Honestly
    labeled ``plane="framework"`` — these are the contract's declared caller-side transformations
    (packing/accumulator/epilogue/layout), not a trace of the framework's internal IR."""
    from .framework_contracts import load_contract
    contract = load_contract(framework)
    steps: list[TransformStep] = []
    for key, name, stage, phase in _CONTRACT_STEP_SPECS:
        section = contract.get(key)
        if not section:
            continue
        steps.append(TransformStep(
            name=name, plane="framework", stage=stage, summary=_contract_summary(section),
            entry=f"kernels/framework_contracts/{framework}.yaml:{key}", phase=phase))
    return steps


def build_expert_trace(framework: str, stream, *, op: str, kernel_id: str) -> LoweringTrace:
    """A LoweringTrace for an expert kernel: declared-transformation steps (from the contract) threaded
    to the asm region we decode. ``graph=None`` by design (we don't have the expert's flattened graph;
    experts are matched at the contract + asm level)."""
    return LoweringTrace(
        kernel=kernel_id, target="rvv", source=framework, graph=None,
        steps=expert_steps_from_contract(framework),
        asm=asm_region_from_stream(stream, op=op, source=framework, label=kernel_id),
        provenance={"level": "contract+asm", "framework": framework})


#: The stages an expert BUILD passes through, in order. These are the real steps of somebody else's
#: toolchain, not ours, which is exactly why they are worth recording: a divergence attributed to "their
#: compiler is better" is unactionable until you can see WHICH step produced the difference.
_BUILD_STAGES = (
    ("preprocess", "frontend", "macro expansion + header inclusion (the .insn / intrinsic surface)"),
    ("frontend", "frontend", "C/C++ -> LLVM IR"),
    ("llvm-pipeline", "kernel-codegen", "the -O pipeline: inlining, vectorization, unrolling"),
    ("asm-emission", "emission", "instruction selection + register allocation -> the emitted stream"),
)

#: Build-command tokens that carry a decision worth recording, and what each names. A flag not listed
#: here is still recorded verbatim under ``flags``; this only decides which ones get NAMED as decisions.
_FLAG_MEANING = {
    "-O": "optimization level",
    "-march": "the ISA the code may use",
    "-mabi": "the calling convention",
    "-mcpu": "the scheduling model",
    "-ffast-math": "float reassociation permitted",
    "-ffp-contract": "fma contraction policy",
    "-funroll-loops": "unrolling forced on",
    "-fno-vectorize": "vectorization forced off",
}


def _flag_decisions(argv) -> list[tuple[str, str]]:
    """(flag, what it decides) for the build tokens that carry a decision. Structural, no regex."""
    out: list[tuple[str, str]] = []
    for token in argv or ():
        tok = str(token)
        if not tok.startswith("-"):
            continue
        head = tok.split("=", 1)[0]
        for known, meaning in _FLAG_MEANING.items():
            if head == known or (known == "-O" and head.startswith("-O") and len(head) <= 3):
                out.append((tok, meaning))
                break
    return out


def expert_build_steps(build_cmd, *, tool: str = "", version: str = "") -> list[TransformStep]:
    """The expert's BUILD as ordered TransformSteps — the steps of a compiler we do not control.

    ``modifiable_by`` is None on every one of them, and that is the point rather than an omission. Our
    own trace records, per step, the seam that can change it; theirs has no such seam, and saying so
    explicitly is what stops a reader from treating an expert step as an action we could take. A
    divergence whose cause sits in their `-O` pipeline is a fact to design against, not a task.
    """
    argv = list(build_cmd or ())
    decisions = _flag_decisions(argv)
    flags = " ".join(str(a) for a in argv if str(a).startswith("-"))
    who = " ".join(t for t in (tool, version) if t)
    entry = " :: ".join(part for part in (who, flags) if part) or None
    steps: list[TransformStep] = []
    for name, phase, what in _BUILD_STAGES:
        detail = what
        if name == "llvm-pipeline" and decisions:
            detail = f"{what}; decided by " + ", ".join(f"{f} ({m})" for f, m in decisions)
        steps.append(TransformStep(
            name=name, plane="expert-build", stage=name, summary=detail,
            entry=entry, phase=phase,
            modifiable_by=None,           # deliberate: we cannot edit their compiler
        ))
    return steps


def expert_trace(source: str, stream, *, op: str, kernel_id: str, target: str,
                 build_cmd=None, tool: str = "", version: str = "",
                 obj: "str | Path | None" = None,
                 hand_written: bool = False, endpoint=None,
                 geometry: dict | None = None, loop_spans=None) -> LoweringTrace:
    """A LoweringTrace for an expert kernel, from its BUILD rather than from a declared contract.

    ``build_expert_trace`` reads the framework contract, i.e. what the expert framework SAYS it does.
    This reads what its toolchain actually ran, so both sides of a comparison end in an asm region
    reached by recorded steps and a divergence can be attributed to a step rather than to a vendor.

    ``hand_written`` records that the expert side has NO lowering to reconstruct: a corpus of
    hand-written assembly was not produced by a compiler, so an absent trace there is a PROPERTY of the
    endpoint and not a missing feature. Stamping it is what keeps someone from later "fixing" it.

    Every step is stamped ``modifiable_by=None``; see :func:`expert_build_steps`.
    """
    steps: list[TransformStep] = []
    prov: dict[str, Any] = {"level": "build+asm", "source": source, "target": target}
    if hand_written:
        prov["no_lowering"] = ("hand-written assembly: there is no compiler lowering to reconstruct on "
                               "the expert side, which is a property of this endpoint rather than a gap")
    else:
        steps = expert_build_steps(build_cmd, tool=tool, version=version)
        prov["build_cmd"] = [str(a) for a in (build_cmd or ())]
        prov["tool"], prov["tool_version"] = tool, version
    if obj is not None:
        prov["object"] = str(obj)
        try:
            from merlin.common import provenance as _p
            prov["source_digest"] = _p.source_digest([obj])
        except Exception:  # noqa: BLE001 — a digest we cannot take is recorded absent, never faked
            prov["source_digest"] = None
    # `endpoint` selects the lifter: an accelerator command stream through the role lifter, an RVV
    # InsnStream through the vector one. Passing an accelerator stream to the RVV lifter yields an empty
    # facet rather than an error, so the choice has to be explicit.
    region = (asm_region_from_roles(stream, endpoint, op=op, source=source, label=kernel_id,
                                    geometry=geometry, loop_spans=loop_spans)
              if endpoint is not None
              else asm_region_from_stream(stream, op=op, source=source, label=kernel_id))
    prov["lifter"] = "roles" if endpoint is not None else "rvv"
    return LoweringTrace(
        kernel=kernel_id, target=target, source=source, graph=None, steps=steps,
        asm=region, provenance=prov)


def traces_agree(ours: LoweringTrace, theirs: LoweringTrace) -> dict:
    """Do the two traces describe the same REGION, and where does each side's authority end?

    Not a comparison of the kernels -- ``cca_compare`` does that. This answers the prior question of
    whether the two traces are about the same thing and how far each is editable, so a divergence is
    never attributed to a step nobody can reach.
    """
    ours_editable = [s.name for s in ours.steps if s.modifiable_by]
    theirs_editable = [s.name for s in theirs.steps if s.modifiable_by]
    return {
        "same_op": (ours.asm or AsmRegion("")).label == (theirs.asm or AsmRegion("")).label,
        "our_editable_steps": ours_editable,
        # Expected to be EMPTY. A non-empty list means something claimed we can edit the expert's
        # compiler, which is false and would route a divergence to a seam that does not exist.
        "their_editable_steps": theirs_editable,
        "their_trace_absent": not theirs.steps,
        "notes": ([theirs.provenance["no_lowering"]] if theirs.provenance.get("no_lowering") else []),
    }


def build_our_trace(stream, *, op: str | None = None, record=None, recognized_op: str | None = None,
                    target: str = "rvv") -> LoweringTrace:
    """A LoweringTrace for OUR compiler's output: the flattened-graph region (from a MatmulRecord, if
    given) → our pipeline transformation steps → the emitted asm region. The full graph→steps→asm thread.

    Op identity is threaded from the graph as the single source of truth: when a ``record`` is given the
    asm is decoded under the graph's op (``op`` still overrides for the record-less path), and
    ``op_agree`` cross-checks the two — a mismatch is recorded and quarantines the trace. ``recognized_op``
    (the region-level identity from ``recognize_regions``) is carried as context on the GraphRegion."""
    graph = graph_region_from_record(record, recognized_op=recognized_op) if record is not None else None
    # Single source of truth for the fine op: prefer the graph's, fall back to the explicit arg.
    asm_op = op or (graph.op if graph is not None else None) or "unknown"
    asm = asm_region_from_stream(stream, op=asm_op, source="ours", label=asm_op)
    prov: dict[str, Any] = {"level": "graph+pipeline+asm"}
    if graph is not None:
        from .cca import lift_asm
        report = op_agree(graph, lift_asm(stream, op=asm_op, source="ours"))
        prov["op_agreement"] = {"agree": report.agree, "disagreements": report.disagreements}
        prov["quarantined"] = not report.agree
        if graph.recognized_op:
            prov["recognized_op"] = graph.recognized_op
    return LoweringTrace(
        kernel=asm_op, target=target, source="ours", graph=graph,
        steps=pipeline_steps(target), asm=asm, provenance=prov)


def emit_trace(trace: LoweringTrace, *, version: int = 1):
    """Write a LoweringTrace as a versioned product under ``out/artifacts/lowering-trace/<target>/v<ver>/``
    (yaml + LLM-digestible markdown + manifest). Returns the product dir. Convention-compliant provenance
    (TS/version/git_sha) via ``new_product``; never writes outside ``out/``."""
    import yaml

    from ..common.artifacts import new_product
    p = new_product("lowering-trace", version=version, target=trace.target,
                    notes=f"{trace.source}:{trace.kernel}")
    stem = f"trace_{trace.source}_{trace.kernel}".replace("/", "_")
    p.add_artifact(f"{stem}.yaml").write_text(yaml.safe_dump(trace.to_dict(), sort_keys=False),
                                              encoding="utf-8")
    p.add_artifact(f"{stem}.md").write_text(trace.to_markdown(), encoding="utf-8")
    p.write_manifest()
    return p.path
