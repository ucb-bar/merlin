"""SDK-free complete-buffer reconstruction for frozen scalar/SCF proof emitters.

Old capture/intake/receipt/proof/kernel functions are NEVER called. The frozen
emitters construct proof terms over the unchanged evaluator. New query outcomes
are UNOBSERVED until a separately reviewed native context checks them.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .lean_graph_pipeline_gate_profiles import INTEGER, Settings, selected
from .lean_graph_pipeline_gate_projection import canonical, project_lean_sources, sha, source_ref


@dataclass(frozen=True)
class PreparedProfile:
    schema: str
    profile: str
    settings: Settings
    kinds: tuple[str, ...]
    domain: str
    relation: str
    premises: tuple[str, ...]
    python_version: str
    buffers: tuple[str, ...]
    attribution: str
    capture: str
    query: str
    roots: tuple[str, ...]
    boundary_roots: tuple[tuple[str, tuple[str, ...]], ...]
    sources: tuple

    @property
    def identity(self):
        return sha(canonical(asdict(self)))

    def describe(self):
        return {
            **asdict(self),
            "identity": self.identity,
            "query_status": "UNCOMPILED",
            "outcome": "UNOBSERVED",
            "qualification": "UNQUALIFIED",
            "compiler_pass_chain": "not_established",
            "public_route": "UNSUPPORTED/HELD",
        }


def _sources(query):
    # Project SOURCES only, never installed objects, runtimes or old guard pins.
    imports = tuple(word for line in query.splitlines() if line.startswith("import ") for word in line.split()[1:])
    if not imports or any(not name.startswith("Merlin.") for name in imports):
        raise ValueError("closed project-only query import roster required")
    paths = {
        Path(__file__),
        Path(__file__).with_name("lean_graph_pipeline_gate_profiles.py"),
        Path(__file__).with_name("lean_graph_pipeline_gate_profile_requests.py"),
    }
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith(("merlin.", "xdsl.")) and path and Path(path).suffix == ".py":
            paths.add(Path(path))
    return tuple(source_ref(path) for path in sorted(paths)) + project_lean_sources(imports)


def _admit(modules):
    from xdsl.dialects import arith, builtin

    from .lean_graph_input_admission import admit_module

    for module in modules:
        admit_module(module, calls="external-only")
        operations = tuple(module.walk())
        if len(operations) > 2048:
            raise ValueError("bounded complete scalar/SCF syntax required")
        for op in operations:
            values = (*op.operands, *op.results, *(a for r in op.regions for b in r.blocks for a in b.args))
            for value in values:
                t = value.type
                if not (
                    type(t) is builtin.IndexType
                    or type(t) is builtin.IntegerType
                    and t.signedness.data == builtin.Signedness.SIGNLESS
                    and 1 <= t.width.data <= 256
                ):
                    raise ValueError("larger adapter supports only signless scalar words/index values")
            if type(op) in (
                arith.AddiOp,
                arith.SubiOp,
                arith.MuliOp,
            ) and op.overflow_flags != arith.IntegerOverflowAttr("none"):
                raise ValueError("overflow-flagged arithmetic outside explicit larger profile")


def _recheck(buffers, modules, profile, settings):
    from merlin.transforms.certificates import TRANSFORM_SCOPES, ir_text
    from merlin.transforms.loops import segment_loops
    from merlin.transforms.normalization import simplify_integers
    from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants
    from merlin.transforms.unrolling import unroll_ordered_loops

    runs = {
        "exact_integer_simplification": simplify_integers,
        "scalar_redundancy_elimination": eliminate_scalar_redundancy,
        "scalar_loop_invariant_motion": hoist_scalar_invariants,
        "ordered_loop_unrolling": lambda m: unroll_ordered_loops(m, (settings.factor,), index_bits=settings.index_bits),
        "ordered_loop_segmentation": lambda m: segment_loops(m, (settings.tile,), index_bits=settings.index_bits),
    }
    rows = []
    for i, kind in enumerate(profile.kinds):
        clone = modules[i].clone()
        witnesses = runs[kind](clone)
        clone.verify()
        if not witnesses or buffers[i] == buffers[i + 1] or ir_text(clone) != buffers[i + 1]:
            raise ValueError("actual changed adjacent output differs from registered " + kind)
        rows.append(
            {
                "kind": kind,
                "scope": TRANSFORM_SCOPES[kind],
                "before": sha(buffers[i]),
                "after": sha(buffers[i + 1]),
                "rechecked_witnesses_diagnostic_only": [asdict(w) for w in witnesses],
                "semantic_proof": False,
            }
        )
    return rows


def _projection(modules, settings, integer_prefix):
    from .lean_graph_state_pipeline_capture import index_operation
    from .lean_graph_state_scalar_capture import metadata, project

    stages = [project(a, b, context=modules) for a, b in zip(modules, modules[1:])]
    first = stages[0]
    for p in stages:
        if any(p[key] != first[key] for key in ("registry", "types", "module_header")):
            raise ValueError("all complete stages must share one descriptor/type/module universe")
        for a, b in zip(first["functions"], p["functions"], strict=True):
            if any(a[k] != b[k] for k in ("name", "header", "inputs", "outputs", "declaration")):
                raise ValueError("complete roster/interface changed between stages")
    for a, b in zip(stages, stages[1:]):
        if any(x["after"] != y["before"] for x, y in zip(a["functions"], b["functions"], strict=True)):
            raise ValueError("exact projected intermediate program join differs")
    controls = {}
    for module in modules:
        for op in module.walk():
            control = index_operation(op)
            if control is not None:
                key = canonical(metadata(op))
                if key in controls and controls[key] != control:
                    raise ValueError("ambiguous index control descriptor")
                controls[key] = control
    concrete = [
        {"identity": i, **controls[canonical(row["descriptor"])]}
        for i, row in enumerate(first["registry"])
        if canonical(row["descriptor"]) in controls
    ]
    suffix = {**asdict(settings), "stages": stages[1:] if integer_prefix else stages, "index_operations": concrete}
    return {"prefix": stages[0] if integer_prefix else None, "pipeline": suffix}


def _domain(capture, *, flexible_entries=False):
    from .lean_graph_state_pipeline_emit import static_controls

    suffix = capture["pipeline"]
    factor, tile, bits = (suffix[k] for k in ("factor", "tile", "index_bits"))
    functions = suffix["stages"][0]["functions"]
    changed = []
    for index, function in enumerate(functions):
        if function["declaration"]:
            continue
        records = [function["before"], *[p["functions"][index]["after"] for p in suffix["stages"]]]
        loops = [[r["loop"] for r in rows if "loop" in r] for rows in records]
        if not any(loops):
            continue
        if any(len(x) != 1 for x in loops):
            raise ValueError("one top-level static changed loop required")
        if not flexible_entries and len(function["inputs"]) != 2:
            raise ValueError("two entry arguments required")
        if records[2] == records[4]:
            raise ValueError("unchanged loop entry lacks the selected explicit bridge")
        if not flexible_entries and any("loop" in row for row in loops[0][0]["body"]):
            raise ValueError("region-free original loop body required")
        carries = loops[0][0]["carries"]
        if (not flexible_entries and not 1 <= len(carries) <= 4) or any(set(t) != {"bits"} for t in carries):
            raise ValueError("ordered word carry count outside selected scalar/SCF profile")
        lower, upper, step = static_controls(records[2], suffix, input_count=len(function["inputs"]))
        if not 0 <= lower <= upper or step <= 0:
            raise ValueError("static nonnegative ordered control and positive stride required")
        count = (upper - lower + step - 1) // step
        if count <= factor or count % factor:
            raise ValueError("empty/inline/tail or nondivisible bulk outside supported submode")
        bulk_upper, bulk_step = lower + count * step, factor * step
        if max(upper + step * factor, bulk_upper + bulk_step * tile, bulk_upper + bulk_step) >= 2 ** (bits - 1):
            raise ValueError("derived unroll/segment signed bounds exceed selected profile")
        if any(
            row["kind"] == "index-constant" and not -(2 ** (bits - 1)) <= row["value"] < 2 ** (bits - 1)
            for row in suffix["index_operations"]
        ):
            raise ValueError("actual control literal outside signed profile")
        changed.append(index)
    if not changed:
        raise ValueError("profile requires actual changed scalar/SCF functions")
    return tuple(changed)


def _adjacent_rosters(capture, pbase, changed):
    """Proof glue over the SAME frozen dispatcher; no new evaluator/model."""
    from .lean_graph_state_pipeline_emit import entry_arguments

    functions = capture["pipeline"]["stages"][0]["functions"]
    defined = [i for i, f in enumerate(functions) if not f["declaration"]]
    text = "namespace Merlin.GraphGateProfileAdjacent\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    text += f"abbrev registry := {pbase}registry\n"
    for stage in range(5):
        entries = [
            f"{{inputs := {pbase}F{i}.inputs, outputs := {pbase}F{i}.outputs, body := {pbase}F{i}.S{stage}}}"
            for i in defined
        ]
        text += f"def entries{stage} : List Entry := [" + ",".join(entries) + "]\n"
    roots = []
    for i in defined:
        base = pbase + f"F{i}."
        for stage, lemma in ((2, "unroll_actual"), (3, "segment_actual")):
            name = f"step{stage}F{i}"
            text += f"theorem {name} {{World Error : Type}} (primitive : Prim → List Int → Int)\n"
            text += "    (external : External World Error registry) (arguments : List Int) (world : World)\n"
            text += f"    (arity : arguments.length = {base}inputs.length) :\n"
            text += f"    evalProgram ({pbase}M primitive external) {base}S{stage} arguments world =\n"
            text += f"    evalProgram ({pbase}M primitive external) {base}S{stage + 1} arguments world := by\n"
            if i not in changed:
                stages = capture["pipeline"]["stages"]
                if stages[stage]["functions"][i]["before"] != stages[stage]["functions"][i]["after"]:
                    raise ValueError("unchanged-loop-mode adjacent function body differs")
                # Arity is intentionally consumed; no new unused-binder warning.
                text += "  revert arity\n  intro _\n  rfl\n"
            else:
                text += entry_arguments(len(functions[i]["inputs"]), base + lemma, base + "inputs")
            roots.append("Merlin.GraphGateProfileAdjacent." + name)
    complete = []
    for stage in range(4):
        proof = f"entries_nil ({pbase}M primitive external)"
        for i in reversed(defined):
            if stage < 2:
                accepted = "firstAccepted" if stage == 0 else "secondAccepted"
                body = f"(check_sound {pbase}F{i}.{accepted} Int World Error ({pbase}M primitive external))"
            else:
                body = f"(step{stage}F{i} primitive external)"
            proof = f"entries_cons ({pbase}M primitive external) rfl rfl {body} ({proof})"
        text += f"theorem complete{stage} {{World Error : Type}} (primitive : Prim → List Int → Int)\n"
        text += "    (external : External World Error registry) :\n"
        text += f"    PreservesEntries ({pbase}M primitive external) entries{stage} entries{stage + 1} :=\n  {proof}\n"
        complete.append("Merlin.GraphGateProfileAdjacent.complete" + str(stage))
    text += f"theorem original_join : entries0 = {pbase}originalEntries := rfl\n"
    text += f"theorem final_join : entries4 = {pbase}finalEntries := rfl\n"
    text += "end Merlin.GraphGateProfileAdjacent\n"
    return (
        text,
        tuple(
            roots
            + complete
            + ["Merlin.GraphGateProfileAdjacent.original_join", "Merlin.GraphGateProfileAdjacent.final_join"]
        ),
        complete,
    )


def _query(capture, profile, changed):
    from .lean_graph_state_integer_emit import emit as integer_emit
    from .lean_graph_state_pipeline_emit import emit as pipeline_emit

    integer_prefix = profile == INTEGER
    query = (
        integer_emit(capture["prefix"], pipeline_capture=capture["pipeline"])
        if integer_prefix
        else pipeline_emit(capture["pipeline"])
    )
    pbase = "Merlin.GraphStateIntegerInstance.Suffix." if integer_prefix else "Merlin.GraphStatePipelineInstance."
    roots = [
        pbase + n
        for n in (
            "complete",
            "controlAccepted",
            "literalsAccepted",
            "literalsFit",
            "originalEntriesChecked",
            "finalEntriesChecked",
        )
    ]
    adjacent = [[], [], [], []]
    for i, function in enumerate(capture["pipeline"]["stages"][0]["functions"]):
        if function["declaration"]:
            continue
        base = pbase + f"F{i}."
        roots += [
            base + n
            for n in ("actual", "firstAccepted", "secondAccepted", "firstTwo", *(f"typed{k}" for k in range(5)))
        ]
        adjacent[0].append(base + "firstAccepted")
        adjacent[1].append(base + "secondAccepted")
        if i in changed:
            roots += [
                base + n
                for n in (
                    "body_group",
                    "innerAccepted",
                    "inner_same",
                    "outer_same",
                    "suffixAccepted23",
                    "suffixAccepted34",
                    "suffix_same23",
                    "suffix_same34",
                    "unrollBoundsChecked",
                    "segmentBoundsChecked",
                    "form2",
                    "form3",
                    "form4",
                    "unroll_actual",
                    "segment_actual",
                )
            ]
            adjacent[2].append(base + "unroll_actual")
            adjacent[3].append(base + "segment_actual")
    adjacent_text, adjacent_roots, complete_roots = _adjacent_rosters(capture, pbase, changed)
    query += adjacent_text
    roots += list(adjacent_roots)
    adjacent = [(root, *local) for root, local in zip(complete_roots, adjacent, strict=True)]
    if integer_prefix:
        base = "Merlin.GraphStateIntegerInstance."
        roots += [base + n for n in ("originalRegistry", "prefix_actual", "exactMiddle", "complete")]
        prefix_roots = []
        for i, f in enumerate(capture["prefix"]["functions"]):
            if not f["declaration"]:
                names = [
                    base + f"F{i}." + n
                    for n in (
                        "actual",
                        "reached",
                        "localRewrites",
                        "beforeTyped",
                        "afterTyped",
                        "firstNormalized",
                        "secondNormalized",
                    )
                ]
                roots += names
                prefix_roots.append(names[0])
        adjacent.insert(0, [base + "prefix_actual", base + "exactMiddle", *prefix_roots])
        query += """theorem merlin_profile_transform {World Error : Type}
    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStateIntegerInstance.registry)
    (typed : ∀ identity arguments world values final,
      Merlin.GraphState.Integer.Values Merlin.GraphState.Integer.Canonical
        (Merlin.GraphState.Scalar.signature Merlin.GraphStateIntegerInstance.registry identity).inputs arguments →
      external.run identity arguments world = .ok (values,final) →
      Merlin.GraphState.Integer.Values Merlin.GraphState.Integer.Canonical
        (Merlin.GraphState.Scalar.signature Merlin.GraphStateIntegerInstance.registry identity).outputs values) :
    Merlin.GraphState.Integer.PreservesEntriesCanonical
      (Merlin.GraphStateIntegerInstance.Suffix.M Merlin.GraphState.Integer.primitive external)
      Merlin.GraphStateIntegerInstance.originalEntries Merlin.GraphStateIntegerInstance.Suffix.finalEntries :=
  Merlin.GraphStateIntegerInstance.complete external typed
"""
    else:
        query += """theorem merlin_profile_transform {World Error : Type}
    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)
    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineInstance.registry) :
    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInstance.M primitive external)
      Merlin.GraphStatePipelineInstance.originalEntries Merlin.GraphStatePipelineInstance.finalEntries :=
  Merlin.GraphStatePipelineInstance.complete primitive external
"""
    roots += [
        "Merlin.GraphState.Scalar.check_sound",
        "Merlin.GraphState.Pipeline.entries_cons",
        "Merlin.GraphState.Pipeline.unroll_loop",
        "Merlin.GraphState.Pipeline.segment_loop",
        "Merlin.GraphState.Pipeline.Scope.check_sound",
        "Merlin.GraphState.Pipeline.checkControl_sound",
        "Merlin.GraphState.Pipeline.checkLiterals_sound",
    ]
    if integer_prefix:
        roots += [
            "Merlin.GraphState.Integer." + name
            for name in (
                "programs_sound",
                "Relation.sound",
                "typed_result",
                "typed_reached",
                "compose_suffix",
                "index_model_typing",
                "canonical_iff",
            )
        ]
    roots.append(profile.final_root)
    return query, tuple(roots), tuple((k, tuple(v)) for k, v in zip(profile.kinds, adjacent, strict=True))


def prepare(profile_name, buffers, settings):
    from merlin.transforms.certificates import parse_ir

    from .lean_graph_input_admission import admit_text

    profile = selected(profile_name)
    if type(settings) is not Settings:
        raise ValueError("typed explicit settings required")
    settings.validate()
    if (
        type(buffers) is not tuple
        or len(buffers) != len(profile.kinds) + 1
        or any(type(text) is not str or len(text.encode()) > 4 * 1024 * 1024 for text in buffers)
    ):
        raise ValueError("exact bounded complete five/six buffer tuple required")
    for text in buffers:
        admit_text(text)
    modules = tuple(parse_ir(text) for text in buffers)
    _admit(modules)
    capture = _projection(modules, settings, profile == INTEGER)
    changed = _domain(capture)
    attribution = _recheck(buffers, modules, profile, settings)
    query, roots, boundary_roots = _query(capture, profile, changed)
    if len(query.encode()) > 4 * 1024 * 1024 or len(roots) != len(set(roots)) or any(not r for _, r in boundary_roots):
        raise ValueError("bounded query with distinct complete endpoint/adjacent roots required")
    return PreparedProfile(
        "merlin.graph_larger_profile_prepared.v1",
        profile.name,
        settings,
        profile.kinds,
        profile.domain,
        profile.relation,
        profile.premises,
        sys.version,
        tuple(sha(text) for text in buffers),
        canonical(attribution),
        canonical(capture),
        query,
        roots,
        boundary_roots,
        _sources(query),
    )
