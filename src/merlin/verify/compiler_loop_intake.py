"""Actual compiler loop boundaries over the existing scalar/SCF evaluator.

This path has no pinned historical producer requirement. Actual buffers are
independently projected, and Lean checks the local body/control/carry joins.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .loop_transform_intake import ROOT, Settings, Unavailable, Unsupported, canonical, digest, file_ref

UNROLL = "ordered_loop_unrolling"
TILE = "ordered_loop_segmentation"
DOMAIN = "compiler-static-ordered-scf-index64-v1"


@dataclass(frozen=True)
class PreparedCompilerLoop:
    buffers: tuple[str, str, str]
    settings: Settings
    kinds: tuple[str, ...]
    capture_json: str
    query: str
    roots: tuple[str, ...]
    sources_json: str
    domain: str = DOMAIN

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    @property
    def actual_target(self):
        return self.buffers[1] if self.kinds == (UNROLL,) else self.buffers[-1]

    def describe(self):
        return {
            "schema": "merlin.compiler-loop.prepared.v1",
            "identity": self.identity,
            "domain": self.domain,
            "buffers": [digest(x) for x in self.buffers],
            "actual_target_sha256": digest(self.actual_target),
            "settings": asdict(self.settings),
            "kinds": self.kinds,
            "query_sha256": digest(self.query),
            "roots": self.roots,
            "authority": False,
        }


def prepare(buffers, *, kinds, factor=2, tile=3, index_bits=64):
    from merlin.transforms.certificates import parse_ir
    from merlin.transforms.loops import segment_loops

    from .lean_graph_input_admission import parse_admitted
    from .lean_graph_pipeline_gate_profile_prepare import _adjacent_rosters, _admit, _domain
    from .lean_graph_pipeline_gate_projection import project_lean_sources
    from .lean_graph_state_pipeline_capture import project_modules
    from .lean_graph_state_pipeline_emit import emit, static_controls

    if kinds not in ((UNROLL,), (TILE,), (UNROLL, TILE)):
        raise Unsupported("explicit ordered compiler loop boundary roster required")
    if type(index_bits) is not int or index_bits != 64:
        raise Unsupported("compiler loop proof requires index64")
    if type(factor) is not int or not 2 <= factor <= 4 or type(tile) is not int or not 2 <= tile <= 16:
        raise Unsupported("compiler loop proof requires factor2..4 and tile2..16")
    expected = 3 if len(kinds) == 2 else 2
    if (
        type(buffers) is not tuple
        or len(buffers) != expected
        or any(type(x) is not str or len(x.encode()) > 4 * 1024 * 1024 for x in buffers)
    ):
        raise Unsupported("complete bounded actual compiler buffers required")
    if any(a == b for a, b in zip(buffers, buffers[1:])):
        raise Unsupported("unchanged compiler boundaries require no proof job")
    try:
        modules = [parse_admitted(x, parse_ir, calls="external-only") for x in buffers]
        _admit(modules)
        if any(sum(1 for _ in m.walk()) > 2048 for m in modules):
            raise ValueError("bounded complete scalar/SCF module required")
        if kinds == (UNROLL,):
            # Only the unroll adjacent theorem authorizes this boundary. This
            # auxiliary segmentation supplies the existing compositional proof
            # scaffold; its result is never delivered as the compiler output.
            auxiliary = modules[-1].clone()
            segment_loops(auxiliary, (tile,), index_bits=64)
            from merlin.transforms.certificates import ir_text

            buffers = (*buffers, ir_text(auxiliary))
            modules.append(auxiliary)
        elif kinds == (TILE,):
            buffers = (buffers[0], buffers[0], buffers[1])
            modules = [modules[0], modules[0], modules[1]]
            factor = 1
        stages, controls = project_modules([modules[0], modules[0], *modules])
        capture = {
            "stages": stages,
            "index_operations": controls,
            "factor": factor,
            "tile": tile,
            "index_bits": index_bits,
        }
        if kinds == (TILE,):
            changed = []
            for i, function in enumerate(stages[0]["functions"]):
                if function["declaration"]:
                    continue
                records = function["before"]
                loops = [row["loop"] for row in records if "loop" in row]
                if not loops:
                    continue
                if len(loops) != 1 or len(function["inputs"]) > 16 or not 1 <= len(loops[0]["carries"]) <= 16:
                    raise ValueError("one loop/at most sixteen inputs/one to sixteen word carries required")
                if any(set(t) != {"bits"} for t in loops[0]["carries"]) or any("loop" in r for r in loops[0]["body"]):
                    raise ValueError("region-free scalar word body required")
                lower, upper, step = static_controls(records, capture, input_count=len(function["inputs"]))
                if not 0 <= lower < upper or step <= 0 or upper + step * tile >= 2**63:
                    raise ValueError("nonempty static fitting positive-stride controls required")
                changed.append(i)
            if not changed:
                raise ValueError("at least one changed top-level loop required")
        else:
            changed = _domain({"pipeline": capture}, flexible_entries=True)
            if not changed:
                raise ValueError("at least one changed top-level loop required")
        query = emit(capture, identity_unroll=kinds == (TILE,))
        adjacent, _, _ = _adjacent_rosters({"pipeline": capture}, "Merlin.GraphStatePipelineInstance.", changed)
        query += adjacent
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise Unsupported(str(exc)) from exc
    roots, namespaces = [], []
    for line in query.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "namespace":
            namespaces.append(words[1])
        elif words[0] == "end":
            namespaces.pop()
        elif words[0] == "theorem":
            roots.append(".".join([*namespaces, words[1]]))
    if ROOT not in roots or len(query.encode()) > 4 * 1024 * 1024:
        raise Unsupported("complete bounded operational proof required")
    imports = [w for line in query.splitlines() if line.startswith("import ") for w in line.split()[1:]]
    sources = [asdict(row) for row in project_lean_sources(imports)]
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / "loop_transform_gate.py",
        parent / "loop_transform_intake.py",
        parent / "compiler_transforms.py",
        parent / "gate_checks.py",
        parent / "gate_worker.py",
        parent / "pass_validation.py",
        parent / "lean/Merlin/GraphStateAudit.lean",
        parent / "lean_graph_input_admission.py",
        parent / "lean_graph_pipeline_gate_projection.py",
        parent / "lean_graph_pipeline_gate_profile_prepare.py",
    }
    paths.update(parent.glob("lean_graph_state*.py"))
    paths.update((parent.parent / "transforms").glob("*.py"))
    paths.update((parent.parent / "common").glob("owned_*.py"))
    sources.extend(file_ref(p) for p in sorted(paths))
    return PreparedCompilerLoop(
        buffers, Settings(factor=factor, tile=tile), kinds, canonical(capture), query, tuple(roots), canonical(sources)
    )


def refresh(prepared):
    if type(prepared) is not PreparedCompilerLoop or prepared.domain != DOMAIN:
        raise Unavailable("exact prepared compiler loop required")
    buffers = prepared.buffers
    if prepared.kinds == (UNROLL,):
        buffers = buffers[:2]
    elif prepared.kinds == (TILE,):
        buffers = (buffers[0], buffers[2])
    return prepare(buffers, kinds=prepared.kinds, factor=max(2, prepared.settings.factor), tile=prepared.settings.tile)


def revalidate(prepared):
    if type(prepared) is not PreparedCompilerLoop or prepared.domain != DOMAIN:
        raise Unavailable("exact prepared compiler loop required")
    for ref in json.loads(prepared.sources_json):
        actual = file_ref(ref["path"])
        if actual["sha256"] != ref["sha256"] or actual["bytes"] != ref.get("bytes", ref.get("size")):
            raise Unavailable("bound compiler loop proof/host source changed")
