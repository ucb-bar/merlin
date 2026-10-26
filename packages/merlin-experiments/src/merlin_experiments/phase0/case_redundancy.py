"""Derived Phase 0 stage: collapse duplicate test cases and give each performance shape one scorer.

Independent derivations reach the same program from different directions: the isolated device group,
the exact application signature and the reduced model form of one captured layer are three cases of
one contraction, and two hand-selected suites repeated each other's shapes. The run this was written
for held 44 groups of identical shapes with 75 extra copies. Every copy is graded and, before this
stage, every copy also counted in the performance headline, so the headline weighted a shape by how
many derivations happened to reach it rather than by the work it stands for.

Two relations are derived from the written capsules, never from a name:

* an EXACT DUPLICATE builds the same program (:func:`.hidden_disjointness.capsule_point`: operation,
  operand roles/shapes/dtypes, attributes, modes, numeric policy) from the same stimulus range and the
  same comparison rule. Exactly one is kept -- the one with the richest declared semantics -- and the
  others are collapsed into it;
* a SHAPE GROUP holds cases whose contraction work is identical (same operation family and
  ``(M, K, N)`` extents of every contraction, a convolution counted by its im2col extents) but whose
  semantics differ (epilogue, readout dtype, layout, stimulus). All stay for correctness; one is the
  group's performance scorer and the others are declared ``performance_scope: excluded``.

A case that states the share of a program's work it represents (a ``source-shape-binding.json`` or a
``performance_demand`` block) but is not a member of that program's derived performance demand is
excluded from scoring as well: it is too small a fraction of its program to stand for it.

Family-contract members (a declared ``performance`` comparison or cohort) are never collapsed; they
can only be excluded from the headline, so a paired or fitted family keeps every point it needs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase0.case_redundancy.v1"
SCORED, EXCLUDED = "scored", "excluded"
SCOPES = (SCORED, EXCLUDED)


def _shape(capsule: Mapping[str, Any], name: str) -> list[int] | None:
    for row in capsule.get("inputs") or ():
        if isinstance(row, Mapping) and row.get("name") == name:
            return [int(v) for v in row.get("shape") or ()]
    return None


def _conv_extents(capsule: Mapping[str, Any]) -> list[int] | None:
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    ifm = _shape(capsule, str(attrs.get("ifm")))
    weight = _shape(capsule, str(attrs.get("weight")))
    if not ifm or len(ifm) != 4 or not weight:
        return None
    kernel = attrs.get("kernel")
    kh, kw = (kernel[0], kernel[1]) if kernel else (attrs.get("kh"), attrs.get("kw"))
    cin = kernel[2] if kernel else attrs.get("ci", ifm[3])
    cout = kernel[3] if kernel else attrs.get("co", weight[-1])
    stride = list(attrs.get("stride") or (1, 1))
    pad = list(attrs.get("padding") or (0, 0, 0, 0))
    dilation = list(attrs.get("dilation") or (1, 1))
    if any(type(v) is not int for v in (kh, kw, cin, cout, *stride, *pad, *dilation)):
        return None
    ho = (ifm[1] + pad[0] + pad[1] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wo = (ifm[2] + pad[2] + pad[3] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    return [ifm[0] * ho * wo, kh * kw * cin, cout]


def work_shape(capsule: Mapping[str, Any]) -> str:
    """The contraction work a case performs, independent of its readout semantics and operand labels."""
    operation = capsule.get("operation") or {}
    op = str(operation.get("op"))
    attrs = operation.get("attributes") or {}
    if op == "conv2d":
        extents = _conv_extents(capsule)
        if extents is not None:
            return json.dumps(["contraction", extents])
    if op in {"matmul", "fused_matmul_bias"}:
        lhs, weight = _shape(capsule, str(attrs.get("lhs"))), _shape(capsule, str(attrs.get("weight")))
        if lhs and weight and len(lhs) == 2 and len(weight) == 2 and lhs[1] == weight[0]:
            return json.dumps(["contraction", [lhs[0], lhs[1], weight[1]]])
    roles = sorted((str(row.get("role")), list(row.get("shape") or ())) for row in capsule.get("inputs") or ()
                   if isinstance(row, Mapping) and row.get("role") not in {"bias", "scale"})
    return json.dumps([op, roles])


def program_sha256(directory: Path, capsule: Mapping[str, Any]) -> str | None:
    """Digest of the case's capsule-local mathematical program, or ``None`` when it has none."""
    source = capsule.get("interface_mlir") or capsule.get("linalg_mlir")
    if not isinstance(source, str) or Path(source).name != source or not (Path(directory) / source).is_file():
        return None
    return hashlib.sha256((Path(directory) / source).read_bytes()).hexdigest()


def exact_identity(capsule: Mapping[str, Any], program: str | None = None) -> str | None:
    """Same program bytes, point, stimulus and comparison: a second copy tests nothing new.

    A whole model, or a case without capsule-local program bytes, has no identity here: its stated
    point does not describe the program it runs, so it is never collapsed.
    """
    from .hidden_disjointness import capsule_point

    if program is None or capsule.get("kind") == "model":
        return None
    return json.dumps([capsule_point(capsule), capsule.get("stimulus_range"), capsule.get("numeric_policy"),
                       (capsule.get("operation") or {}).get("op"), program], sort_keys=True, default=str)


def richness(capsule: Mapping[str, Any]) -> tuple:
    """More declared semantics first: epilogue stages, bias, a readout narrower than the accumulator,
    declared operation attributes (a window, a layout), then declared capsule blocks."""
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    accumulator = (capsule.get("numeric_policy") or {}).get("accumulator_dtype", "i32")
    narrowed = attrs.get("output_dtype") not in (None, accumulator)
    return (len(attrs.get("epilogue") or ()), bool(attrs.get("bias")), narrowed, len(attrs), len(capsule))


def _family_member(capsule: Mapping[str, Any]) -> bool:
    performance = capsule.get("performance") or {}
    return bool(performance.get("comparison_group") or performance.get("cohort") or performance.get("family"))


def derive(
    cases: Sequence[Mapping[str, Any]],
    *,
    demand: Mapping[str, Any] | None = None,
    protected: Iterable[str] = (),
) -> dict[str, Any]:
    """Collapse exact duplicates and assign one performance scorer per shape group.

    ``cases`` rows are ``{"name", "member", "cohort", "capsule", "binding", "program_sha256"}`` where
    ``capsule`` is the written capsule document, ``program_sha256`` its program bytes' digest
    (:func:`program_sha256`; without it a case is never collapsed), ``member`` its corpus path, ``cohort`` the selection that supplied it
    (e.g. a generated cohort or a staged suite) and ``binding`` an optional source-shape binding.
    ``demand`` is a :mod:`.performance_demand` plan; its members are preferred as scorers.
    ``protected`` names family points whose ``performance`` block a staging step no longer carries;
    they are never collapsed, like any declared family member.
    """
    protected = set(protected)
    names = [str(row["name"]) for row in cases]
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique across the selected corpus")
    demanded = {m["name"] for m in (demand or {}).get("members") or () if m.get("split") == "public"}
    demanded_programs = {(r["model"], r["program"]) for m in (demand or {}).get("members") or ()
                         for r in m.get("represents") or ()}
    by_identity: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(cases):
        identity = exact_identity(row["capsule"], row.get("program_sha256"))
        by_identity.setdefault(identity if identity is not None else f"unique:{index}", []).append(row)
    collapsed, kept = [], []
    for identity, rows in sorted(by_identity.items()):
        ordered = sorted(rows, key=lambda r: (r["name"] not in demanded,
                                              not (_family_member(r["capsule"]) or r["name"] in protected),
                                              tuple(-v for v in richness(r["capsule"])), r["name"]))
        keeper = ordered[0]
        kept.append(keeper)
        for row in ordered[1:]:
            if _family_member(row["capsule"]) or row["name"] in protected:
                kept.append(row)  # a family keeps every point; it is only excluded from scoring below
                continue
            collapsed.append({"removed": row["name"], "member": row.get("member"), "kept": keeper["name"],
                              "identity_sha256": hashlib.sha256(identity.encode()).hexdigest()})
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in kept:
        groups.setdefault(work_shape(row["capsule"]), []).append(row)
    scope, excluded = {}, []
    for shape, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: (r["name"] not in demanded,
                                              tuple(-v for v in richness(r["capsule"])), r["name"]))
        scope[ordered[0]["name"]] = SCORED
        for row in ordered[1:]:
            scope[row["name"]] = EXCLUDED
            excluded.append({"case": row["name"], "reason": "shares_work_shape", "scorer": ordered[0]["name"],
                             "work_shape": json.loads(shape)})
    for row in kept:
        binding = row.get("binding") or {}
        program = (binding.get("model"), binding.get("program"))
        if (scope.get(row["name"]) == SCORED and demand is not None and None not in program
                and program in demanded_programs and row["name"] not in demanded):
            scope[row["name"]] = EXCLUDED
            excluded.append({"case": row["name"], "reason": "not_a_derived_representative_of_its_program",
                             "program": "/".join(program)})
    record = {
        "schema": SCHEMA, "cases": len(cases), "kept": len(kept), "collapsed": collapsed,
        "performance_scope": dict(sorted(scope.items())), "excluded": excluded,
        "shape_groups": sum(len(rows) > 1 for rows in groups.values()),
        "scope": "written capsule semantics only; collapsed members are removed, excluded ones stay graded",
    }
    return {**record, "sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()}


def read_cases(roots: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    """``(cohort, directory)`` pairs to case rows; each directory holds ``<category>/<case>`` or cases."""
    import yaml

    rows = []
    for cohort, root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            capsule = yaml.safe_load(path.read_text()) or {}
            binding_path = path.parent / "source-shape-binding.json"
            binding = json.loads(binding_path.read_text()) if binding_path.is_file() else None
            rows.append({"name": str(capsule.get("name")), "member": str(path.parent.relative_to(root)),
                         "cohort": cohort, "capsule": capsule, "program_sha256": program_sha256(path.parent, capsule),
                         "binding": binding if isinstance(binding, dict) else None})
    return rows


SIZE_SCHEMA = "merlin.phase0.correctness_size_classes.v1"


def size_class(capsule: Mapping[str, Any], *, tile: int, edges: Sequence[int]) -> str:
    """Bin of a case's largest contraction extent on the tile scale (``sub_tile`` below one tile)."""
    work = json.loads(work_shape(capsule))
    extents = work[1] if work[0] == "contraction" else [v for _, shape in work[1] for v in shape]
    largest = max((int(v) for v in extents), default=0)
    if largest < tile:
        return "sub_tile"
    bound = next((edge for edge in edges if largest <= edge * tile), None)
    return f"<={bound}t" if bound is not None else f">{edges[-1]}t"


def size_distribution(rows: Sequence[Mapping[str, Any]], *, tile: int | None,
                      policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """The correctness cohort's size-class histogram against its declared policy.

    A record, not a gate: the policy states the intended classes (``edges`` in tile multiples) and its
    rationale, and the record shows where every written case fell, so a skew is visible and reviewed
    rather than an accident of which derivations happened to fire.
    """
    if policy is None or tile is None:
        return {"schema": SIZE_SCHEMA, "status": "undeclared" if policy is None else "tile_underivable"}
    edges = policy.get("edges") if isinstance(policy, Mapping) else None
    if (not isinstance(edges, list) or not edges or any(type(v) is not int or v < 1 for v in edges)
            or edges != sorted(set(edges))):
        raise ValueError("correctness_size_classes.edges must be strictly increasing positive tile multiples")
    counts: dict[str, int] = {}
    for row in rows:
        cls = size_class(row["capsule"], tile=int(tile), edges=edges)
        counts[cls] = counts.get(cls, 0) + 1
    return {"schema": SIZE_SCHEMA, "status": "declared", "tile_edge": int(tile), "policy": dict(policy),
            "classes": dict(sorted(counts.items())), "cases": len(rows)}


def apply_to_generation(written: list, out_root: Path, *, hidden_category: str = "hidden",
                        tile: int | None = None, size_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Collapse exact duplicates among the public capsules a Phase 0 run just wrote, in place.

    Removed directories leave ``written`` before the provenance manifest is built; every excluded case
    gets ``performance_scope: excluded``. The complete record, with the correctness cohort's size-class
    distribution against the recipe's declared ``correctness_size_classes``, is written next to the
    corpus as ``case-redundancy.json``. Hidden capsules are never read here: their disjointness is
    checked separately and they are never scored.
    """
    import shutil

    import yaml

    public = [Path(path) for path in written
              if isinstance(path, Path) and path.parent.name != hidden_category and (path / "capsule.yaml").is_file()]
    rows = []
    for path in public:
        capsule = yaml.safe_load((path / "capsule.yaml").read_text(encoding="utf-8")) or {}
        rows.append({"name": str(capsule.get("name") or path.name), "member": f"{path.parent.name}/{path.name}",
                     "cohort": "generated", "capsule": capsule, "path": path,
                     "program_sha256": program_sha256(path, capsule)})
    by_name = {row["name"]: row for row in rows}
    record = derive(rows)
    for row in record["collapsed"]:
        path = by_name[row["removed"]]["path"]
        shutil.rmtree(path)
        written[:] = [item for item in written if item != path]
    for name, scope in record["performance_scope"].items():
        if scope == EXCLUDED and by_name[name]["capsule"].get("performance_scope") != EXCLUDED:
            capsule = dict(by_name[name]["capsule"], performance_scope=EXCLUDED)
            (by_name[name]["path"] / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False),
                                                                encoding="utf-8")
    kept = [row for row in rows if row["name"] not in {c["removed"] for c in record["collapsed"]}]
    sizes = size_distribution(kept, tile=tile, policy=size_policy)
    raw = (json.dumps({**record, "size_classes": sizes}, sort_keys=True, indent=2) + "\n").encode()
    (Path(out_root) / "case-redundancy.json").write_bytes(raw)
    return {"schema": SCHEMA, "collapsed": len(record["collapsed"]),
            "performance_excluded": sum(v == EXCLUDED for v in record["performance_scope"].values()),
            "size_classes": {key: sizes[key] for key in ("status", "classes") if key in sizes},
            "record": "case-redundancy.json", "sha256": hashlib.sha256(raw).hexdigest()}
