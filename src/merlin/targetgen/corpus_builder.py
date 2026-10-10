"""Validate common capsule declarations after their selected operation builder."""

from __future__ import annotations

import copy


def build_entry(entry: dict, binding, *, builders: dict, semantic_block) -> tuple[dict, str]:
    """Dispatch an abstract capsule entry to its op builder -> (capsule dict, interface MLIR)."""
    op = entry.get("op", "matmul")
    if op not in builders:
        raise ValueError(f"no corpus builder for op {op!r} (have {sorted(builders)})")
    cap, mlir = builders[op](entry, binding)
    if entry.get("prelude") is not None:
        cap, mlir = _with_prelude(entry, binding, builders, cap, mlir)
    if entry.get("input_palette") is not None:
        from .input_palette import realize, validate

        palette = validate(entry["input_palette"])
        for index, row in enumerate(cap.get("inputs") or []):
            realize(palette, name=row["name"], shape=row["shape"], dtype=row["dtype"], index=index)
        cap["input_palette"] = copy.deepcopy(palette)
    # AN EPILOGUE THE BUILDER DID NOT CARRY IS A COVERAGE LIE, so it is refused here rather than in each
    # builder. Only `matmul` and `conv2d` read the entry's `epilogue:`; every other builder writes its
    # own (often empty) list, so a stage declared on, say, a `movement` entry vanished from the capsule
    # -- while `_semantic_block` still credited that stage's FAMILY in `composed_families` off the same
    # entry. The capsule would then be counted as evidence for a family whose arithmetic no engine ever
    # performed, which is the exact failure the pooling epilogue was implemented to close. One check
    # here covers every present and future builder; per-matmul epilogues (resident_reuse) use their own
    # key and are unaffected.
    declared_epilogue = [str(x) for x in (entry.get("epilogue") or [])]
    if declared_epilogue:
        carried = [str(x) for x in ((cap.get("operation") or {}).get("attributes") or {}).get("epilogue", [])]
        dropped = [x for x in declared_epilogue if x not in carried]
        if dropped:
            raise ValueError(
                f"{entry.get('name', op)}: the {op!r} builder dropped epilogue stage(s) {dropped} "
                f"(carried: {carried}). The capsule would still be CREDITED for those stages' semantic "
                f"families, so it would count as evidence for arithmetic nothing computed"
            )
    # THE STIMULUS RANGE AN ENTRY DECLARES, carried here for the same reason: no builder reads it, so
    # an entry that asked for a signed stimulus produced a capsule on the non-negative default, and
    # the sign-sensitive stage it was written to test could not fail. Validated by the ABI's own
    # reader, so a malformed range is refused at generation and not at grading.
    if entry.get("stimulus_range") is not None:
        from merlin.runtime.commandbuffer import STIMULUS_RANGE_KEY, stimulus_range

        cap[STIMULUS_RANGE_KEY] = list(stimulus_range({"params": {STIMULUS_RANGE_KEY: list(entry["stimulus_range"])}}))
    # Stamped once here rather than in each builder, so every capsule a target emits carries the same
    # declaration and a new builder cannot silently forget it.
    if binding.inapplicable_tiers:
        cap["inapplicable_oracle_tiers"] = dict(binding.inapplicable_tiers)
    # A MUST-REFUSE ENTRY says so on the capsule, with what is refused and the evidence, so the grade
    # inverts (merlin.targetgen.expected_refusal) and a reader sees why no program is the right answer.
    if entry.get("outcome") is not None:
        from .expected_refusal import OUTCOMES, REFUSE, declaration

        if entry["outcome"] not in OUTCOMES:
            raise ValueError(f"{entry.get('name', op)}: outcome must be one of {list(OUTCOMES)}")
        if entry["outcome"] == REFUSE:
            refusal = entry.get("refusal") or {}
            cap.setdefault("expected", {}).update(
                declaration(
                    stage=refusal.get("stage"),
                    reason=str(refusal.get("reason") or ""),
                    evidence=str(refusal.get("evidence") or ""),
                )
            )
    sem = semantic_block(entry, binding)
    if sem:
        cap["semantic"] = sem
    return cap, mlir


#: Operand names of a residual-state prelude. Distinct from every builder's defaults so the two
#: programs of one capsule can never alias each other's tensors.
PRELUDE_NAMES = {"lhs": "P_A", "weight": "P_W", "out": "P_Y"}


def _module_body(mlir: str) -> list[str]:
    """The operation lines of one emitted interface module: between its header and its closing brace."""
    lines = mlir.rstrip("\n").split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith("module ")), None)
    if start is None or lines[-1] != "}":
        raise ValueError("interface module has no recognisable header and closing brace")
    return lines[start + 1 : -1]


def _with_prelude(entry: dict, binding, builders: dict, cap: dict, mlir: str) -> tuple[dict, str]:
    """Prefix the capsule's program with a dense contraction that leaves the device's stores non-zero.

    THE RESIDUAL-STATE AXIS. A program can be right on a device whose stores start empty and wrong on
    one where an earlier command left data behind -- a padded convolution that reads rows it never
    wrote, a first reduction step that accumulates onto rows it never cleared. The tensor-level tiers
    have no store and cannot see this; the instruction-level and RTL tiers can, but only if something
    ran first. The prelude is that something: an independent, signed, non-zero contraction issued
    before the operation under test, whose own result is graded too, so it cannot be dropped.
    """
    spec = dict(entry["prelude"] or {})
    unknown = sorted(set(spec) - {"M", "K", "N"})
    if unknown or any(type(spec.get(axis)) is not int or spec[axis] < 1 for axis in ("M", "K", "N")):
        raise ValueError(f"{entry.get('name')}: prelude must declare positive integer M, K, N only (got {spec})")
    prelude_entry = {
        "name": f"{entry.get('name')}_prelude",
        "kind": entry.get("kind", "layer"),
        "op": "matmul",
        "source_role": entry.get("source_role", "derived_sweep"),
        "source_reference": "residual-state prelude",
        **PRELUDE_NAMES,
        **spec,
    }
    pre_cap, pre_mlir = builders["matmul"](prelude_entry, binding)
    names = {row["name"] for row in cap.get("inputs") or []} | {
        str(((cap.get("operation") or {}).get("attributes") or {}).get("out", "Y0"))
    }
    if names & set(PRELUDE_NAMES.values()):
        raise ValueError(
            f"{entry.get('name')}: operand names collide with the prelude's {sorted(PRELUDE_NAMES.values())}"
        )
    head = mlir.rstrip("\n").split("\n")
    start = next(i for i, line in enumerate(head) if line.startswith("module "))
    # The prelude's accumulator values are renamed so a contraction under test cannot redefine them.
    prelude_body = [line.replace("%acc", "%P_acc") for line in _module_body(pre_mlir)]
    merged = [*head[: start + 1], *prelude_body, *_module_body(mlir), "}"]
    cap = copy.deepcopy(cap)
    cap["inputs"] = [*pre_cap["inputs"], *cap["inputs"]]
    cap["operation"].setdefault("attributes", {})["prelude"] = {
        **PRELUDE_NAMES,
        **spec,
        "output_dtype": pre_cap["operation"]["attributes"]["output_dtype"],
    }
    return cap, "\n".join(merged) + "\n"
