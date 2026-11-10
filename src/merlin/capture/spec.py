"""Capture authored SpecIR programs, independent reference values and declared coverage."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import env as _env
from merlin.common.paths import repo_root


def _specir_root() -> str:
    """The specir checkout: ``$SPECIR_ROOT``, else a sibling of this repo (same convention as
    :func:`_m2m_dir` -- external repo, machine-independent fallback, absent means unavailable)."""
    return _env("SPECIR_ROOT") or str(repo_root().parent / "spec")


class SpecProgramUnavailable(RuntimeError):
    """The selected SpecIR source cannot emit a compiler-consumable program for this gen:op."""


@dataclass
class SpecArtifacts:
    """A spec-derived capsule source: the spec's own PROGRAM (RoCC command buffer for a systolic RoCC gen,
    an MXU command sequence / SIMT warp schedule for the others), the deterministic operands + the golden
    the spec's refmodel computes over them, the issue/command sequence, and the ``coverage_goal`` /
    ``test_intent`` the spec declares. All from ``specir`` — INDEPENDENT of the target RTL (the spec is the
    reference). Operands are keyed by ROLE (``lhs`` / ``weight``) so they map onto the merlin_iface interface
    input names regardless of the gen's own tensor names; the golden is keyed ``out``."""

    gen: str
    op: str
    command_buffer: dict  # the spec-emitted program (cb / mxu sequence / warp schedule) — ships as grounding
    operands: dict  # role ("lhs"/"weight") -> rows of values (int for RoCC, float for fp8/simt)
    golden: dict  # {"out": rows} — the spec refmodel result (bit-exact int, or decoded float)
    instructions: list  # RoCC instructions / MXU commands in issue order
    coverage_goal: list  # [{node, kind, text, oracle, tolerance, covers}]
    opcode_backing: dict  # merlin opcode -> [authored RoCC commands backing it] (RoCC gens only)
    workload: tuple
    compare: str = "exact_int"  # "exact_int" for a RoCC int datapath, "tolerance_float" for fp8/simt
    program_emitter: dict[str, str] | None = None


def _selected_spec_program_emitter(value: object) -> dict[str, str] | None:
    """Validate a recipe's explicit SpecIR float-program adapter without importing arbitrary modules."""
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or not {"module", "function"} <= set(value)
        or set(value) - {"module", "function", "composed_from"}
    ):
        raise SpecProgramUnavailable(
            "spec_program_emitter must declare module and function, with only optional composed_from"
        )
    module = value["module"]
    function = value["function"]
    if (
        not isinstance(module, str)
        or not module.startswith("specir.interface.")
        or not all(part.isidentifier() for part in module.split("."))
        or not isinstance(function, str)
        or not function.isidentifier()
    ):
        raise SpecProgramUnavailable("spec_program_emitter must name a function in a specir.interface submodule")
    selected = {"module": module, "function": function}
    if "composed_from" in value:
        leaf = value["composed_from"]
        if not isinstance(leaf, str) or not leaf.startswith("op.") or not leaf[3:].isidentifier():
            raise SpecProgramUnavailable("spec_program_emitter composed_from must name one op.<identifier> leaf")
        selected["composed_from"] = leaf
    return selected


def _parse_spec_ref(spec_ref: str) -> tuple[str, str]:
    """``"<gen>:op.<name>"`` -> ``(gen, "op.<name>")``. The op token is the last colon-delimited field so a
    gen may carry no colon; everything before is the gen id (matched against the specir manifest)."""
    gen, _, op = spec_ref.rpartition(":")
    if not gen or not op:
        raise ValueError(
            f"spec_ref {spec_ref!r} must be '<gen>:op.<name>' (a gen id from the specir "
            f"manifest, then a spec op symbol)"
        )
    return gen, op


def _covers(node) -> list[str]:
    """The op/claim symbols a coverage node DECLARES it covers, from its ``refs`` dictionary.

    The linkage lives in node refs rather than in the flat attrs, which is why an earlier reader said
    it could not be followed and surfaced every one of the gen's declarations for every op. It can be
    followed: ``refs`` is a ``DictionaryAttr`` whose ``covers`` entry is an array of ``SymbolRefAttr``.
    """
    refs = getattr(node, "refs", None)
    table = getattr(refs, "data", None)
    # NOT `isinstance(table, dict)`: xDSL's DictionaryAttr holds an `immutabledict`, which is a Mapping
    # and NOT a dict subclass. The isinstance form looked like careful defensive coding and silently
    # returned "covers nothing" for every node, which disables the op-scoping this function exists to
    # do -- a guard that cannot pass is the same defect as a check that cannot fail.
    if table is None or not hasattr(table, "get"):
        return []
    covers = table.get("covers")
    out: list[str] = []
    for elem in getattr(covers, "data", ()) or ():
        sym = getattr(elem, "root_reference", None)
        name = getattr(sym, "data", None) if sym is not None else None
        if name:
            out.append(str(name))
    return out


def declared_ops(module) -> list[str]:
    """Every ``spec.op`` this gen declares, by name (``op.matmul``, ``op.isa_flush``, ...).

    The op vocabulary a cell cannot express: 13 for one target, 11 for another, all of them collapsed
    into ``contraction``/``movement`` by the cell key.
    """
    from specir.graph import all_nodes, name_of

    return [str(name_of(n)) for n in all_nodes(module) if (getattr(n, "name", "") or "") == "spec.op"]


def _coverage_goals(module, op: str) -> list[dict]:
    """The spec's ``spec.coverage_goal`` / ``spec.test_intent`` nodes that cover ``op``.

    ⚠️ OP-SCOPED, AND IT WAS NOT. Every declaration in the gen was returned for every op, so a matmul
    capsule carried a transcendental's test intent -- the wrong oracle and the wrong tolerance, stated
    with the same confidence as the right ones. The ``covers`` linkage is followed by :func:`_covers`.

    A node covering NOTHING is still returned: a gen-wide coverage goal with no ``covers`` list applies
    to the whole gen by construction, and dropping it would lose a real obligation. What is dropped is
    a node that names other ops and not this one.
    """
    from specir.graph import all_nodes, attrs_of, name_of

    goals: list[dict] = []
    for n in all_nodes(module):
        mnem = getattr(n, "name", "") or ""
        if mnem not in ("spec.coverage_goal", "spec.test_intent"):
            continue
        covers = _covers(n)
        if covers and op not in covers:
            continue  # authored for other ops; not this capsule's contract
        a = attrs_of(n) or {}
        goals.append(
            {
                "node": name_of(n),
                "kind": mnem,
                "covers": covers,
                "text": a.get("text"),
                "oracle": a.get("oracle"),
                "tolerance": a.get("tolerance"),
                "test_type": a.get("test_type"),
            }
        )
    return goals


def _role_of(meta: dict) -> str:
    """Normalize a program tensor's declared role to ``lhs`` / ``weight`` (the merlin_iface interface roles)."""
    return "weight" if str(meta.get("role", "")).startswith("weight") else "lhs"


def _cb_role_operands(cb: dict, operands: dict) -> dict:
    """RoCC path: map the command buffer's named operands (deterministic int rows) onto lhs/weight roles."""
    tmeta = cb.get("tensors", {})
    return {_role_of(tmeta.get(name, {})): rows for name, rows in operands.items()}


def _decode_program_tensors(program: dict) -> dict:
    """Decode a float program's OPERAND tensor BITS to values via specir's codec, keyed by lhs/weight role.
    Only the input operands ship as raw bits; the output tensor (role accumulator/out) has no bits — it IS
    the golden (already decoded) — so it is skipped here."""
    from specir.oracle.dtypes import decode_float, float_format

    out: dict = {}
    for meta in program.get("tensors", {}).values():
        role = str(meta.get("role", ""))
        if "bits" not in meta or role.startswith(("accum", "out", "result")):
            continue  # the output tensor, not an operand
        fmt = float_format(meta["dtype"])
        out[_role_of(meta)] = [[decode_float(b, fmt) for b in row] for row in meta["bits"]]
    return out


def _float_program_artifacts(
    gen: str, op: str, program: dict, cov: list, workload, program_emitter: dict[str, str]
) -> SpecArtifacts:
    """Normalize an fp8/simt program (an MXU sequence / a warp schedule) into SpecArtifacts:
    role-keyed decoded operands + the already-decoded golden + the program as grounding (float compare)."""
    return SpecArtifacts(
        gen=gen,
        op=op,
        command_buffer=program,
        operands=_decode_program_tensors(program),
        golden={"out": program["golden"]["out"]["values"]},
        instructions=program.get("commands", []) if isinstance(program.get("commands"), list) else [],
        coverage_goal=cov,
        opcode_backing={},
        workload=tuple(workload),
        compare="tolerance_float",
        program_emitter=program_emitter,
    )


class SpecRefSource:
    """Capture a capsule from a ``specir`` verification spec: the spec's own PROGRAM + the golden its refmodel
    computes + the declared coverage. ``specir`` imports in-process (pure xDSL). Without a selected
    emitter, only the generic RoCC command-buffer adapter runs. Other SpecIR program shapes require an
    explicit recipe-owned ``spec_program_emitter`` and fail closed if it cannot produce a program."""

    def __init__(self, root: str | None = None):
        self.root = root or _specir_root()

    def available(self) -> bool:
        return (Path(self.root) / "specir" / "__init__.py").exists()

    def capture(
        self,
        spec_ref: str,
        *,
        workload=(16, 16, 16),
        tile_dim: int = 16,
        program_emitter: dict[str, str] | None = None,
    ) -> SpecArtifacts:
        if not self.available():
            raise SpecProgramUnavailable(f"specir root {self.root} missing; set SPECIR_ROOT")
        from merlin.integrations.specir import importable

        selected_emitter = _selected_spec_program_emitter(program_emitter)
        with importable(self.root):
            from specir.gate import load_targets
            from specir.loading import parse_spec_file
            from specir.registry import _SPEC_ROOT

            gen, op = _parse_spec_ref(spec_ref)
            entry = {t.get("id"): t for t in load_targets(_SPEC_ROOT)}.get(gen)
            if entry is None:
                raise SpecProgramUnavailable(f"gen {gen!r} not registered in the specir manifest")
            spec_path = Path(_SPEC_ROOT) / entry["spec"]
            gen_dir = spec_path.parent
            module = parse_spec_file(spec_path)

            # ⚠️ THE OP TOKEN WAS IGNORED, AND THE REF FAILED OPEN. Measured: `gemmini:op.matmul`,
            # `gemmini:op.isa_flush` and `gemmini:op.TOTALLY_BOGUS` returned a BYTE-IDENTICAL command
            # buffer, golden and coverage goal, because the emitter takes `op` and never reads it. A
            # typo'd or renamed spec_ref therefore silently produced a matmul capsule and called it
            # whatever the ref said. Deriving from a source that fails open is worse than not deriving
            # from it, so the token is checked against the gen's own declared ops before anything is
            # emitted. (The emitter's own indifference to `op` is an upstream defect; this makes it
            # unreachable from here rather than pretending it is fixed.)
            ops = declared_ops(module)
            composed_from = selected_emitter.get("composed_from") if selected_emitter else None
            if composed_from is not None and composed_from not in ops:
                raise SpecProgramUnavailable(
                    f"{gen} declares no composed_from leaf {composed_from!r}; it declares {sorted(ops)}"
                )
            if op not in ops and composed_from is None:
                raise SpecProgramUnavailable(
                    f"{gen} declares no {op!r}; it declares {sorted(ops)}. The program emitter does not "
                    f"read the op token, so an unchecked ref here would emit a DIFFERENT op's program "
                    f"under this name"
                )
            cov = _coverage_goals(module, composed_from or op)

            if selected_emitter is None:
                # Generic RoCC class: an actual command buffer and successful lowering are required.
                from specir.interface.emit_capsule import emit_command_buffer
                from specir.interface.rocc_lower import RoccLoweringError, lower_buffer

                cb, prov = emit_command_buffer(module, gen, op, gen_dir, workload=tuple(workload))
                if cb is not None:
                    try:
                        p = lower_buffer(module, cb, dim=tile_dim)
                        return SpecArtifacts(
                            gen=gen,
                            op=op,
                            command_buffer=cb,
                            operands=_cb_role_operands(cb, dict(p.operands)),
                            golden={"out": next(iter(p.golden.values()))}
                            if len(p.golden) == 1
                            else {"out": p.golden.get("Y0") or next(iter(p.golden.values()))},
                            instructions=list(p.instructions),
                            coverage_goal=cov,
                            opcode_backing=prov.get("opcode_backed_by_spec_rocc", {}),
                            workload=tuple(workload),
                            compare="exact_int",
                            program_emitter={
                                "module": "specir.interface.emit_capsule",
                                "function": "emit_command_buffer",
                            },
                        )
                    except RoccLoweringError:
                        pass
                raise SpecProgramUnavailable(
                    f"no generic RoCC program for {gen}:{op}; select a spec_program_emitter in the recipe "
                    "for another SpecIR program interface"
                )

            try:
                selected_module = importlib.import_module(selected_emitter["module"])
            except ImportError as exc:
                raise SpecProgramUnavailable(
                    f"selected spec_program_emitter module {selected_emitter['module']!r} is unavailable"
                ) from exc
            emit = getattr(selected_module, selected_emitter["function"], None)
            if not callable(emit):
                raise SpecProgramUnavailable(
                    f"selected spec_program_emitter function {selected_emitter['function']!r} is unavailable"
                )
            result = emit(module, op, gen_dir, workload=tuple(workload))
            if not isinstance(result, tuple) or len(result) != 2:
                raise SpecProgramUnavailable("selected spec_program_emitter must return (program, provenance)")
            program, provenance = result
            if program is None:
                raise SpecProgramUnavailable(f"selected spec_program_emitter produced no program for {gen}:{op}")
            if (
                not isinstance(program, dict)
                or not isinstance(program.get("tensors"), dict)
                or not isinstance(program.get("golden"), dict)
                or not isinstance(program["golden"].get("out"), dict)
                or not isinstance(program["golden"]["out"].get("values"), list)
            ):
                raise SpecProgramUnavailable(
                    "selected spec_program_emitter must produce a float program with tensors and golden.out.values"
                )
            if composed_from is not None:
                kernel = program.get("kernel")
                if (
                    program.get("target") != gen
                    or program.get("composed_from") != composed_from
                    or not isinstance(provenance, dict)
                    or provenance.get("composed_from") != composed_from
                    or not isinstance(kernel, list)
                    or not kernel
                    or any(not isinstance(step, dict) or step.get("op") != composed_from for step in kernel)
                ):
                    raise SpecProgramUnavailable(
                        "selected spec_program_emitter's target, composed_from provenance and kernel "
                        "must match the declared composed_from leaf"
                    )
            try:
                return _float_program_artifacts(gen, op, program, cov, workload, selected_emitter)
            except (KeyError, TypeError, ValueError) as exc:
                raise SpecProgramUnavailable(
                    "selected spec_program_emitter produced a malformed float program"
                ) from exc
