"""Construct capsules from the authored SpecIR program and its independent golden."""

from __future__ import annotations

from pathlib import Path

from merlin.capture.artifacts import _shape_of
from merlin.capture.spec import SpecRefSource
from merlin.targetgen.golden_store import write_golden


def write_spec_capsule(entry: dict, binding, out_root, *, source: SpecRefSource | None = None):
    """Materialize a capsule from a specir verification spec (``spec_ref: '<gen>:op.<name>'``): the agent
    compiles the merlin_iface interface for the op; the golden + the exact operands come from the spec's own
    command-buffer program (bit-exact refmodel, INDEPENDENT of the target RTL); the spec's command buffer +
    coverage_goal ship as grounding. Fails closed if specir has no program emitter for the gen."""
    import json

    import yaml

    from merlin.targetgen import corpus_spec as CS

    src = source or SpecRefSource()
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    art = src.capture(
        entry["spec_ref"],
        workload=(M, N, K),
        tile_dim=D,
        program_emitter=entry.get("spec_program_emitter"),
    )

    cap, mlir = CS.build(entry, binding)  # merlin_iface interface for the op (names align)
    cap["spec_ref"] = entry["spec_ref"]
    cap["spec_program_emitter"] = art.program_emitter
    out_name = entry.get("out", "Y0")
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    # map the spec's role-keyed operands onto the interface's input names (lhs -> the activation input,
    # weight -> the resident weight) so the grader feeds the exact values the golden was computed over.
    role_to_name = {"lhs": entry.get("lhs", "A0"), "weight": entry.get("weight", "W")}
    prov_inputs = {
        role_to_name.get(role, role): {"shape": _shape_of(rows), "decoded": rows} for role, rows in art.operands.items()
    }
    golden = {
        "golden_source": f"specir_program_{art.gen}",
        "oracle_provenance": {
            "engine": "specir spec program + refmodel golden (INDEPENDENT of the target RTL; the spec is "
            "the reference)",
            "spec_ref": entry["spec_ref"],
            "program_emitter": art.program_emitter,
            "workload": list(art.workload),
            "compare": art.compare,
            "coverage_goal": art.coverage_goal,
            "opcode_backed_by_spec_rocc": art.opcode_backing,
            "program": "capsule.command_buffer.json",
            "inputs": prov_inputs,
        },
        "outputs": {out_name: art.golden.get("out")},
    }
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    # the program ships as VISIBLE grounding — strip any embedded golden (the MXU/SIMT programs carry the
    # output values under "golden"; the answer lives ONLY in the masked golden files, never in a visible one)
    program = {k: v for k, v in art.command_buffer.items() if k != "golden"}
    (d / "capsule.command_buffer.json").write_text(json.dumps(program, indent=1), encoding="utf-8")
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    write_golden(d, golden)
    return d
