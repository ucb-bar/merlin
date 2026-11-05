"""The read-command-buffer skill: its summary script on real-format excerpts, and its facet table."""

from __future__ import annotations

import json

import pytest
from author_skill_support import (
    BODY_LIMIT,
    SKILLS,
    front_matter,
    load_script,
    target_mentions,
    top_level_imports,
)
from merlin_experiments.phase1.feedback import mechanism_hints

SKILL = SKILLS / "read-command-buffer"
SCRIPT = SKILL / "scripts" / "summarize_command_buffer.py"

# Anonymized excerpts of a workspace's real files: class and resource vocabulary renamed, structure,
# keys and numbers kept. The decode table lives where the target's RTL facts keep it.
FACTS = {
    "schema_version": 1,
    "facts": {
        "interfaces": [
            {"name": "command_port"},
            {
                "name": "funct_decode_table",
                "width": 7,
                "legal_funct": [0, 1, 2, 3, 4, 5, 6],
                "names": {
                    "0": "SETUP",
                    "1": "FETCH_B",
                    "2": "FETCH",
                    "3": "WRITE_OUT",
                    "4": "MAC",
                    "5": "MAC_KEEP",
                    "6": "STAGE",
                },
            },
        ]
    },
}
CONTRACT = """\
name: neutral_target
corpus_instruction_roles:
  by_role:
    acc_readout: [WRITE_OUT]
    matmul: [MAC]
    memory: [FETCH]
    weight_load: [STAGE]
output_epilogue:
  accumulate:
    value:
      instructions:
        operand_load: [FETCH_B, FETCH]
rocc_operand_roles:
  instructions:
  - {class: SETUP, funct: 0, operands: {}}
"""


def _op(index, cls, funct):
    operand = {"raw": 0, "kind": "const", "arg_index": None, "offset": None}
    row = {"index": index, "class": cls, "funct": funct, "decoded": {}}
    return row if funct is None else {**row, "rs1": operand, "rs2": operand}


TRACE = {
    "source": "generated/lowered.llvm.mlir",
    "abi": {"custom_opcode": "0x0", "funct3": "0x0"},
    "instructions": [
        _op(0, "FENCE", None),
        _op(1, "SETUP", 0),
        _op(2, "SETUP", 0),
        _op(3, "FETCH", 2),
        _op(4, "FETCH_B", 1),
        _op(5, "STAGE", 6),
        _op(6, "MAC", 4),
        _op(7, "WRITE_OUT", 3),
        _op(8, "FENCE", None),
    ],
    "summary": {
        "class_histogram": {},
        "counted_loops": {
            "basis": "original_llvm_cfg_constant_induction_v1",
            "instruction_multiplicity": "static_sites_only",
            "functions": [
                {
                    "function": "kernel_entry",
                    "loops": [
                        {"header_block": 1, "latch_block": 6, "initial": 0, "bound": 16, "step": 1, "trip_count": 16}
                    ],
                }
            ],
        },
    },
}
BUFFER = {
    "abi_version": "0.1",
    "target": "neutral_target",
    "tensors": {
        "W": {"shape": [16, 16], "dtype": "i8", "role": "weight"},
        "A0": {"shape": [1, 16], "dtype": "i8", "role": "input"},
        "Y0": {"shape": [1, 16], "dtype": "i32", "role": "output"},
    },
    "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
        {
            "opcode": "COMMIT",
            "operands": {"src": "acc0", "dst": "Y0"},
            "attributes": {"epilogue": [], "output_dtype": "i32"},
        },
        {"opcode": "EVICT", "operands": {"handle": "W_res"}},
    ],
}
DEBUG = {
    "halted": True,
    "capsule": "CASE_A",
    "metrics": {
        "cycles": 900,
        "bytes_moved": 4608,
        "bytes_read": 3072,
        "bytes_written": 1024,
        "command_count": 5,
        "accumulator_commits": 2,
        "evictions": 8,
        "intermediate_write_bytes": 512,
    },
    "raw_metrics": {"num_matmuls": 2, "spad_reads": 192, "acc_writes": 16, "cycles": 900, "refill_rows": 8},
    "per_command": [
        {
            "opcode": "COMMIT",
            "dst": "Y0",
            "shape": [16, 16, 64],
            "cycles": 600,
            "spad_reads": 128,
            "acc_writes": 8,
            "refill_rows": 8,
        },
        {
            "opcode": "COMMIT",
            "dst": "Y1",
            "shape": [16, 16, 32],
            "cycles": 300,
            "spad_reads": 64,
            "acc_writes": 8,
            "refill_rows": 0,
        },
    ],
    "oracle": {"kind": "arc", "derived_from_rtl": True, "hw_sha": "0" * 12},
    "n_output_tensors": 2,
    "redacted": ["outputs", "correct"],
}


def _resource(name, activity, busy):
    return {"name": name, "capacity": 1, "busy_unit_cycles": busy, "capacity_normalized_activity": activity}


RESPONSE = {
    "schema": "selfcheck",
    "cases": [
        {"capsule": "OTHER", "status": "pass", "performance": {}},
        {
            "capsule": "CASE_A",
            "status": "pass",
            "tiers": {"L2": {"status": "pass"}},
            "performance": {
                "estimated_whole_soc_cycles": 833298,
                "feature_counts": {
                    "target.class.compute_ws": 32768,
                    "target.class.compute": 0,
                    "target.class.config": 4,
                    "target.class.fence": 19,
                    "target.class.load": 3072,
                    "target.class.preload": 32768,
                    "target.class.store": 1024,
                    "target.commands": 69637,
                    "target.local_read_bytes": 50331648,
                    "target.requested_read_bytes": 786432,
                    "device.requested_write_bytes": 262144,
                    "cpu.integer": 183983,
                },
                "scoped_resource_feedback": {
                    "commands": {
                        "total": 69637,
                        "fences": 19,
                        "config": 4,
                        "load": 3072,
                        "preload": 32768,
                        "compute": 32768,
                        "store": 1024,
                    },
                    "resources": [
                        _resource("cpu.issue", 0.36, 300038),
                        _resource("accelerator.storage.accumulator.all", 0.7275, 606208),
                        _resource("accelerator.controller.execute", 0.705, 587453),
                    ],
                },
            },
        },
    ],
    "performance_brief": {
        "largest_headroom": [
            {
                "test_case": "CASE_A",
                "cycles": 833298,
                "bound_by": "compute",
                "roofline_bound_cycles": 524288,
                "gap_ratio": 1.59,
                "array_utilization_pct": 62.92,
                "dominant_modeled_resource": {"name": "accelerator.storage.accumulator.all", "modeled_activity": 0.727},
                "suggested_mechanisms": [{"id": "final_k_accumulator_readout_frontier", "facet": "overlap"}],
                "inspect": "python isa_tools.py debug selfcheck_out/CASE_A/command_buffer.json --capsule CASE_A",
            }
        ]
    },
}


@pytest.fixture
def summarize():
    return load_script(SCRIPT, "author_skill_summarize_command_buffer").summarize


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "rtl_facts").mkdir()
    (tmp_path / "rtl_facts" / "facts.json").write_text(json.dumps(FACTS))
    (tmp_path / "target_contract.yaml").write_text(CONTRACT)
    case = tmp_path / "selfcheck_out" / "CASE_A"
    case.mkdir(parents=True)
    (case / "command_buffer.json").write_text(json.dumps(BUFFER))
    (case / "instruction_trace.json").write_text(json.dumps(TRACE))
    return tmp_path


def test_command_buffer_counts_bytes_and_sibling_trace_classes(summarize, workspace):
    out = summarize(workspace / "selfcheck_out" / "CASE_A" / "command_buffer.json")
    assert out["kind"] == "command_buffer" and out["commands"] == 4
    assert out["tensor_bytes"] == {"W": 256, "A0": 16, "Y0": 64}
    assert out["by_opcode"]["RES_PACK"] == {
        "count": 1,
        "tensor_bytes": 256,
        "handles": ["W_res"],
        "unknown_bytes": False,
    }
    assert out["by_opcode"]["COMMIT"]["tensor_bytes"] == 64
    trace = out["instruction_trace"]
    assert trace["decode_table_checked"] and trace["class_mismatches"] == []
    assert trace["by_class"]["SETUP"] == {"sites": 2, "funct": 0, "roles": []}
    assert trace["by_class"]["FENCE"]["funct"] is None
    # Roles are whatever lists of class names the contract declares, keyed by their path.
    assert trace["by_class"]["FETCH"]["roles"] == ["by_role/memory", "instructions/operand_load"]
    assert trace["by_class"]["WRITE_OUT"]["roles"] == ["by_role/acc_readout"]
    assert trace["declared_never_emitted"] == ["MAC_KEEP"]
    assert trace["multiplicity"] == "static_sites_only"
    assert trace["counted_loops"][0]["trip_count"] == 16


def test_trace_class_that_disagrees_with_the_targets_decode_table_is_reported(summarize, workspace, tmp_path_factory):
    wrong = json.loads(json.dumps(TRACE))
    wrong["instructions"][3]["class"] = "MAC"  # funct 2 is not MAC in the target's table
    path = workspace / "selfcheck_out" / "CASE_A" / "instruction_trace.json"
    path.write_text(json.dumps(wrong))
    out = summarize(path)
    assert out["class_mismatches"] == [{"index": 3, "class": "MAC", "funct": 2, "table": "FETCH"}]
    # Without facts nothing is checked, and nothing is invented.
    alone = tmp_path_factory.mktemp("alone")
    (alone / "instruction_trace.json").write_text(json.dumps(wrong))
    bare = summarize(alone / "instruction_trace.json")
    assert not bare["decode_table_checked"] and bare["declared_never_emitted"] is None
    assert all(row["roles"] == [] for row in bare["by_class"].values())


def test_debug_output_groups_readouts_and_ranks_their_cycles(summarize, tmp_path):
    path = tmp_path / "debug.json"
    path.write_text(json.dumps(DEBUG))
    out = summarize(path)
    assert out["kind"] == "debug" and out["capsule"] == "CASE_A"
    assert out["by_opcode"]["COMMIT"] == {
        "count": 2,
        "cycles": 900,
        "spad_reads": 192,
        "acc_writes": 16,
        "refill_rows": 8,
    }
    assert [row["share_of_cycles"] for row in out["heaviest_commands"]] == [0.667, 0.333]
    assert out["bytes_per_cycle"] == round(4608 / 900, 3)
    path.write_text(json.dumps({"error": "debug oracle unavailable (model absent)"}))
    assert summarize(path) == {"kind": "tool_error", "error": "debug oracle unavailable (model absent)"}


def test_selfcheck_response_reports_dynamic_classes_bytes_resources_and_brief(summarize, tmp_path):
    path = tmp_path / "response.json"
    path.write_text(json.dumps(RESPONSE))
    out = summarize(path, case="CASE_A")
    assert out["dynamic_classes"] == {
        "compute_ws": 32768,
        "config": 4,
        "fence": 19,
        "load": 3072,
        "preload": 32768,
        "store": 1024,
    }
    assert out["per_compute_command"]["preload"] == 1.0 and out["per_compute_command"]["load"] == 0.094
    assert out["bytes"] == {
        "target.local_read_bytes": 50331648,
        "target.requested_read_bytes": 786432,
        "device.requested_write_bytes": 262144,
    }
    assert [r["name"] for r in out["busiest_resources"]][0] == "accelerator.storage.accumulator.all"
    assert out["brief"]["bound_by"] == "compute" and out["brief"]["inspect"].startswith("python isa_tools.py debug")
    assert "CASE_A" in summarize(path)["error"]


def test_skill_front_matter_size_imports_and_no_target_names():
    fields, body = front_matter(SKILL / "SKILL.md")
    assert fields["name"] == "read-command-buffer"
    assert "dominant_modeled_resource" in fields["description"] and "inspect" in fields["description"]
    assert len(body.encode()) <= BODY_LIMIT
    for command in ("debug", "disasm", "lint", "--capsule", "per_command", "scripts/summarize_command_buffer.py"):
        assert command in body
    assert top_level_imports(SCRIPT) <= {"__future__", "argparse", "json", "sys", "pathlib", "yaml"}
    assert target_mentions(SKILL) == []


def test_skill_facet_table_is_the_brief_mapping():
    """The prose table must say what mechanism_hints does; a changed mapping fails here."""
    body = " ".join((SKILL / "SKILL.md").read_text(encoding="utf-8").split())
    expected = {
        "storage busy": mechanism_hints.resource_facets("accelerator.storage.unit"),
        "transfers": mechanism_hints.resource_facets("memory.level"),
        "issue": mechanism_hints.resource_facets("cpu.issue"),
        "execute": mechanism_hints.resource_facets("accelerator.controller.execute"),
        "`requests` bound": mechanism_hints.bound_facets("read requests"),
        "`compute` bound": mechanism_hints.bound_facets("compute"),
    }
    for label, facets in expected.items():
        assert f"{label}: {'/'.join(facets)}" in body, label
