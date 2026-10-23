"""Put a whole window of dataflow-connected groups to a package as ONE region capsule.

Exercises the real walk (`corpus_spec.build` per member, textual stitching at the seam,
`whole_program._spliced`'s role binding, the by-name coverage check) against a STUB package that
never runs a real compiler -- the same pattern `test_whole_program_buffer.py` uses for a single
group's ask (`_stub_package` / a fake ``invoke``).
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

from example_binder import binder

from merlin.llvmlower import region_capsule as RC

_TARGET = "gemmini"

#: A matmul (member 0) whose committed output the residual add (member 1) reads as its own ``lhs``.
_MEMBERS = [
    {
        "tag": "g1",
        "entry": {
            "name": "g1",
            "kind": "op",
            "source_role": "model_derived",
            "source_reference": "test",
            "op": "matmul",
            "M": 4,
            "K": 8,
            "N": 16,
            "epilogue": [],
            "operand_dtype": "int8",
        },
        "operands": {"lhs": "image", "rhs": "W", "dst": "B_g1"},
        "opcode": "MATMUL",
        "output_dtype": "i8",
    },
    {
        "tag": "g2",
        "entry": {
            "name": "g2",
            "kind": "op",
            "source_role": "model_derived",
            "source_reference": "test",
            "op": "residual_add",
            "lhs_scale": 1.0,
            "rhs_scale": 1.0,
            "bound_lsb": 1,
            "epilogue": [],
        },
        "operands": {"lhs": "B_g1", "rhs": "leaf_y", "dst": "B_g2"},
        "opcode": "RESIDUAL_ADD",
        "output_dtype": "i8",
    },
]

_SHAPES = {
    "image": {"shape": [4, 8], "dtype": "i8", "role": "intermediate"},
    "W": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
    "leaf_y": {"shape": [4, 16], "dtype": "i8", "role": "input"},
    "B_g1": {"shape": [4, 16], "dtype": "i8", "role": "intermediate"},
    "B_g2": {"shape": [4, 16], "dtype": "i8", "role": "intermediate"},
}


def _stub_package():
    commands = ("parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact")
    return SimpleNamespace(
        manifest={"target": _TARGET, "commands": {name: {"argv": ["{tool}", "{input_mlir}"]} for name in commands}},
        tool=pathlib.Path("/nonexistent/tool"),
        directory=pathlib.Path("/nonexistent"),
    )


def _region_reply(*, drop_first_commit: bool = False) -> dict:
    """The package's reply to the STITCHED capsule: a matmul into an accumulator, a commit that is
    member 0's own output, and a residual add reading it that is member 1's own output. The two
    ``__region_out_...`` names are exactly what `ask_package_region` assigns each member's own
    output, in member order -- this reply plays the honest package that keeps both committed."""
    commands = [
        {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": "acc0"}},
        {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "__region_out_0000"}},
        # The residual add reads member 0's committed output back from memory. The contract obliges
        # must_fence_dram_read_after_write, so the honest package fences the hand-off.
        {"opcode": "FENCE", "operands": {}},
        {"opcode": "RESIDUAL_ADD", "operands": {"lhs": "__region_out_0000", "rhs": "X1", "dst": "__region_out_0001"}},
    ]
    if drop_first_commit:
        # A package that silently keeps member 0's result only on-chip: the commit never appears.
        commands = [commands[0], commands[-1]]
    return {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": {
            "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "W": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
            "X1": {"shape": [4, 16], "dtype": "i8", "role": "input"},
            "__region_out_0000": {"shape": [4, 16], "dtype": "i8", "role": "output"},
            "__region_out_0001": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": commands,
    }


def _invoke(reply: dict):
    def run(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        if destination is not None:
            pathlib.Path(destination).write_text(json.dumps(reply), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="; target artifact\n", stderr="")

    return run


def test_a_legal_region_is_spliced_and_its_staged_member_is_bound_by_name(tmp_path):
    record: dict = {}
    commands, scratch, why, cause, region_info = RC.ask_package_region(
        _stub_package(),
        _MEMBERS,
        _TARGET,
        tmp_path,
        30,
        _invoke(_region_reply()),
        shapes=_SHAPES,
        views=[],
        record=record,
        binder=binder(_TARGET),
    )
    assert commands and why == "" and cause == "", (why, cause)
    # REGRESSION: a region that succeeds must record where its reply lives, exactly as a single-group
    # ask does -- `whole_model_build.bind_groups` reads `asked["command_buffer"]`/`["artifact"]` for
    # EVERY package-answered row, region member or not, and crashed with a bare `KeyError` when a
    # region's own record carried neither.
    assert record.get("command_buffer") and pathlib.Path(record["command_buffer"]).name == "command_buffer.json"
    assert record.get("artifact") and pathlib.Path(record["artifact"]).is_file()
    assert region_info == {
        "id": "region_g1_g2",
        "members": ["g1", "g2"],
        "boundary": "g2",
        "declared_opcodes": ["MATMUL", "RESIDUAL_ADD"],
        "graded_at": "g2",
        "staged": ["g1"],
    }
    dsts = [c["operands"]["dst"] for c in commands if "dst" in c.get("operands", {})]
    # This reply STAGES its internal member in the member's own program buffer; it is bound by name.
    assert "B_g1" in dsts and "B_g2" in dsts
    # The activation and the residual's other operand are bound to THIS PROGRAM's own names, never the
    # capsule's local ones -- the seam ("__region_out_0000") never leaks into the final buffer.
    assert not any("__region_out" in str(v) for c in commands for v in c.get("operands", {}).values())
    shares = RC.split_region_commands(commands, ["B_g1", "B_g2"])
    assert shares is not None and len(shares) == 2
    assert all(shares[i] for i in range(2)), "every member keeps a non-empty share of the commands"
    assert sum(len(s) for s in shares) == len(commands)


def test_a_fused_reply_that_never_forms_an_internal_members_output_is_accepted_at_its_boundary(tmp_path):
    """A FUSED region: the internal member's value is the kernel's own business and is never required --
    the region is accepted for the boundary it commits, and graded there."""
    reply = _region_reply(drop_first_commit=True)
    del reply["tensors"]["__region_out_0000"]  # never formed at all
    reply["commands"][-1]["operands"]["lhs"] = "acc0"
    commands, _scratch, why, _cause, region_info = RC.ask_package_region(
        _stub_package(),
        _MEMBERS,
        _TARGET,
        tmp_path,
        30,
        _invoke(reply),
        shapes=_SHAPES,
        views=[],
        record={},
        binder=binder(_TARGET),
    )
    assert commands and why == "", why
    assert region_info["graded_at"] == "g2" and region_info["staged"] == []
    dsts = [c["operands"]["dst"] for c in commands if "dst" in c.get("operands", {})]
    assert "B_g2" in dsts and "B_g1" not in dsts


def test_a_reply_that_never_commits_its_boundary_is_refused_whole(tmp_path):
    reply = _region_reply()
    reply["commands"] = reply["commands"][:2]  # the boundary's own output is never committed
    commands, _scratch, why, _cause, region_info = RC.ask_package_region(
        _stub_package(),
        _MEMBERS,
        _TARGET,
        tmp_path,
        30,
        _invoke(reply),
        shapes=_SHAPES,
        views=[],
        record={},
        binder=binder(_TARGET),
    )
    assert not commands and region_info is None and "B_g2" in why


def test_fewer_than_two_members_is_refused_before_anything_is_asked(tmp_path):
    commands, _scratch, why, _cause, region_info = RC.ask_package_region(
        _stub_package(),
        _MEMBERS[:1],
        _TARGET,
        tmp_path,
        30,
        _invoke(_region_reply()),
        shapes=_SHAPES,
        views=[],
        record={},
        binder=binder(_TARGET),
    )
    assert not commands and region_info is None and "at least two" in why


def test_split_region_commands_partitions_contiguously_and_refuses_an_uncommitted_member():
    commands = [
        {"opcode": "A", "operands": {"dst": "x"}},
        {"opcode": "B", "operands": {"dst": "y"}},
        {"opcode": "C", "operands": {"dst": "z"}},
    ]
    shares = RC.split_region_commands(commands, ["x", "z"])
    assert shares == [[commands[0]], [commands[1], commands[2]]]
    assert RC.split_region_commands(commands, ["x", "not_committed"]) is None
