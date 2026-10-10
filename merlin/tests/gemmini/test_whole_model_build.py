"""The whole-model build: one library call from a compiler package and a model capsule to a program.

Two kinds of test live here. The first half needs nothing but the tree: the refusals `bind_groups`
names, the oracle grade, and the machine-header assertion. The second half builds the statement of
SY_model_resnet50 with a real loop-produced package and holds what the bridge PASSED to it against
the capture itself -- the per-group scales, epilogue and operand order a unit-scale, no-epilogue
capsule can never exercise. It needs the capsule's gitignored weights and the package, and is skipped
without them; with the package's replies recorded (by content, see `whole_model_build._ReplyCache`)
it takes seconds and asks the package nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import selected_driver

from merlin.common.paths import merlin_dir, runs_dir
from merlin.perf import whole_model_build as W

if selected_driver.selected_is_generic("gemmini"):
    pytest.skip(
        "tests compiler/harness modules a gemmini support package ships itself; the selected "
        "generic data provider (merlin.runtime.backends.chipyard_rocc) does not ship them",
        allow_module_level=True,
    )

pytestmark = pytest.mark.target("gemmini")

_TARGET = "gemmini"
_CAPSULE = merlin_dir() / "contract" / "capsules" / "model" / "SY_model_resnet50"

#: The loop-produced package the whole-model arms measured. Override with MERLIN_WHOLE_MODEL_PACKAGE.
_PACKAGE = Path(
    os.environ.get("MERLIN_WHOLE_MODEL_PACKAGE")
    or runs_dir() / "gemmini/capsule-bench/merlin_assisted/merlincirct_p1froz/_qa_work/cand_07/submission"
)

#: Groups the package answers, per package CONTENT. A RATCHET: it may only rise. When a change to the
#: bridge makes the package answer more, raise the entry to the new count in the same change; a
#: change that answers fewer is a regression this refuses, whatever it fixed elsewhere.
#: 68 -> 69 (2026-09-26): the full-width readout is declared to carry the preloaded bias, so the
#: classifier (g71, a biased int32 commit) is no longer refused at its interface capsule.
_ANSWERED_FLOOR = {"92752cf9b6de26ac07b29d74b832321a9caf3ff20d7633c980170c19367f2038": 69}


# ------------------------------------------------------------------------------ refusals, by name


def _reply(tmp_path, tensors, commands, **extra):
    path = tmp_path / "command_buffer.json"
    path.write_text(json.dumps({"tensors": tensors, "commands": commands, **extra}), encoding="utf-8")
    return str(path)


def _buffer(row):
    return {"target": _TARGET, "whole_program": {"per_group": [row], "prepacked": {}, "input_domain": None}}


_WHOLE = {"kind": "whole_program", "outputs": ["Y0"]}


def test_shape_facts_are_read_for_a_covered_op_and_none_for_others():
    conv_entry = {
        "op": "conv2d",
        "N": 8,
        "ci": 4,
        "Himg": 10,
        "Wimg": 10,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "operand_dtype": "i8",
        "output_dtype": "i8",
        "epilogue": [],
    }
    facts = W._shape_facts("conv2d", conv_entry)
    assert facts == {
        "N": 8,
        "ci": 4,
        "Himg": 10,
        "Wimg": 10,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "operand_dtype": "i8",
        "output_dtype": "i8",
    }
    assert W._shape_facts("residual_add", {"op": "residual_add"}) is None
    assert W._shape_facts("conv2d", {"op": "conv2d", "N": 8}) is None  # missing extents
    assert W._shape_facts("conv2d", None) is None


def test_bind_groups_carries_shape_facts_on_both_a_refusal_and_a_splice(tmp_path):
    """The estimate is diagnostic, not a binding: it must survive REGARDLESS of who answers the
    group, so a gap holder later routed to the host or the library still carries its own bound."""
    conv_entry = {
        "op": "conv2d",
        "N": 8,
        "ci": 4,
        "Himg": 10,
        "Wimg": 10,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "operand_dtype": "i8",
        "output_dtype": "i8",
    }
    refused_row = {
        "group": 5,
        "op": "conv2d",
        "on": "reference",
        "operands": {},
        "entry": conv_entry,
        "cause": "package_declined",
        "why": "the package declined this group",
    }
    (refused,) = W.bind_groups(_buffer(refused_row))
    assert refused["on"] == W.ON_VENDOR and refused["shape_facts"] == W._shape_facts("conv2d", conv_entry)

    tensors = {
        "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
        "Y0": {"shape": [4, 8], "dtype": "i8", "role": "output"},
    }
    abi = dict(_WHOLE, args=[{"tensor": "A0", "access": "read"}, {"tensor": "Y0", "access": "write"}])
    spliced_row = {
        "group": 6,
        "op": "conv2d",
        "on": "submission",
        "operands": {"lhs": "B_g5", "dst": "B_g6"},
        "entry": conv_entry,
        "asked": {
            "command_buffer": _reply(tmp_path, tensors, [{"opcode": "X"}], kernel_abi=abi),
            "artifact": "unused",
            "binding": {
                "A0": {"program": "B_g5", "role": "input", "declared": [4, 8], "bound": True},
                "Y0": {"program": "B_g6", "role": "output", "declared": [4, 8], "bound": True},
            },
        },
    }
    (spliced,) = W.bind_groups(_buffer(spliced_row))
    assert spliced["on"] == W.ON_PACKAGE and spliced["shape_facts"] == W._shape_facts("conv2d", conv_entry)


def test_groups_sharing_a_reply_resolve_their_argument_order_once(tmp_path, monkeypatch):
    """The order is a function of the reply; a model's repeated layers share replies, so it is resolved
    once per distinct reply, and every group still binds as it would alone."""
    tensors = {
        "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
        "Y0": {"shape": [4, 8], "dtype": "i8", "role": "output"},
    }
    abi = dict(_WHOLE, args=[{"tensor": "A0", "access": "read"}, {"tensor": "Y0", "access": "write"}])
    reply = _reply(tmp_path, tensors, [{"opcode": "X"}], kernel_abi=abi)
    calls = []
    real = W._kernel_arg_order

    def counted(target, command_buffer):
        calls.append(target)
        return real(target, command_buffer)

    monkeypatch.setattr(W, "_kernel_arg_order", counted)

    def row(group):
        return {
            "group": group,
            "op": "matmul",
            "on": "submission",
            "operands": {"lhs": f"B_g{group - 1}", "dst": f"B_g{group}"},
            "entry": {"op": "matmul"},
            "asked": {
                "command_buffer": reply,
                "artifact": "unused",
                "binding": {
                    "A0": {"program": f"B_g{group - 1}", "role": "input", "declared": [4, 8], "bound": True},
                    "Y0": {"program": f"B_g{group}", "role": "output", "declared": [4, 8], "bound": True},
                },
            },
        }

    buffer = _buffer(row(1))
    buffer["whole_program"]["per_group"] = [row(1), row(2), row(3)]
    bound = W.bind_groups(buffer)
    assert len(calls) == 1
    alone = [W.bind_groups(_buffer(row(g)))[0] for g in (1, 2, 3)]
    assert bound == alone


def _region_rows(tmp_path, *, kernel_abi=True):
    """A fused region g1 (matmul) -> g2 (residual add) as the statement records it: the BOUNDARY's row
    carries the region's one kernel; the internal member's row carries no command of its own."""
    tensors = {
        "A": {"shape": [4, 8], "dtype": "i8", "role": "input"},
        "W": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
        "Y": {"shape": [4, 16], "dtype": "i8", "role": "input"},
        "B_g2": {"shape": [4, 16], "dtype": "i8", "role": "output"},
    }
    args = [{"tensor": t, "access": "read"} for t in ("A", "W", "Y")] + [{"tensor": "B_g2", "access": "write"}]
    extra = {"kernel_abi": dict(_WHOLE, args=args)} if kernel_abi else {}
    region = {"id": "region_g1_g2", "members": ["g1", "g2"], "boundary": "g2", "member_groups": [1, 2]}
    asked = {
        "command_buffer": _reply(tmp_path, tensors, [{"opcode": "MATMUL"}], **extra),
        "artifact": "unused",
        "binding": {
            "A": {"program": "image", "role": "input", "declared": [4, 8], "bound": True},
            "W": {"program": "W", "role": "weight", "declared": [8, 16], "bound": True},
            "Y": {"program": "leaf_y", "role": "input", "declared": [4, 16], "bound": True},
            "B_g2": {"program": "B_g2", "role": "output", "declared": [4, 16], "bound": True},
        },
    }
    internal = {
        "group": 1,
        "op": "matmul",
        "on": "submission",
        "commands": 0,
        "operands": {"lhs": "image", "rhs": "W", "dst": "B_g1"},
        "region": {**region, "role": "internal"},
        "asked": asked,
    }
    boundary = {
        "group": 2,
        "op": "residual_add",
        "on": "submission",
        "commands": 1,
        "operands": {"lhs": "B_g1", "rhs": "leaf_y", "dst": "B_g2"},
        "entry": {"op": "residual_add"},
        "region": {**region, "role": "boundary"},
        "asked": asked,
    }
    buffer = {
        "target": _TARGET,
        "whole_program": {"per_group": [internal, boundary], "prepacked": {}, "input_domain": None},
    }
    return buffer


def test_a_fused_region_binds_one_kernel_at_its_boundary_and_credits_every_member(tmp_path):
    """A FUSED REGION is ONE kernel: its boundary's row binds the region's arguments (reading every
    member's operands), and its internal member is package-answered with no kernel of its own --
    graded at the boundary. Coverage counts both groups as the package's."""
    internal, boundary = W.bind_groups(_region_rows(tmp_path))
    assert boundary["on"] == W.ON_PACKAGE and [a["tensor"] for a in boundary["args"]] == ["A", "W", "Y", "B_g2"]
    assert boundary["member_operands"] == {"1": {"lhs": "image", "rhs": "W", "dst": "B_g1"}, "2": boundary["operands"]}
    assert internal["on"] == W.ON_PACKAGE and internal["graded_at"] == 2
    assert "args" not in internal and "artifact" not in internal, "the region's kernel is the boundary's alone"
    assert W.linked_regions([internal, boundary]) == [{"members": [1, 2], "boundary": 2}]


def test_a_region_whose_kernel_cannot_be_linked_puts_every_member_back_on_the_library(tmp_path):
    """An internal member is never credited to the package while its region's kernel does not run."""
    internal, boundary = W.bind_groups(_region_rows(tmp_path, kernel_abi=False))
    assert boundary["on"] == W.ON_VENDOR
    assert internal["on"] == W.ON_VENDOR and internal["cause"] == W.REGION_UNLINKED
    assert W.linked_regions([internal, boundary]) == []


def test_a_private_buffer_without_the_packages_recipe_is_refused(tmp_path):
    """The kernel reads a buffer the program does not hold. Built only from the package's OWN recipe;
    re-deriving its geometry here would be a second definition of the matrix the kernel reads."""
    tensors = {
        "X": {"shape": [4, 8], "dtype": "i8", "role": "input"},
        "X_im2col": {"shape": [4, 72], "dtype": "i8", "role": "input"},
        "Y0": {"shape": [4, 8], "dtype": "i8", "role": "output"},
    }
    abi = dict(_WHOLE, args=[{"tensor": "X_im2col", "access": "read"}, {"tensor": "Y0", "access": "write"}])
    row = {
        "group": 3,
        "op": "conv2d",
        "on": "submission",
        "operands": {"lhs": "B_g2", "dst": "B_g3"},
        "asked": {
            "command_buffer": _reply(tmp_path, tensors, [{"opcode": "X"}], kernel_abi=abi),
            "artifact": "unused",
            "binding": {
                "X": {"program": "B_g2", "role": "input", "declared": [4, 8], "bound": True},
                "X_im2col": {"program": "g3_X_im2col", "role": "input", "declared": [4, 72], "bound": False},
                "Y0": {"program": "B_g3", "role": "output", "declared": [4, 8], "bound": True},
            },
        },
    }
    (got,) = W.bind_groups(_buffer(row))
    assert got["on"] == W.ON_VENDOR and got["cause"] == W.PRIVATE_WITHOUT_RECIPE
    assert "'X_im2col'" in got["why"]


def test_arguments_come_in_the_order_the_package_declares(tmp_path):
    """The contract row the package's own buffer resolves to decides the order -- here its explicit
    whole-program ABI -- and each argument names the program tensor the splice bound it to."""
    tensors = {
        n: {"shape": [4, 8], "dtype": "i8", "role": r} for n, r in (("X1", "input"), ("X0", "input"), ("Y0", "output"))
    }
    abi = dict(
        _WHOLE, args=[{"tensor": n, "access": "read"} for n in ("X1", "X0")] + [{"tensor": "Y0", "access": "write"}]
    )
    row = {
        "group": 6,
        "op": "residual_add",
        "on": "submission",
        "operands": {"lhs": "B_g4", "rhs": "B_g5", "dst": "B_g6"},
        "asked": {
            "command_buffer": _reply(tmp_path, tensors, [{"opcode": "RESIDUAL_ADD"}], kernel_abi=abi),
            "artifact": "unused",
            "binding": {
                "X0": {"program": "B_g4", "role": "input", "declared": [4, 8], "bound": True},
                "X1": {"program": "B_g5", "role": "input", "declared": [4, 8], "bound": True},
                "Y0": {"program": "B_g6", "role": "output", "declared": [4, 8], "bound": True},
            },
        },
    }
    (got,) = W.bind_groups(_buffer(row))
    assert got["on"] == W.ON_PACKAGE and got["shape"] == "whole_program"
    assert [(a["tensor"], a["program"]) for a in got["args"]] == [("X1", "B_g5"), ("X0", "B_g4"), ("Y0", "B_g6")]


def test_a_group_the_package_did_not_answer_keeps_the_packages_reason():
    row = {"group": 1, "op": "conv2d", "on": "reference", "cause": "package_declined", "why": "no room", "operands": {}}
    (got,) = W.bind_groups(_buffer(row))
    assert (got["on"], got["cause"], got["why"]) == (W.ON_VENDOR, "package_declined", "no room")


def _answered(group, op):
    return {"group": group, "op": op, "on": W.ON_PACKAGE, "shape": "native_whole_op", "args": [], "artifact": "a"}


def test_a_caller_declined_op_keeps_the_library_call_and_says_whose_decision_it_was():
    rows = [
        _answered(1, "conv2d"),
        _answered(2, "residual_add"),
        {"group": 3, "op": "residual_add", "on": W.ON_VENDOR, "cause": "package_declined", "why": "no room"},
    ]
    got = {r["group"]: r for r in W.decline_ops(rows, ["residual_add"])}
    assert got[1]["on"] == W.ON_PACKAGE and got[1]["args"] == []
    assert (got[2]["on"], got[2]["cause"]) == (W.ON_VENDOR, W.CALLER_DECLINED)
    assert "residual_add" in got[2]["why"] and "args" not in got[2] and "artifact" not in got[2]
    # The package's own reason is never overwritten by the caller's.
    assert (got[3]["cause"], got[3]["why"]) == ("package_declined", "no room")


def test_declining_nothing_changes_nothing():
    rows = [_answered(1, "conv2d")]
    assert W.decline_ops([dict(r) for r in rows], []) == rows


def test_a_declined_op_the_model_does_not_have_is_refused():
    with pytest.raises(W.WholeModelBuildError, match=r"\['residual_ad'\]"):
        W.decline_ops([_answered(1, "residual_add")], ["residual_ad"])


@pytest.mark.parametrize("name", [33, "g33", "33"])
def test_a_caller_declined_group_is_named_by_index_and_only_that_group_moves(name):
    rows = [_answered(33, "conv2d"), _answered(35, "conv2d"), _answered(36, "residual_add")]
    got = {r["group"]: r for r in W.decline_ops(rows, [name, "residual_add"])}
    assert (got[33]["on"], got[33]["cause"], got[33]["declined_as"]) == (W.ON_VENDOR, W.CALLER_DECLINED, "group")
    assert "g33" in got[33]["why"]
    assert got[35]["on"] == W.ON_PACKAGE
    assert (got[36]["on"], got[36]["declined_as"]) == (W.ON_VENDOR, "op")


def test_a_declined_group_the_model_does_not_have_is_refused():
    with pytest.raises(W.WholeModelBuildError, match=r"\['g99'\]"):
        W.decline_ops([_answered(1, "conv2d")], ["g99"])


# ------------------------------------------------------------------------------------------- grade


_ORACLE = {
    "groups": {
        "2": {"compare": "exact", "sum": 10, "fnv1a": 99},
        "6": {"compare": "bounded", "bound_lsb": 2, "sum": 0, "fnv1a": 0},
    },
    "argmax": 21,
}


def test_a_run_agreeing_everywhere_is_quotable():
    uart = "\n".join(
        [
            "GM_GROUP 2 matmul 100 sum=10 fnv1a=99",
            "GM_GROUP 6 sum 50 sum=3 fnv1a=5",
            "GM_BOUND 6 max_abs=1 over=0 bound=2",
            "GM_LOCAL 2 mismatches=0 of=4 first=-1",
            "GM_ARGMAX got=21 want=21 agrees=1",
        ]
    )
    verdict = W.grade(uart, _ORACLE)
    assert verdict["quotable"] and verdict["agree"] == ["2", "6"] and verdict["gate"] == "local"


def test_a_bounded_group_is_graded_on_its_bound_and_an_unprinted_one_is_absent():
    uart = (
        "GM_GROUP 2 matmul 100 sum=10 fnv1a=98\nGM_LOCAL 2 mismatches=1 of=4 first=3\nGM_ARGMAX got=21 want=21 agrees=1"
    )
    verdict = W.grade(uart, _ORACLE)
    assert not verdict["quotable"]
    assert verdict["disagree"][0]["group"] == "2" and verdict["absent"] == ["6"]


_CHAIN = {
    "groups": {
        "5": {"compare": "exact", "sum": 1, "fnv1a": 11},
        "6": {"compare": "bounded", "bound_lsb": 2, "sum": 2, "fnv1a": 22},
        "7": {"compare": "exact", "sum": 3, "fnv1a": 33},
    },
    "argmax": 21,
}


def test_a_group_wrong_on_its_own_inputs_fails_even_when_upstream_is_identical():
    """Every upstream digest agrees with the oracle, so group 7's inputs are the reference's own -- and
    its local check still sees 5 wrong elements. The chained digest (made up to agree here) is not the
    gate: a run is refused on the group's own arithmetic."""
    uart = "\n".join(
        [
            "GM_GROUP 5 matmul 1 sum=1 fnv1a=11",
            "GM_GROUP 6 sum 1 sum=2 fnv1a=22",
            "GM_GROUP 7 matmul 1 sum=3 fnv1a=33",
            "GM_LOCAL 5 mismatches=0 of=8 first=-1",
            "GM_BOUND 6 max_abs=0 over=0 bound=2",
            "GM_LOCAL 7 mismatches=5 of=8 first=2",
            "GM_ARGMAX got=21 want=21 agrees=1",
        ]
    )
    verdict = W.grade(uart, _CHAIN)
    assert not verdict["quotable"]
    assert verdict["disagree"] == [{"group": "7", "mismatches": "5", "of": "8", "first": "2"}]
    assert verdict["chained"]["agree"] == ["5", "6", "7"]


def test_an_upstream_bounded_difference_does_not_fail_downstream_groups():
    """Group 6 rounds each operand -- within its declared bound -- so group 7 is handed different
    inputs than the reference chain and its digest differs from the chained oracle. Its own arithmetic
    on those inputs is right, and that is what it is graded on."""
    uart = "\n".join(
        [
            "GM_GROUP 5 matmul 1 sum=1 fnv1a=11",
            "GM_GROUP 6 sum 1 sum=4 fnv1a=40",
            "GM_GROUP 7 matmul 1 sum=9 fnv1a=90",
            "GM_LOCAL 5 mismatches=0 of=8 first=-1",
            "GM_BOUND 6 max_abs=1 over=0 bound=2",
            "GM_LOCAL 7 mismatches=0 of=8 first=-1",
            "GM_ARGMAX got=21 want=21 agrees=1",
        ]
    )
    verdict = W.grade(uart, _CHAIN)
    assert verdict["quotable"] and verdict["agree"] == ["5", "6", "7"]
    assert [d["group"] for d in verdict["chained"]["disagree"]] == ["6", "7"]


def test_a_group_with_no_local_check_is_absent_not_agreed():
    uart = "GM_GROUP 5 matmul 1 sum=1 fnv1a=11\nGM_BOUND 6 max_abs=0 over=0 bound=2\nGM_ARGMAX got=21 want=21 agrees=1"
    verdict = W.grade(uart, _CHAIN)
    assert not verdict["quotable"] and verdict["absent"] == ["5", "7"]


def test_a_memory_dump_is_graded_by_the_recorded_buffer_map():
    """``host_dump`` leaves every check to the host: the program prints no digest, and a reader of the
    run's memory grades each group off the map the build recorded -- an exact group by the oracle's
    digest, a tolerance-declared one by recomputing its contract reference from the operands the device
    actually read."""
    import numpy as np

    from merlin.perf.layer_bench import reference as ref

    memory = bytearray(64)
    out = np.array([1, -2, 3, 4], dtype=np.int8)
    lhs, rhs = np.array([10, 20, -30, 100], dtype=np.int8), np.array([2, 2, 2, 100], dtype=np.int8)
    summed = np.array([6, 11, 0, 100], dtype=np.int8)  # relu(sat(roundeven(0.5*lhs + 0.5*rhs)))
    for base, values in ((0, out), (16, lhs), (32, rhs), (48, summed)):
        memory[base : base + 4] = values.tobytes()

    def place(address):
        return {"address": address, "bytes": 4, "elements": 4, "element_bytes": 1}

    layout = {
        "groups": [
            {"group": 2, "compare": "exact", **place(0)},
            {
                "group": 6,
                "compare": "bounded",
                "bound_lsb": 0,
                "lhs_scale": 0.5,
                "rhs_scale": 0.5,
                "relu": True,
                "lhs": place(16),
                "rhs": place(32),
                **place(48),
            },
        ]
    }
    digest = ref.fnv1a64_words(out.astype("<i8").tobytes()) & ref.DIGEST_MASK
    oracle = {"groups": {"2": {"compare": "exact", "fnv1a": digest}}}

    def read(address, size):
        return memory[address : address + size]

    class _Echo:  # group 2's reference, given its (absent) inputs, is the device's own output here
        def expected(self, group, inputs):
            return out.astype(np.int64)

    assert W.grade_memory(read, layout, oracle, local=_Echo())["quotable"]
    # Without a local reference an exact group is UNVERIFIED, never agreed on its chained digest.
    unguided = W.grade_memory(read, layout, oracle)
    assert not unguided["quotable"] and unguided["unverified"][0]["group"] == "2"
    assert unguided["chained"]["agree"] == ["2"]
    memory[49] = 13  # the sum is out of its declared bound by two
    verdict = W.grade_memory(read, layout, oracle, local=_Echo())
    assert verdict["agree"] == ["2"] and verdict["disagree"] == [{"group": "6", "max_abs": 2, "over": 1}]


def _two_matmuls():
    """A two-group statement: g1 = X @ W1, g2 = B_g1 @ W2, both read out whole."""
    tensors = {
        "X": {"shape": [2, 3], "dtype": "i8", "role": "input"},
        "W1": {"shape": [3, 2], "dtype": "i8", "role": "weight"},
        "ACC_g1": {"shape": [2, 2], "dtype": "i32", "role": "intermediate"},
        "B_g1": {"shape": [2, 2], "dtype": "i32", "role": "output"},
        "W2": {"shape": [2, 2], "dtype": "i8", "role": "weight"},
        "ACC_g2": {"shape": [2, 2], "dtype": "i32", "role": "intermediate"},
        "B_g2": {"shape": [2, 2], "dtype": "i32", "role": "output"},
    }
    readout = {"epilogue": [], "output_dtype": "i32"}
    commands = [
        {"opcode": "MATMUL", "operands": {"lhs": "X", "rhs": "W1", "dst": "ACC_g1"}, "attributes": {}},
        {"opcode": "COMMIT", "operands": {"src": "ACC_g1", "dst": "B_g1"}, "attributes": dict(readout)},
        {"opcode": "MATMUL", "operands": {"lhs": "B_g1", "rhs": "W2", "dst": "ACC_g2"}, "attributes": {}},
        {"opcode": "COMMIT", "operands": {"src": "ACC_g2", "dst": "B_g2"}, "attributes": dict(readout)},
    ]
    per_group = [
        {"group": 1, "commands": 2, "operands": {"lhs": "X", "rhs": "W1", "dst": "B_g1"}},
        {"group": 2, "commands": 2, "operands": {"lhs": "B_g1", "rhs": "W2", "dst": "B_g2"}},
    ]
    reference = {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": tensors,
        "commands": commands,
        "whole_program": {"per_group": per_group},
    }
    leaves = {"X": [[1, 2, 3], [4, 5, 6]], "W1": [[1, 0], [0, 1], [1, 1]], "W2": [[2, 0], [1, 3]]}
    return reference, leaves


def test_a_local_reference_recomputes_a_group_from_the_inputs_the_run_gave_it():
    import numpy as np

    reference, leaves = _two_matmuls()
    local = W.LocalReference(reference, leaves)
    assert local.inputs_of(1) == [] and local.inputs_of(2) == ["B_g1"]
    np.testing.assert_array_equal(local.expected(1, {}), [4, 5, 10, 11])
    handed = np.array([7, -1, 0, 2])  # NOT X @ W1: an upstream that legitimately differs
    np.testing.assert_array_equal(
        local.expected(2, {"B_g1": handed}), (handed.reshape(2, 2) @ [[2, 0], [1, 3]]).reshape(-1)
    )
    with pytest.raises(W.WholeModelBuildError, match="B_g1"):
        local.expected(2, {})


def test_a_memory_dump_is_graded_locally_group_by_group():
    """The dump holds g1 = X @ W1 and g2 computed from g1's DUMPED value. Corrupting g1 by a (declared
    legitimate, here simulated) amount leaves g2 locally right -- its chained digest disagrees and does
    not matter -- while one wrong element of g2 on identical inputs is caught."""
    import numpy as np

    from merlin.perf.layer_bench import reference as ref

    reference, leaves = _two_matmuls()
    local = W.LocalReference(reference, leaves)
    w2 = np.array([[2, 0], [1, 3]])
    g1 = np.array([4, 5, 10, 11], dtype="<i4")

    def dump(g1_values, g2_values):
        memory = bytearray(32)
        memory[0:16] = np.asarray(g1_values, dtype="<i4").tobytes()
        memory[16:32] = np.asarray(g2_values, dtype="<i4").tobytes()
        return lambda address, size: memory[address : address + size]

    def place(address):
        return {"address": address, "bytes": 16, "elements": 4, "element_bytes": 4}

    layout = {
        "groups": [
            {"group": 1, "compare": "exact", "symbol": "B_g1", **place(0), "inputs": []},
            {"group": 2, "compare": "exact", "symbol": "B_g2", **place(16),
             "inputs": [{"symbol": "B_g1", "produced_by": 1, **place(0)}]},
        ]
    }  # fmt: skip
    chain_g2 = (g1.reshape(2, 2) @ w2).reshape(-1)
    oracle = {
        "groups": {
            str(g): {"compare": "exact", "fnv1a": ref.fnv1a64_words(v.astype("<i8").tobytes()) & ref.DIGEST_MASK}
            for g, v in ((1, g1), (2, chain_g2))
        }
    }
    right = W.grade_memory(dump(g1, chain_g2), layout, oracle, local=local)
    assert right["quotable"] and right["agree"] == ["1", "2"]

    # Identical upstream, one wrong element in g2: refused on g2's own arithmetic.
    bad = chain_g2.copy()
    bad[3] += 1
    wrong = W.grade_memory(dump(g1, bad), layout, oracle, local=local)
    assert not wrong["quotable"]
    assert wrong["disagree"] == [{"group": "2", "mismatches": 1, "of": 4, "first": 3}]

    # g1 as the device left it differs (graded on g1's own row, which is left out here to model an
    # upstream bounded group); g2 computed right FROM THAT VALUE passes, though its chained digest fails.
    shifted = g1 + np.array([1, 0, -1, 0], dtype="<i4")
    downstream = (shifted.reshape(2, 2) @ w2).reshape(-1)
    only_g2 = {"groups": [layout["groups"][1]]}
    verdict = W.grade_memory(dump(shifted, downstream), only_g2, oracle, local=local)
    assert verdict["quotable"] and verdict["agree"] == ["2"] and verdict["chained"]["disagree"] == ["2"]


def test_the_memory_map_names_what_a_dump_must_hold_for_local_grading(monkeypatch, tmp_path):
    """Each group's row lists the buffers ANOTHER group produced that it reads; ``dump`` is their union
    with every output. A reader capturing exactly that can grade every group locally."""
    elf = tmp_path / "p.elf"
    elf.write_bytes(b"x")
    from merlin.perf import whole_model_memory as MEM

    # `memory_map` (re-exported on `W` for every existing caller) lives in `whole_model_memory` and
    # calls its OWN module-level `_symbols`; patching `W._symbols` would rebind a different name.
    monkeypatch.setattr(MEM, "_symbols", lambda _elf: {"B_g1": (100, 4), "B_g2": (200, 4), "B_g3": (300, 4)})
    model = {
        "steps": [
            {"group": 1, "kind": "matmul", "out": "B_g1"},
            {"group": 2, "kind": "matmul", "out": "B_g2"},
            {"group": 3, "kind": "matmul", "out": "B_g3"},
        ],
        "buffers": [{"name": f"B_g{g}", "elements": 4} for g in (1, 2, 3)],
    }
    buffer = {
        "whole_program": {
            "per_group": [
                {"group": 1, "operands": {"lhs": "image", "rhs": "W1", "dst": "B_g1"}},
                {"group": 2, "operands": {"lhs": "B_g1", "rhs": "W2", "dst": "B_g2"}},
                {"group": 3, "operands": {"lhs": "B_g1", "rhs": "B_g2", "dst": "B_g3"}},
            ]
        }
    }
    layout = W.memory_map(elf, model, buffer)
    assert layout["schema"] == "whole_model_memory_map_v2"
    inputs = {r["group"]: [(i["symbol"], i["produced_by"]) for i in r["inputs"]] for r in layout["groups"]}
    assert inputs == {1: [], 2: [("B_g1", 1)], 3: [("B_g1", 1), ("B_g2", 2)]}
    assert layout["dump"] == {"symbols": ["B_g1", "B_g2", "B_g3"], "bytes": 12}


# ---------------------------------------------------------------------------- the machine's header


_CURATED = (
    merlin_dir()
    / "experiments/capsule_bench/targets/gemmini/contracts/harness_curated/gemmini-rocc-tests/include/gemmini_params.h"
)


def test_the_header_is_the_one_the_registry_declares_for_the_machine(tmp_path):
    """The header decides which machine a program is for. It is an explicit input, asserted by content
    against the registry, and never whatever a checkout happens to hold."""
    got = W.machine_header("gemmini_gsim_model_testharness", _CURATED)
    assert got["status"] == "registry_declared" and got["declared_by"] == "gemmini_gsim_model_testharness"
    other = tmp_path / "gemmini_params.h"
    other.write_bytes(_CURATED.read_bytes() + b"\n/* one more line is another machine */\n")
    with pytest.raises(W.WholeModelBuildError, match="different machine"):
        W.machine_header("gemmini_gsim_model_testharness", other)


def test_a_machine_with_no_declared_header_needs_the_caller_to_assert_one(tmp_path):
    """A bitstream whose ABI header the registry does not declare (the 30 MHz U250 one). Building for it is possible only on
    the caller's explicit assertion, and the record says the assertion is the caller's."""
    machine = "firesim_gemmini_rocket_u250_30mhz"
    with pytest.raises(W.WholeModelBuildError, match="UNKNOWN"):
        W.machine_header(machine, _CURATED)
    digest = W._sha256(_CURATED)
    got = W.machine_header(machine, _CURATED, digest)
    assert got["status"] == "caller_asserted" and got["registry_notes"]
    with pytest.raises(W.WholeModelBuildError, match="the caller"):
        W.machine_header(machine, _CURATED, "0" * 64)


# ------------------------------------------------ SY_model_resnet50 with a real loop-produced package


@pytest.fixture(scope="module")
def resnet(tmp_path_factory):
    if not (_CAPSULE / "capsule.weights.safetensors").is_file():
        pytest.skip("the model capsule's weights are gitignored and absent from this checkout")
    if not (_PACKAGE / "manifest.yaml").is_file():
        pytest.skip(f"no package at {_PACKAGE} (set MERLIN_WHOLE_MODEL_PACKAGE)")
    from merlin.common import mlir_query as mq
    from merlin.common.ir_lock import IR_LOCK
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    capsule = W.load_model_capsule(_CAPSULE)
    buffer = W.state(capsule, target=_TARGET, package_dir=_PACKAGE, work=tmp_path_factory.mktemp("lower"))
    with IR_LOCK:
        groups = CG.form_groups(mq.parse(capsule.interface.read_text(encoding="utf-8")), _TARGET)
    return SimpleNamespace(buffer=buffer, rows=W.bind_groups(buffer), groups=groups)


def test_the_package_answers_at_least_the_ratcheted_floor(resnet):
    digest = resnet.buffer["whole_program"]["package_replies"]["package_digest"]
    answered = sum(1 for r in resnet.rows if r["on"] == W.ON_PACKAGE)
    floor = _ANSWERED_FLOOR.get(digest)
    if floor is None:
        pytest.skip(f"no floor recorded for package {digest[:12]} (it answers {answered})")
    assert answered >= floor, f"the package now answers {answered} groups, below the floor of {floor}"
    assert answered == floor, f"the package now answers {answered} groups: raise the floor to {answered}"


def test_every_group_is_attributed_and_every_refusal_is_named(resnet):
    device = [g for g in resnet.groups if g.placement != "host"]
    rows = {r["group"]: r for r in resnet.rows}
    assert set(rows) == {g.index for g in resnet.groups}, "a group with no attribution row"
    assert sum(1 for r in resnet.rows if r["on"] == W.ON_HOST) == len(resnet.groups) - len(device)
    for row in resnet.rows:
        if row["on"] == W.ON_VENDOR:
            assert row.get("cause") and row.get("why"), f"g{row['group']} falls back with no named reason"


def test_every_kernel_is_called_in_the_order_its_package_declares(resnet):
    for row in resnet.rows:
        if row["on"] != W.ON_PACKAGE:
            continue
        declared = json.loads(Path(row["command_buffer"]).read_text(encoding="utf-8"))
        order, shape, _why = W._kernel_arg_order(_TARGET, declared)
        assert [a["tensor"] for a in row["args"]] == order and shape == row["shape"], f"g{row['group']}"
        abi = declared.get("kernel_abi") or {}
        if abi.get("kind") == "whole_program":
            assert [a["tensor"] for a in row["args"]] == [a["tensor"] for a in abi["args"]], f"g{row['group']}"


def _scale(op) -> float:
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    return float(GN._scale_source(op).value)


def _stated(row) -> dict:
    """The one arithmetic command of what the bridge PASSED to the package, read back off the file."""
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    parsed = parse_interface_mlir(Path(row["interface"]).read_text(encoding="utf-8"))
    for command in parsed["commands"]:
        if command["opcode"] in ("RESIDUAL_ADD", "CONV2D", "COMMIT"):
            return command
    raise AssertionError(f"g{row['group']}: the interface states no arithmetic command")


def test_what_the_bridge_passes_each_group_is_the_captures_own_facts(resnet):
    """Scales, activation and operand order, recomputed from the CAPTURE and compared with what was
    written into each group's interface and bound as each kernel argument.

    A residual add carries two scales that differ per operand, so a swapped pair is a numerically
    wrong program that every shape check passes -- and the corpus's unit-scale, no-epilogue residual
    capsules could never see it. This is the check that could."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    rows = {r["group"]: r for r in resnet.rows}
    produced_by = {id(r): g.index for g in resnet.groups for m in g.members for r in m.results}
    entry = resnet.buffer["whole_program"]["input_domain"]["tensor"]

    def held(value) -> str:
        """The program tensor a capture value is committed to: its producing group's output."""
        producer = produced_by[id(value)]
        return entry if rows[producer]["on"] == W.ON_HOST else rows[producer]["operands"]["dst"]

    checked = {"sum": 0, "contraction": 0}
    for group in resnet.groups:
        row = rows[group.index]
        if row["on"] == W.ON_HOST or "interface" not in row or group.window_mean is not None:
            continue
        kinds = [CG.classify(m).kind for m in group.members]
        quantize = next((m for m, k in zip(group.members, kinds) if k == CG.QUANTIZE), None)
        command = _stated(row)
        attrs = command.get("attributes") or {}
        epilogue = list(attrs.get("epilogue") or [])
        assert ("relu" in epilogue) == (CG.RELU in kinds), f"g{group.index}: the activation was mispassed"
        bound = {a["tensor"]: a.get("program") for a in row.get("args") or ()}
        if group.operand_sum is not None:
            dequantizes = [CG._input_chain(operand)[1] for operand in list(group.root.operands)[:2]]
            out = _scale(quantize)
            for side, dequantize in zip(("lhs", "rhs"), dequantizes, strict=True):
                want = _scale(dequantize) / out
                assert attrs[f"{side}_scale"] == pytest.approx(want, rel=1e-7), f"g{group.index} {side}_scale"
                if bound:
                    tensor = command["operands"][side]
                    assert bound[tensor] == held(dequantize.operands[0]), (
                        f"g{group.index}: {side} bound to another buffer"
                    )
            checked["sum"] += 1
            continue
        dequantizes = [m for m, k in zip(group.members, kinds) if k == CG.DEQUANTIZE]
        if quantize is not None:
            want = _scale(dequantizes[0]) * _scale(dequantizes[1]) / _scale(quantize)
            assert attrs["acc_scale"] == pytest.approx(want, rel=1e-6), f"g{group.index} acc_scale"
        else:
            assert "acc_scale" not in epilogue, f"g{group.index} leaves as the accumulator and was told to scale"
        if bound:
            # The activation is the root operand that is not a stored argument; whatever the kernel
            # reads it through (the tensor itself, or the package's gather from it), it is that buffer.
            from xdsl.ir import BlockArgument

            chains = [CG._input_chain(o)[1] for o in list(group.root.operands)[:2]]
            (activation,) = [d for d in chains if d is not None and not isinstance(d.operands[0], BlockArgument)]
            reads = {
                a.get("program") or (a.get("gather") or {}).get("source") for a in row["args"] if a["role"] == "input"
            }
            assert reads == {held(activation.operands[0])}, f"g{group.index}: the activation bound to {reads}"
        checked["contraction"] += 1
    assert checked["sum"] >= 1 and checked["contraction"] >= 1, checked


def test_no_argument_is_read_as_a_permutation_of_what_the_program_holds(resnet):
    """Equal counts prove nothing about order. Every argument the package reads from a program tensor
    is that tensor under the same shape, or the same bytes in the same order."""
    from merlin.llvmlower.whole_program import _same_bytes

    tensors = resnet.buffer["tensors"]
    for row in resnet.rows:
        for arg in row.get("args") or ():
            program = arg.get("program")
            if program is None or arg["role"] not in ("input", "output"):
                continue
            mine = list(tensors[program]["shape"])
            assert _same_bytes(mine, arg["declared"]), f"g{row['group']}: {arg['tensor']} {arg['declared']} vs {mine}"


def test_a_viewed_input_reaches_a_convolution_and_a_matmul_in_the_form_each_reads():
    """B_g1 is committed flat [4, 2] and declared rank-4 because a convolution reads it; a matmul
    reads the same bytes flat. Each consumer is re-run on the dumped bytes in the form it reads."""
    import numpy as np

    readout = {"epilogue": [], "output_dtype": "i32"}
    tensors = {
        "X": {"shape": [4, 3], "dtype": "i8", "role": "input"},
        "W1": {"shape": [3, 2], "dtype": "i8", "role": "weight"},
        "ACC_g1": {"shape": [4, 2], "dtype": "i32", "role": "intermediate"},
        "B_g1": {"shape": [1, 2, 2, 2], "dtype": "i32", "role": "intermediate"},
        "W2": {"shape": [2, 2], "dtype": "i8", "role": "weight"},
        "B_g2": {"shape": [4, 2], "dtype": "i32", "role": "intermediate"},
        "W3": {"shape": [2, 3], "dtype": "i8", "role": "weight"},
        "ACC_g3": {"shape": [4, 3], "dtype": "i32", "role": "intermediate"},
        "B_g3": {"shape": [4, 3], "dtype": "i32", "role": "intermediate"},
    }
    conv = {"kernel": [1, 1, 2, 2], "stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc"}
    commands = [
        {"opcode": "MATMUL", "operands": {"lhs": "X", "rhs": "W1", "dst": "ACC_g1"}, "attributes": {}},
        {"opcode": "COMMIT", "operands": {"src": "ACC_g1", "dst": "B_g1"}, "attributes": dict(readout)},
        {
            "opcode": "CONV2D",
            "operands": {"ifm": "B_g1", "weight": "W2", "dst": "B_g2"},
            "attributes": {**conv, **readout},
        },
        {"opcode": "MATMUL", "operands": {"lhs": "B_g1", "rhs": "W3", "dst": "ACC_g3"}, "attributes": {}},
        {"opcode": "COMMIT", "operands": {"src": "ACC_g3", "dst": "B_g3"}, "attributes": dict(readout)},
    ]
    reference = {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": tensors,
        "commands": commands,
        "whole_program": {
            "per_group": [
                {"group": 1, "commands": 2, "operands": {"lhs": "X", "rhs": "W1", "dst": "B_g1"}},
                {"group": 2, "commands": 1, "operands": {"ifm": "B_g1", "weight": "W2", "dst": "B_g2"}},
                {"group": 3, "commands": 2, "operands": {"lhs": "B_g1", "rhs": "W3", "dst": "B_g3"}},
            ],
            "views": [{"tensor": "B_g1", "committed": [4, 2], "read_as": [1, 2, 2, 2]}],
        },
    }
    w2, w3 = np.array([[1, 2], [3, -1]]), np.array([[1, 0, 2], [0, 1, -1]])
    leaves = {"X": np.arange(12).reshape(4, 3) % 5, "W1": [[1, 0], [0, 1], [1, 1]], "W2": w2, "W3": w3}
    local = W.LocalReference(reference, leaves)
    dumped = np.array([3, -2, 0, 7, 1, 1, -4, 5])  # flat, as the device left it
    np.testing.assert_array_equal(local.expected(2, {"B_g1": dumped}), (dumped.reshape(4, 2) @ w2).reshape(-1))
    np.testing.assert_array_equal(local.expected(3, {"B_g1": dumped}), (dumped.reshape(4, 2) @ w3).reshape(-1))


# --------------------------------------- allow_regions/allow_passes never affect a package that opts into neither


#: A worktree checked out beside this one may carry the gitignored weights this one lacks. Overridable
#: (never a hard-coded personal path): set MERLIN_WHOLE_MODEL_CAPSULE to another checkout's capsule
#: directory to let this regression actually run, rather than skip, where its bug was found.
_REAL_CAPSULE = _CAPSULE
if not (_REAL_CAPSULE / "capsule.weights.safetensors").is_file() and os.environ.get("MERLIN_WHOLE_MODEL_CAPSULE"):
    _REAL_CAPSULE = Path(os.environ["MERLIN_WHOLE_MODEL_CAPSULE"])


def test_a_package_that_declares_no_region_or_pass_builds_identically_with_the_flags_on(tmp_path_factory):
    """The regression this guards: ``allow_regions``/``allow_passes`` are builder-level opt-ins, but a
    package that declares neither must never see a different statement -- not a different route, not
    a different compiled object -- because the offer is unconditional on the model's own dataflow and
    was, before this test existed, made to every package regardless of whether it asked. A package
    whose real compiler happens to succeed at lowering a merged capsule it never requested is not
    "asking"; the KeyError this once caused (`bind_groups` reading a `command_buffer` path a region
    ask never recorded) is exactly what a silent behavior change through this flag would look like.
    """
    if not (_REAL_CAPSULE / "capsule.weights.safetensors").is_file():
        pytest.skip("the model capsule's weights are gitignored and absent from every checkout tried")
    if not (_PACKAGE / "manifest.yaml").is_file():
        pytest.skip(f"no package at {_PACKAGE} (set MERLIN_WHOLE_MODEL_PACKAGE)")
    import yaml

    manifest = yaml.safe_load((_PACKAGE / "manifest.yaml").read_text(encoding="utf-8")) or {}
    assert not manifest.get("whole_model_regions") and not manifest.get("whole_model_passes"), (
        f"{_PACKAGE} now declares regions or passes; this regression needs a package that declares neither"
    )

    capsule = W.load_model_capsule(_REAL_CAPSULE)

    def built(*, allow_regions: bool, allow_passes: bool):
        buffer = W.state(
            capsule,
            target=_TARGET,
            package_dir=_PACKAGE,
            work=tmp_path_factory.mktemp("lower"),
            allow_regions=allow_regions,
        )
        rows = W.bind_groups(buffer)
        objects_out = tmp_path_factory.mktemp("objects")
        W._kernel_objects(rows, target=_TARGET, out=objects_out, jobs=8)
        return buffer, rows

    off_buffer, off_rows = built(allow_regions=False, allow_passes=False)
    on_buffer, on_rows = built(allow_regions=True, allow_passes=True)

    assert not any(row.get("region") for row in on_buffer["whole_program"]["per_group"]), (
        "a package that never declared whole_model_regions was offered one anyway"
    )
    off_by_group = {r["group"]: r for r in off_rows}
    on_by_group = {r["group"]: r for r in on_rows}
    assert set(off_by_group) == set(on_by_group)
    for group, off_row in off_by_group.items():
        on_row = on_by_group[group]
        assert on_row["on"] == off_row["on"], f"group {group}: route changed ({off_row['on']} -> {on_row['on']})"
        if off_row["on"] == W.ON_PACKAGE:
            assert on_row.get("object_sha256") == off_row.get("object_sha256"), (
                f"group {group}: compiled object changed with the flags on"
            )
            assert on_row.get("args") == off_row.get("args"), f"group {group}: kernel arguments changed"


# --------------------------------------------------- a region's members share ONE compile, not N


#: A minimal, REAL, compilable kernel body -- the same entry symbol a package's own artifact declares,
#: doing nothing. This is not a stand-in for a real region's arithmetic (see
#: `merlin/tests/runtime/test_region_capsule.py` for that); it is the smallest input that exercises
#: the REAL toolchain (`llvm_mlir_to_object`, `llvm-objcopy`, `llvm-nm`) this test needs to be real
#: about, so the "one compile, N renamed symbols" contract is checked against the actual tools that
#: link a program, not a mock of them.
_TINY_KERNEL_LLVM = """builtin.module {
  llvm.func @gemmini_kernel(%0: !llvm.ptr, %1: !llvm.ptr) {
    llvm.return
  }
}
"""


@pytest.fixture()
def _real_toolchain(monkeypatch):
    """Use an already-built LLVM if one is resolvable (MERLIN_CLANG, or this repo's own toolchain
    convention); skip otherwise. Building one from scratch is a multi-GB, multi-hour step this test
    must not trigger on its own."""
    from merlin.llvmlower import toolchain

    try:
        clang = toolchain.clang()
    except Exception as error:  # noqa: BLE001 -- no toolchain resolvable at all
        pytest.skip(f"no LLVM toolchain resolvable (set MERLIN_CLANG): {error}")
    if not Path(clang).is_file():
        pytest.skip(f"no already-built LLVM at {clang} (set MERLIN_CLANG to reuse one rather than building it)")


def test_a_regions_members_compile_once_and_each_keep_their_own_symbol(_real_toolchain, tmp_path, monkeypatch):
    """Two rows sharing one region's artifact (exactly what `whole_program.emit_region` gives every
    member) must invoke the expensive compile step ONCE -- not once per member -- while still ending
    up with two real, independently linkable objects, each defining only ITS OWN renamed symbol and
    neither the original ``gemmini_kernel``. This is the "link it once, with the right args" contract
    for however many groups one region kernel answers.
    """
    from merlin.targetgen.contract import compile as compile_mod

    # The persistent object cache would serve a warm entry and hide the compile being counted.
    monkeypatch.setenv(W.OBJECT_CACHE_DISABLE_ENV, "1")
    artifact = tmp_path / "region_g5_g6.artifact.txt"
    artifact.write_text(_TINY_KERNEL_LLVM, encoding="utf-8")
    calls: list[Path] = []
    real = compile_mod.llvm_mlir_to_object

    def counting(text, work, *, target):
        calls.append(Path(work))
        return real(text, work, target=target)

    # `_kernel_objects` does `from ...compile import llvm_mlir_to_object` INSIDE the function body, so
    # it re-resolves this name from the module every call; patching the module attribute here is what
    # the function actually calls, with no need to reload anything.
    monkeypatch.setattr(compile_mod, "llvm_mlir_to_object", counting)
    rows = [
        {"group": 5, "op": "matmul", "on": W.ON_PACKAGE, "artifact": str(artifact), "region": {"id": "r"}},
        {"group": 6, "op": "residual_add", "on": W.ON_PACKAGE, "artifact": str(artifact), "region": {"id": "r"}},
    ]
    W._kernel_objects(rows, target=_TARGET, out=tmp_path / "objects", jobs=2)

    assert len(calls) == 1, f"the region's shared artifact was compiled {len(calls)} time(s), expected 1"
    for row in rows:
        assert row.get("cause") is None, row.get("why")
        assert row["symbol"] == f"gemmini_kernel_g{row['group']}"
        assert Path(row["object"]).is_file()
    assert rows[0]["object"] != rows[1]["object"], "each member still gets its own linkable object file"


# ------------------------------------------------------------------- fused regions, graded and scanned


class _ChainReference:
    """A LocalReference over a two-member region (g3 -> g4): g3 sums its input's two rows, g4 adds 1."""

    slices = {3: ([], "B_g3"), 4: ([], "B_g4")}

    def inputs_of(self, group):
        return {3: ["B_g2"], 4: ["B_g3"]}[group]

    def expected(self, group, inputs):
        import numpy as np

        if group == 3:
            return np.asarray(inputs["B_g2"]).reshape(2, 4).sum(axis=0)
        return np.asarray(inputs["B_g3"]) + 1


def _region_layout(memory):
    """A memory map whose g4 row is a fused region graded at its boundary, and a reader over ``memory``."""
    import numpy as np

    places = {}
    address = 0x1000
    for name, values, width in (("B_g2", memory["B_g2"], 1), ("B_g3", memory["B_g3"], 4), ("B_g4", memory["B_g4"], 4)):
        data = np.asarray(values, dtype=f"<i{width}").tobytes()
        places[name] = {
            "symbol": name,
            "address": address,
            "bytes": len(data),
            "elements": len(values),
            "element_bytes": width,
            "data": data,
        }
        address += 0x100
    row = {
        "group": 4,
        "kind": "region",
        **{k: v for k, v in places["B_g4"].items() if k != "data"},
        "inputs": [{**{k: v for k, v in places["B_g2"].items() if k != "data"}, "produced_by": 2}],
        "compare": "region",
        "members": [3, 4],
        "boundary": {"compare": "exact"},
    }
    by_address = {p["address"]: p["data"] for p in places.values()}
    return {"groups": [row]}, lambda address, size: by_address[address][:size]


def test_a_fused_region_is_graded_from_a_dump_at_its_boundary_and_a_wrong_one_is_caught():
    """HOST-SIDE: a region's internal member is recomputed from the region's dumped inputs -- its own
    buffer, which the region's kernel never wrote, is never read -- and the boundary is graded from it."""
    source = list(range(8))
    right = [s + 1 for s in (sum(source[i::4]) for i in range(4))]
    layout, read = _region_layout({"B_g2": source, "B_g3": [77] * 4, "B_g4": right})
    graded = W.grade_memory(read, layout, {"groups": {}}, local=_ChainReference())
    assert graded["agree"] == ["4"] and graded["quotable"]
    layout, read = _region_layout({"B_g2": source, "B_g3": [77] * 4, "B_g4": [*right[:3], right[3] + 2]})
    graded = W.grade_memory(read, layout, {"groups": {}}, local=_ChainReference())
    assert graded["disagree"] == [{"group": "4", "mismatches": 1, "of": 4, "first": 3}] and not graded["quotable"]
    # Without a local reference a region is unverified -- never agreed.
    assert W.grade_memory(read, layout, {"groups": {}})["unverified"][0]["group"] == "4"


def test_the_instruction_scan_covers_a_fused_regions_kernel(monkeypatch, tmp_path):
    """The no-FSM scan reads the WHOLE ELF: a prohibited instruction inside a region's one kernel (linked
    at its boundary) is found and attributed to that group, and the region's internal member -- answered
    by the same kernel -- is not mistaken for library code."""
    from merlin.perf import isa_prohibition as ISA

    elf, compiler = tmp_path / "program.elf", tmp_path / "riscv64-unknown-elf-gcc"
    elf.write_bytes(b"\x7fELF")
    (tmp_path / "g4.o").write_bytes(b"\x7fELF")
    record = {
        "program": {"compiler": str(compiler)},
        "linked_objects": [{"path": str(tmp_path / "g4.o")}],
        "attribution": {
            "per_group": [
                {"group": 2, "on": "vendor"},
                {"group": 3, "on": "package", "graded_at": 4},
                {"group": 4, "on": "package"},
            ]
        },
    }
    (tmp_path / "whole_model_build.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        ISA, "_declared_by_selector", lambda target: {9: {"name": "LOOP_WS", "roles": ["loop_descriptor"]}}
    )
    monkeypatch.setattr(ISA, "disassembler_for", lambda compiler: tmp_path / "riscv64-unknown-elf-objdump")
    monkeypatch.setattr(ISA, "custom_opcode", lambda target: 0x7B)
    monkeypatch.setattr(ISA, "defined_symbols", lambda obj, nm: {"merlin_kernel_g4"} if obj.name == "g4.o" else set())
    monkeypatch.setattr(
        ISA,
        "custom_instructions",
        lambda elf, objdump, opcode: [{"function": "merlin_kernel_g4", "selector": 9, "address": 16}],
    )
    report = ISA.check_build(
        {"elf": str(elf), "notes": {"build_record": str(tmp_path / "whole_model_build.json")}},
        target=_TARGET,
        roles=["loop_descriptor"],
    )
    assert not report["clean"] and report["summary"] == {"LOOP_WS in g4": 1}
    assert report["library_groups"] == ["2"], "a region's internal member is the package's, not library code"


def test_the_targets_driver_says_whether_it_links_a_fused_region():
    """The offer is gated on the TARGET's own declaration, read from its driver, never assumed."""
    from selected_driver import require_support

    require_support(_TARGET)
    facts = W.region_facts(_TARGET)
    assert facts["links"] is True and "matmul" in facts["internal_ops"] and "residual_add" not in facts["internal_ops"]


def test_a_fused_regions_expectations_grade_it_once_at_its_boundary(monkeypatch, tmp_path):
    """The verdict's expectations name exactly the lines the program prints: a fused region's internal
    member has NO line (its value is the kernel's own), its boundary keeps the oracle's digest of the
    boundary tensor and reads what the region reads from outside. Coverage still counts the member."""
    from selected_driver import require_support

    require_support(_TARGET)
    from merlin.perf import whole_model_builder as WB
    from merlin.perf import whole_model_verdict as V
    from merlin.runtime.backends import base as backends

    oracle = {
        "argmax": 3,
        "golden_argmax": 3,
        "groups": {
            "1": {"compare": "exact", "sum": 1, "fnv1a": 11},
            "2": {"compare": "bounded", "sum": 2, "fnv1a": 12, "bound_lsb": 1},
            "3": {"compare": "exact", "sum": 3, "fnv1a": 13},
            "4": {"compare": "exact", "sum": 4, "fnv1a": 14},
        },
    }
    (tmp_path / "oracle.json").write_text(json.dumps(oracle), encoding="utf-8")
    attribution = [
        {"group": 1, "op": "conv2d", "on": "vendor"},
        {"group": 2, "op": "residual_add", "on": "vendor"},
        {"group": 3, "op": "matmul", "on": "package", "region": {"role": "internal"}, "graded_at": 4},
        {"group": 4, "op": "matmul", "on": "package", "region": {"role": "boundary"}, "graded_at": 4},
    ]
    record = {
        "elf": "program.elf",
        "elf_sha256": "e" * 64,
        "program": {"abi_header": {"sha256": "a" * 64}},
        "oracle": {"path": str(tmp_path / "oracle.json")},
        "attribution": {"per_group": attribution, "counts": {}},
        "regions": {"stepped": [{"members": [3, 4], "boundary": 4}]},
    }
    steps = [
        {"kind": "conv2d", "group": 1, "in": "IMAGE", "out": "B_g1", "weight": "W_g1", "bias": "BIAS_g1", "relu": True,
         "scale": 0.05, "in_dim": 6, "ci": 2, "n": 4, "out_dim": 6, "stride": 1, "padding": 1, "kernel": 3,
         "pool": {"size": 0, "stride": 0, "padding": 0}},
        {"kind": "sum", "group": 2, "lhs": "B_g1", "rhs": "B_g1", "out": "B_g2", "rows": 36, "cols": 4,
         "lhs_load": 0.37, "rhs_load": 0.61, "readout": 1.0, "relu": False, "bound_lsb": 1},
        {"kind": "mean", "group": 3, "in": "B_g2", "out": "B_g3", "rows": 4, "window": 36, "multiplier": 0.04},
        {"kind": "matmul", "group": 4, "in": "B_g3", "out": "B_g4", "weight": "W_g4", "bias": None, "relu": False,
         "scale": None, "dequantize": 0.5, "m": 1, "k": 4, "n": 5},
    ]  # fmt: skip
    driver = backends.whole_model_driver(_TARGET)
    monkeypatch.setattr(driver.program, "extract", lambda *a, **k: {"steps": [dict(s) for s in steps]})
    monkeypatch.setattr(WB, "_output_widths", lambda out_dir: {})
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_open as WO

    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    monkeypatch.setattr(WMB, "build", lambda *a, **k: record)
    monkeypatch.setattr(
        WMB,
        "load_model_capsule",
        lambda path: SimpleNamespace(
            inputs={"x": 0}, outputs={"y": 0}, interface="i", weights_manifest="m", weights="w"
        ),
    )
    built = WB.build(None, target=_TARGET, out_dir=tmp_path, model_capsule="m", machine="mach", header="h")
    expectations = built["expectations"]
    assert set(expectations["groups"]) == {"1", "2", "4"} and expectations["group_count"] == 3
    assert expectations["graded_at_boundary"] == {"3": "4"}
    assert expectations["groups"]["4"]["fnv1a"] == 14 and expectations["groups"]["4"]["inputs_from"] == ["2"]
    V.Expectations.from_record(expectations)  # admissible: nothing reads from a group it does not name
    assert [r["on"] for r in built["groups"]] == ["vendor", "vendor", "package", "package"]
