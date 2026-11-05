"""Original routed membership cannot shrink the ordinary required kernel set.

These controls check the actual sidecar/argument transport, not device effects
or source-body equivalence. A repeated call to one kernel is a valid route.
"""

import copy

import pytest

from merlin.llvmlower import device_offload as DO
from merlin.llvmlower.device_build import DeviceRouting


def _sidecar():
    return DO.DeviceRewrite(
        device="independent_fixture",
        granularity=DO.BY_GROUP,
        signatures={"first": (3, 7, 5), "second": (2, 4, 6)},
        routed=(
            DO.Routed("first", (3, 7), (5,), ("i8", "i8", "i8"), group=0),
            DO.Routed("second", (2, 4), (6,), ("i8", "i8", "i8"), group=1),
        ),
        entries={"first": {"op": "matmul", "epilogue": ["relu"]}, "second": {"op": "matmul"}},
    ).to_sidecar()


def test_complete_routed_membership_and_precision_are_preserved():
    sidecar = _sidecar()
    before = copy.deepcopy(sidecar)
    arguments = DO.build_arguments(sidecar)
    assert set(arguments["signatures"]) == {row["symbol"] for row in sidecar["routed"]}
    assert arguments["dtypes"] == {"first": ("i8", "i8", "i8"), "second": ("i8", "i8", "i8")}
    assert arguments["entries"] == sidecar["entries"]
    assert sidecar == before


@pytest.mark.parametrize("damage", ["missing_signature", "extra_signature", "missing_row", "empty_rows"])
def test_routed_membership_cannot_shrink_or_expand_the_required_kernel_set(damage):
    sidecar = _sidecar()
    if damage == "missing_signature":
        del sidecar["signatures"]["second"]
        del sidecar["entries"]["second"]
    elif damage == "extra_signature":
        sidecar["signatures"]["third"] = [2, 4, 6]
        sidecar["entries"]["third"] = {"op": "matmul"}
    elif damage == "missing_row":
        sidecar["routed"].pop()
    else:
        sidecar["routed"] = []
    with pytest.raises(ValueError, match="routed.*membership"):
        DO.build_arguments(sidecar)


def test_repeated_routed_calls_keep_one_consistent_kernel_without_losing_rows():
    sidecar = _sidecar()
    repeated = copy.deepcopy(sidecar["routed"][0])
    repeated["group"] = 2
    sidecar["routed"].append(repeated)
    before = copy.deepcopy(sidecar)
    arguments = DO.build_arguments(sidecar)
    assert len(sidecar["routed"]) == 3 and len(arguments["signatures"]) == 2
    assert arguments["dtypes"]["first"] == ("i8", "i8", "i8")
    assert sidecar == before


@pytest.mark.parametrize("conflict_first", [False, True])
def test_repeated_routed_symbol_cannot_select_the_last_conflicting_precision(conflict_first):
    sidecar = _sidecar()
    repeated = copy.deepcopy(sidecar["routed"][0])
    repeated["dtypes"] = ["i8", "i8", "i32"]
    sidecar["routed"].insert(0 if conflict_first else len(sidecar["routed"]), repeated)
    with pytest.raises(ValueError, match="routed.*precision"):
        DO.build_arguments(sidecar)


@pytest.mark.parametrize("dtypes", [None, [], ["i8", "i8"], ["i8", "i8", 8], "i32"])
def test_routed_precision_must_be_complete_original_metadata(dtypes):
    sidecar = _sidecar()
    sidecar["routed"][0]["dtypes"] = dtypes
    with pytest.raises(ValueError, match="routed.*precision"):
        DO.build_arguments(sidecar)


@pytest.mark.parametrize("symbol", [None, "", "undeclared"])
def test_every_routed_row_needs_its_declared_symbol(symbol):
    sidecar = _sidecar()
    sidecar["routed"][0]["symbol"] = symbol
    with pytest.raises(ValueError, match="routed.*membership"):
        DO.build_arguments(sidecar)


@pytest.mark.parametrize("changed", [DO.BY_CONTRACTION, None, "unknown"])
def test_selected_group_route_cannot_be_changed_to_contraction_fallback(changed):
    import ast
    import inspect
    import textwrap

    from merlin.runtime.backends import spike_model

    routing = DeviceRouting("independent_fixture", "unselected_package", "i8", "i8", granularity=DO.BY_GROUP)
    sidecar = _sidecar()
    sidecar["granularity"] = changed
    sidecar["entries"] = {}
    # Execute the ordinary caller's actual argument-construction statement,
    # without its compiler, provider or link boundaries. Omitting the original
    # routing selection here would accept a contraction downgrade.
    body = ast.parse(textwrap.dedent(inspect.getsource(spike_model.build)))
    statements = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_dev_args" for target in node.targets)
    ]
    assert len(statements) == 1
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    namespace = {"_device_build_arguments": DO.build_arguments, "_dev_side": sidecar, "device": routing}
    with pytest.raises(ValueError, match="selected.*granularity"):
        exec(compile(module, inspect.getsourcefile(spike_model.build), "exec"), namespace)


def test_legacy_contraction_metadata_matches_an_explicit_contraction_selection():
    sidecar = _sidecar()
    del sidecar["granularity"]
    sidecar["entries"] = {}
    routing = DeviceRouting("independent_fixture", "unselected_package", "i8", "i8")
    assert DO.build_arguments(sidecar, expected_granularity=routing.granularity)["entries"] is None


def test_empty_unselected_legacy_route_stays_inert():
    assert DO.build_arguments({}) == {"signatures": {}, "dtypes": {}, "entries": None, "call_buffers": None}
