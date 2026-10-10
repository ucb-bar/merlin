"""Keep original stated programs at the ordinary offload/build boundary.

These are structural transport controls. They establish no device semantics,
source-body equivalence, numerical permission or execution authority.
"""

import copy

import pytest

from merlin.llvmlower import device_offload as DO
from merlin.llvmlower.device_build import FROM_GROUP, kernel_entry


def _rewrite():
    return DO.DeviceRewrite(
        device="independent_fixture",
        granularity=DO.BY_GROUP,
        signatures={"first": (3, 7, 5), "second": (2, 4, 6)},
        routed=(
            DO.Routed("first", (3, 7), (5,), ("i8", "i8", "i8"), group=0),
            DO.Routed("second", (2, 4), (6,), ("i8", "i8", "i8"), group=1),
        ),
        entries={
            "first": {
                "name": "first",
                "op": "matmul",
                "M": 3,
                "N": 7,
                "K": 5,
                "epilogue": ["bias_add", "relu"],
                "acc_scale": 0.25,
                "result_dtype": "i8",
            },
            "second": {
                "name": "second",
                "op": "matmul",
                "M": 2,
                "N": 4,
                "K": 6,
                "epilogue": ["acc_scale"],
                "acc_scale": 0.5,
                "result_dtype": "i8",
            },
        },
    )


def _sidecar(tmp_path):
    rewrite = _rewrite()
    rewrite.write_sidecar(tmp_path)
    return rewrite, DO.load_sidecar(tmp_path)


def test_group_arguments_preserve_every_original_statement(tmp_path):
    rewrite, sidecar = _sidecar(tmp_path)
    sidecar["call_buffers"] = {
        "first": [
            {"role": "activation", "shape": [3, 5], "dtype": "i8"},
            {"role": "bias", "shape": [7], "dtype": "i32"},
            {"role": "output", "shape": [3, 7], "dtype": "i8"},
        ]
    }
    before = copy.deepcopy(sidecar)
    arguments = DO.build_arguments(sidecar)
    assert arguments["entries"] == rewrite.entries
    assert arguments["signatures"] == rewrite.signatures
    assert arguments["call_buffers"] == sidecar["call_buffers"]
    assert set(arguments["dtypes"]) == set(rewrite.signatures)
    for symbol, shape in arguments["signatures"].items():
        entry, provenance, refusal = kernel_entry(symbol, shape, arguments["entries"][symbol], rewrite.device)
        assert entry == rewrite.entries[symbol]
        assert provenance == FROM_GROUP and not refusal
    assert sidecar == before


@pytest.mark.parametrize("damage", ["absent", "null", "empty", "partial", "extra"])
def test_group_arguments_refuse_missing_or_changed_statement_roster(tmp_path, damage):
    _rewrite_value, sidecar = _sidecar(tmp_path)
    if damage == "absent":
        del sidecar["entries"]
    elif damage == "null":
        sidecar["entries"] = None
    elif damage == "empty":
        sidecar["entries"] = {}
    elif damage == "partial":
        del sidecar["entries"]["second"]
    else:
        sidecar["entries"]["unexpected"] = copy.deepcopy(sidecar["entries"]["first"])
    with pytest.raises(ValueError, match="stated group program"):
        DO.build_arguments(sidecar)


@pytest.mark.parametrize("entries", [False, [], "", {"first": None, "second": {}}, {"first": {}, "second": {}}])
def test_group_arguments_refuse_malformed_statements(tmp_path, entries):
    _rewrite_value, sidecar = _sidecar(tmp_path)
    sidecar["entries"] = entries
    with pytest.raises(ValueError, match="stated group program"):
        DO.build_arguments(sidecar)


@pytest.mark.parametrize("granularity", [DO.BY_CONTRACTION, None])
@pytest.mark.parametrize("entries", ["absent", None, {}])
def test_nongroup_arguments_preserve_legacy_unstated_contraction(tmp_path, granularity, entries):
    _rewrite_value, sidecar = _sidecar(tmp_path)
    if granularity is None:
        del sidecar["granularity"]
    else:
        sidecar["granularity"] = granularity
    if entries == "absent":
        del sidecar["entries"]
    else:
        sidecar["entries"] = entries
    assert DO.build_arguments(sidecar)["entries"] is None


def test_nongroup_explicit_statements_keep_existing_behavior(tmp_path):
    rewrite, sidecar = _sidecar(tmp_path)
    sidecar["granularity"] = DO.BY_CONTRACTION
    assert DO.build_arguments(sidecar)["entries"] == rewrite.entries


def test_empty_group_route_has_no_required_statement(tmp_path):
    DO.DeviceRewrite(device="independent_fixture", granularity=DO.BY_GROUP).write_sidecar(tmp_path)
    assert DO.build_arguments(DO.load_sidecar(tmp_path)) == {
        "signatures": {},
        "dtypes": {},
        "entries": {},
        "call_buffers": None,
    }


def test_missing_group_statement_refuses_before_package_or_synthesized_entry(tmp_path, monkeypatch):
    from merlin.llvmlower import device_build as DB

    _rewrite_value, sidecar = _sidecar(tmp_path)
    sidecar["entries"] = {}
    calls = []

    def synthesized(*_args, **_kwargs):
        calls.append("synthesized entry")
        raise AssertionError("a missing original group must not reach synthesized contraction emission")

    monkeypatch.setattr(DB, "kernel_entry", synthesized)
    # This is the ordinary caller sequence: decode all original build arguments
    # before starting the package/toolchain boundary.
    with pytest.raises(ValueError, match="stated group program"):
        arguments = DO.build_arguments(sidecar)
        DB.build_device_objects(
            "independent_fixture",
            **arguments,
            package_dir=tmp_path / "unselected_package",
            workdir=tmp_path / "build",
            operand_dtype="i8",
            accum_dtype="i32",
        )
    assert calls == []
