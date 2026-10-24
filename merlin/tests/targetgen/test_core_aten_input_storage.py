"""Native raw reads use documented zero-initialized backing storage."""

import copy
import importlib.util
import itertools
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.capture_refusals import InputStorageUnavailable
from merlin.targetgen.core_aten_capture import case_capture_name, loader_source
from merlin.targetgen.core_aten_input_storage import native_input_storage_plan

OVERLOAD = "aten.native_layer_norm_backward.default"


def tensor(shape, stride, offset=0):
    def values(dims):
        return [values(dims[1:]) for _ in range(dims[0])] if dims else 1.0

    return dict(
        kind="tensor",
        dtype="float32",
        shape=shape,
        stride=stride,
        storage_offset=offset,
        requires_grad=False,
        values=values(shape),
    )


def case(shape, stride, *, rows=2, offset=0):
    x = tensor([rows, 3], [3, 1])
    stats = tensor(shape, stride, offset)
    return dict(
        overload=OVERLOAD,
        case_id="independent-storage-case",
        rng_seed=0,
        arguments=dict(
            args=dict(kind="tuple", items=[x, x, [3], stats, copy.deepcopy(stats), None, None, [True] * 3]), kwargs={}
        ),
        expected=dict(kind="tuple", items=[]),
    )


@pytest.mark.parametrize("shape,stride", [([2], [1]), ([2, 2], [1, 2]), ([2, 2], [4, 1]), ([], [])])
@pytest.mark.parametrize("offset", [0, 5])
def test_dense_or_partially_read_storage_is_covered(shape, stride, offset):
    document = case(shape, stride, rows=1 if not shape else 2, offset=offset)
    before = copy.deepcopy(document)
    compile(loader_source(document), "<covered input storage>", "exec")
    assert document == before


@pytest.mark.parametrize("stride,cause", [([0], "raw_read_out_of_bounds"), ([2], "unrecorded_storage_hole")])
def test_native_span_is_source_derived_and_zero_filled(stride, cause):
    import numpy as np

    from merlin.llvmlower.input_layout import storage_array

    document = case([2], stride, offset=4)
    del document["expected"]  # The convention never consults expected outputs.
    plan = native_input_storage_plan(OVERLOAD, document["arguments"])
    assert plan["unrecorded_bytes"] == "zero"
    assert [r["input_index"] for r in plan["raw_reads"]] == [2, 3]
    for read in plan["raw_reads"]:
        assert read["minimum_storage_elements"] == 6
        assert read["previously_unrecorded"]["cause"] == cause
        record = dict(shape=[2], stride=stride, storage_offset=4, minimum_storage_elements=6)
        backing = storage_array(np.array([1, 1], dtype=np.float32), record)
        assert backing.size >= 6
        assert backing[4] == 1 and backing[5] == 0
        assert np.array_equal(backing[:4], np.zeros(4, dtype=np.float32))


def test_invalid_native_read_geometry_still_refuses():
    document = case([2], [-1])
    with pytest.raises(InputStorageUnavailable):
        loader_source(document)


def test_keyword_and_tuple_normalized_arguments_preserve_coverage():
    document = case([2], [1])
    args = document["arguments"]["args"]["items"]
    document["arguments"] = dict(
        args=dict(kind="tuple", items=[args[0]]),
        kwargs=dict(input=args[1], normalized_shape=dict(kind="tuple", items=[3]), mean=args[3], rstd=args[4]),
    )
    native_input_storage_plan(OVERLOAD, document["arguments"])


def test_device_metadata_does_not_change_cpu_storage_convention():
    document = case([2], [0])
    document["arguments"]["args"]["items"][1]["device"] = "cuda"
    plan = native_input_storage_plan(OVERLOAD, document["arguments"])
    assert plan["raw_reads"][0]["minimum_storage_elements"] == 2
    assert native_input_storage_plan("aten.alias.default", {}) is None


def test_coverage_matches_independent_address_enumeration():
    from merlin.targetgen.core_aten_input_storage import _raw_coverage

    for shape in ([], [0], [1], [2], [3], [2, 2], [2, 3], [1, 2, 2]):
        for stride in itertools.product(range(4), repeat=len(shape)):
            addresses = {
                sum(i * s for i, s in zip(index, stride)) for index in itertools.product(*(range(n) for n in shape))
            }
            for rows in range(6):
                assert (_raw_coverage(tensor(shape, list(stride)), rows) is None) == (set(range(rows)) <= addresses)


def test_native_batch_provenance_binds_input_qualification_sources(tmp_path):
    from merlin.targetgen.core_aten_provenance import batch_provenance

    receipt = batch_provenance(runner_options={"spike_binary": str(tmp_path / "absent")})
    for filename in ("core_aten_capture.py", "core_aten_input_storage.py", "capture_refusals.py"):
        assert str(repo_root() / "src/merlin/targetgen" / filename) in receipt["sources"]


@pytest.mark.parametrize("runner", [False, True])
@pytest.mark.parametrize("stride", [[0], [2]])
def test_capture_paths_record_storage_convention(tmp_path, monkeypatch, runner, stride):
    from merlin.targetgen import capsule_source, core_aten_capture, core_aten_cover

    calls = []

    class Source:
        m2m_dir = tmp_path

        def __init__(self, **kwargs):
            pass

        def capture_loader(self, loader, *args, **kwargs):
            calls.append(loader)
            from xdsl.dialects.builtin import StringAttr

            layout = StringAttr(json.dumps({"inputs": [{}, {}, {}, {}], "results": []}))
            return SimpleNamespace(
                linalg_mlir=f"builtin.module attributes {{m2m.result_layout = {layout}}} {{}}",
                meta={}, inputs=[], golden=[],
            )

    monkeypatch.setattr(capsule_source, "PytorchRefSource", Source)
    monkeypatch.setattr(core_aten_capture, "PytorchRefSource", Source)
    monkeypatch.setattr(core_aten_capture, "_git_revision", lambda _: None)
    for module in (core_aten_capture, core_aten_cover):
        monkeypatch.setattr(module, "_provenance_ops", lambda _: {"aten.alias.default"})
    bad = case([2], stride)
    good = dict(
        overload="aten.alias.default",
        case_id="next-case",
        rng_seed=0,
        arguments=dict(args=dict(kind="tuple", items=[tensor([2], [1])]), kwargs={}),
        expected=dict(kind="tensor"),
    )
    if runner:
        spec = importlib.util.spec_from_file_location(
            "storage_capture_runner", repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = Namespace(
            output=tmp_path,
            m2m_dir=tmp_path,
            m2m_python=sys.executable,
            capture_timeout=1,
            label="track",
            numeric_policy="fp32",
        )
        monkeypatch.setenv("PYTHONPATH", "unchanged test environment")
        for document in (bad, good):
            module.capture(args, document, {})
        root = tmp_path / "captures"

    else:
        result = core_aten_capture.capture_case_corpus(dict(cases=[bad, good], pytorch_version="fixture"), tmp_path)
        assert result["case_count"] == 2
        assert result["status_counts"] == {"captured_lost_exact_provenance": 1, "captured_exact": 1}
        root = tmp_path
    recorded = json.loads((root / case_capture_name(bad) / "capture.json").read_text())
    assert recorded["status"] == "captured_lost_exact_provenance"
    assert recorded["input_storage_convention"]["unrecorded_bytes"] == "zero"
    assert recorded["input_storage_convention"]["raw_reads"][0]["element_count"] == 2
    assert json.loads((root / case_capture_name(good) / "capture.json").read_text())["status"] == "captured_exact"
    assert calls == [root / case_capture_name(c) / "capsule.pytorch.py" for c in (bad, good)]
