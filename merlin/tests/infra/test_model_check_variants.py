"""Shape variants keep each kernel's form, are a function of the seed alone, and are answered exactly."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from merlin.model_check import catalog as CAT
from merlin.model_check import kernel_reference as KR
from merlin.model_check import report as R
from merlin.model_check import variants as V

TARGET = "gemmini"  # the target whose readout routes these epilogues
ROUTING = {"operand_dtype": "int8", "accum_dtype": "i32", "numeric_policy": None}
#: The boundaries as data: these tests exercise the rule, not the facts (see test_limits_are_derived).
LIMITS = V.Limits(edge=16, operand_elements=262144, accumulator_elements=16384, sources={"test": "fixture"})
_BASE = {
    "kind": "op",
    "operand_dtype": "int8",
    "scale_granularity": "tensor",
    "cat": "layers",
    "label": "dev",
    "source_role": "model_derived",
    "source_reference": "test",
}
#: Model-like forms: strided stem, same-padded, strided pointwise, a mean (N = 1, narrowing readout),
#: a GEMV (M = 1), a runtime-stationary GEMM and a convolution with a bias + scale + relu readout.
_ENTRIES = {
    "k_0": {"op": "conv2d", "N": 64, "ci": 3, "Himg": 224, "Wimg": 224, "kh": 7, "kw": 7, "stride": [2, 2],
            "padding": [3, 3, 3, 3], "epilogue": []},
    "k_1": {"op": "conv2d", "N": 64, "ci": 64, "Himg": 56, "Wimg": 56, "kh": 3, "kw": 3, "stride": [1, 1],
            "padding": [1, 1, 1, 1], "epilogue": []},
    "k_2": {"op": "conv2d", "N": 512, "ci": 256, "Himg": 56, "Wimg": 56, "kh": 1, "kw": 1, "stride": [2, 2],
            "padding": [0, 0, 0, 0], "epilogue": []},
    "k_3": {"op": "matmul", "M": 2048, "K": 49, "N": 1, "epilogue": ["acc_scale"], "acc_scale": 0.0386802452319798},
    "k_4": {"op": "matmul", "M": 1, "K": 2048, "N": 1000, "epilogue": []},
    "k_5": {"op": "matmul", "M": 768, "K": 768, "N": 1024, "epilogue": [], "stationary_operand": "activation"},
    "k_6": {"op": "conv2d", "N": 24, "ci": 8, "Himg": 13, "Wimg": 11, "kh": 3, "kw": 3, "stride": [2, 1],
            "padding": [1, 1, 1, 1], "epilogue": ["bias_add", "acc_scale", "relu"], "acc_scale": 0.0137},
}  # fmt: skip
_SHAPE_KEYS = {"M", "K", "N", "ci", "Himg", "Wimg", "acc_scale", "name", "source_reference"}


def _catalog() -> CAT.ProgramCatalog:
    from merlin.llvmlower.device_shim import logical_buffers

    sidecar = {"signatures": {}, "entries": {}, "call_buffers": {}, "routed": []}
    interfaces = {}
    build = V.interface_builder(TARGET, ROUTING)
    for symbol, form in _ENTRIES.items():
        entry = {**_BASE, "name": symbol, **form}
        interface = build(entry)
        buffers, why = logical_buffers(interface)
        assert buffers is not None, why
        calls = [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers]
        dtypes = [b.dtype for b in buffers if b.role in ("input_0", "weight_0", "out_0")]
        sidecar["signatures"][symbol] = [form["M"], form["N"], form["K"]] if form["op"] == "matmul" else []
        sidecar["entries"][symbol] = entry
        sidecar["call_buffers"][symbol] = calls
        sidecar["routed"].append({"symbol": symbol, "dtypes": dtypes, "source_region_id": f"region_{symbol}"})
        interfaces[symbol] = interface
    return CAT.ProgramCatalog(TARGET, Path("."), Path("."), Path("."), dict(ROUTING), sidecar, interfaces)


@pytest.fixture(scope="module")
def catalog() -> CAT.ProgramCatalog:
    return _catalog()


@pytest.fixture(scope="module")
def built(catalog, tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("bundle")
    out = V.root(root, "m") / "p"
    V.build_program(catalog, out_dir=out, blobs=CAT.BlobStore(V.root(root, "m") / "blobs"), seed=11, model="m",
                    program="p", lim=LIMITS)  # fmt: skip
    return root


def _doc(root: Path) -> dict:
    return json.loads((V.root(root, "m") / "p" / "variants.json").read_text())


def test_every_kernel_gets_every_class(built):
    doc = _doc(built)
    by_kernel: dict[str, set[str]] = {}
    for row in doc["variants"]:
        by_kernel.setdefault(row["variant_of"], set()).add(row["class"])
    assert doc["skipped"] == []
    assert by_kernel == {symbol: set(V.CLASSES) for symbol in _ENTRIES}


def _by_form(root: Path) -> dict[tuple[str, str], tuple]:
    """``(varied kernel, class) -> (entry without its name, interface, operand keys, expected keys)``."""
    calls = {c["symbol"]: c for c in CAT.read_calls(V.root(root, "m") / "p" / "calls.json")["calls"]}
    out = {}
    for row in _doc(root)["variants"]:
        entry = {k: v for k, v in row["entry"].items() if k != "name"}
        call = calls[row["symbol"]]
        out[(row["variant_of"], row["class"])] = (entry, row["interface"], call["operands"], call["expected"])
    return out


def test_variants_are_a_function_of_the_seed_alone(catalog, built, tmp_path):
    """Same seed: the same variants, operands and results, whichever kernels are visited and in what order."""
    import dataclasses

    keep = ("k_6", "k_3")
    subset = dataclasses.replace(
        catalog,
        interfaces={s: catalog.interfaces[s] for s in keep},
        sidecar={**catalog.sidecar, "routed": [r for r in catalog.sidecar["routed"] if r["symbol"] in keep]},
    )
    for seed in (11, 12):
        V.build_program(subset, out_dir=V.root(tmp_path / str(seed), "m") / "p", seed=seed, model="m", program="p",
                        blobs=CAT.BlobStore(V.root(tmp_path / str(seed), "m") / "blobs"), lim=LIMITS)  # fmt: skip
    full, same, other = _by_form(built), _by_form(tmp_path / "11"), _by_form(tmp_path / "12")
    assert set(same) == set(other) == {(s, c) for s in keep for c in V.CLASSES}
    assert all(same[key] == full[key] for key in same)
    assert all(other[key][0] != same[key][0] for key in same)


def test_variants_keep_the_form_and_change_the_sizes(catalog, built):
    edge, restate = LIMITS.edge, V.interface_builder(TARGET, ROUTING)
    for row in _doc(built)["variants"]:
        own, entry = {**_BASE, "name": row["variant_of"], **_ENTRIES[row["variant_of"]]}, row["entry"]
        # Every key that is not a size is carried verbatim: op, epilogue, window, stationary operand, dtypes.
        assert {k: v for k, v in entry.items() if k not in _SHAPE_KEYS} == {
            k: v for k, v in own.items() if k not in _SHAPE_KEYS
        }
        keys = ("M", "K", "N") if own["op"] == "matmul" else ("ci", "N")
        for key in keys:
            if own[key] == 1:
                assert entry[key] == 1, (row["class"], key)
            else:
                assert entry[key] >= 2 and entry[key] % edge and entry[key] != own[key], (row["class"], key)
        if own["op"] == "conv2d":
            assert entry["Himg"] != entry["Wimg"] and (entry["Himg"], entry["Wimg"]) != (own["Himg"], own["Wimg"])
            assert entry["Himg"] > max(own["padding"][0::2]) and entry["Wimg"] > max(own["padding"][1::2])
        assert row["macs"] <= V.MAX_MACS and row["operand_elements"] <= V.MAX_OPERAND_ELEMENTS
        assert [(b["role"], b["dtype"]) for b in row["call_buffers"]] == [
            (b["role"], b["dtype"]) for b in catalog.sidecar["call_buffers"][row["variant_of"]]
        ]
        if "acc_scale" in own:
            scale = entry["acc_scale"]
            assert float(np.float32(scale)) == scale and scale != own["acc_scale"]
            assert V.SCALE_STEP[0] * 0.9 <= abs(scale / own["acc_scale"] - 1) <= V.SCALE_STEP[1] * 1.1
        # The stored interface is exactly what a device build restates the stored entry as.
        assert restate(entry) == row["interface"]
    assert KR.interface_for(entry, target=TARGET, **ROUTING) == row["interface"]  # the build's own path


def test_each_class_lands_in_its_regime(built):
    edge = LIMITS.edge
    for row in _doc(built)["variants"]:
        entry, regimes = row["entry"], row["regimes"]
        if row["class"] == "sub_tile":
            sizes = [entry[k] for k in ("M", "K", "N")] if entry["op"] == "matmul" else [entry["ci"], entry["N"]]
            assert all(v < edge for v in sizes)
        elif row["class"] == "accumulator_spill":
            assert V.SPILL_RATIO[0] <= regimes["accumulator_ratio"] <= V.SPILL_RATIO[1] + 0.05
        elif row["class"] == "operand_spill":
            assert V.SPILL_RATIO[0] <= regimes["operand_ratio"] <= V.SPILL_RATIO[1] + 0.05


def test_expected_results_equal_the_integer_reference(built):
    """``call_outputs`` (numpy) and the integer reference engine agree on every small stored variant."""
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.runtime import reference as REF
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    doc = _doc(built)
    blobs = CAT.BlobStore(V.root(built, "m") / "blobs")
    calls = {c["symbol"]: c for c in CAT.read_calls(V.root(built, "m") / "p" / "calls.json")["calls"]}
    checked = 0
    for row in doc["variants"]:
        if row["macs"] > 40000 and row["class"] != "sub_tile":
            continue
        variant, call = V.Variant.from_record(row), calls[row["symbol"]]
        values = dict(zip(call["roles"], (blobs.get(k) for k in call["operands"]), strict=True))
        expected = [blobs.get(k) for k in call["expected"]]
        cb = parse_interface_mlir(variant.interface)
        buffers, _ = logical_buffers(variant.interface)
        leaves = {b.name: np.asarray(values[b.role]).reshape(b.shape).tolist() for b in buffers if b.kind == "input"}
        slow = REF.reference_outputs(cb, leaves)
        outs = [b for b in buffers if b.kind == "output"]
        assert len(outs) == len(expected)
        for buffer, want in zip(outs, expected, strict=True):
            got = np.asarray(slow[buffer.name], dtype=np.int64).reshape(-1)
            assert np.array_equal(got, want.astype(np.int64).reshape(-1)), row["symbol"]
        checked += 1
    assert checked >= len(_ENTRIES)  # at least every sub_tile variant


def test_operands_span_the_range_and_never_overflow(built):
    doc = _doc(built)
    blobs = CAT.BlobStore(V.root(built, "m") / "blobs")
    calls = {c["symbol"]: c for c in CAT.read_calls(V.root(built, "m") / "p" / "calls.json")["calls"]}
    for row in doc["variants"]:
        variant, call = V.Variant.from_record(row), calls[row["symbol"]]
        values = dict(zip(call["roles"], (blobs.get(k) for k in call["operands"]), strict=True))
        bounds = V.amplitudes(variant, accum_dtype="i32")
        act, weight = (
            np.abs(values["input_0"].astype(np.int64)).max(),
            np.abs(values["weight_0"].astype(np.int64)).max(),
        )
        reduction = (
            row["entry"]["K"]
            if row["entry"]["op"] == "matmul"
            else row["entry"]["kh"] * row["entry"]["kw"] * row["entry"]["ci"]
        )
        bias = np.abs(values["bias_0"].astype(np.int64)).max() if "bias_0" in values else 0
        assert reduction * act * weight + bias < 2**31
        if bounds["activation"] == 128 and values["input_0"].size >= 2 * V.EXTREME_ELEMENTS:
            assert values["input_0"].min() == -128 and values["input_0"].max() == 127
        if row["entry"].get("epilogue") == ["acc_scale"]:  # a narrowing readout is not all saturated
            out = blobs.get(call["expected"][0]).astype(np.int64)
            assert np.mean(np.abs(out) >= 127) < 0.5


def test_stored_variants_load_as_sealed_cases(catalog, built):
    varied = V.load(built, "m", "p", catalog)
    assert varied is not None and len(varied.cases()) == len(_ENTRIES) * len(V.CLASSES)
    assert set(varied.catalog.interfaces) == set(varied.catalog.sidecar["entries"]) == set(varied.variants)
    assert not set(varied.catalog.interfaces) & set(catalog.interfaces)  # never a recorded symbol's name
    assert {c.label.split("/")[0] for c in varied.cases()} == {V.SOURCE}
    assert len(varied.cases(group="region_k_3")) == len(V.CLASSES) == len(varied.cases(group="k_3"))
    case = varied.cases(group="k_4")[0]
    want = KR.call_outputs(varied.catalog.interfaces[case.symbol], list(case.operands), list(case.roles))
    assert all(np.array_equal(a, b) for a, b in zip(want, case.expected, strict=True))
    kernels = [{"symbol": case.symbol, "status": "pass"}]
    varied.annotate(kernels)
    assert kernels[0]["variant_of"] == "k_4" and kernels[0]["variant_class"] in V.CLASSES
    assert V.load(built, "m", "absent", catalog) is None


def test_agent_view_keeps_only_aggregates_of_variants():
    rep = R.new_report("check", model="m", split="sealed", submission={})
    rep["programs"] = [
        {"program": "p", "kernels": [{"symbol": "k_0", "status": "pass"}]},
        {
            "program": "p",
            "split": "sealed",
            "source": V.SOURCE,
            "kernels": [{"symbol": "k_9", "status": "fail", "mismatches": ["sealed_variants/p/k_9#0"]}],
            "refused": {"k_10": "package: M=37 is not a multiple of the tile"},
        },
    ]
    rep["failures"] = [{"kind": "gate", "gate": "kernel_exact", "split": "sealed", "detail": "sealed_variants/p/k_9#0"}]
    view = R.agent_view(rep)
    assert view["programs"][0]["kernels"] == [{"symbol": "k_0", "status": "pass"}]
    assert view["programs"][1]["kernels"] == [] and view["programs"][1]["summary"] == {"fail": 1}
    assert "refused" not in view["programs"][1] and "detail" not in view["failures"][0]


def test_variants_build_parses():
    from merlin.model_check import cli

    args = cli.build_parser().parse_args(["variants-build", "--model", "m", "--seed", "3", "--out", "o.json"])
    assert args.command == "variants-build" and args.seed == 3


def test_limits_are_derived_from_the_facts():
    try:
        lim = V.limits(TARGET, "int8")
    except Exception as error:  # noqa: BLE001 -- facts not selected in this environment
        pytest.skip(f"no RTL facts for {TARGET}: {error}")
    assert lim.edge > 2 and lim.operand_elements > lim.edge**2 and lim.accumulator_elements > lim.edge**2
    assert all(lim.sources.values())


def test_a_scale_step_is_exact_in_float32():
    rng = np.random.default_rng(0)
    for value in (0.0386802452319798, 0.0137, 1.7, 3.1e-5):
        moved = V._perturbed_scale(rng, value)
        assert float(np.float32(moved)) == moved and moved != value
        assert math.isclose(moved, value, rel_tol=V.SCALE_STEP[1] * 1.1)


def test_variant_replays_build_beside_the_private_bundle(tmp_path):
    """A variant replay's interfaces and images restate its shapes: they never go under ``--work``."""
    bundle = tmp_path / "private" / "v1"
    assert V.scratch(bundle, "m") == bundle.resolve().parent / f"v1.{V.SOURCE}-work" / "m"
    assert bundle.resolve().parent in V.scratch(bundle, "m").parents
