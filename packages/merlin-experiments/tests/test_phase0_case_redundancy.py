"""Exact duplicates collapse to one case; every contraction shape has exactly one performance scorer."""

from __future__ import annotations

import json

import yaml

from merlin_experiments.phase0 import case_redundancy as R


def matmul(name, m, k, n, *, epilogue=(), out="i32", stimulus=(-4, 3), lhs="A0", **extra):
    return {"name": name, "kind": "layer", "label": "public",
            "inputs": [{"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
                       {"name": lhs, "role": "input", "shape": [m, k], "dtype": "i8"}],
            "operation": {"op": "matmul", "attributes": {"lhs": lhs, "weight": "W", "out": "Y0",
                                                         "epilogue": list(epilogue), "output_dtype": out}},
            "numeric_policy": {"compare": "exact_int", "dtype": out},
            "expected": {"modes": {"relu": False}}, "stimulus_range": list(stimulus), **extra}


def conv(name, *, side, cin, cout, k, stride, pad):
    return {"name": name, "kind": "layer", "label": "public",
            "inputs": [{"name": "IFM", "role": "input", "shape": [1, side, side, cin], "dtype": "i8"},
                       {"name": "W", "role": "weight", "shape": [k * k * cin, cout], "dtype": "i8"}],
            "operation": {"op": "conv2d", "attributes": {"ifm": "IFM", "weight": "W", "out": "Y0", "kh": k, "kw": k,
                                                         "stride": [stride, stride], "padding": [pad] * 4,
                                                         "epilogue": [], "output_dtype": "i32"}},
            "numeric_policy": {"compare": "exact_int", "dtype": "i32"}}


def rows(*capsules, cohort="generated", program="p" * 64):
    return [{"name": c["name"], "cohort": cohort, "capsule": c, "program_sha256": program} for c in capsules]


def test_exact_duplicates_collapse_into_the_richest_and_labels_do_not_matter():
    record = R.derive(rows(matmul("G_a", 12, 48, 80), matmul("SY_b", 12, 48, 80, lhs="X"),
                           matmul("MF_c", 12, 48, 80, stimulus=(-104, 104))))
    # Same program and stimulus (operand label ignored) collapse; a wider stimulus is a different test.
    assert [(row["removed"], row["kept"]) for row in record["collapsed"]] == [("SY_b", "G_a")]
    assert record["performance_scope"] == {"G_a": "scored", "MF_c": "excluded"}
    assert record["excluded"][0]["reason"] == "shares_work_shape"


def test_models_and_cases_without_program_bytes_never_collapse():
    model = dict(matmul("SY_source_a", 4, 4, 4), kind="model")
    record = R.derive([*rows(model, dict(model, name="SY_source_b")),
                       {"name": "c", "cohort": "generated", "capsule": matmul("c", 4, 4, 4)},
                       {"name": "d", "cohort": "generated", "capsule": matmul("d", 4, 4, 4)}])
    assert record["collapsed"] == []


def test_family_points_are_never_collapsed_only_excluded():
    point = matmul("PK00", 16, 16, 16, performance={"family": "PK"})
    record = R.derive(rows(matmul("PF01", 16, 16, 16), point, matmul("PL00", 16, 16, 16)), protected={"PL00"})
    # The plain copy collapses into the family point; the protected point stays and is only excluded.
    assert [(row["removed"], row["kept"]) for row in record["collapsed"]] == [("PF01", "PK00")]
    assert record["performance_scope"] == {"PK00": "scored", "PL00": "excluded"}


def test_richer_semantics_score_a_shared_shape_and_a_convolution_counts_by_its_im2col_work():
    fused = matmul("fused", 100, 147, 12, epilogue=["bias_add"], out="i8")
    raw = matmul("raw", 100, 147, 12)
    stem = conv("stem", side=20, cin=3, cout=12, k=7, stride=2, pad=3)
    assert json.loads(R.work_shape(stem)) == ["contraction", [100, 147, 12]]
    record = R.derive(rows(raw, fused, stem))
    assert record["performance_scope"] == {"fused": "scored", "raw": "excluded", "stem": "excluded"}


def test_program_bound_cases_outside_the_derived_demand_do_not_score():
    demand = {"members": [{"name": "PD_big", "split": "public",
                           "represents": [{"model": "net", "program": "main"}]}]}
    cases = [{"name": "PD_big", "cohort": "scalability", "capsule": matmul("PD_big", 50, 720, 2048)},
             {"name": "JM_tiny", "cohort": "scalability", "capsule": matmul("JM_tiny", 50, 32, 720),
              "binding": {"model": "net", "program": "main"}},
             {"name": "other", "cohort": "scalability", "capsule": matmul("other", 8, 8, 8),
              "binding": {"model": "elsewhere", "program": "main"}}]
    record = R.derive(cases, demand=demand)
    assert record["performance_scope"] == {"JM_tiny": "excluded", "PD_big": "scored", "other": "scored"}


def test_generation_collapses_in_place_and_records_the_declared_size_classes(tmp_path):
    written = []
    for category, cap in (("layers", matmul("G_a", 12, 48, 80)), ("layers", matmul("SY_b", 12, 48, 80)),
                          ("layers", matmul("MF_c", 12, 48, 80, stimulus=(-9, 9))),
                          ("isa", matmul("SY_d", 3, 5, 7)), ("hidden", matmul("H_e", 12, 48, 80))):
        path = tmp_path / category / cap["name"]
        path.mkdir(parents=True)
        (path / "capsule.yaml").write_text(yaml.safe_dump({**cap, "interface_mlir": "p.mlir"}))
        (path / "p.mlir").write_text("program\n")
        written.append(path)
    summary = R.apply_to_generation(written, tmp_path, tile=16, size_policy={"edges": [1, 8], "intent": "x"})
    assert summary["collapsed"] == 1 and summary["performance_excluded"] == 1
    assert not (tmp_path / "layers/SY_b").exists() and tmp_path / "layers/SY_b" not in written
    assert (tmp_path / "hidden/H_e").exists()  # the hidden cohort is never read here
    assert yaml.safe_load((tmp_path / "layers/MF_c/capsule.yaml").read_text())["performance_scope"] == "excluded"
    record = json.loads((tmp_path / "case-redundancy.json").read_text())
    assert record["size_classes"]["classes"] == {"<=8t": 2, "sub_tile": 1}
    assert summary["size_classes"]["status"] == "declared"
    assert R.size_distribution([], tile=16, policy=None)["status"] == "undeclared"
