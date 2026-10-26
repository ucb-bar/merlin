"""Coverage-table replay uses byte-pinned source suites, never a compiler verdict.

Put corpus_193.json, bounded.json and facts.json in
out/artifacts/core-aten-coverage, or set MERLIN_CORE_ATEN_COVERAGE_INPUTS to their
directory. These large operator-owned source artifacts are not vendored here.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import artifacts_dir, repo_root


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _import(repo_root() / "build_tools/scripts/make_core_aten_coverage.py", "coverage_generator")


@pytest.fixture(scope="module")
def target_inputs():
    return {
        "target": "gemmini",
        "descriptor": repo_root() / "examples/gemmini/target/descriptor.yaml",
        "host_capabilities": repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml",
    }


@pytest.fixture(scope="module")
def table():
    """Read generated JSON without importing torch or executing the loader."""
    source = repo_root() / "examples/workloads/core_aten_coverage/loader.py"
    result = {}
    for node in ast.parse(source.read_bytes()).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in {
            "ROWS",
            "CENSUS",
            "SUITE_SHA256",
            "ADMISSION_SHA256",
        }:
            calls = node.value.elts if isinstance(node.value, ast.List) else [node.value]
            values = [json.loads(ast.literal_eval(call.args[0])) for call in calls]
            result[node.targets[0].id] = values if isinstance(node.value, ast.List) else values[0]
    return result


def test_generated_table_equals_derivation(generator, table, target_inputs):
    directory = Path(os.environ.get("MERLIN_CORE_ATEN_COVERAGE_INPUTS", artifacts_dir() / "core-aten-coverage"))
    paths = {key: directory / f"{key}.json" for key in ("corpus_193", "bounded")}
    if any(not path.is_file() for path in [*paths.values(), directory / "facts.json"]):
        pytest.skip("full source-suite replay requires MERLIN_CORE_ATEN_COVERAGE_INPUTS (193, bounded, facts)")
    assert {key: generator.sha(path) for key, path in paths.items()} == table["SUITE_SHA256"]
    assert generator.sha(directory / "facts.json") == table["ADMISSION_SHA256"]["facts"]
    suites = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    selected = generator.admission(directory / "facts.json", **target_inputs)
    excluded = generator.excluded_cores(target_inputs["descriptor"])
    missing, rows = generator.select_rows(suites["corpus_193"], suites["bounded"], selected, table["CENSUS"], excluded)
    assert rows == table["ROWS"]
    assert sorted({row["overload"] for row in rows}) == missing
    # File/list order is not a case-selection tie breaker.
    suites["corpus_193"]["cases"].reverse()
    suites["bounded"]["selected_cases"].reverse()
    assert generator.select_rows(suites["corpus_193"], suites["bounded"], selected, table["CENSUS"], excluded) == (
        missing,
        rows,
    )
    print(
        json.dumps(
            {
                "missing": missing,
                "cases": [{key: row[key] for key in ("overload", "dtype", "source", "case_id")} for row in rows],
            },
            indent=2,
        )
    )


def test_census_is_bound_to_the_existing_roster(generator, table):
    target = repo_root() / "examples/gemmini/target"
    applications = yaml.safe_load((target / "phase0-workloads.yaml").read_bytes())["applications"]
    descriptor = yaml.safe_load((target / "descriptor.yaml").read_bytes())
    for key, path in (
        ("software_spec", target / "software-spec.yaml"),
        ("host_capabilities", target.parent / "platform/host-capabilities-core-aten.yaml"),
        ("contract", target / "contracts/target_contract.yaml"),
    ):
        assert generator.sha(path) == table["ADMISSION_SHA256"][key]
    assert set(applications) == set(descriptor["workload_spec"]["applications"])
    previous = {label for label in applications if not label.startswith("core_aten_coverage_")}
    assert set(table["CENSUS"]) == previous
    for label, census in table["CENSUS"].items():
        entry = applications[label]
        assert census["loader_sha256"] == generator.sha(target / entry["loader"])
        assert census["profile"] == entry.get("profile")
    groups = {row["group"] for row in table["ROWS"]}
    assert {entry["profile"]["group"] for label, entry in applications.items() if label not in previous} == groups


def test_every_tensor_stays_outside_held_out_shapes(generator, table, target_inputs):
    excluded = generator.excluded_cores(target_inputs["descriptor"])
    operands = []
    for row in table["ROWS"]:
        assert generator.boundary_safe(row, excluded), (row["source"], row["case_id"])
        operands.extend(generator.tensors(row["arguments"]))
    assert not generator.held_out(generator.capsule_document("all_workload_inputs", operands), excluded)
    # A hidden layer cannot be disguised by operand order, dtype, optional bias,
    # a unit batch, or by declaring its weight an auxiliary input.
    core = next(core for core in excluded if all(core) and sum(core.values()) >= 2)
    hidden = [{"shape": [1, *shape]} for shape, count in core.items() for _ in range(count)]
    document = generator.capsule_document("renamed", [*hidden, {"shape": [7]}])
    assert generator.held_out(document, excluded)


def test_selection_skips_held_out_cases_and_breaks_ties_by_case_id(generator, monkeypatch):
    def case(identity, dtype, shape):
        return {
            "overload": "aten.alias.default",
            "case_id": identity,
            "arguments": {"args": [{"kind": "tensor", "dtype": dtype, "shape": shape}]},
        }

    monkeypatch.setattr(
        generator, "decision", lambda *_: {"status": "admitted", "observed_signature": {"family": "movement"}}
    )
    canonical = {"cases": [case("canonical", "float32", [4, 5])]}
    bounded = {
        "selected_cases": [case("z", "int8", [3, 3]), case("held_out", "int8", [2, 2]), case("a", "int8", [3, 3])]
    }
    missing, rows = generator.select_rows(canonical, bounded, {}, {}, [generator.Counter({(2, 2): 1})])
    assert missing == ["aten.alias.default"]
    assert [(row["source"], row["case_id"]) for row in rows] == [("corpus_193", "canonical"), ("bounded", "a")]


def test_export_covers_missing_overloads_and_preserves_all_operands(generator, table, monkeypatch, target_inputs):
    torch = pytest.importorskip("torch", reason="torch is required to export the iteration workload")
    from merlin.targetgen.core_aten_cases import decode_arguments

    assert generator.workload_census(target_inputs["descriptor"]) == table["CENSUS"]
    module = _import(repo_root() / "examples/workloads/core_aten_coverage/loader.py", "core_coverage")
    seen = set()
    for group in module.GROUPS:
        monkeypatch.setattr(module, "_selected_profile", lambda group=group: ({"groups": (group,)}, None))
        model, inputs = module.get_model_and_inputs()
        assert not list(model.parameters()) and not list(model.buffers())
        rows = [row for row in table["ROWS"] if row["group"] == group]
        expected_inputs = [item for row in rows for item in generator.tensors(row["arguments"])]
        assert len(inputs) == len(expected_inputs)
        for actual, recorded in zip(inputs, expected_inputs, strict=True):
            assert list(actual.shape) == recorded["shape"]
            assert list(actual.stride()) == recorded["stride"]
            assert actual.storage_offset() == recorded["storage_offset"]
            assert str(actual.dtype) == "torch." + recorded["dtype"]
        with torch.no_grad():
            expected = []
            for row in rows:
                args, kwargs = decode_arguments(row["arguments"])
                expected.extend(module._flatten(module._op(row["overload"])(*args, **kwargs)))
            torch.testing.assert_close(model(*inputs), tuple(expected), rtol=0, atol=0, equal_nan=True)
            exported = torch.export.export(model, inputs)
        ops = {str(node.target) for node in exported.graph.nodes}
        assert {row["overload"] for row in rows} <= ops
        live = set()
        pending = [node for node in exported.graph.nodes if node.op == "output"]
        while pending:
            node = pending.pop()
            if node not in live:
                live.add(node)
                pending.extend(node.all_input_nodes)
        assert all(
            node in live for node in exported.graph.nodes if str(node.target) in {row["overload"] for row in rows}
        )
        seen.update(ops)
        actual_documents = [
            {"shape": list(tensor.shape)} for tensor in [*inputs, *model.parameters(), *model.buffers()]
        ]
        assert not generator.held_out(
            generator.capsule_document(group, actual_documents), generator.excluded_cores(target_inputs["descriptor"])
        )
    assert {row["overload"] for row in table["ROWS"]} <= seen


def test_mixed_outputs_preserve_integer_precision_and_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    module = _import(repo_root() / "examples/workloads/core_aten_coverage/loader.py", "typed_coverage")
    monkeypatch.setattr(module, "_selected_profile", lambda: ({"groups": module.GROUPS}, None))
    model, inputs = module.get_model_and_inputs()
    outputs = model(*inputs)
    assert isinstance(outputs, tuple)
    rows = [row for row in module.ROWS if row["group"] in module.GROUPS]
    offset = 0
    found_repeat = False
    from merlin.targetgen.core_aten_cases import decode_arguments

    for row in rows:
        args, kwargs = decode_arguments(row["arguments"])
        expected = module._flatten(module._op(row["overload"])(*args, **kwargs))
        actual = outputs[offset : offset + len(expected)]
        torch.testing.assert_close(actual, tuple(expected), rtol=0, atol=0, equal_nan=True)
        if row["overload"] == "aten.repeat.default" and row["dtype"] == "int64":
            assert actual[0].dtype == torch.int64
            assert 9223372036854775807 in actual[0].tolist()
            found_repeat = True
        offset += len(expected)
    assert offset == len(outputs)
    assert found_repeat


def test_coverage_generator_passes_target_literal_gate():
    import runpy

    gate = runpy.run_path(str(repo_root() / "build_tools/scripts/check_no_target_name.py"))
    assert gate["_scan_file"](repo_root() / "build_tools/scripts/make_core_aten_coverage.py") == []


@pytest.mark.parametrize("target", ["device_a", "device_b"])
def test_generator_uses_explicit_target_and_host_selection(generator, tmp_path, monkeypatch, target):
    directory = tmp_path / target
    (directory / "contracts").mkdir(parents=True)
    contract = {"name": target, "compute_units": []}
    (directory / "contracts/target_contract.yaml").write_text(yaml.safe_dump(contract))
    facts, host, descriptor = (directory / name for name in ("facts.json", "host.yaml", "descriptor.yaml"))
    seen = []

    def select(selected_target, **kwargs):
        from merlin.targetgen.target_registry import load_contract

        assert load_contract(selected_target) == contract
        seen.append((selected_target, kwargs))
        return {"selected": selected_target}

    monkeypatch.setattr(generator, "select_admission", select)
    assert generator.admission(facts, target=target, descriptor=descriptor, host_capabilities=host) == {
        "selected": target
    }
    assert seen == [
        (target, {"rtl_facts": facts, "software_spec": directory / "software-spec.yaml", "host_capabilities": host})
    ]


def test_generator_reproduces_committed_section_without_suite_replay(generator, table):
    source = (repo_root() / "examples/workloads/core_aten_coverage/loader.py").read_text()
    committed = source.split(generator.BEGIN)[1].split(generator.END)[0]
    regenerated = generator.generated_section(
        table["ROWS"], table["CENSUS"], table["SUITE_SHA256"], table["ADMISSION_SHA256"]
    )
    assert ast.dump(ast.parse(regenerated)) == ast.dump(ast.parse(committed))
