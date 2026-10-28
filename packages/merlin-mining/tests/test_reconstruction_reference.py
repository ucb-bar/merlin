"""Automatic jobs derive frontend math and preserve real layout/scope boundaries."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.kernels.reference_inventory import SCHEMA, digest
from merlin.mining.reconstruction_cli import execute
from merlin.mining.reconstruction_reference import check_rhs_view, dense_source, dense_source_region


def problem():
    return dict(
        m=4,
        k=6,
        n=8,
        dtypes=["i8", "i8", "i32"],
        numeric=dict(
            accumulator_dtype="i32", bias="none", scale=1, activation="none", rounding="none", saturation="none"
        ),
    )


def test_dense_source_is_independently_admitted_and_keeps_numerical_boundary():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32

    source = dense_source(problem())
    region = dense_source_region(source)
    assert region.operations[0].attributes["k"] == 6
    assert region.complete and region.operations[0].numeric_complete
    constant = next(op for op in source.walk() if type(op) is arith.ConstantOp)
    constant.properties["value"] = IntegerAttr(1, i32)
    with pytest.raises(ValueError, match="exact integer zero"):
        dense_source_region(source)
    with pytest.raises(ValueError, match="epilogue"):
        dense_source({**problem(), "numeric": {**problem()["numeric"], "saturation": "signed8"}})


def test_packed_weight_view_is_a_bijection_without_same_buffer_or_preparation_claim():
    p = problem()
    panel = 4
    view = dict(
        kind="column_panels",
        panel=panel,
        physical_shape=[2, 6, panel],
        logical_shape=[6, 8],
        source_evidence=["source AST wp[n//P,k,n%P]; selected trace access sites"],
    )
    checked = check_rhs_view(p, view)
    logical = np.arange(48).reshape(6, 8)
    packed = logical.reshape(6, 2, panel).transpose(1, 0, 2).copy()
    for k in range(6):
        for n in range(8):
            assert packed[n // panel, k, n % panel] == logical[k, n]
    np.testing.assert_array_equal(packed.transpose(1, 0, 2).reshape(6, 8), logical)
    assert checked["checked"] and not checked["same_input_buffer"]
    assert checked["preparation_cost"] == "not established"
    identity = check_rhs_view(
        p, dict(kind="identity", physical_shape=[6, 8], source_evidence=["observed full row-major source accesses"])
    )
    assert identity["checked"] and identity["logical_shape"] == (6, 8)
    for wrong in (None, {**view, "panel": 3}, {**view, "source_evidence": []}):
        with pytest.raises(ValueError):
            check_rhs_view(p, wrong)


def selected_job(tmp_path, monkeypatch, *, extra_scope=True, native_counts=False):
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/tests/kernels"))
    capture_name = "capture_protocol_" + uuid.uuid4().hex
    capture_path = tmp_path / (capture_name + ".py")
    p = problem()
    capture_path.write_text(f"""
from merlin.kernels.cca import CCA, MemoryFacet, DispatchFacet
from merlin.mining.reconstruction_reference import dense_source, dense_source_region
def capture_reference(document,index,*,root,workspace):
    import importlib
    importlib.import_module(document['target_binding']['factory'].split(':')[0])
    workspace.mkdir(parents=True,exist_ok=True)
    problem={p!r}
    return dict(problem=problem,selected_reference=index['references'][0]['id'],
        semantic_region=dense_source_region(dense_source(problem)).to_dict(),
        rhs_view=dict(kind='column_panels',panel=4,physical_shape=[2,6,4],logical_shape=[6,8],
                      source_evidence=['synthetic source access witness']),
        cca=CCA(op='matmul',backend=['neutral'],memory=MemoryFacet(onchip_resident={'"a"' if extra_scope else "None"}),
                dispatch=DispatchFacet(n_dispatches=100) if {native_counts!r} else None,
                provenance={{'level':'native_issue'}} if {native_counts!r} else {{}}),
        evidence={{'synthetic':True}})
""")
    backend = tmp_path / "selected-backend"
    backend.mkdir()
    backend_name = "target_protocol_" + uuid.uuid4().hex
    backend_path = backend / (backend_name + ".py")
    backend_path.write_text("""
from test_reconstruction import adapter
def bind_dense(source,*,backend_root,output_dir=None):
    selected=adapter()
    from dataclasses import replace
    return replace(selected,source=source)
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    reference = tmp_path / "expert.py"
    reference.write_text("def kernel(a,b):\n    return source_matmul(a,b)\n")
    document = dict(
        schema=SCHEMA,
        mode="reference_guided",
        references=[
            dict(
                id="selected",
                family="dense",
                visibility="operator_only",
                format="python_schedule",
                artifact=dict(path=reference.name, sha256=digest(reference.read_bytes())),
            )
        ],
        capture_binding=dict(factory=capture_name + ":capture_reference", sha256=digest(capture_path.read_bytes())),
        target_binding=dict(
            factory=backend_name + ":bind_dense", backend_root=str(backend), sha256=digest(backend_path.read_bytes())
        ),
    )
    return document


def test_automatic_capture_target_binding_has_no_handwritten_region_graph(tmp_path, monkeypatch):
    document = selected_job(tmp_path, monkeypatch)
    result = execute("analyze", document, root=tmp_path, workspace=tmp_path / "job")
    assert result["status"] == "analyzed"
    assert result["region_comparison"]["status"] == "equivalent"
    auto = result["automatic_reference"]
    assert auto["selected_reference"] == "selected"
    assert auto["rhs_view"]["checked"] and not auto["rhs_view"]["same_input_buffer"]
    assert auto["measurement_status"].startswith("unknown")
    assert auto["mechanism_requirements"][0]["goal"] == "memory.lhs_retention_scope_covered"
    assert "reference_region" not in document and "source_region" not in document
    reference = result["inventory"]["references"][0]
    assert reference["gaps"] == []
    assert reference["resolved_gaps"][0]["next_action"] == "recover_semantic_region"
    assert reference["resolved_gaps"][0]["resolution"] == "complete source/trace semantic recovery"


def test_limited_panel_reuse_does_not_satisfy_unknown_expert_lifetime(tmp_path, monkeypatch):
    pytest.importorskip("z3")
    document = selected_job(tmp_path, monkeypatch)
    result = execute("reconstruct", document, root=tmp_path, workspace=tmp_path / "job", max_candidates=4)
    assert result["status"] == "unresolved"
    assert result["learning_tasks"]
    assert any("memory.lhs_retention_scope_covered" in row["goals"] for row in result["learning_tasks"])
    assert any(path["program_sha256"] for path in result["reconstruction"]["candidates"])
    assert all(not path["satisfied"] for path in result["reconstruction"]["candidates"])


def test_protocol_pin_and_missing_gate_owner_are_machine_readable(tmp_path, monkeypatch):
    document = selected_job(tmp_path, monkeypatch, extra_scope=False)
    document["capture_binding"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        execute("analyze", document, root=tmp_path, workspace=tmp_path / "bad-pin")
    capture_path = tmp_path / (document["capture_binding"]["factory"].split(":")[0] + ".py")
    document["capture_binding"]["sha256"] = digest(capture_path.read_bytes())
    result = execute("learn", document, root=tmp_path, workspace=tmp_path / "job")
    assert result["status"] == "unresolved"
    assert result["gaps"][0]["next_action"] == "bind_existing_source_numeric_storage_target_emission_owner"


def test_unsupported_captured_epilogue_is_an_automatic_adapter_gap(tmp_path, monkeypatch):
    document = selected_job(tmp_path, monkeypatch)
    name = document["capture_binding"]["factory"].split(":")[0]
    import importlib

    module = importlib.import_module(name)
    original = module.capture_reference

    def with_epilogue(*args, **kwargs):
        record = original(*args, **kwargs)
        record["problem"]["numeric"]["saturation"] = "signed8"
        return record

    monkeypatch.setattr(module, "capture_reference", with_epilogue)
    # The injected callback lives in this test module, so update its actual pin.
    document["capture_binding"]["sha256"] = digest(Path(__file__).read_bytes())
    result = execute("reconstruct", document, root=tmp_path, workspace=tmp_path / "unsupported")
    assert result["status"] == "unresolved"
    assert result["gaps"][0]["kind"] == "adapter_gap"
    assert "epilogue" in result["gaps"][0]["reason"]
    assert tuple(result["learning_tasks"][0]["goals"]) == ("numeric.complete_source_epilogue",)


def test_native_dispatch_counts_are_not_compared_with_logical_macro_counts(tmp_path, monkeypatch):
    pytest.importorskip("z3")
    document = selected_job(tmp_path, monkeypatch, extra_scope=False, native_counts=True)
    result = execute("reconstruct", document, root=tmp_path, workspace=tmp_path / "job", max_candidates=1)
    assert result["status"] == "reconstructed"
    row = result["reconstruction"]["candidates"][0]
    ours = row["observation"]["cca"]
    assert ours["dispatch"]["n_dispatches"] is None
    assert ours["provenance"]["uncompared_dispatch"]["value"] > 0
    assert any(
        axis == "dispatch.n_dispatches" and owner == "ours" for axis, owner in row["cca_comparison"]["unknown_axes"]
    )
    assert result["gaps"][0]["goals"] == ("dispatch.actual_instruction_projection",)
    assert any("dispatch.actual_instruction_projection" in task["goals"] for task in result["learning_tasks"])
