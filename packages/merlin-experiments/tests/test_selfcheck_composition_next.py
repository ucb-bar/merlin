"""composition_next: composition routes and source-composition requests inside each case's ``optimization_next``.

A chain case (product -> pool or epilogue, residual add, shared-input siblings) gets one advisory line
naming the route, its library symbol, whether ``submission/`` imports it and a paste-ready command,
beside the library pass rather than in place of it. A model or program case also gets the
``source_composition.json`` options its IR or declaration supports, with a file the owner's reader
accepts. Host-compute flags put the epilogue routes first. Nothing here gates, ranks or claims a
speedup, and nothing private reaches the line.
"""

from __future__ import annotations

import json

import composition_fixtures as F
import pytest
from merlin_experiments.phase1.feedback import composition_next as C
from merlin_experiments.phase1.feedback import performance_brief as PB


@pytest.fixture
def workspace(tmp_path):
    F.declare(tmp_path, "layers", F.conv_pool("SY_chain"))
    F.declare(tmp_path, "layers", F.residual("G_residual"))
    F.declare(tmp_path, "layers", F.matmul("G_plain"))
    F.declare(tmp_path, "projections", F.siblings("PS_siblings"))
    F.declare(tmp_path, "model", F.model("SY_model", group="layers/G_residual"), ir=F.PROGRAM_IR)
    F.program(tmp_path, "PG_program")
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "backend.py").write_text("import json\n")
    (tmp_path / ".agents" / "skills" / "heavy-pass-recipes").mkdir(parents=True)
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'connected'} for name in
                    ('SY_chain', 'G_residual', 'G_plain', 'PS_siblings', 'SY_model', 'PG_program')}}))
    return tmp_path


def _next(document, key="per_capsule"):
    return {row.get("capsule") or row.get("name"): row.get("optimization_next") or {} for row in document[key]}


def test_a_chain_case_gets_its_route_symbol_import_state_and_commands(workspace):
    rows = [{"capsule": name, "pass": True} for name in ("SY_chain", "G_residual", "G_plain")]
    document = PB.attach({"per_capsule": rows}, rows, workspace=workspace)
    found = _next(document)
    pointer = found["SY_chain"][C.KEY]
    line = pointer["line"]
    assert "\n" not in line and line.startswith("composition routes: producer_pool: declared epilogue pools")
    assert (
        "-> merlin.compile.scheduling.rolling_spatial:emit_rolling_spatial_producer (not imported by submission/"
        in line
    )
    assert "run: python .agents/skills/heavy-pass-recipes/scripts/spatial_chain_recipe.py" in line
    assert (
        "all routes for this case: python .agents/skills/heavy-pass-recipes/scripts/composition_routes.py "
        "--case SY_chain"
    ) in line
    assert [r["route"] for r in pointer["routes"]] == ["producer_pool", "producer_epilogue"]
    assert pointer["trigger"] == "declared_or_ir_chain" and "no speedup" in pointer["basis"]
    residual = found["G_residual"][C.KEY]
    assert residual["routes"][0]["symbol"] == "merlin.llvmlower.quantized_affine_pair:prove_ordered_loaded_sum"
    assert C.REQUEST_KEY not in found["SY_chain"] and C.REQUEST_KEY not in found["G_residual"]
    assert C.KEY not in found["G_plain"]


def test_an_import_in_submission_is_reported(workspace):
    (workspace / "submission" / "pool.py").write_text(
        "from merlin.compile.scheduling.rolling_spatial import emit_rolling_spatial_producer\n"
    )
    rows = [{"capsule": "SY_chain", "pass": True}]
    PB.attach({"per_capsule": rows}, rows, workspace=workspace)
    pointer = rows[0]["optimization_next"][C.KEY]
    assert pointer["routes"][0]["imported_by_submission"] is True
    assert "(imported by submission/" in pointer["line"]


def test_the_pointer_sits_beside_the_library_pass_in_ranked_advice(workspace):
    rows = [{"capsule": "SY_chain", "pass": True}]
    library = {"line": "library pass line"}
    brief = {"per_case_advice": {"SY_chain": {"library_pass": library, "skill": "shape-schedule-selection"}}}
    assert C.annotate({"per_capsule": rows}, rows, brief, workspace=workspace) == 1
    advice = brief["per_case_advice"]["SY_chain"]
    assert advice["library_pass"] is library and advice[C.KEY]["routes"][0]["route"] == "producer_pool"
    assert "optimization_next" not in rows[0]  # the brief's copy step puts the advice on the row


def test_host_compute_flags_lead_with_the_epilogue_routes(workspace):
    rows = [
        {"capsule": "SY_chain", "pass": False, "failure": {"category": C.HOST_VETO}},
        {"capsule": "G_residual", "pass": True},
    ]
    census = {"host_compute": [{"program": "G_residual", "verdict": C.HOST_VETO}], "host_compute_vetoed": []}
    document = {"per_capsule": rows, "offload_census": census}
    assert C.host_flagged(document, rows) == {"SY_chain", "G_residual"}
    PB.attach(document, rows, workspace=workspace)
    for row in rows:
        pointer = row["optimization_next"][C.KEY]
        assert pointer["trigger"] == C.HOST_VETO
        assert pointer["line"].startswith(f"host computes this group's epilogue ({C.HOST_VETO}); composition routes")
        assert pointer["routes"][0]["route"] in C.EPILOGUE_ROUTES


def test_a_model_case_gets_source_composition_options_the_reader_accepts(workspace, tmp_path_factory):
    from merlin.targetgen import source_composition_preferences as P

    rows = [{"capsule": "SY_model", "pass": True}]
    PB.attach({"per_capsule": rows}, rows, workspace=workspace)
    found = rows[0]["optimization_next"]
    request = found[C.REQUEST_KEY]
    options = [m["option"] for m in request["options"]]
    assert options == [
        "prepack_weights",
        "share_exact_tensor_preparations",
        "dense_input_views",
        "propagate_layouts",
        "reuse_intermediate_storage",
        "shared_input_siblings",
        "independent_heads",
        "op_profile",
    ]
    line = request["line"]
    assert line.startswith("source_composition.json options this case may use: prepack_weights (")
    assert "| request: write submission/source_composition.json {" in line
    assert ".agents/skills/heavy-pass-recipes/reference/source-composition.md" in line
    assert "no speedup is implied" in line and "%" not in line
    assert "original software permissions" in line and "missing permission keeps original code" in line
    host_options = {m["option"]: m for m in request["options"]
                    if m["option"] in {"propagate_layouts", "reuse_intermediate_storage"}}
    assert "logical_host/propagate_layout" in host_options["propagate_layouts"]["requires"]
    assert "proven final allocation/free lifetimes" in host_options["reuse_intermediate_storage"]["requires"]
    folder = tmp_path_factory.mktemp("request")
    (folder / "source_composition.json").write_text(json.dumps(request["example"]))
    accepted, _ = P.read(folder)
    assert accepted["prepack_weights"] and accepted["independent_heads"] and "op_profile" not in accepted
    routes = {r["route"] for r in found[C.KEY]["routes"]}
    assert {"producer_pool", "residual_add", "shared_input_siblings", "independent_heads"} <= routes
    # The model's own declaration carries capture paths and digests; none reach the feedback.
    text = json.dumps(found)
    assert "materialized" not in text and "golden" not in text and "sha256" not in text


def test_requested_options_are_not_suggested_again_and_stay_in_the_example(workspace):
    (workspace / "submission" / "source_composition.json").write_text(
        json.dumps(
            {
                "schema": "merlin.source-composition-preferences.v1",
                "shared_input_siblings": False,
                "prepack_weights": True,
            }
        )
    )
    rows = [{"capsule": "SY_model", "pass": True}]
    PB.attach({"per_capsule": rows}, rows, workspace=workspace)
    request = rows[0]["optimization_next"][C.REQUEST_KEY]
    assert "prepack_weights" not in [m["option"] for m in request["options"]]
    assert request["already_requested"] == ["prepack_weights"] and request["example"]["prepack_weights"] is True
    # The row's own composition report is the fresher record of what was requested.
    features = {
        name: {"requested": True}
        for name in (
            "prepack_weights",
            "share_exact_tensor_preparations",
            "dense_input_views",
            "propagate_layouts",
            "reuse_intermediate_storage",
            "shared_input_siblings",
            "independent_heads",
            "op_profile",
        )
    }
    rows = [{"capsule": "SY_model", "pass": True, "source_composition": {"features": features}}]
    PB.attach({"per_capsule": rows}, rows, workspace=workspace)
    assert C.REQUEST_KEY not in rows[0]["optimization_next"] and C.KEY in rows[0]["optimization_next"]


def test_program_rows_from_the_program_suite_get_requests_and_slices_do_not(workspace):
    cases = [{"capsule": "PG_program", "status": "pass"}, {"capsule": "PS_siblings", "status": "pass"}]
    document = PB.attach({"cases": cases}, cases, workspace=workspace, debuggable=False)
    found = _next(document, "cases")
    assert [m["option"] for m in found["PG_program"][C.REQUEST_KEY]["options"]][:2] == [
        "prepack_weights",
        "share_exact_tensor_preparations",
    ]
    assert found["PS_siblings"][C.KEY]["routes"][0]["route"] == "shared_input_siblings"
    assert C.REQUEST_KEY not in found["PS_siblings"]


def test_annotation_is_bounded_and_never_raises(workspace, monkeypatch):
    rows = [{"capsule": "SY_chain"}] * (C.LIMIT + 5)
    assert C.annotate({}, [dict(row) for row in rows], None, workspace=workspace) == C.LIMIT
    assert C.annotate(None, None, None, workspace=workspace) == 0
    monkeypatch.setattr(C, "_signals", lambda *a, **k: 1 / 0)
    document = {"per_capsule": [{"capsule": "SY_chain", "pass": True}]}
    assert C.annotate(document, document["per_capsule"], None, workspace=workspace) == 0
    assert PB.attach(document, document["per_capsule"], workspace=workspace) is document
