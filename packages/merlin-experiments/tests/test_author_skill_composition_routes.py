"""composition_routes.py: the per-case discovery command for composition-catalog routes and requests.

The routes must name real catalog rows and symbols, chains come from a case's declared structure or
its IR (never its name), the suggested ``source_composition.json`` is one the owner's reader accepts,
and the command runs from a staged workspace the way the author runs it.
"""

from __future__ import annotations

import json
import subprocess
import sys

import composition_fixtures as F
import pytest
from author_skill_fixtures import load_script, skill_dir

from merlin.compile.composition_catalog import composition_capabilities

SKILL = "heavy-pass-recipes"


@pytest.fixture(scope="module")
def routes():
    return load_script(SKILL, "composition_routes")


def test_every_route_names_catalog_rows_a_listed_symbol_and_existing_files(routes):
    rows = {row["name"]: row for row in composition_capabilities()["capabilities"]}
    skills = {path.parent.name for path in skill_dir(SKILL).parent.glob("*/SKILL.md")}
    for route in routes.ROUTES:
        assert set(route["rows"]) <= set(rows), route["id"]
        module, _, name = route["symbol"].partition(":")
        assert f"{module}.{name}" in rows[route["rows"][0]]["symbols"], route["id"]
        assert set(route["skills"]) <= skills, route["id"]
        if route["recipe"]:
            assert (skill_dir(SKILL) / "scripts" / route["recipe"]).is_file()
    assert len(routes.BY_ID) == len(routes.ROUTES)


def test_request_options_match_the_owner_reader_and_examples_are_accepted(routes, tmp_path):
    from merlin.targetgen import source_composition_preferences as P

    names = [name for name, _ in routes.REQUESTS]
    example = routes.request_example(names)
    assert set(example) == {"schema", *names} and example["schema"] == routes.REQUEST_SCHEMA
    (tmp_path / "source_composition.json").write_text(json.dumps(example))
    accepted, pin = P.read(tmp_path)
    assert accepted == {name: True for name in names} and pin is not None
    minimal = routes.request_example(["independent_heads"])
    assert minimal == {
        "schema": routes.REQUEST_SCHEMA,
        "shared_input_siblings": False,
        "prepack_weights": False,
        "independent_heads": True,
    }
    (tmp_path / "source_composition.json").write_text(json.dumps(minimal))
    assert P.read(tmp_path)[0]["independent_heads"] is True
    for route in routes.ROUTES:
        assert route.get("request") in (None, *names)


@pytest.mark.parametrize(
    "declaration, expected",
    [
        (F.conv_pool(), {"producer_pool", "producer_epilogue"}),
        (F.matmul(epilogue=["relu"]), {"producer_epilogue"}),
        (F.matmul(acc_scale_by_column=[0.5, 0.25]), {"producer_epilogue"}),
        (F.residual(), {"residual_add"}),
        (F.siblings(), {"shared_input_siblings"}),
        (F.model(), {"connected_stages", "parameter_preparation"}),
        (F.matmul(), set()),
        ({"name": "empty"}, set()),
        (None, set()),
    ],
)
def test_chains_come_from_the_declared_structure(routes, declaration, expected):
    assert set(routes.declaration_signals(declaration)) == expected


def test_the_case_name_does_not_select_a_route(routes):
    plain = F.matmul("conv_pool_residual_attention_heads")
    assert routes.declaration_signals(plain) == {}


def test_the_ir_inventory_finds_pool_residual_siblings_heads_and_parameter_preparation(routes, tmp_path):
    path = tmp_path / "program.mlir"
    path.write_text(F.PROGRAM_IR)
    inventory = routes.ir_inventory(path)
    assert inventory["status"] == "observed" and inventory["products"] == 3
    assert inventory["layout_regions"] > 0 and inventory["connected_tensor_edges"] > 0
    assert inventory["shared_input_groups"] == [["matmul_0", "matmul_1"]]
    assert inventory["batched_products"] == ["matmul_2"] and inventory["pools_fed_by_products"] == ["max_pool2d_0"]
    assert inventory["residual_adds"] == ["add_0"] and inventory["parameters_prepared_per_call"] == 1
    signals = routes.ir_signals(inventory)
    assert set(signals) == {
        "producer_pool",
        "producer_epilogue",
        "residual_add",
        "shared_input_siblings",
        "independent_heads",
        "parameter_preparation",
        "host_layout",
        "intermediate_storage",
    }
    options = [match["option"] for match in routes.request_matches(signals, inventory)]
    assert options == [
        "prepack_weights",
        "share_exact_tensor_preparations",
        "dense_input_views",
        "propagate_layouts",
        "reuse_intermediate_storage",
        "shared_input_siblings",
        "independent_heads",
    ]
    assert routes.ir_inventory("not mlir {")["status"] == "unavailable"


def test_a_model_reads_its_ir_and_its_device_group_cases(routes, tmp_path):
    F.declare(tmp_path, "layers", F.residual("group_residual"))
    folder = F.declare(tmp_path, "model", F.model(group="layers/group_residual"), ir=F.PROGRAM_IR)
    signals, inventory, declaration = routes.case_signals(folder, workspace=tmp_path)
    assert inventory["status"] == "observed" and declaration["kind"] == "model"
    assert any("device group case layers/group_residual" in text for text in signals["residual_add"])
    assert signals["parameter_preparation"][0] == "the model declares immutable weights"
    options = [m["option"] for m in routes.request_matches(signals, inventory, declaration)]
    assert options[0] == "prepack_weights" and options[-1] == "op_profile"


def test_report_lists_routes_with_imports_and_commands(routes, tmp_path):
    F.declare(tmp_path, "layers", F.conv_pool())
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "backend.py").write_text(
        "from merlin.compile.scheduling.rolling_spatial import emit_rolling_spatial_producer\n"
    )
    out = routes.report("conv_pool", workspace=tmp_path)
    by_route = {row["route"]: row for row in out["routes"]}
    assert list(by_route) == ["producer_pool", "producer_epilogue"]
    pool = by_route["producer_pool"]
    assert pool["imported_by_submission"] is True and pool["next"].endswith("spatial_chain_recipe.py")
    assert pool["catalog_rows"]["bounded_spatial_packing_and_rolling"]["status"] in (
        "modules_selected",
        "missing_selected_modules",
    )
    epilogue = by_route["producer_epilogue"]
    assert epilogue["imported_by_submission"] is False
    assert epilogue["next"] == (
        "python .agents/skills/scheduling-api/scripts/dense_schedule_example.py contract "
        "merlin.compile.scheduling.spatial_readout:prove_spatial_readout"
    )
    assert "source_composition" not in out  # a single layer is not a model or program check
    everything = routes.report(workspace=tmp_path)
    assert [row["route"] for row in everything["routes"]] == [route["id"] for route in routes.ROUTES]
    assert set(everything["request_options"]) == {name for name, _ in routes.REQUESTS}


def test_the_staged_command_runs_from_the_workspace(tmp_path):
    from merlin_experiments.phase1 import skill_staging as SK
    from merlin_experiments.phase1.options import parse_options

    ws, run = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    SK.stage(ws, run, SK.selection(parse_options(["--run-id", "test", "--driver", "codex", "--sandbox", "bwrap"])))
    script = ws / ".agents" / "skills" / SKILL / "scripts" / "composition_routes.py"
    assert (ws / ".agents" / "skills" / SKILL / "reference" / "source-composition.md").is_file()
    F.program(ws)
    (ws / "submission").mkdir()
    (ws / "submission" / "source_composition.json").write_text(
        json.dumps(
            {
                "schema": "merlin.source-composition-preferences.v1",
                "shared_input_siblings": True,
                "prepack_weights": False,
            }
        )
    )
    proc = subprocess.run(
        [sys.executable, "-B", str(script), "--case", "PG_program"], cwd=ws, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["folder"] == "public_program_sources/PG_program"
    assert {"producer_pool", "independent_heads"} <= {row["route"] for row in out["routes"]}
    request = out["source_composition"]
    assert request["current"]["shared_input_siblings"] is True
    assert request["reference"] == ".agents/skills/heavy-pass-recipes/reference/source-composition.md"
    missing = subprocess.run(
        [sys.executable, "-B", str(script), "--case", "nothing_here"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert missing.returncode != 0 and "no public files" in missing.stderr
