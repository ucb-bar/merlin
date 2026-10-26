"""The heavy-pass-recipes author skill: every recipe runs, verifies numerics and shows its mechanism.

Each script derives its tiles from capacity facts (a synthetic body here, or a written facts file),
runs the library passes and executes every emitted program on host integers against the source. The
convolution recipe must move interior input rows as multi-row runs, keep weights resident when asked,
and never write a local row twice before it is consumed (the zero-fill-then-overwrite race). The
policy recipe must switch exactly the class its evidence supports, keep the incumbent where evidence
or boundaries are missing, fall back on legality, and refuse a stale performance domain.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from author_skill_fixtures import BODY_LIMIT, assert_generic_skill, front_matter, load_script, skill_dir

SKILL = "heavy-pass-recipes"
SCRIPTS = (
    "recipe_support",
    "dense_readout_recipe",
    "long_k_recipe",
    "conv_window_recipe",
    "spatial_chain_recipe",
    "library_pipeline_policy",
)
REFERENCES = ("dense-recipes.md", "convolution.md", "composition.md", "pipeline-policy.md")


def _facts(tmp_path, *, edge=4, operand_rows=128, accumulator_rows=32):
    path = tmp_path / f"facts-{edge}-{operand_rows}-{accumulator_rows}.json"
    path.write_text(
        json.dumps(
            {
                "arrays": [{"name": "array", "rows": edge, "cols": edge}],
                "memories": [
                    {"name": "operand_sram", "banks": 1, "depth": operand_rows, "bytes": operand_rows * edge},
                    {"name": "acc_sram", "banks": 1, "depth": accumulator_rows, "bytes": accumulator_rows * edge * 4},
                ],
                "datapaths": [{"name": "operand_sram", "dtype": "i8"}, {"name": "acc_sram", "dtype": "i32"}],
            }
        )
    )
    return str(path)


def test_skill_is_generic_bounded_and_points_at_its_files():
    assert_generic_skill(SKILL, allowed_imports=frozenset({"merlin", "recipe_support"}))
    fields, body = front_matter(SKILL)
    assert len(body.encode()) <= BODY_LIMIT
    for script in SCRIPTS[1:]:
        assert f"{script}.py" in body, script
    for name in REFERENCES:
        assert f"reference/{name}" in body and (skill_dir(SKILL) / "reference" / name).is_file(), name
    for symbol in (
        "dense_readout_overlap",
        "dense_ready_readout",
        "dense_segmented_pipeline",
        "convolution_exploration",
        "rolling_window",
        "rolling_spatial",
        "composition_catalog",
        "PipelineVariant",
        "shape_policy",
    ):
        assert symbol in body, symbol
    for pointer in (
        "scheduling-api",
        "explore-schedules",
        "compose-pass-pipeline",
        "fusion",
        "shape-schedule-selection",
    ):
        assert "`heavy-pass-recipes`" in front_matter(pointer)[1], pointer


def test_every_script_answers_help_with_the_standard_library_only():
    for script in SCRIPTS:
        proc = subprocess.run(
            [sys.executable, "-B", "-I", str(skill_dir(SKILL) / "scripts" / f"{script}.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0 and "usage" in proc.stdout.lower(), (script, proc.stderr)


def test_readout_recipe_overlaps_readout_on_an_accumulator_bound_gemm():
    out = load_script(SKILL, "dense_readout_recipe").recipe()
    assert out["facts"].startswith("synthetic") and out["accumulator_bound"]
    assert out["output_accumulator_rows_needed"] > out["accumulator_rows"]
    variants = out["variants"]
    computes = {name: row["calls"]["compute"] for name, row in variants.items()}
    assert len(set(computes.values())) == 1  # same products everywhere; the host check compared values
    assert variants["readout_overlap"]["disjoint_readout_epochs"] > 0
    assert variants["ready_readout"]["causal_ready_epochs"] > 0
    assert (
        variants["ready_readout"]["epoch_drains"]
        < variants["batched_readouts"]["epoch_drains"]
        < (variants["segmented"]["epoch_drains"])
    )


def test_long_k_recipe_derives_both_bounds_and_segments_past_the_first(tmp_path):
    recipe = load_script(SKILL, "long_k_recipe")
    out = recipe.recipe()
    bounds = out["bounds"]
    assert out["k_tiles"] == bounds["full_k_panel_pair_max_k_tiles"] + 1 < bounds["segmented_max_k_tiles"]
    assert out["retained_full_k"].startswith("declined")
    seg = out["segmented"]
    assert seg["operand_split_rows"]["lhs"] == out["k_tiles"] * 4 and seg["overlapped_compute_pairs"] > 0
    assert seg["calls"]["compute"] == out["compact"]["calls"]["compute"]
    beyond = recipe.recipe(k_tiles=bounds["segmented_max_k_tiles"] + 1)
    assert isinstance(beyond["segmented"], str) and beyond["segmented"].startswith("declined")
    smaller = recipe.recipe(_facts(tmp_path, operand_rows=64))
    assert smaller["bounds"]["full_k_panel_pair_max_k_tiles"] == 8 and smaller["k_tiles"] == 9


@pytest.fixture(scope="module")
def conv():
    return load_script(SKILL, "conv_window_recipe")


def test_conv_recipe_moves_rows_in_runs_keeps_weights_and_writes_each_row_once(conv):
    out = conv.recipe()
    gather = out["one_row_gather_input_transfers"]
    routes = {name: [c for c in route["candidates"] if c.get("counts")] for name, route in out["routes"].items()}
    assert all(len(rows) == 4 for rows in routes.values())  # live_tiles x resident weights, all unique
    for candidate in routes["halo_source"]:
        counts = candidate["counts"]
        assert counts["input_one_row"] == counts["input_zero_rows"] == 0 and counts["input_multi_row"] > 0
    for candidate in routes["border_rows"]:  # tiny layer: border tiles are a large share here
        assert candidate["input_transfers"] < gather and candidate["rows_per_input_transfer"] > 1
    assert min(c["input_transfers"] for c in routes["halo_source"]) * 5 < gather
    by_choice = {(c["live_tiles"], c["resident_weights"]): c["counts"] for c in routes["border_rows"]}
    assert by_choice["capacity", True]["weight_loads"] < by_choice["capacity", False]["weight_loads"]
    assert by_choice["capacity", False]["input_rows"] < by_choice[1, False]["input_rows"]  # input retained
    assert out["rolling_window"]["stage_rows"] < out["rolling_window"]["unstaged_reference_rows"]
    assert "no local row written twice" in out["verified"]


def test_the_conv_hazard_check_catches_zero_fill_then_overwrite(conv):
    source = conv.plan(conv._support().capacities(None), conv.derived_geometry(conv._support().capacities(None)))
    machine = conv.Machine(source, [1] * 4096, [[1] * 64 for _ in range(512)])
    machine("load_input", dict(first_element=0, row_stride_elements=4, local_row=0, rows=1, columns=4, zero=True))
    machine("load_input", dict(first_element=0, row_stride_elements=4, local_row=0, rows=4, columns=4, zero=False))
    assert machine.hazards == [("lhs", 0)]


def test_conv_count_only_scales_to_a_wider_layer(conv, tmp_path):
    geometry = dict(shape=(1, 12, 12, 16, 16), kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1))
    out = conv.recipe(_facts(tmp_path, edge=8, operand_rows=1024, accumulator_rows=256), geometry, numerics=False)
    halo = [c for c in out["routes"]["halo_source"]["candidates"] if c.get("counts")]
    assert halo and all(c["counts"]["input_one_row"] == 0 for c in halo)
    assert min(c["input_transfers"] for c in halo) * 10 < out["one_row_gather_input_transfers"]


def test_spatial_chain_is_discovered_in_the_catalog_and_matches_the_source():
    recipe = load_script(SKILL, "spatial_chain_recipe")
    out = recipe.recipe()
    rows = out["discovery"]["author_library_rows"]
    assert out["discovery"]["selected_row"] in rows and out["discovery"]["roster"].startswith("library_modules")
    assert out["storage"]["full_intermediate_image_bytes"] == 0 and out["effects"]["stores"] > 0
    assert out["effects"]["zero_rows"] > 0  # padded input rows staged as zeros, once
    # A smaller host workspace shrinks the packed panels (or drops families); the chain still verifies.
    narrow = recipe.recipe(workspace_bytes=out["workspace_bytes"] * 2 // 3)
    panels = {c["stripe_pixels"]: c["panel_rows"] for c in out["choices"]}
    assert all(c["panel_rows"] <= panels[c["stripe_pixels"]] for c in narrow["choices"])
    assert sum(c["panel_rows"] for c in narrow["choices"]) < sum(panels.values())


@pytest.fixture(scope="module")
def policy_module():
    return load_script(SKILL, "library_pipeline_policy")


def test_policy_switches_only_the_supported_class_and_executes_real_passes(policy_module):
    out = policy_module.example()
    assert "SYNTHETIC" in out["evidence"]
    classes = {row["class"]: row for row in out["policy"]}
    assert classes["output_overflows_accumulator"]["selected"] == "segmented"
    fit = classes["output_fits_accumulator"]
    assert fit["status"] == "incumbent_retained" and fit["selected"] == "compact"
    readings = {a["alternative"]: a["reading"] for a in fit["alternatives"]}
    assert "missing" in readings["retained"] and "regresses" in readings["segmented"]
    runs = {tuple(r["shape"]): r for r in out["executions"]}
    for run in runs.values():
        overflow = run["resources"]["accumulator_output_stores_needed"] >= 2
        legal = run["resources"]["segmented_blocks"] >= 1
        assert run["variant"] == ("segmented" if overflow and legal else "compact"), run
    assert any(["segmented", "unsupported"] in [list(d) for d in r["decisions"]] for r in runs.values())


def test_policy_keeps_the_incumbent_without_boundary_cases_or_with_a_stale_domain(policy_module):
    from dataclasses import replace

    from merlin.transforms.shape_policy import PerformanceDomain

    S = policy_module._support()
    caps = S.capacities(None)
    panels, pins = policy_module.demo_panels(caps)
    cases, _ = policy_module.feedback(panels, pins, caps)
    policy = policy_module.assess(cases, {"model_layer", "thin"})
    first = next(o for _, c in cases for _, o in c.observations).identity
    domain = PerformanceDomain.from_observation(first, implementation_sha256=pins["implementation_sha256"])
    overflow = next(c for k, c in cases if k == "tall")
    shape = (*overflow.facts.shapes[0], overflow.facts.shapes[1][1])
    assert policy_module.execute(policy, caps, shape, domain)["variant"] == "segmented"
    stale = replace(domain, implementation_sha256="f" * 64)  # the compiler changed since the panel
    assert policy_module.execute(policy, caps, shape, stale)["variant"] == "compact"
    # Without the only case whose output exactly fills the accumulator, the fit guard's boundary is
    # uncovered: that class cannot switch, whatever its cycles; the other guard is still covered.
    trimmed = {
        name: {**panel, "classes": {k: v for k, v in panel["classes"].items() if k != "wide"}}
        for name, panel in panels.items()
    }
    cases, _ = policy_module.feedback(trimmed, pins, caps)
    rows = policy_module.summary(policy_module.assess(cases, {"model_layer", "thin"}))
    report = {row["class"]: row for row in rows}
    fit = report["output_fits_accumulator"]
    assert fit["status"] == "incumbent_retained"
    assert all("missing boundary cases (value, outside value): 1/0" in a["reading"] for a in fit["alternatives"])
    assert report["output_overflows_accumulator"]["selected"] == "segmented"


def test_every_registered_variant_compiles_through_the_single_dense_entry(policy_module, monkeypatch):
    from merlin.compile.scheduling import dense_ready_readout
    from merlin.transforms.pipeline_policy import PipelinePolicy, execute_policy

    calls = []
    original = dense_ready_readout.schedule_dense_pipeline

    def counted(*args, **kwargs):
        calls.append(kwargs.get("readout_mode", "separate"))
        return original(*args, **kwargs)

    monkeypatch.setattr(dense_ready_readout, "schedule_dense_pipeline", counted)
    S = policy_module._support()
    caps = S.capacities(None)
    shape = (8, 12, 16)
    for variant in policy_module.variants():
        sess = policy_module.session(policy_module.DenseSource(*shape, ((8, 12), (12, 16))), caps)
        facts = policy_module.facts_for(sess, caps)
        variants = (variant,) if not variant.predicates else (variant, policy_module.variants()[0])
        result = execute_policy(
            sess, PipelinePolicy(variants, "compact"), policy_module.registry(caps), facts, output="dense_program"
        )
        assert result.receipt["selected_variant"] == variant.name
        kernel, a, p = result.artifact
        S.execute_dense(kernel, a, p)  # raises unless the emitted kernel reproduces the source product
    assert calls == ["separate", "separate", "disjoint"]


def test_policy_reads_panel_files_and_requires_the_author_pins(policy_module, tmp_path):
    caps = policy_module._support().capacities(None)
    panels, pins = policy_module.demo_panels(caps)
    files = {}
    for name, panel in panels.items():
        path = tmp_path / f"panel_{name}.json"
        path.write_text(json.dumps({"panel": panel}))
        files[name] = str(path)
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    out = policy_module.example(panel_files=files, pins_file=str(tmp_path / "pins.json"))
    assert out["evidence"] == "panel files"
    assert {row["class"]: row["selected"] for row in out["policy"]} == {
        "output_fits_accumulator": "compact",
        "output_overflows_accumulator": "segmented",
    }
    (tmp_path / "partial.json").write_text(json.dumps({k: v for k, v in pins.items() if k != "roi_sha256"}))
    with pytest.raises(SystemExit, match="roi_sha256"):
        policy_module.example(panel_files=files, pins_file=str(tmp_path / "partial.json"))


def test_the_acceptance_check_refuses_outputs_that_do_not_show_the_mechanism():
    from merlin_experiments.phase1 import author_acceptance as AA
    from merlin_experiments.phase1 import heavy_recipe_acceptance as H

    assert "heavy-pass-recipes" in AA.FAST and AA.CHECKS["heavy-pass-recipes"] is AA.check_heavy_pass_recipes
    conv = {
        "one_row_gather_input_transfers": 100,
        "routes": {
            "border_rows": {
                "candidates": [
                    {
                        "resident_weights": False,
                        "input_transfers": 40,
                        "counts": {"input_multi_row": 5, "input_one_row": 30, "input_zero_rows": 5, "weight_loads": 9},
                    },
                    {
                        "resident_weights": True,
                        "input_transfers": 40,
                        "counts": {"input_multi_row": 5, "input_one_row": 30, "input_zero_rows": 5, "weight_loads": 3},
                    },
                ]
            },
            "halo_source": {
                "candidates": [
                    {
                        "resident_weights": True,
                        "input_transfers": 10,
                        "counts": {"input_multi_row": 9, "input_one_row": 0, "input_zero_rows": 1, "weight_loads": 3},
                    }
                ]
            },
        },
    }
    assert "halo route" in H._conv(conv)
    conv["routes"]["halo_source"]["candidates"][0]["counts"]["input_zero_rows"] = 0
    assert H._conv(conv) is None
    conv["routes"]["border_rows"]["candidates"][0]["input_transfers"] = 100
    assert "per-pixel gather" in H._conv(conv)
    assert "declined" not in (H._long_k({"retained_full_k": "applies"}) or "declined")
    assert H._policy(
        {"policy": [{"class": "a", "status": "incumbent_retained"}, {"class": "b", "status": "incumbent_retained"}]}
    )
