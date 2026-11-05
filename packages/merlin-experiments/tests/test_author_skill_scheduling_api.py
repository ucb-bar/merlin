"""The scheduling-api skill: its calls run, its symbols resolve and its discovery matches the catalog."""

from __future__ import annotations

import importlib

import pytest
from author_skill_support import (
    BODY_LIMIT,
    SKILLS,
    front_matter,
    load_script,
    target_mentions,
    top_level_imports,
)

from merlin.compile.scheduling.api_catalog import scheduling_capabilities

SKILL = SKILLS / "scheduling-api"
SCRIPT = SKILL / "scripts" / "dense_schedule_example.py"


@pytest.fixture(scope="module")
def example():
    return load_script(SCRIPT, "author_skill_dense_schedule_example")


def test_discovery_reads_the_selected_release_inventory(example, tmp_path, monkeypatch):
    """TASK.md names scheduling_capabilities(N) for the selected release; discover asks for the same N."""
    assert example.selected_inventory() == 19  # no selected release on this path
    (tmp_path / "shared-library.json").write_text('{"inventory_version": 20}')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert example.selected_inventory() == 20
    assert [r["name"] for r in example.discover()] == [r["name"] for r in example.discover(20)]


def test_discovery_lists_every_v19_capability_and_composition_row(example):
    rows = example.discover()
    catalog = scheduling_capabilities(19)
    names = [row["name"] for row in rows]
    assert names[: len(catalog["capabilities"])] == [row["name"] for row in catalog["capabilities"]]
    assert {f"composition:{row['name']}" for row in catalog["composition"]["capabilities"]} <= set(names)
    assert all(row["symbols"] for row in rows)


def test_contract_reads_signature_fields_and_source(example):
    record = example.contract("merlin.compile.scheduling.dense_segmented_pipeline.DenseSegmentedAdmission")
    assert [f["name"] for f in record["fields"]][:3] == ["overlap", "reduction", "qualification_identity"]
    assert record["fields"][2]["required"] and not record["fields"][3]["required"]
    function = example.contract("merlin.compile.scheduling.dense_segmented_pipeline:schedule_dense_segmented_pipeline")
    assert "schedule_plan" in function["signature"] and function["source"].endswith(tuple(str(n) for n in range(10)))


def test_example_runs_the_pass_and_projected_calls_reproduce_the_product(example):
    out = example.example()
    base, segmented = out["base_calls"], out["segmented_calls"]
    # The segmented pass keeps the LHS panel across K segments: fewer LHS loads, same computes and stores.
    assert segmented["lhs_load"] < base["lhs_load"]
    assert segmented["compute"] == base["compute"] and segmented["store"] == base["store"]
    assert out["overlapped_compute_pairs"] > 0
    assert out["source"]["lhs_loads"] == segmented["lhs_load"]
    assert any("retained_lhs_segmented_k" in line for line in out["schedule_ir"])


def test_a_corrupted_projection_is_caught_by_the_independent_execution(example):
    from dataclasses import replace

    adapter = example.neutral_adapter()
    p, r, s = example.dense_inputs()
    from merlin.compile.scheduling.compact_dense import schedule_compact_dense

    result = schedule_compact_dense(
        p,
        r,
        s,
        adapter,
        name="dense",
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
    )
    example.execute(result.kernel, adapter, p)
    original = adapter.project

    def shifted(call):
        op = original(call)
        return replace(op, dram_col=op.dram_col + 1) if type(op).__name__ == "Store" and op.dram_col == 0 else op

    with pytest.raises((AssertionError, IndexError)):
        example.execute(result.kernel, replace(adapter, project=shifted), p)


def _dotted(body: str) -> set[str]:
    words = "".join(c if (c.isalnum() or c in "._:") else " " for c in body).split()
    return {w.strip(".:").replace(":", ".") for w in words if w.startswith("merlin.") and w.count(".") >= 2}


def test_every_symbol_and_call_in_the_skill_exists():
    _fields, body = front_matter(SKILL / "SKILL.md")
    for dotted in sorted(_dotted(body)):
        module, _, name = dotted.rpartition(".")
        try:
            owner = importlib.import_module(dotted)
        except ImportError:
            owner = getattr(importlib.import_module(module), name)
        assert owner is not None, dotted
    script = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "DenseProblem",
        "DenseLoopStorage",
        "DenseMemoryView",
        "DenseResources",
        "DenseLoopAdapter",
        "schedule_compact_dense",
        "make_dense_schedule_plan",
        "dense_schedule_text",
        "parse_dense_schedule_plan",
        "schedule_dense_segmented_pipeline",
        "revalidate",
        "materialize_scf",
        "dense_blocks",
    ):
        assert name in body and name in script, name


def test_skill_front_matter_scope_links_and_no_target_names():
    fields, body = front_matter(SKILL / "SKILL.md")
    assert fields["name"] == "scheduling-api" and "scheduling_capabilities(N)" in fields["description"]
    assert "inventory_version" in body
    assert len(body.encode()) <= BODY_LIMIT
    for skill in ("compose-pass-pipeline", "fusion", "pass-library", "read-command-buffer"):
        assert f"`{skill}`" in body
    assert "re" not in top_level_imports(SCRIPT)
    assert target_mentions(SKILL) == []
