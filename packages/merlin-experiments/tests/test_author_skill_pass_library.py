"""The generated pass-library skill: committed output is current, complete, screened and tagged."""

from __future__ import annotations

import json

import pytest
from author_skill_support import BODY_LIMIT, SKILLS, front_matter, gate, load_script, target_mentions
from merlin_experiments.phase1.feedback import mechanism_hints

from merlin.common.paths import repo_root

SKILL = SKILLS / "pass-library"


@pytest.fixture(scope="module")
def generator():
    gate()  # the generator imports the gate module from its own directory
    return load_script(repo_root() / "build_tools/scripts/gen_author_pass_library.py", "gen_author_pass_library")


@pytest.fixture(scope="module")
def catalog():
    return json.loads((repo_root() / "docs/design/joint_optimization_catalog.json").read_text(encoding="utf-8"))


def test_committed_skill_matches_a_fresh_render(generator):
    """Regenerate with `python build_tools/scripts/gen_author_pass_library.py` when this fails."""
    assert generator.stale(repo_root()) == []


def test_stale_detects_edited_and_orphaned_files(generator, monkeypatch, tmp_path):
    rendered = generator.render(repo_root())
    monkeypatch.setattr(generator, "render", lambda root: {**rendered, "SKILL.md": rendered["SKILL.md"] + "x"})
    assert generator.stale(repo_root()) == ["SKILL.md"]
    monkeypatch.setattr(
        generator, "render", lambda root: {k: v for k, v in rendered.items() if k != "reference/storage.md"}
    )
    assert generator.stale(repo_root()) == ["reference/storage.md"]


def test_every_mechanism_appears_once_with_a_resolved_call(catalog):
    text = {path.relative_to(SKILL).as_posix(): path.read_text(encoding="utf-8") for path in SKILL.rglob("*.md")}
    for row in catalog["mechanisms"]:
        reference = text[f"reference/{row['facet']}.md"]
        assert reference.count(f"### {row['id']}") == 1, row["id"]
        entry = reference.split(f"### {row['id']}", 1)[1].split("\n### ", 1)[0]
        compiler = row["compiler"]
        assert f"`{compiler['module']}:{compiler['symbol']}`" in entry
        assert "**signature**" in entry, row["id"]  # the symbol exists in the current source
        assert "test that calls it**: `" in entry or "test of its module**" in entry, row["id"]
        for field in ("mechanism", "applicability", "remaining_adapter_or_feedback", "evidence_to_request"):
            assert " ".join(row[field].split()) in entry or "(withheld:" in entry, (row["id"], field)
        assert row["id"] in text["SKILL.md"]


def test_index_maps_busy_resources_exactly_as_the_brief_does(generator):
    index = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    for label, probe in generator.RESOURCE_PROBES:
        concrete = probe.replace("<unit>", "unit").replace("<level>", "level").replace("<lane>", "lane")
        assert f"| {label} | `{probe}` | {', '.join(mechanism_hints.resource_facets(concrete))} |" in index
    for label, bound in generator.BOUND_PROBES:
        assert f"| {label} |  | {', '.join(mechanism_hints.bound_facets(bound))} |" in index


def test_fusion_tags_cover_the_fusion_facet_and_nothing_else(catalog):
    index = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    fusion = sorted(row["id"] for row in catalog["mechanisms"] if row["facet"] == "fusion")
    assert f"Fusion-tagged ({len(fusion)}): {', '.join(fusion)}." in index
    for row in catalog["mechanisms"]:
        reference = (SKILL / "reference" / f"{row['facet']}.md").read_text(encoding="utf-8")
        tagged = f"### {row['id']}  [fusion]" in reference
        assert tagged == (row["id"] in fusion), row["id"]


def test_screen_withholds_any_rostered_target_name(generator):
    forbidden = generator._forbidden()
    name = sorted(forbidden)[0]
    assert generator._screen(f"Needs the {name.upper()} adapter", forbidden) == generator.WITHHELD
    assert generator._screen("Needs the target adapter", forbidden) == "Needs the target adapter"


def test_front_matter_size_and_no_target_names():
    fields, body = front_matter(SKILL / "SKILL.md")
    assert fields["name"] == "pass-library"
    for cue in ("suggested_mechanisms", "dominant_modeled_resource", "bound_by"):
        assert cue in fields["description"]
    assert len(body.encode()) <= BODY_LIMIT
    assert target_mentions(SKILL) == []
