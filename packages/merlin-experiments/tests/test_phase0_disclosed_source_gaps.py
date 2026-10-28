import ast
import hashlib
import inspect

import pytest

from merlin.targetgen.capture_refusals import M2MUnavailable
from merlin_experiments.phase0 import generation
from merlin_experiments.phase0.generation import disclosed_source_gap, withhold_source_gap
from merlin_experiments.phase0.model_qualification import QualificationMissing

SOURCE = {"name": "SY_source_x", "kind": "model", "source_role": "materialized_iteration_capture"}


def _refusal():
    try:
        try:
            raise QualificationMissing("no device group whose arithmetic could be verified")
        except QualificationMissing as missing:
            raise ValueError("source-backed integer contraction lacks a verified isolated i8 matmul") from missing
    except ValueError as exc:
        return exc


def test_a_typed_writer_refusal_is_disclosed_with_its_verbatim_reason_and_digest():
    row = disclosed_source_gap(SOURCE, exc=M2MUnavailable("mixed dtypes"), detail="M2MUnavailable: mixed dtypes")
    assert row == {
        "capsule": "SY_source_x", "kind": "writer_refusal", "reason": "M2MUnavailable: mixed dtypes",
        "reason_sha256": hashlib.sha256(b"M2MUnavailable: mixed dtypes").hexdigest(),
    }
    qualified = disclosed_source_gap(SOURCE, exc=_refusal(), detail="ValueError: lacks a verified isolated i8 matmul")
    assert qualified["kind"] == "writer_refusal"


def test_an_unsupported_screen_of_a_source_capsule_is_disclosed_as_a_host_lane_gap():
    row = disclosed_source_gap(SOURCE, observed={"status": "unsupported", "reason": "m2m_q.quantize.default: x"})
    assert row["kind"] == "host_lane_admission"
    assert row["reason"] == "emitted SW operation admission: m2m_q.quantize.default: x"


@pytest.mark.parametrize(
    "entry,kw",
    [
        ({**SOURCE, "source_role": "derived_sweep"}, {"exc": M2MUnavailable("x")}),
        ({**SOURCE, "kind": "layer"}, {"exc": M2MUnavailable("x")}),
        (SOURCE, {"exc": M2MUnavailable("x"), "must_refuse": True}),
        (SOURCE, {"exc": TypeError("'float' object is not iterable")}),
        (SOURCE, {"exc": ValueError("plain value error")}),
        (SOURCE, {"exc": RuntimeError("anything else")}),
        (SOURCE, {"observed": {"status": "refused", "reason": "x"}}),
        (SOURCE, {"observed": {"status": "unknown", "reason": "x"}}),
        (SOURCE, {"observed": {"status": "admitted"}}),
        (SOURCE, {"observed": {"status": "unsupported"}, "must_refuse": True}),
        (SOURCE, {}),
    ],
)
def test_everything_else_stays_fatal(entry, kw):
    assert disclosed_source_gap(entry, **kw) is None


def test_a_roster_model_is_never_disclosed_as_a_source_gap():
    roster = {**SOURCE, "generalization": {"generalization_axis": "roster"}}
    assert disclosed_source_gap(roster, exc=M2MUnavailable("x")) is None


def test_the_withheld_member_is_removed_from_disk(tmp_path):
    directory = tmp_path / "model" / "SY_source_x"
    directory.mkdir(parents=True)
    (directory / "capsule.yaml").write_text("name: SY_source_x\n")
    other = tmp_path / "model" / "SY_source_y"
    other.mkdir()
    (other / "capsule.yaml").write_text("name: SY_source_y\n")
    assert withhold_source_gap(SOURCE, directory, exc=M2MUnavailable("x"), detail="d")["capsule"] == "SY_source_x"
    assert not directory.exists() and other.is_dir()
    assert sorted(p.parent.name for p in tmp_path.glob("*/*/capsule.yaml")) == ["SY_source_y"]
    kept = tmp_path / "model" / "SY_keep"
    kept.mkdir()
    assert withhold_source_gap(SOURCE, kept, exc=TypeError("bug")) is None and kept.is_dir()


def test_generation_records_the_gap_and_skips_the_member_on_both_paths():
    tree = ast.parse(inspect.getsource(generation.generate_target))
    sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and any(isinstance(n, ast.Name) and n.id == "gap" for n in ast.walk(node.test))
    ]
    assert len(sites) == 2, "the exception path and the admission path must each handle a source gap"
    for site in sites:
        assert any(isinstance(s, ast.Continue) for s in site.body), "a disclosed member must be skipped"
        assert any(
            isinstance(n, ast.Attribute) and n.attr == "append" and isinstance(n.value, ast.Name)
            and n.value.id == "source_gaps" for s in site.body for n in ast.walk(s)
        )
    assert "disclosed_source_gaps" in inspect.getsource(generation.generate_target)


def _qualified_model(root, groups, name="SY_source_x"):
    import yaml

    for group in groups:
        directory = root / "layers" / group
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "capsule.yaml").write_text(f"name: {group}\n")
    directory = root / "model" / name
    directory.mkdir(parents=True)
    rows = [{"capsule": f"layers/{group}"} for group in groups]
    (directory / "capsule.yaml").write_text(yaml.safe_dump({"name": name, "model_qualification": {"group_capsules": rows}}))
    return directory


UNSUPPORTED = {"status": "unsupported", "reason": "host op: x"}


def test_a_qualified_source_model_the_host_lane_refuses_takes_its_qualifying_groups_with_it(tmp_path):
    root = tmp_path / "capsules"
    directory = _qualified_model(root, ["G_a", "G_b"])
    row = withhold_source_gap(SOURCE, directory, observed=UNSUPPORTED, out_root=root, written=[])
    assert row["kind"] == "host_lane_admission"
    assert row["qualifying_groups_withheld"] == ["layers/G_a", "layers/G_b"]
    assert not list(root.glob("*/*/capsule.yaml"))


def test_a_qualifying_group_a_published_or_pending_member_names_is_kept(tmp_path):
    root = tmp_path / "capsules"
    directory = _qualified_model(root, ["G_shared", "G_pending", "G_only"])
    pending = _qualified_model(root, ["G_pending"], name="SY_source_later")
    row = withhold_source_gap(
        SOURCE, directory, observed=UNSUPPORTED, out_root=root, written=[root / "layers" / "G_shared"]
    )
    assert row["qualifying_groups_withheld"] == ["layers/G_only"]
    assert (root / "layers" / "G_shared").is_dir() and (root / "layers" / "G_pending").is_dir() and pending.is_dir()
    assert not directory.exists() and not (root / "layers" / "G_only").exists()


def test_a_fatal_admission_keeps_the_model_and_its_groups(tmp_path):
    root = tmp_path / "capsules"
    directory = _qualified_model(root, ["G_a"])
    for kw in ({"observed": {"status": "refused", "reason": "x"}}, {"observed": UNSUPPORTED, "must_refuse": True}):
        assert withhold_source_gap(SOURCE, directory, out_root=root, written=[], **kw) is None
    assert directory.is_dir() and (root / "layers" / "G_a").is_dir()


def test_the_admission_call_site_passes_the_output_root_and_published_members():
    tree = ast.parse(inspect.getsource(generation.generate_target))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "withhold_source_gap"
        and any(k.arg == "observed" for k in node.keywords)
    ]
    assert len(calls) == 1
    assert {k.arg for k in calls[0].keywords} >= {"out_root", "written"}
