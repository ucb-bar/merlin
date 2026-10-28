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
