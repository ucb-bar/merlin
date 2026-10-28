import pytest

from merlin_experiments.phase0.generation import withhold_unadmitted_micro_model as withhold

MICRO = {"name": "SY_micro_model", "micro_model": True, "source_role": "derived_sweep"}
UNSUPPORTED = {
    "status": "unsupported",
    "reason": "m2m_q.quantize.default: operand_dtype 'fp32' not in ['int8']",
    "placements": {"host": 3},
}


def test_a_synthesized_micro_model_the_host_lane_cannot_admit_is_withheld_with_its_verbatim_reason():
    row = withhold(MICRO, UNSUPPORTED, must_refuse=False)
    assert row["capsule"] == "SY_micro_model"
    assert row["placements"] == {"host": 3} and row["placements"] is not UNSUPPORTED["placements"]
    assert row["reason"].endswith(UNSUPPORTED["reason"])
    assert row["reason"].startswith("emitted program not admitted by the selected host lane")


@pytest.mark.parametrize(
    "entry,observed,must_refuse",
    [
        ({**MICRO, "micro_model": False}, UNSUPPORTED, False),
        ({"name": "SY_x", "source_role": "derived_sweep"}, UNSUPPORTED, False),
        ({**MICRO, "source_role": "model_derived"}, UNSUPPORTED, False),
        ({**MICRO, "source_role": "materialized_iteration_capture"}, UNSUPPORTED, False),
        ({k: v for k, v in MICRO.items() if k != "source_role"}, UNSUPPORTED, False),
        (MICRO, UNSUPPORTED, True),
        (MICRO, {**UNSUPPORTED, "status": "unknown"}, False),
        (MICRO, {**UNSUPPORTED, "status": "refused"}, False),
        (MICRO, {**UNSUPPORTED, "status": "admitted"}, False),
        (MICRO, {"reason": "no status"}, False),
    ],
)
def test_every_other_unadmitted_emission_stays_fatal(entry, observed, must_refuse):
    assert withhold(entry, observed, must_refuse=must_refuse) is None


def _member(tmp_path, name="SY_micro_model"):
    directory = tmp_path / "capsules" / "_diagnostic" / name
    directory.mkdir(parents=True)
    (directory / "capsule.yaml").write_text(f"name: {name}\n")
    return directory


def test_a_withheld_member_leaves_the_output_tree_so_no_corpus_glob_can_see_it(tmp_path):
    from merlin_experiments.phase0.generation import withhold_unadmitted_member

    directory = _member(tmp_path)
    kept = _member(tmp_path, "SY_other")
    row = withhold_unadmitted_member(MICRO, UNSUPPORTED, directory, must_refuse=False)
    assert row["capsule"] == "SY_micro_model" and not directory.exists()
    assert sorted(path.parent.name for path in (tmp_path / "capsules").glob("*/*/capsule.yaml")) == ["SY_other"]
    assert kept.is_dir()


def test_a_member_that_stays_fatal_is_not_removed(tmp_path):
    from merlin_experiments.phase0.generation import withhold_unadmitted_member

    directory = _member(tmp_path)
    assert withhold_unadmitted_member(MICRO, UNSUPPORTED, directory, must_refuse=True) is None
    assert withhold_unadmitted_member({**MICRO, "micro_model": False}, UNSUPPORTED, directory, must_refuse=False) is None
    assert (directory / "capsule.yaml").is_file()


def test_generation_skips_a_withheld_member_instead_of_publishing_it():
    """A withheld member must not fall through to ``written.append``: the call site ends in ``continue``."""
    import ast
    import inspect

    from merlin_experiments.phase0 import generation

    tree = ast.parse(inspect.getsource(generation.generate_target))
    sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and any(isinstance(n, ast.Name) and n.id == "withheld" for n in ast.walk(node.test))
    ]
    assert sites, "the withheld branch is missing from generate_target"
    for site in sites:
        assert any(isinstance(statement, ast.Continue) for statement in site.body), (
            "a withheld member must be skipped, not appended to the written corpus"
        )
        assert any(
            isinstance(n, ast.Attribute) and n.attr == "append"
            and isinstance(n.value, ast.Name) and n.value.id == "host_placed"
            for statement in site.body for n in ast.walk(statement)
        )
