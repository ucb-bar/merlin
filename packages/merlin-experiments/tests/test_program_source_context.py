"""Full-program width premises never confer cut, support, or numerical authority."""

from hashlib import sha256

import pytest
from merlin_experiments.phase0.connected_source_context import require_context
from merlin_experiments.phase0.program_source_context import WholeProgramSourceContext


def observation():
    return dict(schema="merlin.selected-index-lowering.v1", compiler_requested="fixture",
                compiler_resolved="/fixture", compiler_sha256="a" * 64,
                cross_flags=["--target=fixture"], data_layout="e-p:64:64", index_bits=64,
                scope="synthetic unit observation, not selected target evidence")


def fixture(tmp_path):
    paths = {name: tmp_path / (name + ".mlir") for name in ("first", "second")}
    for name, path in paths.items():
        path.write_text("original source " + name)
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    selected = observation()
    return paths, hashes, selected


def test_full_source_roster_is_byte_bound_and_observation_is_owned(tmp_path):
    paths, hashes, selected = fixture(tmp_path)
    context = WholeProgramSourceContext(paths, source_sha256=hashes, selected_index_observation=selected)
    selected["index_bits"] = 8
    require_context(context, paths)
    assert context.for_program(paths["first"])["selected_index_observation"]["index_bits"] == 64
    assert context.control_admission_context(paths["first"], target="fixture") is None
    assert context.index_admission_context(paths["first"]) is None
    assert not hasattr(context, "source_binding")  # no source/window equivalence proof
    with pytest.raises(ValueError, match="another source roster"):
        require_context(context, {"first": paths["first"]})
    paths["second"].write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        context.for_program(paths["first"])


@pytest.mark.parametrize("change", ["hash", "width", "path", "roster"])
def test_context_rejects_mismatched_or_indirect_premises(tmp_path, change):
    paths, hashes, selected = fixture(tmp_path)
    if change == "hash":
        hashes["first"] = "b" * 64
    elif change == "width":
        selected["index_bits"] = 32
    elif change == "path":
        link = tmp_path / "link.mlir"
        link.symlink_to(paths["first"])
        paths["first"] = link
    else:
        hashes.pop("second")
    with pytest.raises(ValueError):
        WholeProgramSourceContext(paths, source_sha256=hashes, selected_index_observation=selected)


def test_unissued_and_serialized_context_do_not_grant_source_premises(tmp_path):
    paths, hashes, selected = fixture(tmp_path)
    valid = WholeProgramSourceContext(paths, source_sha256=hashes, selected_index_observation=selected)
    for invalid in (object.__new__(WholeProgramSourceContext), valid.for_program(paths["first"])):
        with pytest.raises(ValueError, match="issued"):
            require_context(invalid, paths)
