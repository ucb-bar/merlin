"""Source admission and installed resource lookup are distinct selected roots."""

from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import launch_config as L
from test_phase1_launch_config import _checkout


@pytest.mark.parametrize("separate", [False, True])
def test_explicit_resource_root_changes_only_native_lookup(tmp_path, separate):
    selected = "/original/reviewed/resource-checkout"

    def edit(document):
        document["variables"]["SOURCE_REVISION"] = "a" * 40
        if separate:
            document["inputs"]["resource_repo"] = selected

    config = _checkout(tmp_path, edit)
    launch = L.build(L.load(config), task_root=tmp_path / "task", prep=tmp_path / "prep", run_id="resource-root")
    assert launch.path(launch.document["inputs"]["repo"]) == launch.path("${REPO}")
    assert launch.variables["SOURCE_REVISION"] == "a" * 40
    index = launch.argv.index("--repo")
    assert launch.argv[index + 1] == (selected if separate else str(launch.path("${REPO}")))
    assert launch.path(launch.document["launcher"]["cwd"]) == launch.path("${REPO}")


def test_resource_root_requires_nonempty_selected_path(tmp_path):
    root = Path(__file__).resolve().parents[3]
    document = yaml.safe_load((root / "examples/gemmini/phase1/launch/el4-fire-aligned.yaml").read_bytes())
    document["inputs"]["resource_repo"] = ""
    document.pop("release_config", None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        L.load(path)
