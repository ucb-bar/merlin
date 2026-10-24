"""Native source records bind the actual full-call execution and score owners."""

import shutil

import pytest
from merlin_experiments.phase1 import source_inputs
from merlin_experiments.spec import SpecError

from merlin.common.paths import module_source_path

OWNERS = {
    "full_call_oracle": "merlin.targetgen.spike_full_call",
    "full_call_capsules": "merlin.targetgen.core_aten_capsules",
    "full_call_runner": "merlin.targetgen.capsule_runner",
    "full_call_device": "merlin.targetgen.core_aten_device",
    "full_call_numeric_admission": "merlin.targetgen.core_aten_spec_int8",
    "full_call_lane_scores": "merlin.targetgen.capsule_common",
}


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("damage", ["bytes", "missing", "path"])
def test_full_call_source_owner_drift_refuses_prepared_record(tmp_path, monkeypatch, owner, damage):
    copied = {}
    for key, module in OWNERS.items():
        path = tmp_path / (key + ".py")
        shutil.copyfile(module_source_path(module), path)
        copied[module] = path
    original = source_inputs._source
    monkeypatch.setattr(source_inputs, "_source", lambda module: copied.get(module) or original(module))
    arguments = {"repo": tmp_path, "entrypoint": tmp_path / "transport.py"}
    prepared = source_inputs.record(**arguments)
    for key, module in OWNERS.items():
        assert prepared["inputs"]["phase1:startup:" + key]["path"] == str(copied[module])
    source_inputs.verify(prepared, **arguments)
    path = copied[OWNERS[owner]]
    if damage == "bytes":
        path.write_bytes(path.read_bytes() + b"\n# altered qualification owner\n")
    elif damage == "missing":
        path.unlink()
    else:
        other = path.with_name("replacement-" + path.name)
        shutil.copyfile(path, other)
        copied[OWNERS[owner]] = other
    with pytest.raises(SpecError, match="source identity changed|input is absent"):
        source_inputs.verify(prepared, **arguments)
