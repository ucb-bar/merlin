import hashlib
from dataclasses import replace

import pytest
from merlin_experiments.phase1 import task_staging as T
from test_shared_pass_authoring import admitted, stage
from test_shared_pass_v2 import manifest

from merlin.targetgen import generate_prompt as P
from merlin.targetgen import shared_pass_contract as S

old_v1_block = P._shared_pass_prompt_block_v1


def test_existing_v1_saved_task_resumes_unchanged(tmp_path, monkeypatch):
    conf, ws, run = admitted(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(P, "shared_pass_prompt_block", old_v1_block)
        stage(conf, ws, run, m)
    assert T.verify_shared_pass_prompt(conf, ws, run) == S.shared_pass_requirement()


@pytest.mark.parametrize("status", ["unknown", "unsupported"])
def test_required_v2_interface_cannot_hide_as_hardware_unsupported(status):
    req, value = manifest(required=("buffering_pipelining",))
    row = next(r for r in value["shared_pass_contract"]["families"] if r["name"] == "buffering_pipelining")
    row.update(status=status, reason="Hardware does not implement this interface")
    with pytest.raises(ValueError, match="original requirement needs supported"):
        S.validate_shared_pass_contract(value, requirement=req)


@pytest.mark.parametrize("names", [("buffering_pipelining", "buffering_pipelining"), ("not_a_family",)])
def test_v2_original_duplicate_unknown_names_refuse(names):
    with pytest.raises(ValueError, match="duplicate or unknown"):
        S.shared_pass_requirement(version=2, required_supported=names)


@pytest.mark.parametrize("change", ["drop_one", "downgrade_v1"])
def test_v2_resume_cannot_change_original_required_set(tmp_path, monkeypatch, change):
    conf, ws, run = admitted(tmp_path)
    req = S.shared_pass_requirement(version=2, required_supported=("multi_output_fusion", "buffering_pipelining"))
    conf = replace(conf, shared_pass_requirement=req)
    stage(conf, ws, run, monkeypatch)
    next_req = (
        S.shared_pass_requirement(version=2, required_supported=("multi_output_fusion",))
        if change == "drop_one"
        else S.shared_pass_requirement(required_supported=("multi_output_fusion",))
    )
    with pytest.raises(ValueError, match="changed|resume"):
        T.verify_shared_pass_prompt(replace(conf, shared_pass_requirement=next_req), ws, run)


def test_original_v1_block_render_is_byte_exact():
    record = {"requirement": S.shared_pass_requirement(), "original_library": "unchanged-neutral-record"}
    assert (
        hashlib.sha256(P.shared_pass_prompt_block(record).encode()).hexdigest()
        == "3f66ba4bac8fdb847b5369ecf96f2920f7ae4b475ef48b5bfab013abacc6b882"
    )


def test_active_v2_scope_cannot_accept_different_canonical_requirement():
    from merlin.targetgen.shared_pass_context import scoped_shared_pass_requirement, selected_shared_pass_requirement

    original = S.shared_pass_requirement(version=2, required_supported=("buffering_pipelining",))
    with scoped_shared_pass_requirement(original):
        assert selected_shared_pass_requirement() == original
        for changed in (S.shared_pass_requirement(version=2), S.shared_pass_requirement()):
            with pytest.raises(ValueError, match="differs|replace"):
                selected_shared_pass_requirement(changed)
        malformed = S.shared_pass_requirement(version=2)
        malformed["required_supported"] = ["buffering_pipelining", "buffering_pipelining"]
        with pytest.raises(ValueError, match="duplicate or unknown"):
            selected_shared_pass_requirement(malformed)
