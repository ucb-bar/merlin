"""Row-only final policy wiring on actual neutral model images.

The primitives still execute on the host. Declaring the device domain here
exercises final admission plumbing, never device instructions or qualification.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_row_softmax import exp_object as exp_object  # noqa: F401
from test_source_row_model import build_model, capture, reviewed_row_selection
from test_source_row_provider import RowProvider

from merlin.common.digest import sha256_file
from merlin.llvmlower.device_build import DeviceRouting
from merlin.model_check import full_build as FB
from merlin.runtime.backends import spike
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


class DiagnosticPolicy:
    def __init__(self, verdict):
        self.verdict, self.observed = verdict, []

    def evaluate(self, *, elf, evidence_root):
        self.observed.append(sha256_file(elf))
        evidence_root.mkdir(parents=True)
        result = dict(status=self.verdict, elf_sha256=sha256_file(elf))
        path = evidence_root / "result.json"
        path.write_text(json.dumps(result))
        return dict(result, report_path=str(path), report_sha256=sha256_file(path))


def service(policy):
    owner = Path(__file__).resolve()
    return LinkedElfAdmissionService("neutral", policy.evaluate, ((str(owner), sha256_file(owner)),))


@pytest.mark.parametrize("verdict", ["accepted", "refused"])
def test_row_only_link_uses_original_final_policy_without_contraction_signatures(tmp_path, exp_object, verdict):
    from merlin.llvmlower import toolchain

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle, work = tmp_path / "capture", tmp_path / "build"
    expected, trace = capture(bundle, 19)
    provider = RowProvider(tmp_path / "provider", columns=19, chunk=None, exp_object=exp_object)
    config = replace(reviewed_row_selection(provider, trace), execution_domain="device")
    policy = DiagnosticPolicy(verdict)
    route = DeviceRouting("neutral", tmp_path / "unused-package", "f32", "f32", linked_elf_admission=service(policy))
    if verdict == "refused":
        with pytest.raises(ValueError, match="refused by its selected policy"):
            build_model(bundle, work, provider, trace, config=config, device=route)
    else:
        result = build_model(bundle, work, provider, trace, config=config, device=route)
        assert result["linked_elf_admission"]["elf_sha256"] == sha256_file(result["elf"])
        console, _ = FB.run_host(result["elf"], timeout_s=30)
        actual, _ = FB.read(console, bundle, result["build_hash"])
        assert len(actual) == len(expected) == 9
        for got, want in zip(actual, expected, strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape
            np.testing.assert_array_equal(got.tobytes(), want.tobytes())
    assert policy.observed == [sha256_file(work / "model.elf")]
    recipe = json.loads((work / "compilation_recipe.json").read_text())
    assert (recipe["status"] == "completed") == (verdict == "accepted")
    assert not (work / "device").exists()


def test_backend_cannot_replace_original_device_policy_during_preparation(tmp_path, exp_object):
    bundle = tmp_path / "capture"
    _, trace = capture(bundle, 19)
    provider = RowProvider(tmp_path / "provider", columns=19, chunk=None, exp_object=exp_object)
    config = replace(reviewed_row_selection(provider, trace), execution_domain="device")
    policy = DiagnosticPolicy("accepted")
    route = DeviceRouting("neutral", tmp_path / "unused-package", "f32", "f32", linked_elf_admission=service(policy))

    def mutate(bindings, work):
        object.__setattr__(route, "linked_elf_admission", service(DiagnosticPolicy("accepted")))

    config = replace(config, prepare=mutate)
    with pytest.raises(ValueError, match="selection changed"):
        build_model(bundle, tmp_path / "build", provider, trace, config=config, device=route)
    assert not policy.observed
    assert not (tmp_path / "build" / "model.elf").exists()
