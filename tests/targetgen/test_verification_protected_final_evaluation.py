from __future__ import annotations

from verification_witnesses.protected_final_evaluation import (
    VERIFIER,
    F,
    ModuleType,
    Q,
    ReadbackFiles,
    SimpleNamespace,
    StageGateError,
    _json,
    _native_case,
    _rebind,
    _recompile,
    _request,
    json,
    native_case,
    pytest,
    replace,
    sha256_file,
    shutil,
    subprocess,
    sys,
)


def test_actual_observation_controls_cannot_authorize_physical_final_admission(native_case):
    request = _request(native_case)
    observed = F.observe_protected_final_comparison(**vars(request))
    assert observed.accuracy_passed and observed.physical_status == "UNKNOWN"
    capability = request.verifier.qualification
    declaration = {
        "status": "qualified",
        "hardware_sha256": request.binding.hardware_sha256,
        "runtime_sha256": request.binding.runtime_sha256,
        "timer_scope_sha256": request.binding.timer_scope_sha256,
    }

    def must_not_run(**kwargs):
        pytest.fail("a supplied physical callback was treated as authority")

    for authority in (capability, declaration):
        changed = replace(request, physical_execution_domain=authority)
        with pytest.raises(StageGateError, match="physical final admission UNKNOWN"):
            F.admit_protected_final_comparison(**vars(changed))
        with pytest.raises(StageGateError, match="physical final admission UNKNOWN"):
            F.evaluate_protected_final_campaign(
                (changed,),
                original_campaign=request.original_binding.parent / "campaign.json",
                expected_members=(request.binding.member,),
            )
    from merlin.development.backend.optimization.component_final_policy import strict_final_component_campaign_gate

    with pytest.raises(StageGateError, match="typed immutable comparison"):
        strict_final_component_campaign_gate((observed,), expected_members=(observed.member,))


def test_cycle_boolean_is_not_hardware_cycles(native_case):
    request = _request(native_case, candidate_cycles=True)
    with pytest.raises(StageGateError, match="positive reported"):
        F.observe_protected_final_comparison(**vars(request))


def test_two_approximations_do_not_double_original_tolerance(native_case):
    case = native_case
    approximate_dir = case.private / "approximate-reference"
    shutil.copytree(case.arms[0].harness.parent, approximate_dir)
    approximate = ReadbackFiles(*(approximate_dir / path.name for path in case.arms[0].paths()[:6]))
    _recompile(case, approximate)
    _recompile(case, case.arms[1])
    _recompile(case, case.arms[1])
    case.arms.append(approximate)
    _rebind(case, 2)
    _rebind(case, 1)
    request = _request(case, reference=approximate, original_reference=case.arms[0])
    pair_only = F.admit_protected_numerical_readback(
        run_dir=request.run_dir,
        environment=request.environment,
        original_budget=request.original_budget,
        budget=request.budget,
        reference=approximate,
        candidate=request.candidate,
        reference_build=case.service,
        candidate_build=case.service,
    )
    assert dict(pair_only.quality.values)["elementwise_violation_count"] == 0
    row = F.observe_protected_final_comparison(**vars(request))
    assert not row.accuracy_passed
    with pytest.raises(StageGateError, match="physical final admission UNKNOWN"):
        F.evaluate_protected_final_campaign(
            (request,),
            original_campaign=request.original_binding.parent / "campaign.json",
            expected_members=(row.member,),
        )
