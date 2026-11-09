from __future__ import annotations

from verification_witnesses.numerical_readback import (
    BW,
    HARNESS,
    RB,
    BuildOnlyService,
    HarnessBuildRecipe,
    N,
    Path,
    QualityBudget,
    QualityLimit,
    SimpleNamespace,
    _frame,
    _json,
    _rebind,
    base64,
    hashlib,
    json,
    native_case,
    np,
    pytest,
    shutil,
    subprocess,
    yaml,
)


def test_actual_native_full_outputs_are_admitted_without_hardware_claim(native_case):
    case = native_case
    (raw, _) = N.parse_console(case.arms[0].console.read_text())
    values = N.decode_float_readback(raw, {"F": "f32", "H": "bf16"})
    assert values["F"] == np.asarray(2 * (np.arange(10) - 3), np.float32).reshape(2, 5).tolist()
    assert values["I"] == (np.arange(10) ** 2 - 7).reshape(2, 5).tolist()
    observation = N.admit_protected_numerical_readback(**case.freeze())
    assert observation.quality.value_map == {"elementwise_violation_count": 0}
    assert observation.quality.parameters == case.budget.parameters
    assert observation.quality.complete and observation.element_count == 23
    assert observation.outputs == ("F", "H", "I")
    assert observation.missing and (not hasattr(observation, "hardware_verified"))
    assert all((RB.file_sha256(Path(path)) == digest for (path, digest) in observation.evidence))


@pytest.mark.parametrize("profile,expected", [("exact", 1), ("elementwise", 0)])
def test_signed_zero_uses_raw_bits_for_exact_policy(native_case, profile, expected):
    case = native_case
    console = case.arms[1].console.read_text()
    case.arms[1].console.write_text(
        console[: console.index("OUT_B64_BEGIN v1 H")] + _frame("H", (1, 3), [0, 16256, 49152], 2) + "DONE\n"
    )
    budget = QualityBudget.exact(case.budget.reference) if profile == "exact" else case.budget
    _json(case.private / "budget.json", budget.to_dict())
    result = N.admit_protected_numerical_readback(**case.freeze(budget))
    assert result.quality.value_map == {budget.limits[0].metric: expected}


@pytest.mark.parametrize("word,profile", [(2143363909, "exact"), (2143363909, "elementwise")])
def test_nonfinite_raw_exactness_and_elementwise_refusal_count(native_case, word, profile):
    case = native_case
    for arm in case.arms:
        console = arm.console.read_text()
        arm.console.write_text(_frame("F", (2, 5), [word] + [0] * 9) + console[console.index("OUT_B64_BEGIN v1 I") :])
    budget = QualityBudget.exact(case.budget.reference) if profile == "exact" else case.budget
    _json(case.private / "budget.json", budget.to_dict())
    with np.errstate(invalid="ignore", over="ignore"):
        result = N.admit_protected_numerical_readback(**case.freeze(budget))
    assert result.quality.value_map[budget.limits[0].metric] == (0 if profile == "exact" else 1)


@pytest.mark.parametrize("delta,expected", [(0.01, 0), (0.5, 1)])
def test_elementwise_recomputes_every_value_at_fixed_tolerance(native_case, delta, expected):
    case = native_case
    floats = np.asarray(2 * (np.arange(10) - 3), np.float32)
    floats[-1] += delta
    console = case.arms[1].console.read_text()
    case.arms[1].console.write_text(
        _frame("F", (2, 5), floats.view(np.uint32)) + console[console.index("OUT_B64_BEGIN v1 I") :]
    )
    result = N.admit_protected_numerical_readback(**case.freeze())
    assert result.quality.value_map == {"elementwise_violation_count": expected}
    assert result.element_count == 23
