from __future__ import annotations

from verification_witnesses.protected_verifier_qualification import (
    F,
    StageGateError,
    _native_case,
    _request,
    native_case,
    pytest,
    replace,
    sha256_file,
)


def test_json_receipt_and_reconstructed_record_never_mint_capability(native_case):
    verifier = _request(native_case).verifier
    capability = verifier.qualification
    with pytest.raises(StageGateError, match="issued qualification capability"):
        replace(verifier, qualification=capability.receipt).validate()
    with pytest.raises(StageGateError, match="not independently issued"):
        replace(verifier, qualification=replace(capability, _issuer=None)).validate()
    record = F._mapping(capability.receipt)
    assert len(record["controls"]) == 2 * len(F.REQUIRED_WITNESSES)
    assert "synthetic" in record["scope"]
    assert record["status"] == "observation_qualified"
    assert record["qualified_roles"] == ["private_witness_observation"]
    assert {row["status"] for row in record["controls"]} == {"observed", "refused"}


def test_negative_controls_must_actually_refuse(native_case):
    with pytest.raises(StageGateError, match="accepted an independently declared negative"):
        _request(native_case, control_fault="accept")


def test_verifier_crash_is_not_negative_control_success(native_case):
    with pytest.raises(RuntimeError, match="no diagnosed refusal"):
        _request(native_case, control_fault="crash")
