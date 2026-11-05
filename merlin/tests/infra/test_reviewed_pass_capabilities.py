"""Original live capability owners are selected exactly, never from JSON verdicts."""

import hashlib
from dataclasses import replace

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)


class OriginalOwner:
    """A narrow structural test proof, not hardware/timing qualification."""

    def __init__(self, receipt):
        self.receipt = receipt
        self.identity = sha256_file(receipt)
        self.checks = 0

    def revalidate(self):
        if sha256_file(self.receipt) != self.identity:
            raise ValueError("original structural evidence changed")
        self.checks += 1


def case(tmp_path):
    receipt = tmp_path / "structural-evidence.txt"
    receipt.write_text("original structural observation; not event/hardware qualification\n")
    owner = OriginalOwner(receipt)
    digest = hashlib.sha256(b"selected original identity").hexdigest()
    binding = PassCapabilityBinding(
        "rtl:" + digest, "sw:" + digest, "original-numerics", "runtime:" + digest, "library:" + digest
    )
    capability = ReviewedPassCapability("timing_model_correspondence", binding, owner)
    registry = ReviewedPassCapabilities((capability,))
    return binding, owner, capability, registry


def test_exact_structural_owner_selection_rechecks_original_implementation_and_evidence(tmp_path):
    binding, owner, capability, registry = case(tmp_path)
    before = owner.checks
    selected = registry.resolve("timing_model_correspondence", binding)
    assert selected is capability and selected.owner is owner and owner.checks > before
    record = registry.record()
    assert record["capabilities"][0]["owner_identity"] == owner.identity
    assert "no qualification" in record["capabilities"][0]["authority"]
    assert "disabled" in record["missing_capability"]
    record["capabilities"][0]["binding"]["runtime"] = "forged"
    assert registry.resolve("timing_model_correspondence", binding).owner is owner
    assert registry.resolve("configuration_cleanup", binding) is None
    assert ReviewedPassCapabilities().resolve("timing_model_correspondence", binding) is None


@pytest.mark.parametrize("field", ["hardware", "software", "numerical_policy", "runtime", "library"])
def test_other_facts_spec_policy_runtime_or_library_never_use_nearest_owner(tmp_path, field):
    binding, _owner, _capability, registry = case(tmp_path)
    assert (
        registry.resolve("timing_model_correspondence", replace(binding, **{field: "another exact identity"})) is None
    )


@pytest.mark.parametrize("change", ["evidence", "identity", "owner", "callback", "binding", "registry"])
def test_changed_or_reconstructed_owner_cannot_replace_original_selection(tmp_path, monkeypatch, change):
    binding, owner, capability, registry = case(tmp_path)
    if change == "evidence":
        owner.receipt.write_text("new copied observation\n")
    elif change == "identity":
        owner.identity = "forged"
    elif change == "owner":
        capability._owner = OriginalOwner(owner.receipt)
    elif change == "callback":
        monkeypatch.setattr(OriginalOwner, "revalidate", lambda self: None)
    elif change == "binding":
        object.__setattr__(binding, "library", "different release")
    else:
        registry._capabilities = ()
    with pytest.raises(ValueError, match="changed|original"):
        registry.resolve("timing_model_correspondence", binding)


def test_saved_dictionary_label_or_duplicate_key_never_grants_executable_owner(tmp_path):
    binding, owner, capability, registry = case(tmp_path)
    with pytest.raises(ValueError, match="original reviewed owners"):
        ReviewedPassCapabilities((capability.record(),))
    with pytest.raises(ValueError, match="original identity"):
        ReviewedPassCapability("timing_model_correspondence", binding, {"identity": owner.identity})
    with pytest.raises(ValueError, match="ambiguous"):
        ReviewedPassCapabilities((capability, capability))
    with pytest.raises(ValueError, match="immutable"):
        ReviewedPassCapabilities([capability])
    with pytest.raises(ValueError, match="exact original binding"):
        registry.resolve("timing_model_correspondence", vars(binding))


@pytest.mark.parametrize("value", [None, "", 12, "a\nb"])
def test_missing_original_identity_refuses_instead_of_selecting_a_default(value):
    with pytest.raises(ValueError, match="explicit bounded original identity"):
        PassCapabilityBinding("hardware", "software", "policy", value, "library")
