"""A regenerated gemmini input bundle grants nothing from the retired hardware-bringup tree.

``merlin/experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0`` holds an
``isa_include`` link to the vendor ISA headers, which are excluded from candidate bundles.
Historical bundles under that experiment still list it; they are evidence and stay untouched. What
matters for a fresh run is what ``corpus prepare`` emits NOW: it scaffolds every arm's bundle with
:func:`merlin.targetgen.generate_bundles.generate_bundles` from the example descriptor, for the
public, realistic and hardware-bringup variants. This regenerates exactly those grants in memory,
the way ``scaffold`` does (no live resource/contract roots), and proves none reaches the tree --
neither the directory itself, nor anything inside it, nor an ancestor that would expose it.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import python_source_dir, repo_root
from merlin.targetgen.generate_bundles import _ALL_ARMS, generate_bundles
from merlin.targetgen.sandbox.bwrap import resolve_grant
from merlin.targetgen.target_experiment import load_target_experiment

pytestmark = pytest.mark.target("gemmini")

#: The variants ``corpus prepare`` (``preparation.scaffold``) materializes.
_VARIANTS = ("public_v0", "realistic_v0", "hwbringup_v0")
_RETIRED = "merlin/experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0"


def _descriptor():
    descriptor = load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")
    assert descriptor.isa_headers == () and descriptor.hwbringup_set is None
    assert descriptor.curated_harness is None and descriptor.declared_contract is None
    return descriptor


@pytest.mark.parametrize("variant", _VARIANTS)
def test_no_regenerated_bundle_grants_the_retired_bringup_tree(variant):
    root = repo_root().resolve()
    retired = (root / _RETIRED).resolve(strict=False)
    manifests = generate_bundles(
        _descriptor(), variant=variant, arms=tuple(_ALL_ARMS), python_source_root=python_source_dir()
    )
    assert manifests, "no bundle was generated, so this proves nothing"
    for bundle_id, manifest in manifests.items():
        grants = [entry["path"] for entry in manifest["allowed"]]
        assert grants, bundle_id
        for grant in grants:
            assert "hwbringup_gemmini_v0" not in grant and "isa_include" not in grant, (bundle_id, grant)
            resolved = resolve_grant(grant, root)
            if resolved is None:
                continue
            target = resolved.resolve(strict=False)
            assert not target.is_relative_to(retired), (bundle_id, grant)
            assert not retired.is_relative_to(target), (bundle_id, grant, "an ancestor grant exposes it")


def test_no_regenerated_bundle_grants_a_vendor_isa_header():
    """No grant names a file of the vendor header family, wherever it lives."""
    for variant in _VARIANTS:
        for manifest in generate_bundles(_descriptor(), variant=variant, arms=tuple(_ALL_ARMS)).values():
            for entry in manifest["allowed"]:
                assert not entry["path"].rstrip("/").endswith(("gemmini.h", "gemmini_nn.h", "gemmini_params.h"))
