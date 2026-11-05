"""A host-only judge SDK cannot select the author profile's import path."""

import copy
from dataclasses import replace

import pytest

from merlin.targetgen.sandbox import toolchain as TC
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from test_bundle_sdk_import_roots import selected


def profile_bundle(original, name):
    bundle = copy.deepcopy(original)
    profile = A.resolve(name)
    bundle.update(allowed=[], tools=list(profile.tools), author_profile=profile.record())
    # This test starts with an independently valid original judge profile and
    # changes only the frozen treatment-tools field as materialize_bundle does.
    import hashlib
    import json

    judge = bundle['compiler_tool_profile']
    judge['treatment_tools'] = list(profile.tools)
    judge['sha256'] = hashlib.sha256(json.dumps(
        {key: value for key, value in judge.items() if key != 'sha256'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return bundle


@pytest.mark.parametrize('name', (*P.NAMES, *A.SCRATCH_NAMES, *A.SEEDED_NAMES))
def test_host_only_judge_sdk_does_not_select_a_profile_import_root(selected, name):
    bundle = profile_bundle(selected.bundle, name)
    paths = replace(selected.paths, python_import_roots=(str(selected.ws / 'profile-library'),))
    # No judge grant snapshot exists: none is author-readable in this profile.
    # In particular this must not mount or select selected.sdk merely because
    # the independent judge profile still pins its original source release.
    assert TC.paths_for_bundle(selected.ws, bundle, paths=paths) is paths
    assert str(selected.sdk) not in paths.python_import_roots


@pytest.mark.parametrize('damage', ['unknown_profile', 'changed_tools', 'changed_record'])
def test_profile_import_selection_still_requires_its_exact_sealed_identity(selected, damage):
    bundle = profile_bundle(selected.bundle, P.NAMES[0])
    if damage == 'unknown_profile':
        bundle['author_profile']['name'] = 'unreviewed-profile'
    elif damage == 'changed_tools':
        bundle['tools'] = []
    else:
        bundle['author_profile']['live_cca'] = True
    with pytest.raises(ValueError):
        TC.paths_for_bundle(selected.ws, bundle, paths=selected.paths)


def test_profile_does_not_skip_independent_judge_profile_integrity(selected):
    bundle = profile_bundle(selected.bundle, P.NAMES[0])
    bundle['compiler_tool_profile']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='compiler tool profile identity changed'):
        TC.paths_for_bundle(selected.ws, bundle, paths=selected.paths)
