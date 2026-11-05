"""LLVM syntax interoperability must preserve the reviewed author source closures."""
import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin_experiments.phase1 import author_profiles as A


@pytest.mark.parametrize('level,expected_modules', [(1, 0), (2, 15), (3, 34)])
def test_raw_rtl_source_preparation_keeps_exact_reviewed_leaf_inventory(tmp_path, level, expected_modules):
    # Existing profile tests permit a private complete source fixture while a
    # change is under review; normal frozen/check-out runs use their own tree.
    source = Path(os.environ.get('MERLIN_AUTHOR_PROFILE_TEST_REPO', str(repo_root())))
    name = f'xs-el-raw-rtl-v2/el{level}'
    selection = A.prepare_sources(source, tmp_path/'sources', profile=name,
                                  review_id='test LLVM parser source dependency boundary')
    A.verify_sources(name, selection)
    modules = [] if selection is None else [m for _, _, m in selection.members if m is not None]
    assert len(modules) == expected_modules
    assert 'merlin.targetgen.contract.toolchain' not in modules
    assert not any(m.startswith('merlin.llvmlower') for m in modules)
