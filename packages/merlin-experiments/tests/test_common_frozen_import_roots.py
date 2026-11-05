"""No empty optional directory invents an import; nonempty mappings stay exact."""
from pathlib import Path

import pytest

from merlin_experiments.phase1.brokers.common_transforms import _frozen_import_root


def test_empty_manifest_root_is_omitted_and_nonempty_root_requires_one_frozen_owner():
    root = Path('/selected/lib')
    assert _frozen_import_root(root, [Path('/other/X.olean')], [Path('/frozen/other/X.olean')]) is None
    originals = [root / 'A.olean', root / 'Nested/B.olean']
    frozen = [Path('/frozen/lib/A.olean'), Path('/frozen/lib/Nested/B.olean')]
    assert _frozen_import_root(root, originals, frozen) == Path('/frozen/lib')
    with pytest.raises(ValueError, match='exact frozen file closure'):
        _frozen_import_root(root, originals, [frozen[0], Path('/different/Nested/B.olean')])
