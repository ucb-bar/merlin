import hashlib

import pytest

from merlin.common.paths import module_source_path
from merlin.targetgen.application_inventory import _known_rational_reference

_MODULES = (
    "merlin.targetgen.exact_quantized_reference",
    "merlin.targetgen.exact_arithmetic",
    "merlin.runtime.exact_readout",
)


def _entry(name, root):
    digest = hashlib.sha256(module_source_path(name).read_bytes()).hexdigest()
    return dict(path=f"{root}/{name.replace('.', '/')}.py", sha256=digest)


def _source(root="/source/merlin-src", numerics="integer_readout"):
    source = _entry(_MODULES[0], root)
    source.update(arithmetic="exact_rational", numerics=numerics,
                  dependencies=[_entry(name, root) for name in _MODULES[1:]])
    return source


@pytest.mark.parametrize("root", ["/source/merlin-src", "/elsewhere/checkout/src", ""])
def test_sealed_worker_mount_paths_bind_by_module_and_bytes(root):
    assert _known_rational_reference(_source(root))


def test_float_readout_numerics_are_accepted_for_the_same_modules():
    assert _known_rational_reference(_source(numerics="float_readout"))


def test_changed_bytes_other_modules_and_missing_dependencies_are_refused():
    source = _source()
    changed = _source()
    changed["sha256"] = "0" * 64
    assert not _known_rational_reference(changed)
    changed = _source()
    changed["dependencies"][0]["sha256"] = "0" * 64
    assert not _known_rational_reference(changed)
    other = _source()
    other["path"] = "/source/merlin-src/merlin/targetgen/other.py"
    assert not _known_rational_reference(other)
    suffix = _source()
    suffix["path"] = "/source/merlin-src/xmerlin/targetgen/exact_quantized_reference.py"
    assert not _known_rational_reference(suffix)
    short = _source()
    short["dependencies"] = short["dependencies"][:1]
    assert not _known_rational_reference(short)
    swapped = _source()
    swapped["dependencies"].reverse()
    assert not _known_rational_reference(swapped)
    assert not _known_rational_reference(dict(source, arithmetic="float"))
    assert not _known_rational_reference(dict(source, numerics="other"))
    assert not _known_rational_reference(dict(source, dependencies="x"))
    assert not _known_rational_reference("x")
