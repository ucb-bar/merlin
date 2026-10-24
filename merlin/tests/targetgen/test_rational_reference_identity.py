"""A materialized receipt can bind the registered independent rational oracle."""

import copy
import hashlib

from merlin.common.paths import module_source_path
from merlin.targetgen.application_inventory import _known_rational_reference


def test_only_registered_source_bytes_and_complete_dependencies_are_accepted():
    def identity(name):
        path = module_source_path(name).resolve()
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    source = identity("merlin.targetgen.exact_quantized_reference")
    source.update(
        arithmetic="exact_rational",
        numerics="integer_readout",
        dependencies=[identity("merlin.targetgen.exact_arithmetic"), identity("merlin.runtime.exact_readout")],
    )
    assert _known_rational_reference(source)
    for key, value in [
        ("path", "/tmp/other.py"),
        ("sha256", "0" * 64),
        ("dependencies", []),
        ("arithmetic", "fp32"),
        ("numerics", "unchecked"),
    ]:
        changed = copy.deepcopy(source)
        changed[key] = value
        assert not _known_rational_reference(changed)
    changed = copy.deepcopy(source)
    changed["dependencies"][1]["sha256"] = "0" * 64
    assert not _known_rational_reference(changed)
