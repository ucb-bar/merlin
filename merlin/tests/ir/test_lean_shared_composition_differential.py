"""Independent scalar selection and modular sums for composed actual canonicalization."""

import json
from types import SimpleNamespace

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_shared_composition import _constructor_cast_index_source

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("case", ["composed", "cast-roundtrip", "independent-constant-hoist"])
def test_composed_constructor_cast_and_index_native(tmp_path, fidelity_tools, index_bits, case):
    cancel = case == "cast-roundtrip"
    source = _constructor_cast_index_source(cancel=cancel, index_first=case != "independent-constant-hoist")
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    _, before, after = canonicalize(tmp_path, tools, source)
    negative = -(1 << (31 if cancel else 7))
    samples = [[[11], [37], [99]], [[-1], [negative], [0]]]
    expected = [["37", "22"], [str(negative & ((1 << 32) - 1)), "22"]]
    for name, path in [("source", before), ("target", after)]:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, fidelity_tools, function="forward", index_bits=index_bits)
        result = json.loads((work / "results.json").read_text())
        assert result["native"]["results"] == expected
        assert [[b for value in case["values"] for b in value["bits"]] for case in result["lean"]] == expected
