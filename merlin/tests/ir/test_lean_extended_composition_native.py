"""Independent signed-bit references for composed cast, insertion and comparison rewrites."""

from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_extended_composition import source
from test_lean_integer_fold import canonicalize
from test_lean_minmax_commutative_native import compare_expected


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("escape", ["none", "tensor", "wide", "both"])
def test_actual_composed_aliases_preserve_all_observable_bits(tmp_path, fidelity_tools, kind, escape):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(kind, escape))
    certify(stage, before, tools, tmp_path)
    samples, expected = [], []
    for i, bits in enumerate([0, 1, 5, 6, 127, 128, 255]):
        value = bits - 256 if kind == "extsi" and bits >= 128 else bits
        keep = (i * 131 + 37) & 4294967295
        samples.append([[bits], [(i * 97 + 109) & 8191], [keep]])
        outputs = [(value + int(5 < value)) & 8191, keep]
        if escape in {"tensor", "both"}:
            outputs.append(value & 8191)
        if escape in {"wide", "both"}:
            outputs.append(value & 4294967295)
        expected.append(outputs)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)
