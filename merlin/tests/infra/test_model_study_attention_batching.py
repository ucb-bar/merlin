"""Matched diagnostics preserve every product and original output coordinate."""

import importlib.util

import numpy as np
import pytest

from merlin.common.paths import repo_root


def _study():
    path = repo_root() / "examples/gemmini/model-study/attention_batching.py"
    spec = importlib.util.spec_from_file_location("attention_batch_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("groups,heads,rows,k,n", [(3, 2, 1, 7, 5), (2, 5, 3, 13, 11)])
def test_coordinate_mapping_preserves_every_exact_dot_with_signed_operands(groups, heads, rows, k, n):
    original, grouped = _study().products(groups=groups, heads=heads, rows=rows, reduction=k, columns=n, seed=19)
    a, b, expected = original
    ar, br, grouped_expected = grouped
    for g in range(groups):
        for h in range(heads):
            index = g * heads + h
            np.testing.assert_array_equal(a[index], ar[g, h * rows : (h + 1) * rows])
            np.testing.assert_array_equal(b[index], br[g])
            # Independent scalar accumulation, distinct from the vectorized oracle.
            for r in range(rows):
                for c in range(n):
                    value = sum(int(a[index, r, i]) * int(b[index, i, c]) for i in range(k))
                    assert value == expected[index, r, c] == grouped_expected[g, h * rows + r, c]
    assert np.any(a < 0) and np.any(a > 0)


def test_stimulus_refuses_undefined_geometry_and_overflow_before_allocation():
    module = _study()
    with pytest.raises(ValueError, match="positive integer"):
        module.products(groups=1, heads=0, rows=1, reduction=1, columns=1, seed=0)
    with pytest.raises(ValueError, match="accumulator range"):
        module.products(groups=1, heads=1, rows=1, reduction=2**31, columns=1, seed=0)


def test_output_cannot_modify_frozen_packages_even_through_alias(tmp_path):
    compiler, host = (tmp_path / n for n in ("compiler", "host"))
    compiler.mkdir()
    host.mkdir()
    alias = tmp_path / "compiler-alias"
    alias.symlink_to(compiler, target_is_directory=True)
    module = _study()
    for path in (compiler / "new-study", host / "new-study", alias / "new-study"):
        with pytest.raises(ValueError, match="outside compiler/host"):
            module.selected_output(path, compiler=compiler, host_package=host, repetitions=2)
    assert list(compiler.iterdir()) == list(host.iterdir()) == []
    output = module.selected_output(tmp_path / "new-study", compiler=compiler, host_package=host, repetitions=2)
    assert output == tmp_path / "new-study" and not output.exists()
