"""The deterministic test stimulus must expose bugs, stay reproducible, and agree with its C twin.

The fill used to be indexed by the FLAT position with a period-4 expression, which made every row of
every ML-shaped operand identical — 4 of 6 deliberately wrong gemmini matmuls then passed the grader.
These tests pin the properties that failure violated, so a future "simplification" of the fill cannot
quietly reintroduce it.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common import stimulus as STIM
from merlin.common.paths import repo_root
from merlin.runtime.tensor import Tensor


def _rows(t: Tensor, cols: int) -> list[tuple]:
    return [tuple(t.data[r * cols:(r + 1) * cols]) for r in range(len(t.data) // cols)]


def test_fill_is_deterministic_across_calls():
    assert STIM.fill("A0", (16, 16)) == STIM.fill("A0", (16, 16))


def test_distinct_names_give_distinct_data():
    assert STIM.fill("A0", (16, 16)) != STIM.fill("A1", (16, 16))


def test_values_stay_in_range():
    vals = set(STIM.fill("W", (8, 12), lo=0, hi=3))
    assert vals <= {0, 1, 2, 3} and len(vals) == 4


@pytest.mark.parametrize("shape", [(16, 16), (16, 64), (64, 64), (20, 24), (32, 8)])
def test_ml_shaped_operands_have_distinct_rows_and_columns(shape):
    """The regression that mattered: a row length divisible by 4 used to collapse every row to one."""
    rows, cols = shape
    t = Tensor.deterministic("X", shape, "i8")
    grid = _rows(t, cols)
    assert len(set(grid)) == rows, f"duplicate rows in {shape}: {len(set(grid))} of {rows}"
    colset = {tuple(grid[r][c] for r in range(rows)) for c in range(cols)}
    assert len(colset) == cols, f"duplicate columns in {shape}: {len(colset)} of {cols}"


def test_square_operand_is_not_symmetric():
    """A == A^T would hide any transpose / layout bug."""
    t = Tensor.deterministic("A0", (16, 16), "i8")
    g = _rows(t, 16)
    assert any(g[r][c] != g[c][r] for r in range(16) for c in range(16))


def test_rank1_operand_varies_along_its_axis():
    assert len(set(STIM.fill("bias", (16,)))) > 1


def test_grid_shape_collapses_leading_dims():
    assert STIM.grid_shape((4,)) == (1, 4)
    assert STIM.grid_shape((2, 3, 4)) == (6, 4)
    assert STIM.grid_shape(()) == (1, 1)


def test_a_wrong_matmul_changes_the_output():
    """The mutation the old stimulus could not catch: reading the operand rows in reverse order."""
    M = K = N = 16
    A = Tensor.deterministic("A0", (M, K), "i8")
    W = Tensor.deterministic("W", (K, N), "i8")
    ref = [sum(A.data[m * K + k] * W.data[k * N + n] for k in range(K))
           for m in range(M) for n in range(N)]
    reversed_m = [sum(A.data[(M - 1 - m) * K + k] * W.data[k * N + n] for k in range(K))
                  for m in range(M) for n in range(N)]
    assert reversed_m != ref


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a host C compiler")
def test_emitted_c_fill_matches_the_python_fill(tmp_path):
    """The baremetal reference programs fill their own leaves in C. If that C ever disagreed with the
    Python golden, a 'three-way bit-exact' agreement would be comparing different problems."""
    rows, cols, name = 16, 24, "A0"
    src = tmp_path / "fill.c"
    src.write_text(
        "#include <stdint.h>\n#include <stdio.h>\ntypedef signed char elem_t;\n"
        f"#define R {rows}\n#define C {cols}\n#define S {STIM.det_seed(name)}\n"
        "static elem_t A[R][C];\n" + STIM.C_MIX_FN + "int main(void){\n"
        + STIM.c_fill_loop_2d("A", "R", "C", "S") + "\n"
        "for(int r=0;r<R;r++)for(int c=0;c<C;c++)printf(\"%d\\n\",(int)A[r][c]);return 0;}\n")
    exe = tmp_path / "fill"
    subprocess.run(["gcc", "-O2", "-o", str(exe), str(src)], check=True, capture_output=True)
    got = [int(x) for x in subprocess.run([str(exe)], check=True, capture_output=True,
                                          text=True).stdout.split()]
    assert got == Tensor.deterministic(name, (rows, cols), "i8").data


def test_no_capsule_operand_of_the_graded_corpus_is_degenerate():
    """Corpus-level guard: no shipped capsule, on ANY target, may regress to hiding bugs.

    Two blind spots this had, both of the same kind -- a check that examines nothing reports success:

      * it named three capsule roots (`isa`, `layers`, `model_slices`), so the operands of every other
        target's corpus were never looked at. Measured 2026-09-01: six further roots ship capsules.
      * it skipped any operand that was not rank 2, so a batched operand was never checked. A rank-3
        operand collapses to a (rows, cols) grid the same way the fill itself does -- there is nothing
        about it that cannot be compared.

    Roots are DISCOVERED, and the per-root counts are asserted non-zero, so a root that stops
    contributing operands fails here instead of quietly reducing the corpus under test.
    """
    import yaml
    from merlin.targetgen.corpus_operands import stimulus_rigor_findings
    root = repo_root() / "merlin" / "contract" / "capsules"
    flagged: list[str] = []
    checked: dict[str, int] = {}
    for cat_dir in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")):
        cat = cat_dir.name
        if cat in ("conformance", "profiles"):          # specs and policies, not capsules
            continue
        for cy in sorted(cat_dir.rglob("capsule.yaml")):
            try:
                spec = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as e:
                flagged.append(f"{cat}/{cy.parent.name}: unparseable capsule.yaml ({e})")
                continue
            for inp in spec.get("inputs", []) or []:
                if inp.get("role") not in ("input", "weight", "bias"):
                    continue
                shape = tuple(int(d) for d in (inp.get("shape") or ()))
                if not shape:
                    continue
                t = Tensor.deterministic(inp["name"], shape, inp.get("dtype", "i8"))
                checked[cat] = checked.get(cat, 0) + 1
                # `Tensor.deterministic` fills over lo..hi = 0..3 whatever the dtype, so the operand's
                # symbol alphabet is 4. Judging distinctness against the alphabet is what separates a
                # collapse from an arithmetic impossibility: a 16-column single-row operand cannot have
                # more than 4 distinct columns however good the fill is.
                found = stimulus_rigor_findings(t.data, shape, alphabet=4)
                if found:
                    flagged.append(f"{cat}/{cy.parent.name}:{inp['name']}{shape} -> {found[0]}")
    assert checked, f"no capsule operand was examined at all under {root} -- the guard is vacuous"
    assert not flagged, ("degenerate capsule operands (an operand that hides a bug makes its capsule's "
                         f"pass meaningless): {flagged}")
