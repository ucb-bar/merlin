"""Which operators the PT2E quantizer calls a contraction, and what it counts before refusing.

WHY THIS EXISTS. ``_recipe_quantizer.build_quantizer`` maps each family a derived recipe lists onto the
ATen operators it will annotate. Its contraction row named ``conv2d`` and ``linear`` -- the spellings a
module assembled from ``nn.Conv2d``/``nn.Linear`` exports to -- and nothing else. A model that calls
``torch.matmul`` directly exports to ``aten.matmul.default``, matched none of them, and was refused by
``validate`` as "the model has no operator the recipe's families cover".

MEASURED, and it is why three capsules could not be generated: ``M3_host_island_seam_gemmini``'s whole
exported graph is ``2x aten.matmul.default``. That capsule is the accelerator/host-island/accelerator
seam -- the one capsule built to prove that composition -- and it could not be captured at all. The
failure surfaced as ``M2MUnavailable: worker exited non-zero (rc=1)``, which named neither the operator
nor the table, so the corpus simply had a hole in it.

``validate`` had the second half of the same bug: three counters are kept (contractions, operand sums,
window means) and it read one, so a model carrying only the sums or means a recipe also covers was
refused with a message asserting its families were uncovered.

CHECKED STRUCTURALLY, over the module's AST. ``torch`` is deliberately absent from the ordinary test
interpreter -- the capture venv owns it -- and this file's sibling says plainly why that matters: "a
test that skips is a test that passes either way". The operator table is a literal in the source, so
it can be read without importing torch, and reading it is enough to pin the regression.
"""

from __future__ import annotations

import ast

from merlin.common.paths import merlin_dir

_SOURCE = merlin_dir() / "python" / "merlin" / "targetgen" / "_recipe_quantizer.py"


def _operator_table() -> dict[str, list[str]]:
    """``{family: [dotted aten op, ...]}`` read out of the module's ``operators`` literal."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "operators" for t in node.targets):
            continue
        table: dict[str, list[str]] = {}
        for key, value in zip(node.value.keys, node.value.values):
            if not isinstance(key, ast.Constant):
                continue
            table[str(key.value)] = [
                ast.unparse(op) for op in (value.elts if isinstance(value, (ast.Tuple, ast.List)) else [])
            ]
        return table
    raise AssertionError("no `operators = {...}` table found in _recipe_quantizer")


class TestTheContractionFamilyNamesEveryMatmulSpelling:
    def test_the_table_exists_and_declares_a_contraction_family(self):
        table = _operator_table()
        assert "contraction" in table, f"families declared: {sorted(table)}"

    def test_a_bare_matmul_is_a_contraction(self):
        """THE REGRESSION. `M3_host_island_seam_gemmini` exports to `2x aten.matmul.default` and
        matched nothing, so the seam capsule could not be captured."""
        ops = _operator_table()["contraction"]
        assert any("matmul" in op for op in ops), (
            f"the contraction family names {ops}, none of which is a matmul -- a model that calls "
            f"torch.matmul directly annotates zero contractions and is refused as covering no family"
        )

    def test_the_rank_variants_are_named_beside_it(self):
        """Which of matmul/mm/bmm survives export is a property of the input ranks and of whatever
        decomposition ran, not of the model's intent. Matching one and not its siblings is the same
        bug at a different rank."""
        ops = " ".join(_operator_table()["contraction"])
        missing = [name for name in ("mm.default", "bmm.default") if name not in ops]
        assert not missing, f"contraction names no {missing}; it declares: {ops}"

    def test_the_module_spellings_are_still_there(self):
        """The original two must not be lost while adding the others."""
        ops = " ".join(_operator_table()["contraction"])
        assert "conv2d" in ops and "linear" in ops


class TestValidateCountsEveryFamilyItAnnotates:
    def test_the_refusal_reads_all_three_counters(self):
        """`annotate` increments `annotated`, `annotated_sums` and `annotated_means`; a refusal that
        reads only the first asserts the recipe's families are uncovered when two of them may be
        covered. Read structurally: the guard must mention every counter it is deciding about."""
        tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
        guards = [
            ast.unparse(node.test)
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(isinstance(raise_, ast.Raise) and "RecipeError" in ast.unparse(raise_) for raise_ in ast.walk(node))
            and "annotated" in ast.unparse(node.test)
        ]
        assert guards, "no RecipeError guard reading an annotation counter was found"
        joined = " ".join(guards)
        for counter in ("annotated", "annotated_sums", "annotated_means"):
            assert counter in joined, (
                f"the refusal does not read {counter!r}: a model carrying only that family would be "
                f"refused as covering none of the recipe's families. Guards found: {guards}"
            )


class TestAModelAlreadyInTheRecipesIntegersIsNotQuantizedAgain:
    """An observer learns a floating-point range; handed a tensor already on the grid there is none.

    `apply_recipe` quantized unconditionally, so a model whose inputs are already integral reached
    `HistogramObserver`, which calls `torch.histc`, which has no int8 kernel -- and the capture died
    with `NotImplementedError: "histogram_cpu" not implemented for 'Char'`, an error about a missing
    torch kernel several layers below the actual mistake.

    Measured on `M3_host_island_seam_gemmini`: its loader hands out one `torch.int8` input because the
    capsule's arithmetic IS integer. It is the host-island seam capsule, it generated before recipes
    existed, and the recipe path regressed it. With the guard it captures clean (`ok: true,
    opaque: 0`).
    """

    def test_apply_recipe_checks_for_a_floating_point_input_before_observing(self):
        tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
        fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "apply_recipe"),
            None,
        )
        assert fn is not None, "apply_recipe not found"
        body = ast.unparse(fn)
        assert "is_floating_point" in body, (
            "apply_recipe does not ask whether any example input is floating point, so a model already "
            "expressed in the recipe's integers is handed to an observer that cannot observe it"
        )

    def test_the_skip_is_recorded_rather_than_silent(self):
        """Skipping is right; skipping quietly is not. A reader of the capsule's provenance has to be
        able to tell 'quantized under this recipe' from 'already was what the recipe would produce'."""
        tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "apply_recipe")
        body = ast.unparse(fn)
        assert "_recipe_quantization_stats" in body and "'applied': False" in body.replace('"', "'"), (
            "the integer-input path must record that it applied nothing, and why"
        )
