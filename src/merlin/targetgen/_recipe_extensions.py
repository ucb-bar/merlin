"""Explicit generic integer recipe extensions inside the capture interpreter.

The operators in the rewritten graph define the new quantized reference. This
module records their actual contraction count; it supplies no target schedule,
placement, timing or execution verdict.
"""

from __future__ import annotations


class Extensions:
    def __init__(self, recipe, epilogue):
        self.recipe, self.epilogue = recipe, epilogue
        self.residual = None
        self.reports = {}
        keys = ("accumulator_residual", "integer_mean", "projection_input_reuse")
        if any(key in recipe for key in keys) and epilogue is None:
            raise ValueError("integer recipe extensions require an explicit epilogue")
        if "accumulator_residual" in recipe:
            if "operand_sum_grid" in recipe:
                raise ValueError("residual accumulator and shared operand-sum policies have distinct ownership")
            import _integer_residual as residual

            self.residual = residual.AccumulatorResidual(recipe["accumulator_residual"])
            epilogue.residual = self.residual
        if "integer_mean" in recipe:
            import _integer_pool as pool

            pool.check(recipe["integer_mean"])
        if "projection_input_reuse" in recipe:
            import _projection_input_reuse as projection

            projection.validate(recipe["projection_input_reuse"], epilogue.block["scale"]["columns_per_scale"])

    def select(self, nodes, eligible, quantized_reader):
        if self.residual is None:
            return
        candidates = [n for n in nodes if n.target in self.epilogue.contractions and eligible(n)]
        # Select structural ownership first: an earlier residual can also feed
        # the next residual's skip edge. Its observer is a real quantized reader.
        for node in candidates:
            self.epilogue.pattern(node)
        by_name = {n.name: n for n in nodes}
        while True:
            before = set(self.residual.selected)
            for node in candidates:
                site = self.residual.selected.get(node.name)
                if site is None:
                    continue
                last = by_name[site["last"]]
                bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
                if self.epilogue.exit_of(node, last, bias is not None, quantized_reader) != "requant":
                    self.residual.discard(node)
            if set(self.residual.selected) == before:
                break
        self.residual.close_selection()

    def owns(self, node):
        return self.residual is not None and self.residual.owns(node)

    def freeze(self, prepared):
        if self.residual is not None:
            self.residual.freeze(prepared)

    def finish(self, quantized):
        if self.residual is not None:
            self.reports["accumulator_residual"] = self.residual.census()
        if "integer_mean" in self.recipe:
            import _integer_pool as pool

            self.reports["integer_mean"] = pool.rewrite(
                quantized, self.recipe["integer_mean"], epilogue=self.epilogue.block,
            )
        if "projection_input_reuse" in self.recipe:
            import _projection_input_reuse as projection

            self.reports["projection_input_reuse"] = projection.rewrite(
                quantized, self.recipe["projection_input_reuse"],
                columns_per_scale=self.epilogue.block["scale"]["columns_per_scale"],
            )
        return self.reports

    def physical_contractions(self, logical):
        return (logical + self.reports.get("accumulator_residual", {}).get("extra_contractions", 0)
                + self.reports.get("integer_mean", {}).get("new_contractions", 0)
                - self.reports.get("projection_input_reuse", {}).get("contractions_removed", 0))
