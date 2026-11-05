"""Keep an original source-pass selection live across ordinary object/link IO.

Static requests retain their exact source identity. An original host policy
binds once to the actual prepared input; it issues no permission, numerical
policy, target qualification or sandbox grant.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.llvmlower import source_optimization_policy as P
from merlin.llvmlower import source_optimization_selection as S
from merlin.llvmlower.source_optimization_request import request_from_record


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source optimization transport requires a direct regular artifact")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _record(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


class SourceOptimizationBuild:
    """Original snapshot, then an existing guard bound to actual prepared IR."""

    def __init__(self, selection, *, backend=None):
        self.selection, self._source, self._report, self._guard = selection, None, None, None
        self._completed = self._bound_selection = None
        self._original = self._library = None
        if selection is None:
            return
        if type(selection) not in (S.SourceOptimizationSelection, P.SourceOptimizationPolicy):
            raise ValueError("ordinary source choices require the original typed selection or policy")
        if backend is not None and backend not in ("scalar", "rvv"):
            raise ValueError("selected compiler backend has no source optimization consumer")
        self._original = _record(selection.record())
        from merlin.llvmlower import automatic_destinations, automatic_gathers

        directory = Path(S.__file__).parent
        self._library = [
            _pin(path)
            for path in dict.fromkeys(
                (
                    Path(__file__),
                    *((Path(P.__file__),) if type(selection) is P.SourceOptimizationPolicy else ()),
                    *(
                        directory / name
                        for name in (
                            "source_optimization_selection.py",
                            "source_optimization_request.py",
                            "optional_passes.py",
                        )
                    ),
                    *automatic_gathers.implementation_paths(),
                    *automatic_destinations.implementation_paths(),
                )
            )
        ]

    def revalidate(self):
        if self.selection is None:
            return
        if _record(self.selection.record()) != self._original or any(_pin(row["path"]) != row for row in self._library):
            raise ValueError("original source optimization selection changed during build/execution")
        if self._guard is not None:
            self._guard.revalidate()
            if _pin(self._source["path"]) != self._source:
                raise ValueError("actual prepared source changed after source choice consumption")
        if self._report is not None:
            if _pin(self._report["path"]) != self._report:
                raise ValueError("consumed source optimization report changed")
            report = json.loads(Path(self._report["path"]).read_bytes())
            try:
                expected = self._guard.report(
                    gather=report["consumers"]["automatic-host-gathers"],
                    packing=report["consumers"]["automatic-host-permutations"],
                )
            except (KeyError, TypeError) as error:
                raise ValueError("consumed source optimization report is incomplete") from error
            if _record(report) != _record(expected):
                raise ValueError("consumed source optimization report does not match original selection")
        if self._completed is not None:
            from merlin.llvmlower.compilation_recipe import verify_completed_recipe

            recipe, executable = self._completed
            if _pin(recipe["path"]) != recipe or _pin(executable["path"]) != executable:
                raise ValueError("source optimization completed build changed")
            actual = verify_completed_recipe(
                Path(recipe["path"]), executable=Path(executable["path"]), expected_recipe_sha256=recipe["sha256"]
            )
            bound = actual.get("preparation", {})
            if (
                bound.get("source_optimization_input") != self._source
                or bound.get("source_optimization_selection") != self._report
            ):
                raise ValueError("source optimization is absent from actual completed compilation")

    def bind(self, path):
        """Bind once after ordinary preparation, never refresh a stale input."""
        self._bind(path)

    def _bind(self, path, returned_selection=None):
        self.revalidate()
        if self.selection is None:
            return
        if self._guard is not None:
            raise ValueError("source optimization input is already bound")
        source = _pin(path)
        text = Path(source["path"]).read_text()
        if type(self.selection) is P.SourceOptimizationPolicy:
            selected = self.selection.bind(text) if returned_selection is None else returned_selection
            self.selection.verify_selection(selected, text)
            if selected.request.source_sha256 != source["sha256"]:
                raise ValueError("source optimization policy requires exact prepared source bytes")
        else:
            selected = self.selection
        self._guard = S.SourceCompilationOptimization(text, selected)
        self._bound_selection, self._source = selected, source
        self.revalidate()

    def options(self):
        self.revalidate()
        return {} if self.selection is None else {"source_optimizations": self._bound_selection or self.selection}

    def consumed(self, compilation, workdir):
        # The independent exact-preparation flag shares this ordinary lowering
        # boundary. Retain its actual effects even without a pass-policy request.
        report = Path(workdir) / "tensor-preparation-cse.json"
        if report.is_file():
            compilation.bind_preparation("tensor_preparation_sharing", report)
            compilation.bind_preparation("tensor_preparation_shared_source", Path(workdir) / "model.tensor-preparations.mlir")
        if self.selection is None:
            return
        if self._guard is None:
            raise ValueError("source optimization has no actual prepared input")
        self._report = _pin(Path(workdir) / "source-optimization-selection.json")
        self.revalidate()
        compilation.bind_preparation("source_optimization_input", Path(self._source["path"]))
        compilation.bind_preparation("source_optimization_selection", Path(self._report["path"]))

    def identity(self):
        self.revalidate()
        if type(self.selection) is P.SourceOptimizationPolicy:
            if self._bound_selection is None:
                raise ValueError("source optimization policy has no actual prepared input")
            return _record(
                {"policy": self._original, "selection": self._bound_selection.record(), "library": self._library}
            ).encode()
        return (
            b"" if self.selection is None else _record({"selection": self._original, "library": self._library}).encode()
        )

    def selection_record(self):
        self.revalidate()
        return None if self.selection is None else json.loads(self._original)

    def finish(self, compilation, executable):
        self.revalidate()
        compilation.completed(executable)
        if self.selection is not None:
            self._completed = (_pin(compilation.path), _pin(executable))

    def record(self):
        self.revalidate()
        if self.selection is None:
            return None
        if self._source is None or self._report is None or self._completed is None:
            raise ValueError("source optimization has no completed consumption")
        return json.loads(
            _record(
                {
                    "schema": "merlin.source_optimization_build.v2"
                    if type(self.selection) is P.SourceOptimizationPolicy
                    else "merlin.source_optimization_build.v1",
                    **(
                        {"policy": json.loads(self._original)}
                        if type(self.selection) is P.SourceOptimizationPolicy
                        else {}
                    ),
                    "selection": self._bound_selection.record(),
                    "library": self._library,
                    "input": self._source,
                    "report": self._report,
                    "recipe": self._completed[0],
                    "executable": self._completed[1],
                }
            )
        )

    def accept(self, record, *, build_dir, executable):
        """Reopen only a current ordinary producer, under this live selection."""
        self.revalidate()
        if self.selection is None:
            if record is not None:
                raise ValueError("unselected source optimization returned an unexpected receipt")
            return
        policy = type(self.selection) is P.SourceOptimizationPolicy
        expected_fields = {
            "schema",
            "selection",
            "library",
            "input",
            "report",
            "recipe",
            "executable",
        } | ({"policy"} if policy else set())
        expected_schema = "merlin.source_optimization_build.v2" if policy else "merlin.source_optimization_build.v1"
        if type(record) is not dict or set(record) != expected_fields or record["schema"] != expected_schema:
            raise ValueError("selected source optimization returned no exact producer receipt")
        expected = (Path(build_dir) / "lower/source-optimization-selection.json").resolve()
        if Path(record["report"]["path"]) != expected:
            raise ValueError("source optimization receipt names a foreign consumed report")
        returned_selection = None
        if policy:
            if _record(record["policy"]) != self._original:
                raise ValueError("source optimization receipt changed original policy")
            returned = record["selection"]
            if type(returned) is not dict or set(returned) != {
                "request",
                "original_permitted_passes",
                "original_maximum_added_operations",
            }:
                raise ValueError("source optimization receipt has no exact bound selection")
            returned_selection = S.SourceOptimizationSelection(
                request_from_record(returned["request"]),
                self.selection.permitted_passes,
                self.selection.original_maximum_added_operations,
            )
            if _record(returned_selection.record()) != _record(returned):
                raise ValueError("source optimization receipt changed original permissions or ceiling")
        self._bind(record["input"]["path"], returned_selection)
        self._report = _pin(expected)
        self._completed = (_pin(Path(build_dir) / "compilation_recipe.json"), _pin(executable))
        if _record(self.record()) != _record(record):
            raise ValueError("source optimization receipt changed original source, library or selection")
