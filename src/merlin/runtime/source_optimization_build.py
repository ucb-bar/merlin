"""Keep an original source-pass selection live across ordinary object/link IO.

This is transport and freshness checking for the existing source selection. It
creates no permission, source hash, numerical policy or target qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.llvmlower import source_optimization_selection as S


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
        self._completed = None
        self._original = self._library = None
        if selection is None:
            return
        if type(selection) is not S.SourceOptimizationSelection:
            raise ValueError("ordinary source choices require the original typed selection")
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
        self.revalidate()
        if self.selection is None:
            return
        source = _pin(path)
        text = Path(source["path"]).read_text()
        self._guard = S.SourceCompilationOptimization(text, self.selection)
        self._source = source
        self.revalidate()

    def options(self):
        self.revalidate()
        return {} if self.selection is None else {"source_optimizations": self.selection}

    def consumed(self, compilation, workdir):
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
                    "schema": "merlin.source_optimization_build.v1",
                    "selection": json.loads(self._original),
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
        if type(record) is not dict or set(record) != {
            "schema",
            "selection",
            "library",
            "input",
            "report",
            "recipe",
            "executable",
        }:
            raise ValueError("selected source optimization returned no exact producer receipt")
        expected = (Path(build_dir) / "lower/source-optimization-selection.json").resolve()
        if Path(record["report"]["path"]) != expected:
            raise ValueError("source optimization receipt names a foreign consumed report")
        self.bind(record["input"]["path"])
        self._report = _pin(expected)
        self._completed = (_pin(Path(build_dir) / "compilation_recipe.json"), _pin(executable))
        if _record(self.record()) != _record(record):
            raise ValueError("source optimization receipt changed original source, library or selection")
