"""Run a model's programs as a chain on one input, following the models file's input bindings.

Each program step's ``@forward`` model inputs (``in0``, ``in1``...) come from the input's model-ready
arrays, from earlier programs' outputs, from constants, or -- for a recurrent program's later steps --
from its own previous outputs (``carry``). Outputs leave a program in storage form (a bf16 tensor as its
16-bit pattern) and are fed onward in compute form, as a capture's ``inputs.npz`` states them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from merlin.model_check import catalog as CAT
from merlin.model_check.models import Model, Program

if TYPE_CHECKING:
    from merlin.runtime.dispatch_runtime import ForwardArgsCache
    from merlin.runtime.linalg_numpy import ConstantCache


class ChainError(RuntimeError):
    """A binding names something the chain does not have."""


class ChainCache:
    """What every case of one process shares: each program's parsed module, each stage's constant
    arguments, and the work :func:`linalg_numpy.evaluate` does on constants alone.

    Every cached value is a deterministic function of the same bytes, so a case run with the cache
    produces the bytes it produces without it. The cache only grows; use one per process.
    """

    def __init__(self) -> None:
        from merlin.runtime.dispatch_runtime import ForwardArgsCache
        from merlin.runtime.linalg_numpy import ConstantCache

        self.weights: ForwardArgsCache = ForwardArgsCache()
        self.constants: ConstantCache = ConstantCache()
        self._modules: dict[str, Any] = {}

    def module(self, path: Path):
        from merlin.frontends.linalg_mlir import parse_mlir_file

        key = str(Path(path).resolve())
        if key not in self._modules:
            self._modules[key] = parse_mlir_file(path)
        return self._modules[key]

    def arguments(self, catalog: CAT.ProgramCatalog, inputs: Mapping[str, np.ndarray], scratch: Path) -> list:
        """:func:`catalog.capture_arguments` from the cache, its constant arguments marked constant."""
        args = CAT.capture_arguments(catalog.stage, inputs, scratch, weights_cache=self.weights)
        for value in args:
            if self.weights.is_constant(value) and not self.constants.is_constant(value):
                self.constants.mark(value)
        return args


@dataclass
class ChainRun:
    """Every program's outputs per step (compute form), and the judged outputs in order."""

    steps: dict[str, list[list[np.ndarray]]] = field(default_factory=dict)

    def last(self, program: str) -> list[np.ndarray]:
        return self.steps[program][-1]


def result_elements(module) -> list[str]:
    """Element type token of each ``@forward`` result (``f32``, ``bf16``, ``i8``...)."""
    from merlin.runtime import linalg_numpy as LN

    fn = next(op for op in module.walk() if op.name == "func.func" and op.body.blocks)
    return [LN._elem(t) for t in fn.function_type.outputs.data]  # noqa: SLF001


def to_compute(values: list[np.ndarray], elements: list[str]) -> list[np.ndarray]:
    from merlin.runtime import linalg_numpy as LN

    return [np.asarray(LN._load(np.asarray(v), e)) for v, e in zip(values, elements, strict=True)]  # noqa: SLF001


def bind(program: Program, model: Model, arrays: Mapping[str, np.ndarray], step: int, run: ChainRun) -> dict:
    """``{"in0": ..., ...}`` for ``program`` at ``step``."""
    out: dict[str, np.ndarray] = {}
    for index, binding in enumerate(program.inputs):
        source = binding["from"]
        if step > 0 and "carry" in binding:
            value = run.last(program.name)[int(binding["carry"])]
        elif source == "input":
            value = arrays[binding["key"]]
        elif source == "tokens":
            prefill = int(model.chain["prefill_tokens"])
            tokens = np.asarray(arrays["tokens"], np.int64)
            value = tokens[:prefill][None] if binding["slice"] == "prefill" else tokens[prefill + step][None, None]
        elif source == "output":
            if binding["program"] not in run.steps:
                raise ChainError(f"{program.name} binds {binding['program']}, which has not run")
            value = run.last(binding["program"])[int(binding["output"])]
        elif source == "constant":
            value = np.asarray(binding["value"], dtype=binding["dtype"])
        else:  # pragma: no cover - the models loader validates sources
            raise ChainError(f"unknown binding source {source!r}")
        out[f"in{index}"] = np.asarray(value)
    return out


def run_reference(
    model: Model,
    catalogs: Mapping[str, CAT.ProgramCatalog],
    arrays: Mapping[str, np.ndarray],
    scratch: Path,
    *,
    record: Callable[[str, int, CAT.ProgramCatalog, list[np.ndarray]], None] | None = None,
    cache: ChainCache | None = None,
) -> ChainRun:
    """The INT8 reference of every program step on one input; ``record(program, step, catalog, args)``
    is offered each step's bound arguments (to record its device calls). ``cache`` (one per process,
    shared by every case) does each program's constant work once; the bytes are the same without it."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.runtime import linalg_numpy as LN

    run = ChainRun()
    for program in model.programs:
        catalog = catalogs[program.name]
        module = cache.module(catalog.ref_prepared) if cache is not None else parse_mlir_file(catalog.ref_prepared)
        elements = result_elements(module)
        arity = CAT._function_arity(module)  # noqa: SLF001
        run.steps[program.name] = []
        for step in range(program.steps):
            inputs = bind(program, model, arrays, step, run)
            if cache is not None:
                args = cache.arguments(catalog, inputs, scratch / program.name)
            else:
                args = CAT.capture_arguments(catalog.stage, inputs, scratch / program.name)
            if record is not None:
                record(program.name, step, catalog, args)
            values = LN.evaluate(module, args[:arity], constants=cache.constants if cache is not None else None)
            run.steps[program.name].append(to_compute([np.asarray(v) for v in values], elements))
    return run


def judged_outputs(model: Model, run: ChainRun) -> list[np.ndarray]:
    """The judged outputs in order: every step of every judged (program, output)."""
    out: list[np.ndarray] = []
    for ref in model.judged["outputs"]:
        for step_values in run.steps[ref["program"]]:
            out.append(np.asarray(step_values[int(ref["output"])]))
    return out


def summary(run: ChainRun) -> dict[str, Any]:
    return {name: len(steps) for name, steps in run.steps.items()}
