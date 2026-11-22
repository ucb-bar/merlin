"""Shared refusal type for outlined model inputs and reference dispatch."""


class DispatchRuntimeError(RuntimeError):
    """An outlined model cannot be evaluated by the reference runtime."""
