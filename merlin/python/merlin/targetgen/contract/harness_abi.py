"""The harness ABI a target's contract declares: how a runner-owned harness calls into its kernel.

The contract splits responsibility — the package emits a kernel FUNCTION, the runner owns the harness
that embeds the leaf tensors, calls that function, and prints the OUT/METRIC/DONE protocol. To write
that harness the runner needs four facts it cannot derive from RTL, because they are software
conventions of the target's own test environment rather than properties of its hardware:

* the **entry symbol** the kernel is emitted under,
* the **fence symbol** (if any) that must be called before reading results back,
* the **includes** the harness needs for that environment's helpers, and
* the **cycle-window metric** name the runner's parser attributes cycles to.

Before this block existed these were literals in the generic compile path, which meant a target whose
test harness spelled any of them differently could not use that path at all. Note that they are
genuinely undeducible: nothing in a fact bundle says a kernel is called ``foo_kernel`` rather than
``bar_entry``. That is exactly why they belong in a contract the target authors, and why this module
refuses rather than guessing — a substituted default here produces a harness that compiles and then
fails to link, with the error pointing at the linker instead of at the missing declaration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class HarnessAbiError(ValueError):
    """A target's contract does not declare a usable harness ABI (fail closed, never a default)."""


@dataclass(frozen=True)
class HarnessAbi:
    """What a runner needs in order to write a harness that calls this target's kernel."""

    #: The symbol the package's lowered MLIR defines the kernel under.
    entry_symbol: str
    #: Called after the kernel and before results are read; None when the target needs no fence.
    fence_symbol: str | None = None
    #: Header paths the harness must include, verbatim, in declaration order.
    includes: tuple[str, ...] = ()
    #: Metric name marking that the measured cycle window covers the accelerator region. None when the
    #: target does not distinguish, in which case no such METRIC line is printed — an absent metric and
    #: one printed as 0 are read identically by the parser, so emitting a falsy one says nothing.
    cycle_window_metric: str | None = None
    #: Extra declarations the harness emits before ``main`` (e.g. an ``extern`` for the entry symbol).
    extern_decls: tuple[str, ...] = field(default_factory=tuple)

    def declarations(self) -> str:
        """The include + extern preamble, as C source."""
        lines = [f'#include "{h}"' if not h.startswith("<") else f"#include {h}"
                 for h in self.includes]
        lines.extend(self.extern_decls or (f"extern void {self.entry_symbol}();",))
        return "\n".join(lines)

    def call(self, args: str) -> str:
        """The kernel call, plus the fence when the target declares one."""
        call = f"  {self.entry_symbol}({args});"
        return f"{call}\n  {self.fence_symbol}();" if self.fence_symbol else call

    def cycle_window_line(self) -> str:
        """The METRIC line attributing the cycle window, or empty when the target declares none."""
        if not self.cycle_window_metric:
            return ""
        return f'  printf("METRIC {self.cycle_window_metric} 1\\n");'


def from_contract(contract: dict[str, Any], *, target: str) -> HarnessAbi:
    """Read the ``harness_abi`` block, refusing anything incomplete.

    Raises rather than filling in a default. The literals this replaces were one target's, so a default
    would silently reintroduce exactly the weld the block exists to remove — and would do it in the one
    place hardest to notice, since the resulting harness looks correct until it fails to link.
    """
    block = (contract or {}).get("harness_abi")
    if not isinstance(block, dict) or not block:
        raise HarnessAbiError(
            f"target {target!r} declares no `harness_abi` block in its target contract; the runner "
            f"cannot write a harness without the entry symbol it should call. Add the block (see "
            f"merlin/schemas/target_contract.schema.yaml) rather than relying on a default.")
    entry = block.get("entry_symbol")
    if not isinstance(entry, str) or not entry:
        raise HarnessAbiError(f"target {target!r}: harness_abi.entry_symbol is required and must name "
                              f"the symbol the package's lowered MLIR defines the kernel under")
    fence = block.get("fence_symbol") or None
    metric = block.get("cycle_window_metric") or None
    for name, value in (("fence_symbol", fence), ("cycle_window_metric", metric)):
        if value is not None and not isinstance(value, str):
            raise HarnessAbiError(f"target {target!r}: harness_abi.{name} must be a string or absent")
    return HarnessAbi(
        entry_symbol=entry, fence_symbol=fence,
        includes=tuple(block.get("includes") or ()),
        cycle_window_metric=metric,
        extern_decls=tuple(block.get("extern_decls") or ()),
    )


def for_target(target: str) -> HarnessAbi:
    """Resolve a target's harness ABI through the registry (the contract is the single source)."""
    from ..target_registry import resolve

    return from_contract(resolve(target).load_contract(), target=target)
