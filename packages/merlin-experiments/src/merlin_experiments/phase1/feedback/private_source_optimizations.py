"""Selected original source-pass owner at the ordinary full-model build call.

No experiment JSON or candidate manifest supplies the live qualification owner.
The existing sealed compiler-pass transport must reconstruct it independently.
Production targets without the new owner/context protocol remain unavailable.
"""

from __future__ import annotations

import sys


def compile_model(compiler, *, compiler_package, original_inputs, **options):
    """Keep legacy calls exact; selected calls use the original typed bridge.

    The raw original ContextVar is read before any owner callback so the bridge
    can freeze actual inputs first. It then revalidates that live selection.
    This does not restore a selection from saved records or widen any grants.
    """
    module = sys.modules.get("merlin.targetgen.compiler_pass_selection")
    selection = None if module is None else module._ACTIVE.get()
    if selection is None or "source_optimizations_v1" not in selection.mechanisms:
        return compiler(**options)
    from merlin.targetgen.compiler_source_optimizations import compile_with_source_optimizations

    def current_inputs(reader=original_inputs):
        # This original companion and the actual caller reader/default are both
        # part of the core's pre-callback function and source snapshot.
        return reader()

    return compile_with_source_optimizations(
        compiler,
        selection=selection,
        compiler_package=compiler_package,
        original_inputs=current_inputs,
        **options,
    )
