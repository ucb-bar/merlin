"""Scoped FX effects and nested graph cleanup inside the isolated capture interpreter."""

from __future__ import annotations

import contextlib
import threading

from .portable import helper

_specs = helper("_recipe_specs")
RecipeError = _specs.RecipeError


_PT2E_CLEANUP_LOCK = threading.RLock()


@contextlib.contextmanager
def preserve_fx_vmap_effects():
    """Keep exported vmap context operations through every DCE in a capture.

    These exact legacy Python wrappers mutate functorch's nesting stack (or
    initialize its decompositions); their unused return values do not make them
    dead code. Use FX's function-effect registration, not a replacement operator
    or a model rewrite. The worker holds this scope through quantization AND
    later model2mlir capture. Direct recipe callers get the same scope during
    recipe application. Registrations added here are restored on exit.

    Like the PT2E cleanup adapter, this is for the isolated capture interpreter;
    unrelated concurrent framework captures belong in separate workers.
    """
    try:
        from torch._functorch import predispatch
        from torch.fx import node as fx_node
    except ImportError:
        yield
        return
    names = ("_vmap_increment_nesting", "_vmap_decrement_nesting", "lazy_load_decompositions")
    functions = tuple(getattr(predispatch, name) for name in names if hasattr(predispatch, name))
    with _PT2E_CLEANUP_LOCK:
        registry = fx_node._side_effectful_functions
        added = tuple(function for function in functions if function not in registry)
        for function in added:
            fx_node.has_side_effect(function)
        try:
            yield
        finally:
            for function in added:
                registry.discard(function)


@contextlib.contextmanager
def _pt2e_nested_graph_cleanup():
    """Preserve live first-class FX graph state during TorchAO conversion.

    Some Torch builds prune descendants of a GraphModule referenced by get_attr,
    although recursive DCE later requires those descendants. TorchAO creates a
    new GraphModule inside conversion, so an instance override cannot reach its
    cleanup. Temporarily adapt just this method in the isolated capture worker;
    serialize opted-in conversions and restore it even on exceptions. This is
    not an on-disk framework patch or a general multi-threaded Torch override:
    unrelated concurrent framework conversions must use separate workers.
    Graphs without first-class submodules retain the framework's own cleanup.
    """
    import torch

    report = {
        "schema": "merlin.pt2e_nested_graph_cleanup.v1",
        "adapted_calls": 0,
        "retained_graph_references": 0,
        "removed_modules": 0,
    }
    with _PT2E_CLEANUP_LOCK:
        original = torch.fx.GraphModule.delete_all_unused_submodules

        def cleanup(module):
            used, graph_references, visiting = set(), set(), set()

            def mark(path):
                atoms = path.split(".")
                used.update(".".join(atoms[:n]) for n in range(1, len(atoms) + 1))

            def graph_state(gm, prefix=""):
                if id(gm) in visiting:
                    raise RecipeError("cyclic first-class FX graph state is unsupported")
                visiting.add(id(gm))
                try:
                    for node in gm.graph.nodes:
                        if node.op not in {"call_module", "get_attr"}:
                            continue
                        path = str(node.target)
                        full = prefix + path
                        mark(full)
                        value = gm
                        for atom in path.split("."):
                            value = getattr(value, atom)  # Missing live state remains an error.
                        if not isinstance(value, torch.nn.Module):
                            continue
                        if node.op == "get_attr" and isinstance(value, torch.fx.GraphModule):
                            graph_references.add(full)
                            graph_state(value, full + ".")
                        else:
                            # Calls may depend on any descendant. A first-class
                            # opaque Module likewise has no inspectable FX body.
                            for name, _ in value.named_modules():
                                mark(full + ("." + name if name else ""))
                finally:
                    visiting.remove(id(gm))

            graph_state(module)
            if not graph_references:
                return original(module)
            removed = 0
            for name, _ in list(module.named_modules()):
                if name and name not in used:
                    removed += bool(module.delete_submodule(name))
            report["adapted_calls"] += 1
            report["retained_graph_references"] += len(graph_references)
            report["removed_modules"] += removed

        torch.fx.GraphModule.delete_all_unused_submodules = cleanup
        try:
            yield report
        finally:
            torch.fx.GraphModule.delete_all_unused_submodules = original
