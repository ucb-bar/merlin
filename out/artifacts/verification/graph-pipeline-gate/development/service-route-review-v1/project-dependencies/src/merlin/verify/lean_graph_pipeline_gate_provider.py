"""Source-only selection of ONE frozen common provider; no launch authority.

The trusted host must separately authorize construction/use of a fresh context.
This loader never constructs Context, opens runtime images, or discovers an SDK.
Public gate activation remains held in lean_graph_pipeline_gate_native.py.
Changing this selected source cut requires a new reviewed gate source version.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

from .lean_graph_pipeline_gate_projection import GateUnavailable

_DIRECTORY = (Path(__file__).resolve().parents[3] /
    "out/artifacts/verification/graph-regions-declared/original-image-context-review-v5/"
    "source-snapshot/src/merlin/verify")
_PINS = (
    ("lean_graph_original_image_context", "bfdb1a71019dfccc335f35f560dd6e51b7b1bc2671cf033c529ad52f6473bcc6", 24711),
    ("lean_graph_original_image_context_data", "77d30d01f1c839edb52da314c386c307424f5bb2ebcccae2f9ee43b7c59d32e1", 15692),
    ("lean_graph_original_image_context_io", "9ed416a334277acd890c95a75d45ca0366c0eb0797b71574bc467720ad1bdb09", 4512),
)
_PACKAGE = "merlin.verify._graph_pipeline_frozen_provider_v5"
_MODULES = None
_PROJECT_MODULES = frozenset("Merlin." + name for name in (
    "GraphStateGateEntries", "GraphStateScalar", "GraphStateScalarContract",
    "GraphStateScalarNormalize", "GraphStateScalarSound", "GraphStateScalarTyping",
    "GraphStateScalarInitialized"))
_COMPILE_ROOTS = {
    "Merlin.GraphStateScalar": tuple("Merlin.GraphState.Scalar." + name for name in (
        "check_sound", "check_typed", "check_composes", "check_initialized", "check_operands_initialized")),
    "Merlin.GraphStateScalarInitialized": tuple("Merlin.GraphState.Scalar." + name for name in (
        "typed_execution_shaped", "HasReads.initialized", "typed_operands_initialized")),
    "Merlin.GraphStateScalarSound": ("Merlin.GraphState.Scalar.normalize_sound", "Merlin.GraphState.Scalar.loopEval_congr"),
    "Merlin.GraphStateScalarTyping": ("Merlin.GraphState.Scalar.checkTypes_sound",),
}
# Exact reviewed metadata/descriptor/provenance identities. These are data refs,
# never a runtime-discovery mechanism or permission to construct/launch Context.
_REVIEW_PINS = {'base_intent': ('out/build/graph-paper-deadline/original-runtime-candidate-v1/basic-kernel-probe-v1/INTENT.json',
                 '2682ee9260af211714d48a3b41206a0464a5fd18d6495ba071386f87eea67a8b',
                 145093),
 'copy': ('out/build/graph-paper-deadline/original-runtime-candidate-v1/RESULT.json',
          '849060513fb300270eb75fd79518cc187325b4f8472105b34570faeaaf85a7ce',
          3531848),
 'finite_review': ('out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/finite-stdlib-diagnostics-v1/REVIEW.json',
                   '200da90f0ca766a78eca5d18a3b4fd315ac733f396017b20324a71d17ebae16b',
                   392060),
 'metadata': ('out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/runtime-snapshot-review/METADATA-REVIEW-v2.json',
              '8c4a1566fac6ecba764c91e8179b5732d7ccda1a5e9a6f58559d7a130770cd82',
              3017778),
 'probe': ('out/artifacts/verification/graph-regions-declared/original-image-context-review-v5/REVIEWED-RUNTIME-INTENT-v2.json',
           'ccb70d1da9990d786dd59328686a389f4cd891c1a8aa24cf78625e3924a8e2f8',
           214970),
 'trace': ('out/build/graph-paper-deadline/original-runtime-candidate-v1/basic-kernel-probe-v1/trace_parser.py',
           '2aaae713ed80a46735434eca16dbb6fd2ed63e4bd8e7df50308f64ac56d0b7ba',
           9118)}


def selected():
    """Return exact frozen controller/data/IO modules, or fail closed.

    Same bytes imported under another package still yield different Python class
    identities and are NOT accepted. Saved records never select executable code.
    """
    global _MODULES
    paths, contents = [], {}
    for name, digest, size in _PINS:
        path = _DIRECTORY / (name + ".py")
        if path.is_symlink():
            raise GateUnavailable("selected provider source is a symlink")
        raw = path.read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise GateUnavailable("selected frozen provider source changed: " + name)
        paths.append(path)
        contents[name] = (path, raw)
    if _MODULES is None:
        if any(name == _PACKAGE or name.startswith(_PACKAGE + ".") for name in sys.modules):
            raise GateUnavailable("selected provider namespace was populated externally")
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_DIRECTORY)]
        sys.modules[_PACKAGE] = package
        # Execute the already digest-checked bytes, never import from a stale
        # pyc or write __pycache__ into the immutable source snapshot. Relative
        # imports resolve only these preloaded data/IO/controller modules.
        loaded = {}
        for name in (_PINS[1][0], _PINS[2][0], _PINS[0][0]):
            path, raw = contents[name]
            module = types.ModuleType(_PACKAGE + "." + name)
            module.__file__, module.__package__ = str(path), _PACKAGE
            sys.modules[module.__name__] = module
            setattr(package, name, module)
            exec(compile(raw, str(path), "exec"), module.__dict__)
            loaded[name] = module
        _MODULES = tuple(loaded[name] for name, _, _ in _PINS)
    for module, path, (name, _, _) in zip(_MODULES, paths, _PINS, strict=True):
        if (sys.modules.get(_PACKAGE + "." + name) is not module
                or Path(module.__file__) != path):
            raise GateUnavailable("selected provider module identity changed")
    controller, data, io = _MODULES
    if controller.data is not data or controller.io is not io or controller.CheckedQuery is not data.CheckedQuery:
        raise GateUnavailable("mixed frozen provider modules/classes")
    return _MODULES


def bound_context(context):
    """Check frozen class identity AND the context's exact three source refs.

    This is source/host binding, not context validity or permission to launch.
    The common provider still performs all runtime and live-result checks.
    """
    controller, data, io = selected()
    if type(context) is not controller.Context:
        raise GateUnavailable("exact selected frozen Context required; alternate classes/callbacks refused")
    expected = tuple(data.FileRef(str(_DIRECTORY / (name + ".py")), digest, size)
                     for name, digest, size in _PINS)
    if type(context.controls) is not tuple or context.controls != expected:
        raise GateUnavailable("context does not bind the complete selected frozen controller source cut")
    expected_review = {role: data.FileRef(str(Path(__file__).resolve().parents[3] / path), digest, size)
                       for role, (path, digest, size) in _REVIEW_PINS.items()}
    if type(context.review) is not dict or context.review != expected_review:
        raise GateUnavailable("context review/finite-runtime descriptor differs from selected exact policy")
    return controller, data, io


def context_arguments(prepared, *, workspace):
    """SOURCE-ONLY recipe for the future trusted host's explicit construction.

    No Context is created and no runtime image is opened. The host still needs
    separate exact source/workspace authorization before passing these arguments
    to selected()[0].Context.from_review. Saved JSON cannot select a provider,
    policy, module map, compile-root list or query. No public gate is activated.
    """
    from .lean_graph_pipeline_gate_native import request

    _, data, _ = selected()
    selected_request = request(prepared)
    projects = {name: data.FileRef(path, digest, size)
                for name, path, digest, size in selected_request.project_sources}
    if set(projects) != _PROJECT_MODULES:
        raise GateUnavailable("exact minimal seven-module gate closure required")
    # This validates spelling only. Fresh-directory creation is the authorized
    # controller's responsibility, never implicit preparation or an SDK probe.
    destination = str(data.absolute(str(workspace)))
    root = Path(__file__).resolve().parents[3]
    return {
        "review": {role: data.FileRef(str(root / path), digest, size)
                   for role, (path, digest, size) in _REVIEW_PINS.items()
                   if role in {"metadata", "copy", "probe", "trace"}},
        "project_sources": projects,
        "controller_sources": tuple(data.FileRef(str(_DIRECTORY / (name + ".py")), digest, size)
                                    for name, digest, size in _PINS),
        "workspace": destination,
        "compile_audit_roots": dict(_COMPILE_ROOTS),
    }
