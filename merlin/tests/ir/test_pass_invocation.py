"""Shared invocation preserves numerical-policy gates and successful stage prefixes."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from merlin.compiler.passes.native import STAGED_RUNNER
from merlin.llvmlower import normalization_reassociation, passes_xdsl
from merlin.xdsl_dialects import ir_inspection


@pytest.mark.parametrize("selected", [False, True])
def test_preprocessing_keeps_optional_numerical_rewrite_between_observed_stages(monkeypatch, selected):
    events = []
    original_module = object()
    original_callback = object()
    audit = object()

    def rewrite(name, **expected):
        def apply(module, **options):
            assert module is original_module
            assert options == expected
            events.append(name)
            return len(events)

        return apply

    monkeypatch.setattr(passes_xdsl, "prune_dead_pure_tensor_ops", rewrite("prune"))
    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", rewrite("quant", on_rewrite=original_callback))
    monkeypatch.setattr(passes_xdsl, "add_c_interface", rewrite("interface"))
    monkeypatch.setattr(passes_xdsl, "_layer_norm_chunked_sums_selected", lambda: selected)
    monkeypatch.setattr(
        normalization_reassociation,
        "chunk_layer_norm_sums",
        rewrite(
            "reassociate", allow_reassociation=True, assume_finite_intermediates=True, on_rewrite=original_callback
        ),
    )

    def observe(owner, stage, module, *, generic):
        assert owner is audit and module is original_module and generic is True
        events.append(stage)

    monkeypatch.setattr(ir_inspection, "record_stage", observe)
    counts = passes_xdsl._preprocess_module(original_module, audit=audit, on_quant_rewrite=original_callback)
    assert events == [
        "xdsl-parsed", "prune", "xdsl-pruned", "quant", "xdsl-quant-lowered",
        *(["reassociate"] if selected else []), "interface", "xdsl-c-interface",
    ]
    assert counts == {
        "dead_tensor_ops_pruned": 2,
        "quant_ext_lowered": 4,
        **({"layer_norm_sums_chunked": 6} if selected else {}),
        "c_interface_funcs": 7 if selected else 6,
    }


def test_preprocessing_failure_keeps_only_completed_observations(monkeypatch):
    events = []
    failure = ValueError("unsupported realized quantization")
    monkeypatch.setattr(passes_xdsl, "prune_dead_pure_tensor_ops", lambda module: 0)

    def refuse(module):
        raise failure

    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", refuse)
    monkeypatch.setattr(passes_xdsl, "add_c_interface", lambda module: pytest.fail("ran after failed quantization"))
    monkeypatch.setattr(ir_inspection, "record_stage", lambda audit, stage, module, **_: events.append(stage))
    with pytest.raises(ValueError) as raised:
        passes_xdsl._preprocess_module(object(), audit=object())
    assert raised.value is failure
    assert events == ["xdsl-parsed", "xdsl-pruned"]


def test_shared_preprocessing_owners_keep_patch_pickle_and_source_custody(monkeypatch):
    import pickle
    from pathlib import Path

    from merlin.compiler.frontend import preprocess, repairs
    from merlin.compiler.ir import provenance
    from merlin.compiler.lower.scalar import casts
    from merlin.compiler.quantization import linalg
    from merlin.compiler.transforms import dead, matmul
    from merlin.llvmlower import weight_prepack_runtime

    assert passes_xdsl is preprocess
    families = {
        provenance: ("carry_provenance",),
        linalg: ("lower_quant_ext",),
        dead: ("prune_dead_pure_tensor_ops", "_dead_tensor_operation_is_pure"),
        matmul: ("collapse_overrank_matmul", "lower_bf16_matmul_f32acc"),
        casts: ("fix_bool_fptosi", "fix_bool_sitofp"),
        repairs: ("preprocess_text_textual", "_Dequant"),
    }
    for owner, names in families.items():
        for name in names:
            assert getattr(passes_xdsl, name) is getattr(owner, name)
    # Old payloads resolve the exact records; new payloads name their real owner.
    record = repairs._Dequant("", "%0", ["%w", "%s", "%z"], 0, "w", "s", "z", "out")
    payload = pickle.dumps(record, protocol=0)
    assert b"merlin.compiler.frontend.repairs" in payload
    old = payload.replace(b"merlin.compiler.frontend.repairs", b"merlin.llvmlower.passes_xdsl")
    assert pickle.loads(old) == record
    assert pickle.loads(payload) == record
    mapper = object.__new__(preprocess._SourceTransformMap)
    mapper.source_sha256 = "retained-pickle-state"
    payload = pickle.dumps(mapper, protocol=0)
    old = payload.replace(b"merlin.compiler.frontend.preprocess", b"merlin.llvmlower.passes_xdsl")
    restored = pickle.loads(old)
    assert type(restored) is preprocess._SourceTransformMap
    assert restored.__dict__ == mapper.__dict__
    paths = set(preprocess.implementation_paths())
    assert {Path(owner.__file__) for owner in families} <= paths
    assert paths <= set(weight_prepack_runtime.implementation_paths())

    def selected(module):
        return 0

    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", selected)
    assert preprocess.lower_quant_ext is selected
    assert Path(__file__) in preprocess.implementation_paths()
    assert Path(linalg.__file__) in preprocess.implementation_paths()


def test_native_stage_retains_repeated_passes_options_and_refuses_missing_marker(monkeypatch):
    events = []
    binding = ModuleType("torch_mlir.passmanager")

    class PassManager:
        @staticmethod
        def parse(text, context):
            events.append(text)
            return SimpleNamespace(run=lambda operation: None)

    binding.PassManager = PassManager
    monkeypatch.setitem(sys.modules, "torch_mlir.passmanager", binding)
    namespace = {}
    exec(STAGED_RUNNER, namespace)  # noqa: S102 -- execute the actual shipped native runner
    run = namespace["_run_stages"]
    module = SimpleNamespace(operation=object())
    pipeline = "canonicalize,func.func(cse),canonicalize{max-iterations=3}"
    run(None, module, pipeline, False)
    assert events == ["builtin.module(" + pipeline + ")"]
    events.clear()
    with pytest.raises(RuntimeError, match="pipeline marker is absent"):
        run(None, module, pipeline, False, pre_generalize=(("rewrite", lambda *_: pytest.fail("ran unguarded")),))
    assert events == []
