"""Preparation shares exact source parses without sharing mutation authority."""

import hashlib
import json
import os

import pytest
from test_weight_prepack import MLIR, _bundle

from merlin.common import mlir_query as mq
from merlin.compile.complete_group_routing import select_complete_group
from merlin.llvmlower import weight_prepack
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.prepacked_device_capture import original_sibling_source, rebind
from merlin.llvmlower.sibling_source_route import SiblingSourceRoute
from merlin.llvmlower.weight_prepack_runtime import fixed_arguments
from merlin.xdsl_dialects._common import text


def count_file_parses(monkeypatch):
    original = mq.parse_mlir_file
    calls = []

    def counted(path, **kwargs):
        calls.append(path.resolve())
        return original(path, **kwargs)

    monkeypatch.setattr(mq, "parse_mlir_file", counted)
    return calls


def test_nested_invocation_cache_is_bounded_and_released_on_exit(tmp_path, monkeypatch):
    first, second = tmp_path / "first.mlir", tmp_path / "second.mlir"
    for path in (first, second):
        path.write_text("module {}")
    calls = count_file_parses(monkeypatch)
    global_cache = mq.parse(first)
    with mq.parse_scope(max_entries=1):
        local_first = mq.parse(first)
        assert local_first is not global_cache
        with mq.parse_scope():
            assert mq.parse(first) is local_first
        mq.parse(second)
        assert mq.parse(first) is not local_first  # one-entry LRU evicted it
    assert mq._PARSE_SCOPE.get() is None
    with mq.parse_scope():
        mq.parse(first)
    assert calls == [first, first, second, first, first]
    with pytest.raises(RuntimeError, match="caller stopped"), mq.parse_scope():
        mq.parse(first)
        raise RuntimeError("caller stopped")
    assert mq._PARSE_SCOPE.get() is None


def test_source_byte_changes_never_reuse_stale_graph_or_partial_parse(tmp_path, monkeypatch):
    path = tmp_path / "model.mlir"
    original = "module {func.func @first() {return}}"
    changed = original.replace("first", "other")
    path.write_text(original)
    with mq.parse_scope():
        first = mq.parse(path)
        stamp = path.stat()
        path.write_text(changed)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert path.stat().st_size == stamp.st_size
        assert mq.parse(path) is not first
        assert "@other" in text(mq.parse(path))
    reader = mq.parse_mlir_file

    def change_during_parse(source, **kwargs):
        module = reader(source, **kwargs)
        source.write_text(original)
        return module

    monkeypatch.setattr(mq, "parse_mlir_file", change_during_parse)
    with mq.parse_scope(), pytest.raises(ValueError, match="changed during parsing"):
        mq.parse(path)


def test_fixed_ownership_clones_quant_lowering_and_rechecks_session_and_manifest(tmp_path, monkeypatch):
    marker = """%scale = arith.constant dense<0.5>:tensor<f32>
    %zp = arith.constant dense<0>:tensor<i64>
    %q = "quant_ext.quantize_per_tensor"(%1,%scale,%zp)
      <{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}>
      :(tensor<3xf32>,tensor<f32>,tensor<i64>)->tensor<3xi8>
    """
    source = _bundle(tmp_path / "source", mlir=MLIR.replace("    %3 =", marker + "    %3 ="))
    manifest = json.loads((source / "weights.safetensors.manifest.json").read_text())
    calls = count_file_parses(monkeypatch)
    with mq.parse_scope():
        assert fixed_arguments(source, manifest) == (0, 1, 2)
        assert fixed_arguments(source, manifest) == (0, 1, 2)
        assert mq.op_count(mq.parse(source / "model.mlir"), "quant_ext.quantize_per_tensor") == 1
        (source / "session_contract.yaml").write_text("version: 1\nstates: [{input_arg: 0}]\n")
        assert fixed_arguments(source, manifest) == (1, 2)
        manifest["1"]["kind"] = "input"
        assert fixed_arguments(source, manifest) == (2,)
    assert calls == [source / "model.mlir"]


def test_four_ownership_and_two_original_readers_use_one_owned_parse(tmp_path, monkeypatch):
    source, work = _bundle(tmp_path / "source"), tmp_path / "work"
    monkeypatch.setattr(weight_prepack, "_default_cache_root", lambda: tmp_path / "cache")
    prepared = weight_prepack.prepare_build_bundle(source, work, frozenset({weight_prepack.FEATURE}))
    original = mq.parse((source / "model.mlir").read_text())
    digest = hashlib.sha256(text(original, generic=True).encode()).hexdigest()
    selected = SiblingSourceRoute(digest, BorrowedBufferEffects("fixture owner", True, True, True))
    route = DeviceRouting(
        "generic-test",
        tmp_path / "package",
        "int8",
        "i32",
        select=select_complete_group,
        capture=source / "model.mlir",
        granularity="group",
        require_complete_groups=True,
        sibling_source=selected,
    )
    manifest = json.loads((source / "weights.safetensors.manifest.json").read_text())
    calls = count_file_parses(monkeypatch)
    with mq.parse_scope():
        assert fixed_arguments(source, manifest) == (0, 1, 2)
        assert fixed_arguments(source, manifest) == (0, 1, 2)
        # Each route also calls the complete preparation revalidator, including
        # its own fixed_arguments analysis and original hashes/semantic checks.
        bound = rebind(route, source=source, prepared=prepared, work=work)
        reopened, _ = original_sibling_source(bound, selected, actual_source=prepared / "model.mlir", work=work)
        assert text(reopened, generic=True) == text(original, generic=True)
        reopened.body.block.first_op.attributes["fixture_changed"] = reopened.body.block.first_op.sym_name
        assert "fixture_changed" not in text(mq.parse(source / "model.mlir"), generic=True)
        assert calls == [source / "model.mlir"]
    assert mq._PARSE_SCOPE.get() is None


def test_repeated_purity_analysis_reuses_only_exact_source_not_session_or_manifest(tmp_path, monkeypatch):
    from merlin.llvmlower import passes_xdsl

    source = _bundle(tmp_path / "source")
    manifest = json.loads((source / "weights.safetensors.manifest.json").read_text())
    lower = passes_xdsl.lower_quant_ext
    calls = []

    def counted(module):
        calls.append(module)
        return lower(module)

    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", counted)
    assert fixed_arguments(source, manifest) == (0, 1, 2)
    assert fixed_arguments(source, manifest) == (0, 1, 2)
    (source / "session_contract.yaml").write_text("version: 1\nstreams: [{input_arg: 0}]\n")
    assert fixed_arguments(source, manifest) == (1, 2)
    manifest["1"]["kind"] = "input"
    assert fixed_arguments(source, manifest) == (2,)
    assert len(calls) == 1
    # Same path, size and timestamp, but changed source bytes require a new proof.
    path = source / "model.mlir"
    stamp = path.stat()
    path.write_text(path.read_text().replace("%3", "%5"))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert fixed_arguments(source, manifest) == (2,)
    assert len(calls) == 2
    path.write_text(path.read_text().replace("    %5 =", '    "unknown.effect"() : () -> ()\n    %5 ='))
    for _ in range(2):
        with pytest.raises(ValueError, match="effects"):
            fixed_arguments(source, manifest)
    assert len(calls) == 4  # failed purity proofs are not retained


def test_cached_purity_still_rejects_source_mutation_during_lookup(tmp_path, monkeypatch):
    from merlin.llvmlower import weight_prepack_runtime as runtime

    source = _bundle(tmp_path / "source")
    manifest = json.loads((source / "weights.safetensors.manifest.json").read_text())
    assert fixed_arguments(source, manifest) == (0, 1, 2)
    original = runtime._tensor_entry_arity

    def change_after_hit(*args):
        result = original(*args)
        path = source / "model.mlir"
        path.write_text(path.read_text().replace("%3", "%5"))
        return result

    monkeypatch.setattr(runtime, "_tensor_entry_arity", change_after_hit)
    with pytest.raises(ValueError, match="source changed"):
        fixed_arguments(source, manifest)


def test_selected_analysis_implementation_change_requires_new_purity_proof(tmp_path, monkeypatch):
    from merlin.llvmlower import prepared_operand_effects

    source = _bundle(tmp_path / "source")
    manifest = json.loads((source / "weights.safetensors.manifest.json").read_text())
    assert fixed_arguments(source, manifest) == (0, 1, 2)

    def different_library(*_args, **_kwargs):
        raise ValueError("new selected analysis refuses these effects")

    monkeypatch.setattr(prepared_operand_effects, "validate_pure_tensor_function", different_library)
    with pytest.raises(ValueError, match="new selected analysis"):
        fixed_arguments(source, manifest)
