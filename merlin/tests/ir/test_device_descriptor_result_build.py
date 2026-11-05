"""Original model/shim custody using real emitted C and substituted device tools.

The optional lowering control below invokes actual MLIR tools, but no substituted
object is executed or described as qualified hardware.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_device_build_parallel import synthetic_build  # noqa: F401

from merlin.common.digest import sha256_file
from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_shim as D
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCall, BorrowedBufferCalls, BorrowedBufferEffects
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.device_shim import emit_translation_unit
from merlin.llvmlower.toolchain import clang as native_clang
from merlin.llvmlower.toolchain import mlir_translate as native_translate
from merlin.runtime.device_descriptor_results import DeviceDescriptorResults

SOURCE = """module {
 func.func private @first(%a:tensor<2x2xi8> {bufferization.access="read"},
 %b:tensor<2x2xi8> {bufferization.access="read"},
 %c:tensor<2x2xi32> {bufferization.access="write"}) -> tensor<2x2xi32>
 attributes {llvm.emit_c_interface}
 func.func @forward(%a:tensor<2x2xi8>,%b:tensor<2x2xi8>)->tensor<2x2xi32> {
 %c=tensor.empty():tensor<2x2xi32>
 %r=func.call @first(%a,%b,%c):(tensor<2x2xi8>,tensor<2x2xi8>,tensor<2x2xi32>)->tensor<2x2xi32>
 return %r:tensor<2x2xi32>
 }
}"""


def neutral_effects_owner(source, built):
    """Conditional transport fixture only; substituted objects are never qualified."""
    from xdsl.dialects.builtin import MemRefType

    effects = BorrowedBufferEffects("neutral-conditional-premise-v1", True, True, True)
    calls = tuple(
        BorrowedBufferCall(
            value.wrapper_symbol,
            value.borrowed_symbol,
            tuple(str(MemRefType(t.get_element_type(), t.get_shape())) for t in value.argument_types),
            value.result_argument,
            effects,
        )
        for value in built.shim_unit.result_identities
    )
    pins = tuple((path.resolve(), sha256_file(path)) for path in (*built.objects, built.shim_object.with_suffix(".c")))
    return BorrowedBufferCalls(source, sha256_file(source), sha256_file(source), calls, pins, neutral_validate_original)


def neutral_validate_original():
    """Explicit conditional premise for custody tests, not a hardware qualifier."""


@pytest.fixture
def issued(synthetic_build, tmp_path, monkeypatch):  # noqa: F811
    # Keep ordinary lowering's imported tool resolvers independent of the
    # diagnostic device runner below, including when this file runs alone.
    from merlin.llvmlower import lower  # noqa: F401
    build, _, _, _, _, _ = synthetic_build
    monkeypatch.setattr(D, "emit_translation_unit", emit_translation_unit)
    monkeypatch.setattr(D, "kernel_abi_for", lambda _: D.KernelAbi("kernel", version=1))
    from merlin.llvmlower import toolchain

    tools = []
    for name in ("clang", "translate", "objcopy"):
        path = tmp_path / ("diagnostic-" + name)
        path.write_text("owned diagnostic tool identity; substituted runner never launches this file\n")
        path.chmod(0o700)
        tools.append(str(path))
    monkeypatch.setattr(toolchain, "clang", lambda: tools[0])
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: tools[1])
    monkeypatch.setattr(B, "_objcopy", lambda: tools[2])
    route = B.DeviceRouting(
        "synthetic",
        tmp_path,
        "i8",
        "i32",
        descriptor_results=True,
        descriptor_effects=neutral_effects_owner,
        numeric_policy={"reduction": "source"},
    )
    guard = DeviceDescriptorResults(route)
    compilation = CompilationRecipe(tmp_path, producer=Path(__file__))
    guard.bind_sources(compilation)
    source, sidecar = tmp_path / "model.mlir", tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text('{"fixture":"custody only"}')
    guard.bind_input(source, sidecar, compilation)
    built = build(names=("first",), descriptor_results=True, tile_edge=2, numeric_policy=route.numeric_policy)
    return route, guard, compilation, built, source


def test_actual_unit_receipt_and_original_source_are_bound_once(issued):
    _, guard, compilation, built, _ = issued
    guard.bind_shim(built, compilation)
    selected = guard.options()["descriptor_result_identities"]
    assert all(v.borrowed_effects is None for v in built.shim_unit.result_identities)
    assert all(v.borrowed_effects.qualification_identity == "neutral-conditional-premise-v1" for v in selected)
    assert tuple(v.symbol for v in selected) == tuple(v.symbol for v in built.shim_unit.result_identities)
    record = built.shim_object.parent / "descriptor-result-effects.json"
    assert json.loads(record.read_bytes()) == guard._effects.record()
    assert json.loads(guard.identity())["borrowed_calls"] == guard._effects.record()
    names = compilation.record["preparation"]
    assert "descriptor_result_shim_2" in names
    assert guard._inputs[0].path.name == "model.mlir"
    assert any(Path(pin["path"]).name == "device_shim.py" for pin in names.values())
    assert any(pin["path"] == str(record.resolve()) for pin in names.values())
    with pytest.raises(ValueError, match="one original"):
        guard.bind_shim(built, compilation)


def test_repeated_role_binding_deduplicates_same_file_and_rejects_conflicting_identity(issued):
    _, guard, compilation, _, _ = issued
    before = dict(compilation.record["preparation"])
    guard.bind_sources(compilation)
    assert compilation.record["preparation"] == before
    first = next(iter(compilation.record["preparation"].values()))
    first["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="conflicts"):
        guard.bind_sources(compilation)


@pytest.mark.parametrize(
    "options",
    [{"granularity": "other"}, {"catalog_manifest": "catalog.json"}, {"post_offload_transform": lambda *_: None}],
)
def test_unsupported_original_abi_route_refuses_before_emission(tmp_path, options):
    route = B.DeviceRouting("neutral", tmp_path, "i8", "i32", descriptor_results=True, **options)
    with pytest.raises(ValueError, match="original generated contraction"):
        DeviceDescriptorResults(route)


@pytest.mark.parametrize(
    "change",
    ["granularity", "catalog_manifest", "post_offload_transform", "package_dir", "numeric_policy", "select", "owner"],
)
def test_original_route_cannot_change_while_same_boolean_stays_enabled(issued, change):
    route, guard, _, _, _ = issued
    if change == "owner":
        guard.routing = replace(route)
    elif change == "numeric_policy":
        route.numeric_policy["reduction"] = "reassociated"
    else:
        value = {"granularity": "group", "catalog_manifest": "new.json", "package_dir": "/changed"}.get(change)
        object.__setattr__(route, change, value if value is not None else lambda *_: None)
    with pytest.raises(ValueError, match="route|routing"):
        guard.revalidate()


@pytest.mark.parametrize("change", ["access", "name", "empty"])
def test_metadata_mutated_before_first_consumer_snapshot_does_not_match_compiled_issuer(issued, change):
    _, guard, compilation, built, _ = issued
    (identity,) = built.shim_unit.result_identities
    identities = (
        ()
        if change == "empty"
        else (
            replace(
                identity,
                **(
                    {"argument_access": ("read", "read", "read-write")}
                    if change == "access"
                    else {"wrapper_symbol": "changed"}
                ),
            ),
        )
    )
    object.__setattr__(built.shim_unit, "result_identities", identities)
    with pytest.raises(ValueError, match="issued descriptor roster"):
        guard.bind_shim(built, compilation)


@pytest.mark.parametrize(
    "where", ["source", "compiled_object", "unit_after_bind", "input", "recipe_method", "kernels", "objects"]
)
def test_later_changes_cannot_refresh_original_custody(issued, monkeypatch, where):
    _, guard, compilation, built, source = issued
    guard.bind_shim(built, compilation)
    if where == "recipe_method":
        from merlin.llvmlower.descriptor_result_preparation import DescriptorResultPreparation

        monkeypatch.setattr(DescriptorResultPreparation, "bind_recipe", lambda *_: None)
    elif where == "kernels":
        built.kernels["first"] = "another_kernel"
    elif where == "objects":
        object.__setattr__(built, "objects", (built.shim_object,))
    elif where == "unit_after_bind":
        object.__setattr__(built.shim_unit, "text", built.shim_unit.text + "\n/* changed */")
    else:
        path = {"source": built.shim_object.with_suffix(".c"), "compiled_object": built.shim_object, "input": source}[
            where
        ]
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|identity"):
        guard.revalidate()


def test_closed_roster_comment_cannot_be_terminated_by_metadata(issued):
    _, _, _, built, _ = issued
    (identity,) = built.shim_unit.result_identities
    payload = "*/ injected /*"
    marker = D.descriptor_result_marker((replace(identity, wrapper_symbol=payload),))
    assert marker.count("*/") == 1 and marker.count("/*") == 1
    rows = json.loads(marker.removeprefix("/* merlin.descriptor_result_identities.v1 ").removesuffix(" */\n"))
    assert rows[0]["wrapper_symbol"] == payload
    with pytest.raises(ValueError, match="bounded"):
        D.descriptor_result_marker((identity,) * 4097)


def test_unselected_route_is_inert_and_cannot_attach_later(tmp_path):
    route = B.DeviceRouting("neutral", tmp_path, "i8", "i32")
    guard = DeviceDescriptorResults(route)
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))
    guard.bind_sources(recipe)
    assert guard.options() == {} and guard.identity() == b"" and "preparation" not in recipe.record
    object.__setattr__(route, "descriptor_results", True)
    with pytest.raises(ValueError, match="routing changed"):
        guard.revalidate()


def test_real_lowering_consumes_the_actual_emitted_roster_and_joins_llvm(issued, tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.llvmlower.target_data_layout import _query

    _, guard, compilation, built, source = issued
    guard.bind_shim(built, compilation)
    monkeypatch.setattr(toolchain, "clang", native_clang)
    monkeypatch.setattr(toolchain, "mlir_translate", native_translate)
    lowered = lower_model_file(
        source,
        tmp_path / "lower",
        targets=(),
        index_bits=64,
        data_layout=_query(native_clang(), ("-O2",)),
        **guard.options(),
    )
    guard.consumed(lowered, compilation)
    assert lowered.stats["descriptor_result_identity"]["rewrites"] == 1
    assert "call ptr @malloc(" not in lowered.ll_path.read_text()
    assert "descriptor_result_consumed_1" in compilation.record["preparation"]
    paths = [pin["path"] for pin in compilation.record["preparation"].values()]
    assert len(paths) == len(set(paths))
    lowered.ll_path.write_bytes(lowered.ll_path.read_bytes() + b"changed after consumption")
    with pytest.raises(ValueError, match="changed|identity"):
        guard.revalidate()
