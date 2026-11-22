"""Isolated immutable residency publication with explicit provider inventories."""

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from merlin.compiler.schedule.storage import application_residency as residency
from merlin.compiler.schedule.storage.residency import ResidencyOverflow
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options, current_codegen_options
from merlin.compiler.semantic.target_binding import NativeCompilationError


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    source = SimpleNamespace(digest=lambda: "source-digest")
    region = SimpleNamespace(digest=lambda: "region-digest", input_storages=(("arg0", "external"),))
    archive_path = tmp_path / "weights.bin"
    archive_path.write_bytes(b"header" + b"dense")
    entry = SimpleNamespace(offset=0, nbytes=5, name="captured_weight")
    checked_sources, packed_sources = [], []
    parameters = SimpleNamespace(
        node_ids=("arg0",), entries=(entry,), archive_path=archive_path,
        archive_sha256="archive-sha256", payload_sha256="payload-sha256",
        payload_bytes=5,
        check=lambda request: checked_sources.append(request),
        record=lambda: {"ports": [{"node_id": "arg0", "name": entry.name}]},
    )
    inventory = [{"node_id": "arg0", "bytes": 8, "alignment": 8,
                  "layout": {"schema": "test.packed.v1", "padding": "zero"}}]

    def pack(request, node_id, dense, alternative):
        assert alternative.placement == "accelerator"
        packed_sources.append((request, node_id, dense))
        return dense + b"\0" * 3

    adapter = SimpleNamespace(
        inventory_resident_inputs=lambda request, alternative: inventory,
        pack_resident_input=pack,
        resident_memory_geometry=lambda: {"dma_base": 0, "dma_capacity": 1024, "alignment": 8},
    )
    plan = SimpleNamespace(selected=(SimpleNamespace(id="device", placement="accelerator"),))
    monkeypatch.setattr(residency, "application_plan", lambda request, selected: plan)
    monkeypatch.setattr(residency, "outline_semantic_regions", lambda *args, **kwargs:
                        SimpleNamespace(regions=(("device", region),)))
    from merlin.compiler.lower.c import weights as weights_pack

    def checked_payload(path, *, archive_sha256, out_blob):
        out_blob.write_bytes(path.read_bytes()[6:])
        return {"payload_offset": 6, "payload_sha256": parameters.payload_sha256,
                "payload_bytes": parameters.payload_bytes}

    monkeypatch.setattr(weights_pack, "checked_payload", checked_payload)
    options = CodegenOptions(application_residency=True, io_memory_base=32, io_memory_bytes=16,
                             io_memory_capacity_base=32, io_memory_capacity_bytes=64)
    return SimpleNamespace(**locals())


@contextmanager
def _scope(inputs, **option_changes):
    with codegen_options(replace(inputs.options, **option_changes)):
        with residency.residency_scope(inputs.source, inputs.adapter, inputs.parameters, None):
            yield


def _declined(inputs, parameters=True, state=None):
    """The reason, and the options and publication inside the scope, of an application residency declines."""
    selected = inputs.options
    with codegen_options(selected):
        with residency.residency_scope(inputs.source, inputs.adapter,
                                       inputs.parameters if parameters else None, state):
            inside = current_codegen_options(), residency.publication(), residency.declined()
        assert current_codegen_options() is selected
    assert residency.declined() is None and residency.publication() is None
    options, published, reason = inside
    # Inside, the application compiles exactly as with residency off: the same options and no table.
    assert options == replace(selected, application_residency=False) and published is None
    assert not inputs.packed_sources
    return reason


def test_publication_binds_exact_packed_bytes_and_restores_options(inputs):
    previous = current_codegen_options()
    assert residency.publication() is None
    with _scope(inputs):
        files = residency.publication()
        record = json.loads(files["residency.json"])
        load, = record["loads"]
        assert files[load["path"]] == b"dense\0\0\0"
        assert load["sha256"] == sha256(files[load["path"]]).hexdigest()
        assert load["layout"] == inputs.inventory[0]["layout"]
        binding = current_codegen_options().resident_bindings["region-digest"]["arg0"]
        assert binding == {"address": 48, "bytes": 8, "immutable": True}
        assert inputs.checked_sources == [inputs.source]
        assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")]
        rows = {row["name"]: row for row in record["entries"]}
        assert rows[load["name"]]["load_protocol"] == "once_at_load"
        assert rows["regional_io"]["load_protocol"] == "per_call"
    assert current_codegen_options() is previous
    assert residency.publication() is None


def test_repeated_publication_is_byte_stable_and_source_layout_changes_are_bound(inputs):
    with _scope(inputs):
        first = dict(residency.publication())
    with _scope(inputs):
        assert residency.publication() == first
    inputs.inventory[0]["layout"]["padding"] = "explicit-zero"
    with _scope(inputs):
        second = residency.publication()
        assert second["residency.json"] != first["residency.json"]
        first_load = json.loads(first["residency.json"])["loads"][0]
        second_load = json.loads(second["residency.json"])["loads"][0]
        assert first_load["name"] != second_load["name"]
        assert first_load["sha256"] == second_load["sha256"]


def test_one_immutable_layout_is_loaded_once_and_bound_in_each_device_region(inputs, monkeypatch):
    second = SimpleNamespace(digest=lambda: "second-region", input_storages=(("arg0", "external"),))
    plan = SimpleNamespace(selected=(SimpleNamespace(id="first", placement="accelerator"),
                                    SimpleNamespace(id="second", placement="accelerator")))
    monkeypatch.setattr(residency, "application_plan", lambda *args: plan)
    monkeypatch.setattr(residency, "outline_semantic_regions", lambda *args, **kwargs:
                        SimpleNamespace(regions=(("first", inputs.region), ("second", second))))
    inputs.inventory.append(inputs.inventory[0] | {"node_id": "runtime_input"})
    with _scope(inputs):
        record = json.loads(residency.publication()["residency.json"])
        assert len(record["loads"]) == 1
        bindings = current_codegen_options().resident_bindings
        assert bindings["region-digest"] == bindings["second-region"]
        assert set(bindings["region-digest"]) == {"arg0"}
        assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")]


def test_inventory_corruption_is_propagated_instead_of_skipping_region(inputs):
    def bad_inventory(region, alternative):
        raise ValueError("corrupt counted source")

    inputs.adapter.inventory_resident_inputs = bad_inventory
    with pytest.raises(ValueError, match="corrupt counted source"):
        with _scope(inputs):
            pass
    assert not inputs.packed_sources
    assert residency.publication() is None


def test_packing_uses_authenticated_snapshot_after_caller_archive_changes(inputs, monkeypatch):
    def authenticated_then_changed(path, **kwargs):
        checked = inputs.checked_payload(path, **kwargs)
        path.write_bytes(b"headerwrong")
        return checked

    monkeypatch.setattr(inputs.weights_pack, "checked_payload", authenticated_then_changed)
    with _scope(inputs):
        record = json.loads(residency.publication()["residency.json"])
        load, = record["loads"]
        assert residency.publication()[load["path"]] == b"dense\0\0\0"
        assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")]


def test_authenticated_snapshot_identity_and_size_must_match_parameters(inputs, monkeypatch):
    for changed in ({"payload_sha256": "wrong"}, {"payload_bytes": 6}):
        monkeypatch.setattr(inputs.weights_pack, "checked_payload",
                            lambda path, changed=changed, **kwargs: inputs.checked_payload(path, **kwargs) | changed)
        with pytest.raises(ValueError, match="payload identity differs"):
            with _scope(inputs):
                pass
        assert not inputs.packed_sources


def test_an_application_without_a_resident_layout_compiles_without_residency(inputs):
    inputs.region.input_storages = (("runtime_input", "external"),)
    inputs.adapter.inventory_resident_inputs = lambda region, alternative: None
    assert _declined(inputs) == "no device operand of the application has a resident layout"


def test_an_application_without_external_parameters_compiles_without_residency(inputs):
    assert _declined(inputs, parameters=False) == "the application has no external parameters to keep resident"
    assert not inputs.checked_sources


def test_an_unsupported_immutable_consumer_compiles_without_any_residency(inputs, monkeypatch):
    second = SimpleNamespace(digest=lambda: "unsupported-region",
                             input_storages=(("arg0", "external"),))
    plan = SimpleNamespace(selected=(SimpleNamespace(id="first", placement="accelerator"),
                                    SimpleNamespace(id="second", placement="accelerator")))
    monkeypatch.setattr(residency, "application_plan", lambda *args: plan)
    monkeypatch.setattr(residency, "outline_semantic_regions", lambda *args, **kwargs:
                        SimpleNamespace(regions=(("first", inputs.region), ("second", second))))
    for unsupported in (None, [], [{"node_id": "runtime_input"}]):
        inputs.adapter.inventory_resident_inputs = lambda region, alternative, unsupported=unsupported: (
            inputs.inventory if region is inputs.region else unsupported
        )
        assert _declined(inputs) == "a device region reads parameters through an operation with no resident layout: arg0"


def test_staging_must_fit_capacity_even_when_it_is_entirely_outside(inputs):
    for base in (24, 88, 104):
        with pytest.raises(ValueError, match="staging lies outside"):
            with _scope(inputs, io_memory_base=base):
                pass
        assert not inputs.packed_sources


def test_staging_and_parameter_capacity_overflow_precedes_packing(inputs):
    with pytest.raises(ResidencyOverflow) as error:
        with _scope(inputs, io_memory_bytes=12, io_memory_capacity_bytes=16):
            pass
    assert error.value.overflow_bytes == 8
    assert not inputs.packed_sources


def test_one_undivided_region_puts_the_arena_at_its_end_and_stages_in_the_rest(inputs):
    with _scope(inputs, io_memory_bytes=64):
        binding = current_codegen_options().resident_bindings["region-digest"]["arg0"]
        assert binding == {"address": 88, "bytes": 8, "immutable": True}
        assert (current_codegen_options().io_memory_base, current_codegen_options().io_memory_bytes) == (32, 56)
        rows = {row["name"]: row for row in json.loads(residency.publication()["residency.json"])["entries"]}
        assert (rows["regional_io"]["address"], rows["regional_io"]["bytes"]) == (32, 56)
    with pytest.raises(ValueError, match="leaves no staging window"):
        with _scope(inputs, io_memory_bytes=8, io_memory_capacity_bytes=8):
            pass
    assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")]


def test_conflicting_inventory_for_one_parameter_is_rejected(inputs):
    inputs.inventory.append(inputs.inventory[0] | {"bytes": 16})
    with pytest.raises(ValueError, match="inconsistent inventory"):
        with _scope(inputs):
            pass
    inputs.inventory[1] = inputs.inventory[0] | {"layout": {"schema": "other-layout"}}
    with pytest.raises(ValueError, match="incompatible resident layouts"):
        with _scope(inputs):
            pass


def test_bad_packed_byte_count_refuses_without_leaking_publication(inputs):
    inputs.adapter.pack_resident_input = lambda *args: b"short"
    with pytest.raises(ValueError, match="byte count differs"):
        with _scope(inputs):
            pass
    assert residency.publication() is None


def test_missing_provider_abi_is_refused_explicitly(inputs):
    del inputs.adapter.inventory_resident_inputs
    with pytest.raises(ValueError, match="does not declare"):
        with _scope(inputs):
            pass


def test_disabled_scope_needs_no_parameters_provider_or_io_and_clears_outer_publication(inputs):
    with _scope(inputs):
        outer = residency.publication()
        with codegen_options(CodegenOptions()):
            with residency.residency_scope(None, None, None, None):
                assert residency.publication() is None
        assert residency.publication() is outer
    assert residency.publication() is None


def test_default_codegen_identity_is_unchanged_by_disabled_residency():
    identity = CodegenOptions().identity_record()
    assert "application_residency" not in identity
    assert "resident_bindings" not in identity
    assert CodegenOptions(application_residency=False).identity_record() == identity
    active = CodegenOptions(application_residency=True, io_memory_capacity_base=0,
                            io_memory_capacity_bytes=64).identity_record()
    assert active["application_residency"] is True


def test_persistent_state_compiles_without_residency_before_inventory_or_packing(inputs):
    inputs.adapter.inventory_resident_inputs = None  # never reached
    reason = _declined(inputs, state=object())
    assert reason == "the application has persistent state inputs, which stay a host interface"
    assert not inputs.checked_sources


def test_inventory_and_packing_receive_selected_alternative(inputs):
    selected = inputs.plan.selected[0]
    inventory = inputs.adapter.inventory_resident_inputs
    pack = inputs.adapter.pack_resident_input
    def selected_inventory(region, alternative):
        assert alternative is selected
        return inventory(region, alternative)
    def selected_pack(region, node_id, dense, alternative):
        assert alternative is selected
        return pack(region, node_id, dense, alternative)
    inputs.adapter.inventory_resident_inputs = selected_inventory
    inputs.adapter.pack_resident_input = selected_pack
    with _scope(inputs):
        assert residency.publication() is not None


def _compiled_bindings(inputs):
    binding = current_codegen_options().resident_bindings[inputs.region.digest()]["arg0"]
    return SimpleNamespace(description={"resident_bindings": [dict(node_id="arg0", **binding)]})


def test_provider_package_consumes_the_exact_planned_binding(inputs):
    with _scope(inputs):
        package = _compiled_bindings(inputs)
        package.description["resident_bindings"][0]["provider_layout"] = "extra target metadata"
        residency.validate_region_bindings((("device", inputs.region),), (package,))


def test_provider_cannot_change_or_replace_a_requested_binding(inputs):
    with _scope(inputs):
        for change in ({"address": 56}, {"bytes": 7}, {"immutable": False}, {"node_id": "other"},
                       {"address": True}):
            package = _compiled_bindings(inputs)
            package.description["resident_bindings"][0].update(change)
            with pytest.raises(ValueError, match="did not consume"):
                residency.validate_region_bindings((("device", inputs.region),), (package,))


@pytest.mark.parametrize("report", [[], [{"node_id": "arg0"}]])
def test_provider_cannot_ignore_a_binding_or_report_an_incomplete_binding(inputs, report):
    with _scope(inputs):
        description = {"resident_bindings": report}
        with pytest.raises(ValueError, match="resident bindings"):
            residency.validate_region_bindings((("device", inputs.region),), (SimpleNamespace(description=description),))


def test_provider_cannot_report_duplicate_bindings(inputs):
    with _scope(inputs):
        package = _compiled_bindings(inputs)
        package.description["resident_bindings"] *= 2
        with pytest.raises(ValueError, match="duplicate"):
            residency.validate_region_bindings((("device", inputs.region),), (package,))


def test_table_asset_requires_at_least_one_consuming_region(inputs):
    with _scope(inputs):
        record = json.loads(residency.publication()["residency.json"])
        # Model a planning regression that publishes an asset with no requests.
        record["bindings"] = {}
        residency.publication()["residency.json"] = json.dumps(record).encode()
        with codegen_options(replace(current_codegen_options(), resident_bindings={})):
            with pytest.raises(ValueError, match="asset is not bound"):
                residency.validate_region_bindings((("device", inputs.region),),
                                                  (SimpleNamespace(description={}),))


def test_extra_region_bindings_are_rejected(inputs):
    with _scope(inputs):
        with pytest.raises(ValueError, match="uncompiled region"):
            residency.validate_region_bindings((), ())


def test_resident_region_check_reraises_a_deferred_compile_refusal(inputs):
    from merlin.compiler.build.regions import _check_resident_regions
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    refusal = NativeCompilationError("resource_limit", "region does not fit")
    with _scope(inputs), pytest.raises(NativeCompilationError, match="does not fit"):
        _check_resident_regions((("device", inputs.region),), (refusal,))


def test_resident_region_check_refuses_a_package_that_ignores_its_binding(inputs):
    from merlin.compiler.build.regions import _check_resident_regions

    package = SimpleNamespace(description={})
    with _scope(inputs), pytest.raises(ValueError, match="did not consume"):
        _check_resident_regions((("device", inputs.region),), (package,))


def _published(loads):
    """An active publication whose entries sit at the given (address, payload) pairs."""
    files, entries, rows = {}, [], []
    for i, (address, data) in enumerate(loads):
        name, path = f"parameter_{i}", f"residency/parameter_{i}.bin"
        files[path] = data
        entries.append(dict(name=name, address=address, bytes=len(data)))
        rows.append(dict(name=name, path=path, sha256=sha256(data).hexdigest()))
    files["residency.json"] = json.dumps(dict(entries=entries, loads=rows)).encode()
    return residency._PUBLICATION.set(files)


def test_resident_deployment_loads_one_arena_after_the_window_and_inside_the_capacity():
    from merlin.compiler.build import deployment

    token = _published([(4096, b"ab"), (4100, b"cd")])
    try:
        arena = deployment._resident_arena(4096, 8192)
        assert (arena["address"], arena["bytes"]) == (4096, 6)
        assert deployment._resident_bytes(arena, residency.publication()) == b"ab\0\0cd"
        for window_end, capacity_end in ((4097, 8192), (4096, 4101)):
            with pytest.raises(NativeCompilationError, match="resident loads leave the capacity"):
                deployment._resident_arena(window_end, capacity_end)
    finally:
        residency._PUBLICATION.reset(token)


def test_resident_arena_is_its_own_read_only_load_with_checked_bytes():
    from merlin.compiler.build import deployment

    host = dict(address=0, bytes=4096, usable_bytes=4096, measurement="measured_sections", reserved_bytes=64,
                declared_reservations=[dict(name="stack", bytes=64), dict(name="heap", bytes=0)])
    record = dict(
        provider=dict(host_allocation=host, sys_clk_freq_hz=1, window_section=".w", image_section=".i",
                      resident_section=".r"),
        host_allocation=host, host_linker_abi="symbolic_reservations_v1", window=dict(address=4096, bytes=64),
        parameters=None, inputs=[], outputs=[], window_required_bytes=0,
        resident=dict(address=8192, bytes=2, path="handoff/resident.bin", sha256=sha256(b"ab").hexdigest()))
    files = deployment.platform_files(record, {"handoff/resident.bin": b"ab"})
    linker = files["handoff/chip.ld"].decode()
    assert "resident PT_LOAD FLAGS(4);" in linker and "RESIDENT (r) : ORIGIN  =  8192, LENGTH  =  2" in linker
    assert " .r : { KEEP(*(.r)) } >RESIDENT :resident" in linker
    assert files["handoff/window.S"].endswith(b'.section .r,"a",@progbits\n.incbin "resident.bin"\n')
    assert files["handoff/resident.bin"] == b"ab"
    with pytest.raises(NativeCompilationError, match="resident arena bytes differ"):
        deployment.platform_files(record, {"handoff/resident.bin": b"ac"})
    record["provider"]["resident_section"] = ".w"
    with pytest.raises(NativeCompilationError, match="resident section must differ"):
        deployment.platform_files(record)


def test_a_resident_deployment_is_published_only_after_the_provider_admits_it(inputs):
    from merlin.compiler.build.regions import _admit_resident_deployment

    calls = []
    admitting = SimpleNamespace(admit_resident_deployment=lambda *args, **kwargs: calls.append((args, kwargs)))
    resident = dict(resident=dict(address=8192, bytes=2))
    for deployment in (None, dict(resident=None)):
        _admit_resident_deployment(deployment, SimpleNamespace(), "summaries", "regions", "root", None, inputs.source)
    for adapter, target_source in ((SimpleNamespace(), "target"), (admitting, None)):
        with pytest.raises(NativeCompilationError, match="resident deployment requires provider admission"):
            _admit_resident_deployment(resident, adapter, "summaries", "regions", "root", target_source, inputs.source)
    assert calls == []
    _admit_resident_deployment(resident, admitting, "summaries", "regions", "root", "target", inputs.source)
    assert calls == [(("summaries", "regions", "root", "target"), dict(source_request_digest="source-digest"))]
