"""Residency plans place and identify bytes, and an application publishes its resident table or refuses exactly.

Each row runs several checks, and every failure names its `check N` label, which the comment at that check
describes. A check that patches or changes its fixture opens its own nested monkeypatch context and builds its
own `inputs` instance in its own directory. Fake targets declare the typed `ResidentParameters` capability.
"""

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
from corefixtures.row_table import run_rows

from merlin.compiler.backend.protocol import ResidentParameters
from merlin.compiler.schedule.storage import application_residency as residency
from merlin.compiler.schedule.storage.residency import (
    ReservedInterval,
    ResidencyOverflow,
    ResidencyRequest,
    plan_residency,
)
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options, current_codegen_options
from merlin.compiler.semantic.target_binding import NativeCompilationError


@contextmanager
def _source(row):
    """Name the check label `row` in any failure of the block, including an unexpected exception."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as failure:
        if row not in getattr(failure, "__notes__", ()):
            failure.add_note(row)
        raise


@contextmanager
def _refused(row, error=ValueError, match=None):
    """Require the block to raise `error` matching `match`; any other outcome fails naming the check label `row`."""
    with _source(row), pytest.raises(error, match=match) as caught:
        yield caught


def _request(name="weights", **changes):
    fields = dict(
        name=name, role="immutable_parameter", bytes=16, alignment=8,
        lifetime="application", load_protocol="once_at_load",
    )
    return ResidencyRequest(**(fields | changes))


def _plan(requests=(), **changes):
    options = dict(base=32, capacity=128, dma_base=0, dma_capacity=256)
    return plan_residency(requests, **(options | changes))


def _plan_placement(directory, patch):
    with _source("check 135"):
        # check 135: first fit avoids reserved holes and straddling reservations without aliasing.
        plan = _plan(
            [_request("a", bytes=9), _request("b", bytes=4), _request("c", bytes=8)],
            base=33, capacity=80,
            reserved=[ReservedInterval("image", 48, 8)],
        )
        assert [(plan.entry(name).address, plan.entry(name).bytes) for name in ("a", "b", "c")] == [
            (56, 9), (40, 4), (72, 8),
        ], "check 135"
        assert plan.required_bytes == 47, "check 135"
        straddling = _plan([_request(bytes=8)], reserved=[ReservedInterval("image", 24, 16)])
        assert straddling.entry("weights").address == 40, "check 135"
        temporal = _plan([
            _request("a", lifetime="phase:one", load_protocol="per_phase"),
            _request("b", lifetime="phase:two", load_protocol="per_phase"),
        ])
        assert temporal.entry("a").end <= temporal.entry("b").address, "check 135"

    with _source("check 134"):
        # check 134: the table keeps distinct roles and explicit load protocols.
        requests = [
            _request("weights"),
            _request("state", role="mutable_state"),
            _request("input", role="staging", lifetime="call", load_protocol="per_call"),
            _request("scratch", role="scratch", lifetime="phase:decode", load_protocol="per_phase"),
        ]
        plan = _plan(requests, reserved=[ReservedInterval("executable", 192, 32)])
        table = {row["name"]: row for row in plan.record()["entries"]}
        assert {row["role"] for row in table.values()} == {
            "immutable_parameter", "mutable_state", "staging", "scratch", "image",
        }, "check 134"
        for request in requests:
            row = table[request.name]
            assert set(row) == {"name", "role", "bytes", "alignment", "address", "lifetime", "load_protocol"}, (
                "check 134")
            assert row["lifetime"] == request.lifetime, "check 134"
            assert row["load_protocol"] == request.load_protocol, "check 134"
        assert plan.entry("executable").load_protocol == "once_at_load", "check 134"
        assert plan.reserved_names == ("executable",), "check 134"

    with _source("check 141"):
        # check 141: empty and zero-byte declarations consume no storage.
        assert _plan(capacity=0).required_bytes == 0, "check 141"
        plan = _plan([_request("empty", bytes=0), _request("full", bytes=128)],
                     reserved=[ReservedInterval("empty_image", 40, 0)])
        assert plan.entry("empty").address == plan.entry("full").address == 32, "check 141"
        assert plan.required_bytes == 128, "check 141"


def _plan_capacity(directory, patch):
    with _source("check 136"):
        # check 136: exact fit and overflow count alignment and reserved bytes.
        assert _plan([_request(bytes=128)]).required_bytes == 128, "check 136"
        with _refused("check 136", ResidencyOverflow, "overflow by 1 bytes") as error:
            _plan([_request(bytes=129)])
        assert (error.value.required_bytes, error.value.capacity_bytes, error.value.overflow_bytes) == (129, 128, 1), (
            "check 136")
        with _refused("check 136", ResidencyOverflow) as error:
            _plan([_request(bytes=17, alignment=16)], base=33, capacity=40,
                  reserved=[ReservedInterval("image", 48, 16)])
        assert (error.value.required_bytes, error.value.overflow_bytes) == (48, 8), "check 136"
        plan = _plan(reserved=[ReservedInterval("staging", 32, 128, role="staging")])
        assert plan.required_bytes == plan.capacity == 128, "check 136"
        with _refused("check 136", ResidencyOverflow) as error:
            _plan(reserved=[ReservedInterval("staging", 32, 129, role="staging")])
        assert error.value.overflow_bytes == 1, "check 136"
        assert _plan(reserved=[ReservedInterval("image", 160, 1000)]).required_bytes == 0, "check 136"

    with _source("check 137"):
        # check 137: the declared arena must fit a valid DMA aperture.
        for options in (
            dict(base=16, capacity=32, dma_base=32), dict(base=32, capacity=225),
            dict(base=-1), dict(capacity=-1), dict(dma_base=-1), dict(dma_capacity=-1),
            dict(base=True), dict(capacity=1.5), dict(dma_capacity=None),
        ):
            with _refused("check 137"):
                _plan(**options)

    with _source("check 138"):
        # check 138: reserved intervals are disjoint, and adjacent intervals are valid.
        with _refused("check 138", match="overlap"):
            _plan(reserved=[ReservedInterval("a", 0, 12), ReservedInterval("b", 11, 4)])
        plan = _plan(reserved=[ReservedInterval("a", 0, 12), ReservedInterval("b", 12, 4)])
        assert len(plan.entries) == 2, "check 138"


def _plan_declarations(directory, patch):
    with _source("check 139"):
        # check 139: invalid request declarations are refused.
        for changes in (
            dict(name=""), dict(role="parameter"), dict(role=[]), dict(bytes=-1), dict(bytes=True),
            dict(alignment=0), dict(alignment=False), dict(lifetime=""), dict(load_protocol="implicit"),
            dict(load_protocol={}),
        ):
            with _refused("check 139"):
                _request(**changes)

    with _source("check 140"):
        # check 140: reserved addresses and alignment, and distinct names, are validated.
        with _refused("check 140", match="misaligned"):
            ReservedInterval("image", 3, 8, alignment=4)
        with _refused("check 140", match="address"):
            ReservedInterval("image", -1, 8)
        for reservations in ((), (ReservedInterval("weights", 192, 8),)):
            requests = [_request()] if reservations else [_request(), _request()]
            with _refused("check 140", match="distinct names"):
                _plan(requests, reserved=reservations)


def _plan_identity(directory, patch):
    with _source("check 142"):
        # check 142: records and digest are stable under declaration reordering.
        requests = [_request("b"), _request("a", bytes=9)]
        reservations = [ReservedInterval("z_image", 192, 8), ReservedInterval("y_image", 208, 8)]
        first = _plan(requests, reserved=reservations)
        second = _plan(reversed(requests), reserved=reversed(reservations))
        assert first == second, "check 142"
        assert first.to_json() == json.dumps(first.record(), sort_keys=True, separators=(",", ":")), "check 142"
        assert first.digest() == sha256(first.to_json().encode()).hexdigest() == second.digest(), "check 142"
        changed = _plan([replace(requests[0], load_protocol="per_call"), requests[1]], reserved=reservations)
        assert changed.digest() != first.digest(), "check 142"
        record = first.record()
        record["entries"][0]["address"] = -1
        assert first.entry("a").address == 32, "check 142"

    with _source("check 143"):
        # check 143: plain records are accepted, and image changes bind the identity.
        request = dict(name="weights", role="immutable_parameter", bytes=16, alignment=8,
                       lifetime="application", load_protocol="once_at_load")
        first = _plan([request], reserved=[dict(name="image", address=192, bytes=8)])
        second = _plan([request], reserved=[dict(name="image", address=200, bytes=8)])
        assert first.entry("weights").address == second.entry("weights").address, "check 143"
        assert first.digest() != second.digest(), "check 143"
        with _refused("check 143", KeyError):
            first.entry("missing")


class _ResidentTarget(ResidentParameters):
    """A declared residency target whose hooks are instance attributes, so a check can replace one."""

    def __init__(self, **hooks):
        vars(self).update(hooks)

    def resident_memory_geometry(self, *args, **kwargs):
        raise AssertionError("this check does not set the hook")

    inventory_resident_inputs = pack_resident_input = admit_resident_deployment = resident_memory_geometry


def _inputs(patch, directory):
    """One application with one device region reading one external parameter, patched through `patch`."""
    source = SimpleNamespace(digest=lambda: "source-digest")
    region = SimpleNamespace(digest=lambda: "region-digest", input_storages=(("arg0", "external"),))
    archive_path = directory / "weights.bin"
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

    adapter = _ResidentTarget(
        inventory_resident_inputs=lambda request, alternative: inventory,
        pack_resident_input=pack,
        resident_memory_geometry=lambda: {"dma_base": 0, "dma_capacity": 1024, "alignment": 8},
    )
    plan = SimpleNamespace(selected=(SimpleNamespace(id="device", placement="accelerator"),))
    patch.setattr(residency, "application_plan", lambda request, selected, parameters=None: plan)
    patch.setattr(residency, "outline_semantic_regions", lambda *args, **kwargs:
                  SimpleNamespace(regions=(("device", region),)))
    from merlin.compiler.lower.c import weights as weights_pack

    def checked_payload(path, *, archive_sha256, out_blob):
        out_blob.write_bytes(path.read_bytes()[6:])
        return {"payload_offset": 6, "payload_sha256": parameters.payload_sha256,
                "payload_bytes": parameters.payload_bytes}

    patch.setattr(weights_pack, "checked_payload", checked_payload)
    options = CodegenOptions(application_residency=True, io_memory_base=32, io_memory_bytes=16,
                             io_memory_capacity_base=32, io_memory_capacity_bytes=64)
    return SimpleNamespace(**locals())


@contextmanager
def _own(patch, directory, label):
    """The `inputs` instance of one check, in its own directory and nested monkeypatch context."""
    root = directory / f"check_{label}"
    root.mkdir()
    with _source(f"check {label}"), patch.context() as own:
        yield _inputs(own, root)


@contextmanager
def _scope(inputs, **option_changes):
    with codegen_options(replace(inputs.options, **option_changes)):
        with residency.residency_scope(inputs.source, inputs.adapter, inputs.parameters, None):
            yield


def _declined(inputs, row, parameters=True, state=None):
    """The reason, and the options and publication inside the scope, of an application residency declines."""
    selected = inputs.options
    with codegen_options(selected):
        with residency.residency_scope(inputs.source, inputs.adapter,
                                       inputs.parameters if parameters else None, state):
            inside = current_codegen_options(), residency.publication(), residency.declined()
        assert current_codegen_options() is selected, row
    assert residency.declined() is None and residency.publication() is None, row
    options, published, reason = inside
    # Inside, the application compiles exactly as with residency off: the same options and no table.
    assert options == replace(selected, application_residency=False) and published is None, row
    assert not inputs.packed_sources, row
    return reason


def _two_regions(inputs, second):
    """Outline the application as its region and `second`, both selected for the accelerator."""
    plan = SimpleNamespace(selected=(SimpleNamespace(id="first", placement="accelerator"),
                                    SimpleNamespace(id="second", placement="accelerator")))
    inputs.patch.setattr(residency, "application_plan", lambda *args: plan)
    inputs.patch.setattr(residency, "outline_semantic_regions", lambda *args, **kwargs:
                         SimpleNamespace(regions=(("first", inputs.region), ("second", second))))


def _publication_bytes(directory, patch):
    with _own(patch, directory, 102) as inputs:
        # check 102: publication binds exact packed bytes and restores the options.
        previous = current_codegen_options()
        assert residency.publication() is None, "check 102"
        with _scope(inputs):
            files = residency.publication()
            record = json.loads(files["residency.json"])
            load, = record["loads"]
            assert files[load["path"]] == b"dense\0\0\0", "check 102"
            assert load["sha256"] == sha256(files[load["path"]]).hexdigest(), "check 102"
            assert load["layout"] == inputs.inventory[0]["layout"], "check 102"
            binding = current_codegen_options().resident_bindings["region-digest"]["arg0"]
            assert binding == {"address": 48, "bytes": 8, "immutable": True}, "check 102"
            assert inputs.checked_sources == [inputs.source], "check 102"
            assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")], "check 102"
            rows = {row["name"]: row for row in record["entries"]}
            assert rows[load["name"]]["load_protocol"] == "once_at_load", "check 102"
            assert rows["regional_io"]["load_protocol"] == "per_call", "check 102"
        assert current_codegen_options() is previous, "check 102"
        assert residency.publication() is None, "check 102"

    with _own(patch, directory, 103) as inputs:
        # check 103: repeated publication is byte stable, and a changed source layout is bound.
        with _scope(inputs):
            first = dict(residency.publication())
        with _scope(inputs):
            assert residency.publication() == first, "check 103"
        inputs.inventory[0]["layout"]["padding"] = "explicit-zero"
        with _scope(inputs):
            second = residency.publication()
            assert second["residency.json"] != first["residency.json"], "check 103"
            first_load = json.loads(first["residency.json"])["loads"][0]
            second_load = json.loads(second["residency.json"])["loads"][0]
            assert first_load["name"] != second_load["name"], "check 103"
            assert first_load["sha256"] == second_load["sha256"], "check 103"

    with _own(patch, directory, 104) as inputs:
        # check 104: one immutable layout is loaded once and bound in each device region.
        _two_regions(inputs, SimpleNamespace(digest=lambda: "second-region", input_storages=(("arg0", "external"),)))
        inputs.inventory.append(inputs.inventory[0] | {"node_id": "runtime_input"})
        with _scope(inputs):
            record = json.loads(residency.publication()["residency.json"])
            assert len(record["loads"]) == 1, "check 104"
            bindings = current_codegen_options().resident_bindings
            assert bindings["region-digest"] == bindings["second-region"], "check 104"
            assert set(bindings["region-digest"]) == {"arg0"}, "check 104"
            assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")], "check 104"

    with _own(patch, directory, 118) as inputs:
        # check 118: a disabled scope needs no parameters, provider, or I/O, and hides the outer publication.
        with _scope(inputs):
            outer = residency.publication()
            with codegen_options(CodegenOptions()):
                with residency.residency_scope(None, None, None, None):
                    assert residency.publication() is None, "check 118"
            assert residency.publication() is outer, "check 118"
        assert residency.publication() is None, "check 118"

    with _own(patch, directory, 121) as inputs:
        # check 121: inventory and packing receive the selected alternative.
        selected = inputs.plan.selected[0]
        inventory = inputs.adapter.inventory_resident_inputs
        pack = inputs.adapter.pack_resident_input

        def selected_inventory(region, alternative):
            assert alternative is selected, "check 121"
            return inventory(region, alternative)

        def selected_pack(region, node_id, dense, alternative):
            assert alternative is selected, "check 121"
            return pack(region, node_id, dense, alternative)

        inputs.adapter.inventory_resident_inputs = selected_inventory
        inputs.adapter.pack_resident_input = selected_pack
        with _scope(inputs):
            assert residency.publication() is not None, "check 121"


def _publication_custody(directory, patch):
    with _own(patch, directory, 106) as inputs:
        # check 106: packing reads the authenticated snapshot after the caller's archive changes.
        def authenticated_then_changed(path, **kwargs):
            checked = inputs.checked_payload(path, **kwargs)
            path.write_bytes(b"headerwrong")
            return checked

        inputs.patch.setattr(inputs.weights_pack, "checked_payload", authenticated_then_changed)
        with _scope(inputs):
            record = json.loads(residency.publication()["residency.json"])
            load, = record["loads"]
            assert residency.publication()[load["path"]] == b"dense\0\0\0", "check 106"
            assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")], "check 106"

    with _own(patch, directory, 105) as inputs:
        # check 105: an inventory error propagates instead of skipping the region.
        def bad_inventory(region, alternative):
            raise ValueError("corrupt counted source")

        inputs.adapter.inventory_resident_inputs = bad_inventory
        with _refused("check 105", match="corrupt counted source"):
            with _scope(inputs):
                pass
        assert not inputs.packed_sources, "check 105"
        assert residency.publication() is None, "check 105"

    with _own(patch, directory, 107) as inputs:
        # check 107: the authenticated snapshot's identity and size must match the parameters.
        for changed in ({"payload_sha256": "wrong"}, {"payload_bytes": 6}):
            inputs.patch.setattr(
                inputs.weights_pack, "checked_payload",
                lambda path, changed=changed, **kwargs: inputs.checked_payload(path, **kwargs) | changed)
            with _refused("check 107", match="payload identity differs"):
                with _scope(inputs):
                    pass
            assert not inputs.packed_sources, "check 107"

    with _own(patch, directory, 115) as inputs:
        # check 115: conflicting inventory for one parameter is refused.
        inputs.inventory.append(inputs.inventory[0] | {"bytes": 16})
        with _refused("check 115", match="inconsistent inventory"):
            with _scope(inputs):
                pass
        inputs.inventory[1] = inputs.inventory[0] | {"layout": {"schema": "other-layout"}}
        with _refused("check 115", match="incompatible resident layouts"):
            with _scope(inputs):
                pass

    with _own(patch, directory, 116) as inputs:
        # check 116: a wrong packed byte count is refused without leaking a publication.
        inputs.adapter.pack_resident_input = lambda *args: b"short"
        with _refused("check 116", match="byte count differs"):
            with _scope(inputs):
                pass
        assert residency.publication() is None, "check 116"

    with _own(patch, directory, 117) as inputs:
        # check 117: the same hooks on an object that does not declare the residency capability are refused.
        hooks = vars(inputs.adapter) | {"admit_resident_deployment": inputs.adapter.admit_resident_deployment}
        inputs.adapter = SimpleNamespace(**hooks)
        assert not isinstance(inputs.adapter, ResidentParameters), "check 117"
        with _refused("check 117", match="does not declare"):
            with _scope(inputs):
                pass
        assert not inputs.packed_sources, "check 117"


def _declines(directory, patch):
    with _own(patch, directory, 108) as inputs:
        # check 108: an application without a resident layout compiles without residency.
        inputs.region.input_storages = (("runtime_input", "external"),)
        inputs.adapter.inventory_resident_inputs = lambda region, alternative: None
        assert _declined(inputs, "check 108") == "no device operand of the application has a resident layout", (
            "check 108")

    with _own(patch, directory, 109) as inputs:
        # check 109: an application without external parameters compiles without residency.
        assert _declined(inputs, "check 109", parameters=False) == (
            "the application has no external parameters to keep resident"), "check 109"
        assert not inputs.checked_sources, "check 109"

    with _own(patch, directory, 110) as inputs:
        # check 110: an unsupported immutable consumer compiles without any residency.
        _two_regions(inputs, SimpleNamespace(digest=lambda: "unsupported-region",
                                             input_storages=(("arg0", "external"),)))
        for unsupported in (None, [], [{"node_id": "runtime_input"}]):
            inputs.adapter.inventory_resident_inputs = lambda region, alternative, unsupported=unsupported: (
                inputs.inventory if region is inputs.region else unsupported
            )
            assert _declined(inputs, "check 110") == (
                "a device region reads parameters through an operation with no resident layout: arg0"), "check 110"

    with _own(patch, directory, 114) as inputs:
        # check 114: an arena the undivided capacity cannot hold with staging compiles without residency.
        # The board's one region: dense parameters staged in it fit where the packed arena and staging do not.
        inputs.options = replace(inputs.options, io_memory_bytes=8, io_memory_capacity_bytes=8)
        assert _declined(inputs, "check 114") == "the resident arena of 8 bytes leaves no staging window", (
            "check 114")
        inputs.options = replace(inputs.options, io_memory_bytes=4, io_memory_capacity_bytes=4)
        assert _declined(inputs, "check 114") == (
            "the resident arena of 8 bytes does not fit the declared capacity of 4 bytes"), "check 114"
        assert not inputs.packed_sources, "check 114"

    # check 119: an active residency records application_residency in the codegen identity, so compiles with
    # residency on and off never share an identity.
    with _source("check 119"):
        active = CodegenOptions(application_residency=True, io_memory_capacity_base=0, io_memory_capacity_bytes=64)
        assert active.identity_record()["application_residency"] is True, "check 119"
        assert active.identity_record() != replace(active, application_residency=False).identity_record(), (
            "check 119")

    with _own(patch, directory, 120) as inputs:
        # check 120: persistent state declines before inventory or packing and checks no source.
        inputs.adapter.inventory_resident_inputs = None  # never reached
        reason = _declined(inputs, "check 120", state=object())
        assert reason == "the application has persistent state inputs, which stay a host interface", "check 120"
        assert not inputs.checked_sources, "check 120"


def _capacity_refusals(directory, patch):
    with _own(patch, directory, 111) as inputs:
        # check 111: staging must fit the capacity even when it lies entirely outside it.
        for base in (24, 88, 104):
            with _refused("check 111", match="staging lies outside"):
                with _scope(inputs, io_memory_base=base):
                    pass
            assert not inputs.packed_sources, "check 111"

    with _own(patch, directory, 112) as inputs:
        # check 112: a staging and parameter capacity overflow is refused before packing.
        with _refused("check 112", ResidencyOverflow) as error:
            with _scope(inputs, io_memory_bytes=12, io_memory_capacity_bytes=16):
                pass
        assert error.value.overflow_bytes == 8, "check 112"
        assert not inputs.packed_sources, "check 112"

    with _own(patch, directory, 113) as inputs:
        # check 113: one undivided region puts the arena at its end and stages in the rest.
        with _scope(inputs, io_memory_bytes=64):
            binding = current_codegen_options().resident_bindings["region-digest"]["arg0"]
            assert binding == {"address": 88, "bytes": 8, "immutable": True}, "check 113"
            assert (current_codegen_options().io_memory_base, current_codegen_options().io_memory_bytes) == (
                32, 56), "check 113"
            rows = {row["name"]: row for row in json.loads(residency.publication()["residency.json"])["entries"]}
            assert (rows["regional_io"]["address"], rows["regional_io"]["bytes"]) == (32, 56), "check 113"
        assert inputs.packed_sources == [(inputs.region, "arg0", b"dense")], "check 113"


def _compiled_bindings(inputs):
    binding = current_codegen_options().resident_bindings[inputs.region.digest()]["arg0"]
    return SimpleNamespace(description={"resident_bindings": [dict(node_id="arg0", **binding)]})


def _provider_binding_reports(directory, patch):
    with _own(patch, directory, 123) as inputs, _scope(inputs):
        # check 123: a provider cannot change or replace a requested binding.
        for change in ({"address": 56}, {"bytes": 7}, {"immutable": False}, {"node_id": "other"},
                       {"address": True}):
            package = _compiled_bindings(inputs)
            package.description["resident_bindings"][0].update(change)
            with _refused("check 123", match="did not consume"):
                residency.validate_region_bindings((("device", inputs.region),), (package,))

    with _own(patch, directory, 122) as inputs, _scope(inputs):
        # check 122: the exact planned binding is accepted, with an extra provider key.
        package = _compiled_bindings(inputs)
        package.description["resident_bindings"][0]["provider_layout"] = "extra target metadata"
        assert residency.validate_region_bindings((("device", inputs.region),), (package,)) is None, "check 122"

    with _own(patch, directory, 124) as inputs, _scope(inputs):
        # check 124: a provider cannot ignore a binding.
        with _refused("check 124", match="resident bindings"):
            residency.validate_region_bindings((("device", inputs.region),),
                                               (SimpleNamespace(description={"resident_bindings": []}),))

    with _own(patch, directory, 125) as inputs, _scope(inputs):
        # check 125: a provider cannot report an incomplete binding.
        with _refused("check 125", match="resident bindings"):
            residency.validate_region_bindings((("device", inputs.region),),
                                               (SimpleNamespace(description={"resident_bindings": [
                                                   {"node_id": "arg0"}]}),))

    with _own(patch, directory, 126) as inputs, _scope(inputs):
        # check 126: a provider cannot report duplicate bindings.
        package = _compiled_bindings(inputs)
        package.description["resident_bindings"] *= 2
        with _refused("check 126", match="duplicate"):
            residency.validate_region_bindings((("device", inputs.region),), (package,))


def _resident_region_check(directory, patch):
    from merlin.compiler.build.regions import _check_resident_regions

    with _own(patch, directory, 130) as inputs:
        # check 130: the resident region check refuses a package that ignores its binding.
        package = SimpleNamespace(description={})
        with _scope(inputs), _refused("check 130", match="did not consume"):
            _check_resident_regions((("device", inputs.region),), (package,))

    with _own(patch, directory, 127) as inputs, _scope(inputs):
        # check 127: a table asset requires at least one consuming region.
        record = json.loads(residency.publication()["residency.json"])
        # Model a planning regression that publishes an asset with no requests.
        record["bindings"] = {}
        residency.publication()["residency.json"] = json.dumps(record).encode()
        with codegen_options(replace(current_codegen_options(), resident_bindings={})):
            with _refused("check 127", match="asset is not bound"):
                residency.validate_region_bindings((("device", inputs.region),),
                                                   (SimpleNamespace(description={}),))

    with _own(patch, directory, 128) as inputs, _scope(inputs):
        # check 128: a binding for an uncompiled region is refused.
        with _refused("check 128", match="uncompiled region"):
            residency.validate_region_bindings((), ())

    with _own(patch, directory, 129) as inputs:
        # check 129: the resident region check reraises a deferred compile refusal.
        refusal = NativeCompilationError("resource_limit", "region does not fit")
        with _scope(inputs), _refused("check 129", NativeCompilationError, "does not fit"):
            _check_resident_regions((("device", inputs.region),), (refusal,))


def test_residency_plans_publish_and_refuse_exactly(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "plan_placement": _plan_placement,
        "plan_capacity": _plan_capacity,
        "plan_declarations": _plan_declarations,
        "plan_identity": _plan_identity,
        "publication_bytes": _publication_bytes,
        "publication_custody": _publication_custody,
        "declines": _declines,
        "capacity_refusals": _capacity_refusals,
        "provider_binding_reports": _provider_binding_reports,
        "resident_region_check": _resident_region_check,
    })
