"""Dense segment partitions must preserve every byte exactly once and in order."""
import json
import pytest
from merlin.compile.application_deployment import _partition, handoff_interface, PagedParameters
from merlin.semantic_compiler.target_binding import NativeCompilationError

def test_partition_accepts_tail_and_rejects_gaps_overlaps_and_reordering():
    _partition([dict(source_offset=0, bytes=8), dict(source_offset=8, bytes=3)], 11)
    for rows in ([dict(source_offset=0, bytes=8), dict(source_offset=9, bytes=2)],
                 [dict(source_offset=0, bytes=8), dict(source_offset=7, bytes=4)],
                 [dict(source_offset=8, bytes=3), dict(source_offset=0, bytes=8)],
                 [dict(source_offset=0, bytes=8)],
                 [dict(source_offset=0, bytes=0), dict(source_offset=0, bytes=11)]):
        with pytest.raises(NativeCompilationError, match='deployment_partition'):
            _partition(rows, 11)

@pytest.fixture
def case():
    from merlin.tests.runtime.test_application_deployment import DeploymentTest
    instance = DeploymentTest()
    instance.setUp()
    yield instance
    instance.doCleanups()

def test_fitting_tiled_route_retains_exact_v1_publication(case):
    from merlin.compile.application_deployment import publish
    case.interface["model_parameters"] = dict(payload_bytes=128, payload_sha256="0" * 64)
    expected = case.placement()
    files = {}
    publish(expected, case.interface, files, case.root)
    case.provider.update(transport_timeout_seconds=60, transport_clock_hz_bound=1_000_000_000)
    case.provider["refill_groups"] = [dict(context_index=0, operand="arg0",
        groups=[dict(source_offset=0, bytes=64, first_launch=0, launch_count=1)])]
    case.regions[0]["description"] = dict(execution=dict(schema="k32_tiles"))
    actual = case.placement()
    other = {}
    publish(actual, case.interface, other, case.root)
    assert actual == expected and other == files
    assert "refill_groups" not in actual["provider"] and "paging" not in actual

@pytest.mark.parametrize("operations", [["future_operation"], ["invoke", "bind", "hash"],
    ["bind", "bind", "invoke", "hash"], ["bind", "invoke", "invoke", "hash"], ["bind", "invoke", "hash", "hash"],
    ["invoke", "hash"], ["bind", "hash"], ["bind", "invoke"]])
def test_paged_ranges_steps_and_separate_refill_deadline(case, operations):
    from hashlib import sha256
    from types import SimpleNamespace
    from merlin.compile.application_deployment import _paged_record, place, publish
    from merlin.semantic_compiler.codegen import codegen_options
    payload = bytes(2 * 1024 * 1024 + 128)
    case.interface["model_parameters"] = dict(payload_bytes=len(payload),
        payload_sha256=sha256(payload).hexdigest(), path="handoff/parameters.bin", ports=[
        dict(node_id="arg0", nbytes=len(payload)-128, offset=0),
        dict(node_id="arg1", nbytes=128, offset=len(payload)-128)])
    case.provider["refill_groups"] = [dict(context_index=0, operand="arg0", groups=[
        dict(source_offset=i * 32768, bytes=32768, first_launch=i, launch_count=1)
        for i in range(64)])]
    parameter = SimpleNamespace(write_payload=lambda path: path.write_bytes(payload))
    case.provider.update(transport_timeout_seconds=60, transport_clock_hz_bound=1_000_000_000)
    with codegen_options(case.options):
        deployment = place(case.source, case.adapter, case.regions, case.interface, case.root, parameter)
    paging = deployment["paging"]
    assert deployment["parameters"]["source_bytes"] == len(payload) and "sha256" not in deployment["parameters"]
    assert deployment["parameters"]["source_sha256"] == sha256(payload).hexdigest()
    assert paging["mailbox"]["address"] + 8 <= deployment["parameters"]["address"]
    files = {}
    publish(deployment, case.interface, files, case.root)
    record = json.loads(files["deployment.json"])
    mailbox = next(s for s in record["steps"] if s.get("name") == "mailbox")
    assert mailbox["role"] == "refill_mailbox"
    with pytest.raises(NativeCompilationError, match="non-preload deployment steps must be bind, invoke, hash in order"):
        _paged_record({**record, "steps": [dict(operation=op) for op in operations]})
    steps = {s["operation"]: s for s in record["steps"]}
    assert steps["bind"]["completion"] == steps["invoke"]["completion"] == "synchronous"
    assert steps["invoke"]["failure"] == "preserve buffers until device completion or reset"
    assert steps["hash"]["completion"] == "after_successful_run"
    for step in record["steps"]:
        if step["operation"] == "window_preload":
            assert step["lifetime"] and step["completion"] == "before_bind"
    wait = deployment["refill_wait"]
    assert wait["max_cycles"] == deployment["transport_timeout_seconds"] * wait["clock_hz"]
    assert wait["max_cycles"] > deployment["max_polls"]
    header = files["handoff/deployment_adapter.h"]
    assert b"MERLIN_DEPLOYMENT_REFILL_MAX_CYCLES UINT64_C(60000000000)" in header
    assert b"SP5_" not in header and b"SP6_" not in header
    case.provider.update(transport_timeout_seconds=7, transport_clock_hz_bound=123)
    with codegen_options(case.options):
        derived = place(case.source, case.adapter, case.regions, case.interface, case.root, parameter)
    assert derived["refill_wait"]["max_cycles"] == 7 * 123

def test_paged_binding_reuses_parameter_abi(tmp_path):
    from dataclasses import replace
    from merlin.compile.application_handoff import application_handoff
    from merlin.tests.runtime.test_application_parameters import setup
    source, original = setup(tmp_path)
    addresses = {key: 0x1000 + i * 64 for i, key in enumerate(original.node_ids)}
    paged = PagedParameters(original, dict(parameters=dict(address=0x1000, bytes=128),
        paging=dict(bindings=addresses, integrity="driver authenticated")))
    assert paged.record()["source_bytes"] == original.payload_bytes and "payload_sha256" not in paged.record()
    assert paged.record()["source_sha256"] == original.payload_sha256
    interface = dict(source="app.c", header="app.h", entry="app", inputs=[], model_parameters=paged.record())
    handoff = application_handoff(source, handoff_interface(interface), [],
        {"c-package/app.c": b"", "c-package/app.h": b""}, tmp_path)
    abi = json.loads(handoff["handoff/abi.json"])
    assert abi["files"]["parameters.bin"] == original.payload_sha256
    assert abi["model_parameters"]["placement"] == "../deployment.json#/paging"
    assert interface["model_parameters"] == paged.record()
    declarations, implementation, _ = paged.emit_binding("app")
    common, _, _ = replace(original, payload_bytes=128, entries=paged.entries).emit_binding("app")
    assert declarations.split("/*", 1)[0] == common.split("/*", 1)[0]
    assert "refill only at generated request boundaries" in declarations
    assert "merlin_sha256_update" not in implementation
    assert "bytes!=128u" in implementation
