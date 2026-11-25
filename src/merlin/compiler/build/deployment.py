"""Publish the bounded, one-invocation deployment selected by the compiler.

Provider reservations remain opaque. All caller-owned arrays are placed after
them without aliasing; execution and image changes stay in generated runtime C.
"""

from __future__ import annotations

import json
from hashlib import sha256
from math import prod
from pathlib import Path

from merlin.compiler.backend.protocol import PlatformDeployment
from merlin.compiler.lower.c.runtime import DT_BYTES
from merlin.compiler.semantic.codegen import current_codegen_options
from merlin.compiler.semantic.target_binding import NativeCompilationError

from merlin.compiler.schedule.launch_schedule import validate_launch_schedule


def selected(adapter):
    """True when the adapter declares that this compile publishes a bounded on-chip deployment."""
    return isinstance(adapter, PlatformDeployment) and adapter.deployment_required() is True


def _refuse(reason, message):
    status = "resource_limit" if reason.endswith("_overflow") else "unsupported_semantics"
    raise NativeCompilationError(status, reason + ": " + message)


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        _refuse("deployment_provider_contract", "invalid provider " + name)
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def place(source, adapter, regions, interface, native_root, parameters_object=None, measured_host=None):
    """Choose bindings within the provider-published on-chip map."""
    if not selected(adapter):
        return None
    if native_root is None:
        _refuse("deployment_provider_contract", "on-chip deployment requires provider requirements")
    supplied = adapter.deployment_requirements(regions, native_root)
    if supplied is None:
        _refuse("deployment_geometry_mismatch", "provider does not support the selected deployment geometry")
    provider = json.loads(json.dumps(supplied, allow_nan=False))
    if not isinstance(provider, dict) or provider.get("schema") != "merlin.provider_deployment.v1":
        _refuse("deployment_provider_contract", "unsupported provider deployment schema")
    provider_sections(provider)
    resident_section = provider.get("resident_section")
    if current_codegen_options().application_residency and resident_section is None:
        _refuse("deployment_unsupported_residency", "on-chip deployment requires disjoint nonresident storage")
    transport_timeout = provider.pop("transport_timeout_seconds", None)
    transport_clock = provider.pop("transport_clock_hz_bound", None)
    window = provider.get("window")
    if not isinstance(window, dict):
        _refuse("deployment_provider_contract", "provider deployment window must be explicit")
    window_address = _integer(window.get("address"), "window address")
    window_bytes = _integer(window.get("bytes"), "window bytes", positive=True)
    options = current_codegen_options()
    # A resident deployment stages in a prefix of its capacity; every other deployment's capacity is its window.
    capacity_end = window_address + (options.io_memory_capacity_bytes if options.application_residency else window_bytes)
    if (options.io_memory_base != window_address or options.io_memory_bytes != window_bytes
            or options.io_memory_capacity_base != window_address
            or options.io_memory_capacity_base + options.io_memory_capacity_bytes != capacity_end
            or capacity_end < window_address + window_bytes):
        _refuse("deployment_geometry_mismatch", "provider deployment window differs from selected geometry")
    resident = _resident_arena(window_address + window_bytes, capacity_end) if options.application_residency else None
    host_map = provider.get("host_map")
    if not isinstance(host_map, dict):
        _refuse("deployment_provider_contract", "provider host map must be explicit")
    for key in ("imem_address", "imem_bytes", "csr_address"):
        _integer(host_map.get(key), key, positive=key == "imem_bytes")
    runtime = provider.get("runtime")
    if not isinstance(runtime, dict) or any(
            not isinstance(runtime.get(key), str) or not runtime[key].isascii() or not runtime[key].isidentifier()
            for key in ("host_type", "io_type", "start", "wait")):
        _refuse("deployment_provider_contract", "provider runtime identifiers must be explicit")
    for key in ("image_completion", "reload_policy"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            _refuse("deployment_provider_contract", "provider runtime " + key + " must be explicit")
    for key in ("host_context_reservation_bytes", "max_polls", "allocation_alignment"):
        _integer(provider.get(key), key, positive=True)
    host_allocation = provider.get("host_allocation")
    if not isinstance(host_allocation, dict) or host_allocation.get("measurement") not in HOST_MEASUREMENTS:
        _refuse("deployment_provider_contract", "provider host allocation must declare reservations")
    measured = host_allocation["measurement"] == "measured_sections"
    _integer(host_allocation.get("address"), "host allocation address")
    host_bytes = _integer(host_allocation.get("bytes"), "host allocation bytes", positive=True)
    reserved_bytes = _integer(host_allocation.get("reserved_bytes"), "reserved host bytes")
    declared = host_allocation.get("declared_reservations")
    if not isinstance(declared, list):
        _refuse("deployment_provider_contract", "provider host reservations must be explicit")
    host_reservations = {}
    for row in declared:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]
                or row["name"] in host_reservations):
            _refuse("deployment_provider_contract", "invalid provider host reservation identity")
        host_reservations[row["name"]] = _integer(row.get("bytes"), "host reservation bytes")
    if sum(host_reservations.values()) != reserved_bytes or reserved_bytes > host_bytes:
        _refuse("deployment_provider_contract", "provider host reservations exceed or differ from declared capacity")
    if measured and set(host_reservations) != {"stack", "heap"}:
        _refuse("deployment_provider_contract", "measured host sections declare only stack and heap")
    if not measured and host_reservations.get("context", 0) < provider["host_context_reservation_bytes"]:
        _refuse("deployment_provider_contract", "provider context budget exceeds its host reservation")
    if "usable_bytes" in host_allocation:
        declared_usable = _integer(host_allocation.get("usable_bytes"), "usable host bytes", positive=True)
        if declared_usable > host_bytes:
            _refuse("deployment_host_overflow", "usable host region exceeds physical capacity")
        # The reduced aggregate belongs to old saved deployments only.
        host_allocation["usable_bytes"] = host_bytes
    if measured_host is not None and measured_host > host_bytes:
        # The caller linked this application's host once; bench staging gives each step its own smaller host.
        _refuse("deployment_host_overflow",
                f"measured application host {measured_host} bytes exceeds {host_bytes} backed host bytes")
    published_regions = provider.get("regions")
    if not isinstance(published_regions, list) or len(published_regions) != len(regions):
        _refuse("deployment_provider_contract", "provider must publish every region launch count")
    calls = []
    for region, published in zip(regions, published_regions):
        if (not isinstance(published, dict) or published.get("context_index") != region["context_index"]
                or published.get("placement") != region["placement"]):
            _refuse("deployment_provider_contract", "provider region launch identity differs")
        count = _integer(published.get("launches_per_invocation"), "region launch count")
        if (region["placement"] == "host" and count != 0) or (region["placement"] != "host" and count == 0):
            _refuse("deployment_provider_contract", "provider region launch count differs from placement")
        calls.append({"region": region["id"], "context_index": region["context_index"],
                      "placement": region["placement"], "native_programs": region["native_programs"],
                      "launches_per_invocation": count,
                      **({"device_launch_schedule": published["device_launch_schedule"]}
                         if "device_launch_schedule" in published else {})})
    reservations = provider.get("reservations")
    if not isinstance(reservations, list):
        _refuse("deployment_provider_contract", "provider reservations must be explicit")
    occupied, names = [], set()
    cursor = window_address
    for row in reservations:
        if not isinstance(row, dict) or not all(key in row for key in (
            "name", "address", "bytes", "role", "lifetime", "promotion_receipts",
        )) or not isinstance(row["name"], str) or not row["name"] or row["name"] in names:
            _refuse("deployment_provider_contract", "invalid provider reservation identity")
        names.add(row["name"])
        address = _integer(row["address"], "reservation address")
        size = _integer(row["bytes"], "reservation bytes", positive=True)
        if address < window_address or address + size > window_address + window_bytes:
            _refuse("deployment_provider_contract", "provider reservation outside deployment window")
        for start, end, lifetime in occupied:
            if address < end and start < address + size:
                current = row["lifetime"]
                if (not isinstance(lifetime, dict) or not isinstance(current, dict)
                        or lifetime.get("scope") != "region_call" or current.get("scope") != "region_call"
                        or type(lifetime.get("context_index")) is not int
                        or type(current.get("context_index")) is not int
                        or lifetime["context_index"] == current["context_index"]):
                    _refuse("deployment_provider_contract", "provider reservations overlap within a call or lack distinct call lifetimes")
        occupied.append((address, address + size, row["lifetime"]))
        cursor = max(cursor, address + size)
    allocations = []

    def allocate(name, size, role, alignment=None):
        nonlocal cursor
        if alignment is None:
            alignment = provider["allocation_alignment"]
        _integer(size, name + " bytes")
        if name in names or type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            _refuse("deployment_provider_contract", "invalid deployment allocation identity or alignment")
        names.add(name)
        cursor = (cursor + alignment - 1) // alignment * alignment
        row = dict(name=name, address=cursor, bytes=size, role=role, alignment=alignment)
        allocations.append(row)
        cursor += size
        return row

    allocate("device_scratch", max(1, _integer(provider.get("raw_scratch_bytes"), "raw scratch")), "private_scratch")
    allocate("conversion_scratch", max(1, _integer(provider.get("conversion_scratch_bytes"), "conversion scratch")),
             "private_scratch")
    allocate("workspace", max(1, interface["workspace_bytes"]), "application_workspace",
             max(provider["allocation_alignment"], interface["workspace_alignment"]))
    state = interface.get("model_state", {}).get("ports", [])
    if any(row.get("mode") != "read_only" or row.get("initialization") != "captured"
           or not _digest(row.get("initial_sha256")) for row in state):
        _refuse("deployment_state_contract", "deployment supports only authenticated captured read-only state")
    state_inputs = {row["node_id"]: row for row in state}
    state_outputs = {row["output_index"]: row for row in state if row["mode"] == "updated"}

    def boundary(key, index, kind):
        tensor = source.node(key).type
        size = prod(tensor.shape) * DT_BYTES[tensor.dtype]
        state_port = state_inputs.get(key) if kind == "input" else state_outputs.get(index)
        role = ("persistent_state_input" if kind == "input" else "persistent_state_update") if state_port else (
            "public_input" if kind == "input" else "public_output")
        row = allocate(kind + ":" + key, size, role)
        return {**row, "id": key, "index": index, "type": tensor.record(), "layout": "dense",
                **({"path": f"inputs/input_{index:03d}.bin"} if kind == "input" else {}),
                "endianness": "little", "ownership": "caller", **({"state": state_port["name"]} if state_port else {})}

    ordered_inputs = [boundary(key, i, "input") for i, key in enumerate(interface["inputs"])]
    inputs, state_rows = [], []
    for row in ordered_inputs:
        if state:
            row["source_input_index"] = row["index"]
        if row["role"] == "persistent_state_input":
            port = state_inputs[row["id"]]
            if port["mode"] != "read_only" or port["initialization"] != "captured":
                _refuse("deployment_state_contract", "deployment supports only captured read-only state")
            row.update(path=f"handoff/state/{row['id']}.bin", sha256=port["initial_sha256"], state_mode="read_only", initialization="captured")
            state_rows.append(row)
        else:
            row.update(index=len(inputs), path=f"inputs/input_{len(inputs):03d}.bin")
            inputs.append(row)
    outputs = [boundary(key, i, "output") for i, key in enumerate(interface["outputs"])]
    parameters = interface.get("model_parameters")
    parameter = None
    paging = None
    # Instruction images follow the bound bytes at a 16-byte boundary in the same window load.
    image_bytes = window_image_bytes(provider)
    data_limit = window_bytes if not image_bytes else max(0, window_bytes - image_bytes) // 16 * 16
    if parameters:
        alignment = provider["allocation_alignment"]
        full_end = (cursor + alignment - 1) // alignment * alignment + parameters["payload_bytes"]
        if full_end - window_address > data_limit and provider.get("refill_groups"):
            paging, parameter = _page(provider, parameters, regions, allocate, parameters_object,
                                      window_address + data_limit)
        else:
            parameter = {**allocate("parameters", parameters["payload_bytes"], "immutable_parameters"),
                         "path": "handoff/parameters.bin", "sha256": parameters["payload_sha256"]}
    required = cursor - window_address
    if required > data_limit:
        _refuse("deployment_window_overflow",
                f"disjoint on-chip deployment requires {required} bytes"
                f"{f' plus {image_bytes} image bytes' if image_bytes else ''}; "
                f"window {window_bytes} bytes; overflow {required - data_limit} bytes")
    if host_allocation.get("placement") == "after_window_data":
        # The host shares the window's memory and starts after the placed data and images, so the linker measures
        # what the host uses instead of a fixed host reservation.
        if (paging is not None or host_allocation["address"] != window_address
                or host_allocation["bytes"] != window_bytes):
            _refuse("deployment_provider_contract", "a host after the window data shares one complete window")
        used = -(-required // 16) * 16 + image_bytes
        host_address = window_address + -(-used // provider["allocation_alignment"]) * provider["allocation_alignment"]
        if host_address + reserved_bytes > window_address + window_bytes:
            _refuse("deployment_window_overflow", "window data and images leave no room for the host stack and heap")
        host_allocation.update(address=host_address, bytes=window_address + window_bytes - host_address)
        host_allocation["usable_bytes"] = host_allocation["bytes"]
        window["bytes"] = window_bytes = host_address - window_address
        data_limit = window_bytes - image_bytes
    if paging is None:
        provider.pop("refill_groups", None)
    deployment = {
        "schema": "merlin.onchip_deployment.v1", "host_linker_abi": "symbolic_reservations_v1",
        "request_digest": source.digest(),
        "target_identity": source.target_identity, "window": provider["window"], "provider": provider,
        "allocations": allocations, "inputs": inputs, "outputs": outputs, "parameters": parameter,
        **({"state_inputs": state_rows} if state_rows else {}),
        **({"resident": resident} if resident else {}),
        "calls": calls,
        "window_required_bytes": required, "window_remaining_bytes": data_limit - required,
        **({"window_image_reserve_bytes": image_bytes} if image_bytes else {}),
        "storage_policy": "disjoint; no lifetime sharing", "completion": "synchronous generated application run",
        "host_map": host_map, "max_polls": provider["max_polls"],
        "polling_bound": "host polls per generated device wait; independent of functional-model instruction limit",
        "host_allocation": provider["host_allocation"],
    }
    if paging is not None:
        timeout = _integer(transport_timeout, "transport timeout seconds", positive=True)
        clock = _integer(transport_clock, "transport clock Hz bound", positive=True)
        deployment.update(schema="merlin.onchip_deployment.v2", paging=paging, transport_timeout_seconds=timeout,
                          refill_wait=dict(max_cycles=timeout * clock, clock_hz=clock,
                                           clock_assumption=f"host cycle counter, core frequency at most {clock / 1_000_000_000:g} GHz; conservative {timeout}-second minimum"))
    interface["deployment"] = deployment
    return deployment


def _resident_arena(window_end, capacity_end):
    """One read-only load covering every resident entry, which lie after the window and inside the capacity."""
    from merlin.compiler.schedule.storage.application_residency import publication

    files = publication()
    if files is None:
        _refuse("deployment_unsupported_residency", "resident deployment requires the active residency publication")
    record = json.loads(files["residency.json"])
    entries = {row["name"]: row for row in record["entries"]}
    loads = [dict(name=row["name"], path=row["path"], sha256=row["sha256"], address=entries[row["name"]]["address"],
                  bytes=entries[row["name"]]["bytes"]) for row in record["loads"]]
    if not loads:
        _refuse("deployment_unsupported_residency", "resident deployment has no resident loads")
    start = min(row["address"] for row in loads)
    end = max(row["address"] + row["bytes"] for row in loads)
    if start < window_end or end > capacity_end:
        _refuse("deployment_geometry_mismatch", "resident loads leave the capacity after the deployment window")
    return dict(name="resident", address=start, bytes=end - start, path="handoff/resident.bin",
                sha256=sha256(_resident_bytes(dict(address=start, bytes=end - start, loads=loads), files)).hexdigest(),
                role="immutable_resident_parameters", loads=loads)


def _resident_bytes(resident, files):
    """The arena image: each packed load at its planned offset, and zero bytes between loads."""
    payload = bytearray(resident["bytes"])
    for row in resident["loads"]:
        data = files[row["path"]]
        if len(data) != row["bytes"] or sha256(data).hexdigest() != row["sha256"]:
            _refuse("deployment_payload_contract", "resident load bytes differ from the residency publication")
        offset = row["address"] - resident["address"]
        payload[offset:offset + row["bytes"]] = data
    return bytes(payload)


def adapter_c(name, deployment):
    """A fixed consumer ABI; all model-specific binding stays generated."""
    runtime = deployment["provider"]["runtime"]
    declarations = f"""
typedef struct {{ const char *name; const char *role; uintptr_t address; size_t bytes; }} merlin_deployment_hash_range;
extern const merlin_deployment_hash_range merlin_deployment_hash_ranges[];
extern const size_t merlin_deployment_hash_count;
int merlin_deployment_bind({runtime["host_type"]} *,{runtime["io_type"]},size_t max_polls);
int merlin_deployment_run(void);
"""
    def pointer(row, const=False):
        return f"({'const ' if const else ''}void *)(uintptr_t)UINT64_C({row['address']})"

    inputs = ",".join(pointer(row, True) for row in sorted(
        deployment["inputs"] + deployment.get("state_inputs", []),
        key=lambda row: row.get("source_input_index", row["index"]))) or "0"
    outputs = ",".join(pointer(row) for row in deployment["outputs"]) or "0"
    hashes = ",\n".join(
        f"  {{{json.dumps(row['id'])},{json.dumps(row['role'])},UINT64_C({row['address']}),{row['bytes']}u}}"
        for row in deployment["outputs"]
    )
    parameter = deployment["parameters"]
    bind_parameters = (f"  status={name}_bind_parameters(&merlin_deployment_context,{pointer(parameter, True)},{parameter['bytes']}u);\n"
                       "  if (status) return status;\n") if parameter else ""
    implementation = f"""
static {name}_context merlin_deployment_context;
_Static_assert(sizeof(merlin_deployment_context)<={deployment["provider"]["host_context_reservation_bytes"]}u,"deployment context reservation exceeded");
const merlin_deployment_hash_range merlin_deployment_hash_ranges[]={{{hashes}}};
const size_t merlin_deployment_hash_count={len(deployment['outputs'])}u;
int merlin_deployment_bind({runtime["host_type"]} *host,{runtime["io_type"]} io,size_t max_polls) {{
  int status={name}_bind(&merlin_deployment_context,host,io,max_polls);
  if (status) return status;
{bind_parameters}  return 0;
}}
int merlin_deployment_run(void) {{
  const void *inputs[]={{{inputs}}};
  void *outputs[]={{{outputs}}};
  return {name}_run(&merlin_deployment_context,inputs,outputs);
}}
"""
    if deployment.get("paging"):
        declarations += "int merlin_deployment_kernel_status(void);\n"
        implementation += "int merlin_deployment_kernel_status(void) { return merlin_deployment_context.state.kernel_status; }\n"
    return declarations, implementation


def publish(deployment, interface, files, native_root, adapter):
    """Authenticate payload identities and serialize the single invocation.

    The adapter adds its platform members, such as a linker script; its notes extend the handoff README.
    """
    if interface.get("platform_binding", {}).get("completion") != "synchronous":
        _refuse("deployment_completion_contract", "on-chip deployment needs synchronous generated completion")
    steps = []
    for row in deployment["inputs"] + deployment.get("state_inputs", []):
        steps.append({"operation": "window_preload", "name": row["id"], "path": row["path"],
                      "source_offset": 0, "address": row["address"], "bytes": row["bytes"],
                      "alignment": row["alignment"], "role": row["role"], "lifetime": "through_run_completion",
                      "identity": "required_runtime_input", "completion": "before_bind"})
    if deployment["parameters"] and not deployment.get("paging"):
        row = deployment["parameters"]
        steps.append({"operation": "window_preload", **row, "source_offset": 0,
                      "lifetime": "immutable_through_completion_and_recovery", "completion": "before_bind"})
    images = deployment["provider"].get("images")
    # A host-only invocation, such as one bench-staged host step, installs no image.
    if not isinstance(images, list) or (not images and any(call["placement"] != "host" for call in deployment["calls"])):
        _refuse("deployment_provider_contract", "on-chip deployment requires instruction images")
    image_library = {}
    for row in images:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            _refuse("deployment_provider_contract", "invalid deployment image")
        path = Path(row["path"])
        if path.is_absolute() or path.as_posix() != row["path"] or ".." in path.parts:
            _refuse("deployment_provider_contract", "unsafe deployment image path")
        root = Path(native_root)
        if any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            _refuse("deployment_provider_contract", "linked deployment image")
        payload = (root / path).read_bytes()
        if not _digest(row.get("sha256")) or sha256(payload).hexdigest() != row["sha256"] or len(payload) != row.get("bytes"):
            _refuse("deployment_provider_contract", "deployment image identity differs")
        offset = _integer(row.get("imem_offset_bytes"), "instruction image offset")
        capacity = _integer(row.get("capacity_bytes"), "instruction image capacity", positive=True)
        if capacity > deployment["host_map"]["imem_bytes"]:
            _refuse("deployment_provider_contract", "instruction image capacity exceeds provider host map")
        convention = row.get("upload_convention")
        if not isinstance(convention, str) or not convention.strip():
            _refuse("deployment_provider_contract", "provider image upload convention must be explicit")
        if offset + len(payload) > capacity:
            _refuse("deployment_imem_overflow", f"instruction image exceeds {capacity} bytes at offset {offset}")
        image_library[row["path"]] = {"operation": "install_image", **row, "source_offset": 0,
                      "address": deployment["host_map"]["imem_address"] + offset, "owner": "generated_runtime",
                      "completion": deployment["provider"]["runtime"]["image_completion"], "final_upload_bytes": len(payload)}
    runtime_events = []
    for call in deployment["calls"]:
        if call["placement"] == "host":
            runtime_events.append({"operation": "host_call", **call, "owner": "generated_runtime"})
            continue
        paths = [(Path(program) / "program.bin").as_posix() for program in call["native_programs"]]
        if any(path not in image_library for path in paths):
            _refuse("deployment_runtime_contract", "generated call image missing from provider library")
        installs = [image_library[path] for path in paths]
        event = {"operation": "region_call", **call, "owner": "generated_runtime"}
        if "device_launch_schedule" in call:
            event["device_launch_schedule"] = validate_launch_schedule(
                call["device_launch_schedule"], image_sha256s=[row["sha256"] for row in installs],
                placement=call["placement"])
            if event["device_launch_schedule"]["total_launches"] != call["launches_per_invocation"]:
                _refuse("deployment_runtime_contract", "provider launch count differs from schedule")
            event["images"] = installs
        else:
            count = call["launches_per_invocation"]
            if len(installs) != 1 or type(count) is not int or count <= 0:
                _refuse("deployment_runtime_contract", "generated call needs an explicit launch count or launch schedule")
            event["events"] = [installs[0], {
                "operation": "launch_group", "repeat": count, "events": [
                    {"operation": "write_packed_inputs", "destinations": "provider reservations for this call"},
                    {"operation": "start", "entry": deployment["provider"]["runtime"]["start"]},
                    {"operation": "wait", "entry": deployment["provider"]["runtime"]["wait"], "max_polls": deployment["max_polls"]},
                    {"operation": "read_outputs", "completion": "successful wait and generated status checks"},
                ]}, {"operation": "restore_dense_outputs"}]
        runtime_events.append(event)
    steps.extend([
        {"operation": "bind", "entry": "merlin_deployment_bind", "completion": "synchronous"},
        {"operation": "invoke", "entry": "merlin_deployment_run", "generated_entry": interface["platform_binding"]["invoke"],
         "completion": "synchronous", "runtime_events": runtime_events,
         "failure": "preserve buffers until device completion or reset"},
        {"operation": "hash", "ranges": deployment["outputs"], "completion": "after_successful_run"},
    ])
    record = {**deployment, "steps": steps, "hash_ranges": deployment["outputs"],
              "image_library": list(image_library.values()), "reload_policy": deployment["provider"]["runtime"]["reload_policy"],
              "adapter_header": "handoff/deployment_adapter.h", "timeout": "max_polls per synchronous device wait"}
    if deployment.get("resident"):
        from merlin.compiler.schedule.storage.application_residency import publication

        files[deployment["resident"]["path"]] = _resident_bytes(deployment["resident"], publication())
    if deployment.get("paging"):
        files["handoff/refill-mailbox.bin"] = bytes(deployment["paging"]["mailbox"]["bytes"])
        record = _paged_record(record)
    for path, data in adapter.deployment_platform_files(record).items():
        if path in files and path != "handoff/README.md":
            _refuse("deployment_provider_contract", "platform member replaces a generated member: " + path)
        files[path] = files.get(path, b"") + data if path == "handoff/README.md" else data
    files["deployment.json"] = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    header = ("#pragma once\n#include \"" + Path(interface["header"]).name + "\"\n"
              f"#define MERLIN_DEPLOYMENT_WINDOW_BASE UINT64_C({deployment['window']['address']})\n"
              f"#define MERLIN_DEPLOYMENT_WINDOW_BYTES {deployment['window']['bytes']}u\n"
              f"#define MERLIN_DEPLOYMENT_IMEM_BASE UINT64_C({deployment['host_map']['imem_address']})\n"
              f"#define MERLIN_DEPLOYMENT_CSR_BASE UINT64_C({deployment['host_map']['csr_address']})\n"
              f"#define MERLIN_DEPLOYMENT_IMEM_BYTES {deployment['host_map']['imem_bytes']}u\n"
              f"#define MERLIN_DEPLOYMENT_MAX_POLLS {deployment['max_polls']}u\n").encode()
    if deployment.get("paging"):
        box = deployment["paging"]["mailbox"]
        header += (f"#define MERLIN_DEPLOYMENT_REQUEST UINT64_C({box['address']})\n"
                   f"#define MERLIN_DEPLOYMENT_ACK UINT64_C({box['address'] + 4})\n"
                   f"#define MERLIN_DEPLOYMENT_REFILL_MAX_CYCLES UINT64_C({deployment['refill_wait']['max_cycles']})\n"
                   f"#define MERLIN_DEPLOYMENT_REFILL_CLOCK_HZ UINT64_C({deployment['refill_wait']['clock_hz']})\n"
                   f"#define MERLIN_DEPLOYMENT_TRANSPORT_TIMEOUT_SECONDS {deployment['transport_timeout_seconds']}u\n").encode()
    files["c-package/combined/deployment_adapter.h"] = header
    files["handoff/deployment_adapter.h"] = header


def _partition(rows, size):
    cursor = 0
    for row in rows:
        if (type(row.get("source_offset")) is not int or row["source_offset"] != cursor
                or type(row.get("bytes")) is not int or row["bytes"] <= 0):
            _refuse("deployment_partition", "operand segments must partition dense bytes in order")
        cursor += row["bytes"]
    if cursor != size:
        _refuse("deployment_partition", "operand segments leave a gap or overlap")


def _page(provider, parameters, regions, allocate, parameter_object, window_end):
    facts = provider["refill_groups"]
    if len(facts) != 1 or len(regions) != 1 or parameter_object is None:
        _refuse("deployment_paging_contract", "paging requires one operand in one device call")
    fact = facts[0]
    ports = parameters["ports"]
    matches = [p for p in ports if p["node_id"] == fact["operand"]]
    if len(matches) != 1:
        _refuse("deployment_paging_contract", "paged operand is not an immutable parameter")
    paged = matches[0]
    groups = fact["groups"]
    _partition(groups, paged["nbytes"])
    mailbox = allocate("mailbox", 8, "refill_mailbox")
    small, bindings = [], {}
    for port in ports:
        if port is not paged:
            row = allocate("parameter:" + port["node_id"], port["nbytes"], "immutable_parameters")
            small.append({**row, "source_offset": port["offset"], "path": parameters["path"]})
            bindings[port["node_id"]] = row["address"]
    mailbox.update(request_address=mailbox["address"], acknowledge_address=mailbox["address"] + 4,
                   word_bytes=4, byte_order="little")
    last = small[-1] if small else mailbox
    slot_address = (last["address"] + last["bytes"] + 63) // 64 * 64
    available = window_end - slot_address
    largest = max(g["bytes"] for g in groups)
    if available < 2 * largest:
        _refuse("deployment_slot_overflow", "parameter slot holds fewer than two refill groups")
    slot = allocate("parameter_slot", available // largest * largest, "paged_parameter", 64)
    bindings[paged["node_id"]] = slot["address"]
    segments = []
    for i, group in enumerate(groups):
        if (not segments or segments[-1]["bytes"] + group["bytes"] > slot["bytes"]):
            segments.append(dict(id=len(segments), operand_offset=group["source_offset"], bytes=0,
                                 first_group=i, group_count=0, first_launch=group["first_launch"], launch_count=0))
        segment = segments[-1]
        segment["bytes"] += group["bytes"]
        segment["group_count"] += 1
        segment["launch_count"] += group["launch_count"]
    _partition([dict(source_offset=s["operand_offset"], bytes=s["bytes"]) for s in segments], paged["nbytes"])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp) / "parameters.bin"
        parameter_object.write_payload(blob)
        payload = blob.read_bytes()
    for row in [*small, *segments]:
        offset = row.get("source_offset", paged["offset"] + row.get("operand_offset", 0))
        row.update(path=parameters["path"], source_offset=offset, sha256=sha256(payload[offset:offset+row["bytes"]]).hexdigest(),
                   destination=slot["address"] if "id" in row else row["address"], alignment=64)
    base = min(bindings.values())
    physical = slot["address"] + slot["bytes"] - base
    parameter = dict(name="parameters", address=base, bytes=physical, role="paged_parameters",
                     path=parameters["path"], source_sha256=parameters["payload_sha256"],
                     source_bytes=parameters["payload_bytes"])
    return dict(context_index=fact["context_index"], operand=paged["node_id"],
                region_operand=fact.get("region_operand", fact["operand"]), slot=slot, mailbox=mailbox,
                segments=segments, small_parameters=small, bindings=bindings,
                source_parameters=parameters, physical_parameter_bytes=physical,
                validation="per-segment runtime FP8 validation before use; later invalid segments return status 2 after earlier starts, with output unchanged",
                integrity="per-segment driver readback SHA256; no whole-blob bind hash"), parameter


def _paged_record(record):
    operations = [s["operation"] for s in record["steps"] if s["operation"] != "window_preload"]
    if operations != ["bind", "invoke", "hash"]:
        _refuse("deployment_paging_contract", "non-preload deployment steps must be bind, invoke, hash in order")
    paging = record["paging"]
    protected = [r for r in record["allocations"] if r["role"] != "paged_parameter"]
    protected += record["provider"]["reservations"]
    protected += record["outputs"] + record["inputs"]
    for segment in paging["segments"]:
        segment["must_not_touch"] = protected
    failure = "FAIL on wrong request id, timeout, or readback mismatch; no retry"
    steps = [s for s in record["steps"] if s["operation"] == "window_preload"]
    steps += [dict(operation="window_preload", **r, lifetime="immutable_through_completion_and_recovery",
                   completion="before_bind") for r in paging["small_parameters"]]
    mailbox = paging["mailbox"]
    steps += [dict(operation="window_preload", name=mailbox["name"], role=mailbox["role"],
                   path="handoff/refill-mailbox.bin", source_offset=0,
                   address=mailbox["address"], bytes=mailbox["bytes"], alignment=mailbox["alignment"],
                   sha256=sha256(bytes(mailbox["bytes"])).hexdigest(),
                   lifetime="mutable_through_completion_and_recovery", completion="before_bind")]
    segments = paging["segments"]
    original = {s["operation"]: s for s in record["steps"] if s["operation"] != "window_preload"}
    steps += [dict(operation="load_segment", **segments[0]), dict(original["bind"]),
              {**original["invoke"], "transport_completion": "await_completion"}]
    for segment in segments[1:]:
        steps += [dict(operation="await_request", step_id=segment["id"]),
                  dict(operation="load_segment", **segment),
                  dict(operation="acknowledge", step_id=segment["id"])]
    steps += [dict(operation="await_completion"), dict(original["hash"])]
    for step in steps:
        step["transport_failure"] = failure
    return {**record, "steps": steps}


class PagedParameters:
    """Bind the physical slot and whole small operands, without a fictitious blob scan."""
    def __init__(self, original, deployment):
        from dataclasses import replace
        self.original, self.paging = original, deployment["paging"]
        self.base = deployment["parameters"]["address"]
        self.payload_bytes = deployment["parameters"]["bytes"]
        self.node_ids = original.node_ids
        self.entries = tuple(replace(e, offset=self.paging["bindings"][k] - self.base)
                             for k, e in zip(original.node_ids, original.entries, strict=True))

    def check(self, source):
        self.original.check(source)

    def record(self):
        return dict(schema="merlin.paged_application_parameters.v1", payload_bytes=self.payload_bytes,
                    source_blob=self.original.record(), source_sha256=self.original.payload_sha256,
                    source_bytes=self.original.payload_bytes,
                    physical_bindings=self.paging["bindings"],
                    validation=self.paging["integrity"])

    def emit_binding(self, name):
        from dataclasses import replace
        return replace(self.original, payload_bytes=self.payload_bytes, entries=self.entries).emit_binding(
            name, physical_address=self.base)


def handoff_interface(interface):
    """Describe the complete parameters.bin source separately from physical bindings."""
    parameters = interface.get("model_parameters", {})
    if parameters.get("schema") == "merlin.paged_application_parameters.v1":
        return {**interface, "model_parameters": {**parameters["source_blob"],
                "placement": "../deployment.json#/paging"}}
    return interface


# declared_reservations: fixed code, data, context, stack, and heap ceilings.
# measured_sections: measured objects fill host memory; only stack and heap are declared.
HOST_MEASUREMENTS = ("declared_reservations", "measured_sections")


def section_name(value, name):
    """Section spellings belong to the target provider, not the generic compiler: a dot and an ASCII identifier."""
    label = value[1:] if isinstance(value, str) and value.startswith(".") else ""
    if not (label.isascii() and label.isidentifier()):
        _refuse("deployment_provider_contract", "invalid " + name + " section name")
    return value


def provider_sections(provider):
    """The window, image, and optional parameter sections. A provider that names a parameter section loads the
    immutable parameters, which end the window data, and the instruction images in their own read-only load."""
    window = section_name(provider.get("window_section"), "window")
    image = section_name(provider.get("image_section"), "image")
    parameters = provider.get("parameter_section")
    if parameters is not None:
        section_name(parameters, "parameter")
    if len({window, image, parameters}) != 3:
        _refuse("deployment_provider_contract", "window, image, and parameter sections must differ")
    return window, image, parameters


def window_image_bytes(provider):
    """Window bytes reserved for distinct instruction images after the window data.

    Each image is rounded up to 16 bytes, the largest alignment a compiler gives one array, so the
    reservation bounds the linked image section whatever padding the object build inserts.
    """
    if provider["host_allocation"].get("measurement") != "measured_sections":
        return 0
    return sum(-(-size // 16) * 16 for size in {row["sha256"]: row["bytes"] for row in provider.get("images") or ()}.values())
