"""Exercise publication bounds, image integrity, and the generated fixed ABI."""

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from merlin.compile.application_deployment import adapter_c, place, platform_files, publish, selected
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


WINDOW_ADDRESS = 0x200C0000
WINDOW_BYTES = 786432


@dataclass(frozen=True)
class DeploymentOptions(CodegenOptions):
    application_deployment: str | None = None


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tensor = TensorType((8,), "bf16", "raw-bit-pattern")
        self.source = KernelRequest((SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("y", "identity", ("x",), tensor)), ("y",), ("external",),
            (("x", "external"),), "test-target")
        self.interface = dict(inputs=["x"], outputs=["y"], workspace_bytes=128, workspace_alignment=64,
                              header="combined/app.h", platform_binding=dict(completion="synchronous", invoke="app_run"))
        payload = b"\x01\0\0\0"
        (self.root / "program.bin").write_bytes(payload)
        self.provider = dict(schema="merlin.provider_deployment.v1",
            host_map=dict(imem_address=0x20000, imem_bytes=131072, csr_address=0x40000),
            host_allocation=dict(measurement="declared_reservations", address=0x08000000, bytes=65536,
                                 reserved_bytes=65536, declared_reservations=[dict(name="context", bytes=4096),
                                                                           dict(name="other", bytes=61440)]),
            host_context_reservation_bytes=4096, allocation_alignment=64, max_polls=10000000,
            imem_address_convention="host_map.imem_address + image.imem_offset_bytes",
            runtime=dict(host_type="device_host", io_type="device_io", start="device_start", wait="device_wait",
                         image_completion="provider_image_ready", reload_policy="provider_image_reuse"),
            regions=[dict(context_index=0, placement="accelerator", launches_per_invocation=1)],
            window=dict(address=WINDOW_ADDRESS, bytes=WINDOW_BYTES),
            reservations=[dict(name="device_frame", address=WINDOW_ADDRESS, bytes=4096,
                               role="device_scratch", lifetime="through_run_completion", promotion_receipts=[])],
            raw_scratch_bytes=4096, conversion_scratch_bytes=128,
            images=[dict(path="program.bin", sha256=sha256(payload).hexdigest(), bytes=len(payload),
                         imem_offset_bytes=0, capacity_bytes=131072, upload_convention="counted_ensure_loaded")])
        self.adapter = SimpleNamespace(deployment_requirements=lambda *_: deepcopy(self.provider))
        self.regions = [dict(id="r", context_index=0, placement="accelerator", native_programs=["."],
                             description=dict(launches_per_invocation=1))]
        self.options = DeploymentOptions(application_deployment="onchip", io_memory_base=WINDOW_ADDRESS, io_memory_bytes=WINDOW_BYTES,
            io_memory_capacity_base=WINDOW_ADDRESS, io_memory_capacity_bytes=WINDOW_BYTES)

    def placement(self):
        with codegen_options(self.options):
            return place(self.source, self.adapter, self.regions, self.interface, self.root)

    def host_abi_provider(self):
        provider = deepcopy(self.provider)
        provider["host_allocation"].update(usable_bytes=61440, declared_reservations=[
            dict(name=name, bytes=size) for name, size in
            (("code", 36864), ("data", 8192), ("context", 4096), ("stack", 12288), ("heap", 4096))])
        provider["sys_clk_freq_hz"] = 30000000
        return provider

    def test_default_preserves_interface_and_does_not_call_provider(self):
        self.adapter.deployment_requirements = lambda *_: self.fail("default provider call")
        before = deepcopy(self.interface)
        self.assertIsNone(place(self.source, self.adapter, [], self.interface, self.root))
        self.assertEqual(self.interface, before)
        with codegen_options(replace(self.options, application_deployment=None)):
            self.assertFalse(selected())
            self.assertIsNone(place(self.source, self.adapter, [], self.interface, self.root))

    def test_disjoint_arrays_public_ranges_and_hash_roles(self):
        self.interface["model_state"] = {"ports": [dict(node_id="x", output_index=0, mode="updated", name="cache")]}
        deployment = self.placement()
        allocations = deployment["allocations"]
        cursor = WINDOW_ADDRESS + 4096
        for row in allocations:
            self.assertGreaterEqual(row["address"], cursor)
            self.assertEqual(row["address"] % row["alignment"], 0)
            cursor = row["address"] + row["bytes"]
        self.assertEqual([row["name"] for row in allocations[:3]], ["device_scratch", "conversion_scratch", "workspace"])
        self.assertEqual(deployment["outputs"][0]["role"], "persistent_state_update")
        self.assertEqual(sum(row["bytes"] for row in deployment["host_allocation"]["declared_reservations"]), 65536)
        files = {}
        publish(deployment, self.interface, files, self.root)
        record = json.loads(files["deployment.json"])
        self.assertEqual([row["operation"] for row in record["steps"]],
                         ["window_preload", "bind", "invoke", "hash"])
        self.assertEqual(record["steps"][2]["runtime_events"][0]["events"][0]["final_upload_bytes"], 4)
        self.assertEqual(record["steps"][2]["runtime_events"][0]["events"][0]["address"], 0x20000)
        self.assertEqual(record["hash_ranges"], deployment["outputs"])
        self.assertEqual(record["image_library"][0]["completion"], "provider_image_ready")
        self.assertEqual(record["reload_policy"], "provider_image_reuse")
        declarations, implementation = adapter_c("app", deployment)
        self.assertIn("int merlin_deployment_bind(device_host *,device_io,size_t max_polls)", declarations)
        self.assertIn("sizeof(merlin_deployment_context)<=4096u", implementation)
        self.assertIn("persistent_state_update", implementation)

    def test_explicit_selection_and_geometry_mismatch_refuse(self):
        with codegen_options(self.options):
            self.assertTrue(selected())
        self.options = replace(self.options, io_memory_base=WINDOW_ADDRESS + 64)
        with self.assertRaisesRegex(RuntimeError, "deployment_geometry_mismatch"):
            self.placement()
        self.adapter.deployment_requirements = lambda *_: None
        with self.assertRaisesRegex(RuntimeError, "deployment_geometry_mismatch"):
            self.placement()

    def test_provider_must_publish_region_launch_counts(self):
        del self.provider["regions"][0]["launches_per_invocation"]
        with self.assertRaisesRegex(RuntimeError, "invalid provider region launch count"):
            self.placement()

    def test_provider_host_capacity_and_context_budget_refuse(self):
        self.provider["host_allocation"]["bytes"] = 65535
        with self.assertRaisesRegex(RuntimeError, "host reservations exceed or differ"):
            self.placement()
        self.provider["host_allocation"]["bytes"] = 65536
        self.provider["host_context_reservation_bytes"] = 4097
        with self.assertRaisesRegex(RuntimeError, "context budget exceeds"):
            self.placement()

    def test_provider_runtime_image_policies_must_be_explicit(self):
        for key in ("image_completion", "reload_policy"):
            with self.subTest(key=key):
                value = self.provider["runtime"].pop(key)
                with self.assertRaisesRegex(RuntimeError, "provider runtime " + key + " must be explicit"):
                    self.placement()
                self.provider["runtime"][key] = value

    def test_publish_emits_host_abi_platform_files_without_invocation_payload(self):
        self.provider = self.host_abi_provider()
        deployment = self.placement()
        readme = b"Existing handoff instructions.\n"
        files = {"handoff/README.md": readme}
        publish(deployment, self.interface, files, self.root)
        record = json.loads(files["deployment.json"])
        expected = platform_files(record)
        self.assertEqual(set(expected), {"handoff/chip.ld", "handoff/window.S", "handoff/platform-cflags.txt", "handoff/harness_sha256.h", "handoff/platform-requirements.json"})
        for member, payload in expected.items():
            with self.subTest(member=member):
                self.assertEqual(files[member], payload)
        self.assertNotIn("handoff/window.bin", files)
        self.assertEqual(record["host_allocation"]["usable_bytes"], 61440)
        self.assertEqual(record["provider"]["host_allocation"]["usable_bytes"], 61440)
        self.assertEqual(record["provider"]["sys_clk_freq_hz"], 30000000)
        self.assertTrue(files["handoff/README.md"].startswith(readme))
        self.assertIn(b"Deployment provides chip.ld, window.S and platform-cflags.txt.", files["handoff/README.md"])
        self.assertIn(b"platform-cflags.txt to all platform sources, including drivers.", files["handoff/README.md"])

    def test_publish_vcs_map_sha_helper_and_platform_requirements(self):
        from merlin.common.paths import data_path
        self.provider = self.host_abi_provider()
        files = {}
        publish(self.placement(), self.interface, files, self.root)
        record = json.loads(files["deployment.json"])
        self.assertNotIn("vcs_linker_map", record["provider"])
        self.assertNotIn("handoff/vcs.ld", files)
        vcs_linker_map = dict(host_address=0x80000000, window_address=0x81000000)
        vcs = platform_files(record, vcs_linker_map=vcs_linker_map)["handoff/vcs.ld"]
        self.assertIn(b"ORIGIN  =  2147483648, LENGTH  =  65536", vcs)
        self.assertIn(b"ORIGIN  =  2164260864", vcs)
        self.assertIn(b"__stack_end <= ORIGIN(HOST)+LENGTH(HOST)", vcs)
        self.assertEqual(files["handoff/harness_sha256.h"],
                         data_path("runtime", "baremetal", "spike", "output_sha256.h").read_bytes())
        requirements = json.loads(files["handoff/platform-requirements.json"])
        self.assertEqual(requirements["SYS_CLK_FREQ"]["value"], 30000000)
        self.assertEqual(requirements["SP5_LINKER"]["value"], "handoff/chip.ld")
        self.assertEqual(requirements["VCS_LINKER_MAP"], {
            "value": None, "owner": "platform-owned",
            "source": "platform DRAM link map (Kit A used 0x80000000/0x81000000)"})
        for key, row in requirements.items():
            self.assertTrue(row["source"])
            self.assertIn(row["owner"], ("compiler-required", "platform-owned"))
            if row["owner"] == "platform-owned":
                self.assertIsNone(row["value"], key)
        self.assertNotIn("handoff/window.bin", files)
        vcs_linker_map["host_address"] = -1
        with self.assertRaisesRegex(RuntimeError, "deployment_provider_contract"):
            platform_files(record, vcs_linker_map=vcs_linker_map)

        # A historical/edited provider field is never an implicit platform input.
        record["provider"]["vcs_linker_map"] = vcs_linker_map
        self.assertNotIn("handoff/vcs.ld", platform_files(record))

    def test_publish_without_host_abi_preserves_handoff_readme(self):
        readme = b"Existing handoff instructions.\n"
        files = {"handoff/README.md": readme}
        publish(self.placement(), self.interface, files, self.root)
        self.assertEqual(files["handoff/README.md"], readme)
        for member in ("chip.ld", "vcs.ld", "harness_sha256.h", "platform-requirements.json", "window.S", "platform-cflags.txt", "window.bin"):
            with self.subTest(member=member):
                self.assertNotIn("handoff/" + member, files)

    def test_place_refuses_host_abi_stack_and_heap_overflow(self):
        self.provider = self.host_abi_provider()
        self.provider["host_allocation"].update(usable_bytes=32768, declared_reservations=[
            dict(name=name, bytes=size) for name, size in
            (("code", 8192), ("data", 4096), ("context", 4096), ("stack", 24576), ("heap", 24576))])
        with self.assertRaisesRegex(RuntimeError, "deployment_host_overflow.*stack and heap"):
            self.placement()

    def test_place_refuses_host_abi_without_clock(self):
        self.provider = self.host_abi_provider()
        del self.provider["sys_clk_freq_hz"]
        with self.assertRaisesRegex(RuntimeError, "deployment_clock_contract"):
            self.placement()

    def test_provider_geometry_drives_generic_adapter(self):
        base, size = 0x90000000, 65536
        self.options = replace(self.options, io_memory_base=base, io_memory_bytes=size,
                               io_memory_capacity_base=base, io_memory_capacity_bytes=size)
        self.provider["window"] = dict(address=base, bytes=size)
        self.provider["reservations"][0]["address"] = base
        self.provider["host_map"] = dict(imem_address=0x30000, imem_bytes=64, csr_address=0x50000)
        self.provider["host_context_reservation_bytes"] = 1024
        self.provider["images"][0].update(capacity_bytes=64, imem_offset_bytes=4,
                                           upload_convention="provider_loader")
        self.provider["imem_address_convention"] = "provider image offset metadata"
        deployment = self.placement()
        declarations, implementation = adapter_c("app", deployment)
        self.assertIn("merlin_deployment_bind(device_host *,device_io", declarations)
        self.assertIn("sizeof(merlin_deployment_context)<=1024u", implementation)
        files = {}
        publish(deployment, self.interface, files, self.root)
        self.assertIn(b"#define MERLIN_DEPLOYMENT_WINDOW_BASE UINT64_C(2415919104)",
                      files["handoff/deployment_adapter.h"])
        record = json.loads(files["deployment.json"])
        self.assertEqual(record["image_library"][0]["address"], 0x30004)
        self.assertEqual(record["image_library"][0]["upload_convention"], "provider_loader")
        self.assertEqual(record["steps"][2]["runtime_events"][0]["events"][1]["events"][1]["entry"],
                         "device_start")

    def test_overflow_names_required_window_and_excess(self):
        self.provider["raw_scratch_bytes"] = WINDOW_BYTES
        with self.assertRaisesRegex(RuntimeError, r"requires [0-9]+ bytes; window 786432 bytes; overflow [0-9]+ bytes"):
            self.placement()

    @unittest.skipUnless(shutil.which("cc"), "requires C compiler")
    def test_fixed_adapter_c_binds_and_runs_published_addresses(self):
        deployment = self.placement()
        declarations, implementation = adapter_c("app", deployment)
        expected_input = deployment["inputs"][0]["address"]
        expected_output = deployment["outputs"][0]["address"]
        code = f"""
#include <stdint.h>
#include <stddef.h>
typedef struct {{ int initialized; }} device_host;
typedef struct {{ int unused; }} device_io;
typedef struct {{ int initialized; }} app_context;
static int app_bind(app_context *ctx,device_host *host,device_io io,size_t polls) {{
  (void)io; ctx->initialized=host->initialized && polls; return !ctx->initialized;
}}
static int app_run(app_context *ctx,const void *const *inputs,void *const *outputs) {{
  return !ctx->initialized || (uintptr_t)inputs[0]!={expected_input}u ||
         (uintptr_t)outputs[0]!={expected_output}u;
}}
{declarations}
{implementation}
int main(void) {{
  device_host host={{1}};
  device_io io={{0}};
  return merlin_deployment_bind(&host,io,100u) || merlin_deployment_run() || merlin_deployment_hash_count!=1u ||
         merlin_deployment_hash_ranges[0].address!={expected_output}u || merlin_deployment_hash_ranges[0].bytes!=16u;
}}
"""
        source = self.root / "adapter.c"
        executable = self.root / "adapter"
        source.write_text(code)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", str(source), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)

    def test_bad_provider_reservations_and_mutated_image_refuse(self):
        self.provider["reservations"].append({**self.provider["reservations"][0], "name": "overlap"})
        with self.assertRaisesRegex(RuntimeError, "reservations overlap"):
            self.placement()
        self.provider["reservations"].pop()
        deployment = self.placement()
        (self.root / "program.bin").write_bytes(b"\0\0\0\0")
        with self.assertRaisesRegex(RuntimeError, "image identity differs"):
            publish(deployment, self.interface, {}, self.root)

    def test_existing_per_call_reuse_and_two_image_runtime_order(self):
        first = self.provider["reservations"][0]
        first["lifetime"] = dict(scope="region_call", context_index=0)
        self.provider["reservations"].append({**first, "name": "second_frame",
            "lifetime": dict(scope="region_call", context_index=1)})
        folder = self.root / "second"
        folder.mkdir()
        payload = b"\x02\0\0\0"
        (folder / "program.bin").write_bytes(payload)
        self.provider["images"].append({**self.provider["images"][0], "path": "second/program.bin",
                                        "sha256": sha256(payload).hexdigest()})
        self.regions.append(dict(id="r2", context_index=1, placement="accelerator", native_programs=["second"],
                                 description=dict(streaming=dict(launch_count=3))))
        self.provider["regions"].append(dict(context_index=1, placement="accelerator", launches_per_invocation=3))
        deployment = self.placement()
        self.assertEqual(deployment["allocations"][0]["address"], WINDOW_ADDRESS + 4096)
        files = {}
        publish(deployment, self.interface, files, self.root)
        record = json.loads(files["deployment.json"])
        events = record["steps"][2]["runtime_events"]
        self.assertEqual([row["region"] for row in events], ["r", "r2"])
        self.assertEqual([row["events"][0]["path"] for row in events], ["program.bin", "second/program.bin"])
        self.assertEqual([row["events"][1]["repeat"] for row in events], [1, 3])
        self.provider["reservations"][1]["lifetime"]["context_index"] = 0
        with self.assertRaisesRegex(RuntimeError, "reservations overlap within a call"):
            self.placement()

    def test_image_offset_bounds_and_upload_identity(self):
        deployment = self.placement()
        image = deployment["provider"]["images"][0]
        image["imem_offset_bytes"] = -1
        with self.assertRaisesRegex(RuntimeError, "invalid provider instruction image offset"):
            publish(deployment, self.interface, {}, self.root)
        image["imem_offset_bytes"] = 131069
        with self.assertRaisesRegex(RuntimeError, "deployment_imem_overflow"):
            publish(deployment, self.interface, {}, self.root)
        image["imem_offset_bytes"] = 131068
        publish(deployment, self.interface, {}, self.root)
        image["upload_convention"] = ""
        with self.assertRaisesRegex(RuntimeError, "upload convention must be explicit"):
            publish(deployment, self.interface, {}, self.root)

    def test_counted_upload_accepts_exact_capacity_and_refuses_one_word_over(self):
        for size in (131072, 131076):
            payload = bytes(size)
            (self.root / "program.bin").write_bytes(payload)
            self.provider["images"][0].update(bytes=size, sha256=sha256(payload).hexdigest())
            deployment = self.placement()
            if size == 131072:
                publish(deployment, self.interface, {}, self.root)
            else:
                with self.assertRaisesRegex(RuntimeError, "image exceeds 131072"):
                    publish(deployment, self.interface, {}, self.root)


if __name__ == "__main__":
    unittest.main()
