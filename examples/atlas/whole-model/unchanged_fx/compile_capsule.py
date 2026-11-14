"""Build the example runtime's host library from an application's emitted C package."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import torch


def raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def preflight_host_library(manifest, *, execution_host_math=None):
    """Refuse supplier obligations the native ctypes example cannot fulfill."""
    if manifest["application_name"] != "capsule":
        raise ValueError("example runtime packaging requires application-name capsule")
    interface = manifest["application_interface"]
    if execution_host_math is not None:
        if execution_host_math.get("functional_validation") is not True:
            raise ValueError("execution-host math is functional validation only")
        from execution_host_math import validate_execution_host_math_selection
        validate_execution_host_math_selection(manifest, execution_host_math["equivalence"], execution_host_math["archive"], execution_host_math["rv64_output"])
        return interface
    if interface.get("host_linkage"):
        raise ValueError(
            "native ctypes runtime packaging cannot supply selected host linkage: "
            + json.dumps(interface["host_linkage"], sort_keys=True)
            + "; missing pinned-supplier linking and selected host-platform execution support. "
            "Use a platform runner that verifies and links the exact archive SHA-256 "
            "and satisfies each region's host ISA, ABI, rounding and effects requirements."
        )
    return interface


def build_host_library(application, manifest, *, package=None, execution_host_math=None):
    """Build the example runtime shim for a standard capsule application.

    This packages emitted C only; it never captures or compiles source IR.
    The application must use the explicit ``capsule`` C namespace.
    """
    interface = preflight_host_library(manifest, execution_host_math=execution_host_math)
    package = application / "c-package" if package is None else package
    header = interface["header"]
    shim = f'''#include "{header}"
static atlas_host host;
static capsule_context context;
int bind(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get) {{
  if(atlas_host_init(&host,(atlas_host_map){{0x20000u,0x40000u,131072u}},
      (atlas_host_mmio){{read,write,0}},(atlas_host_memory){{1,0,0}})) return -1;
  return capsule_bind(&context,&host,(atlas_tile_memory_io){{put,get,0}},20u);
}}
int run(const void *const *inputs,void *const *outputs) {{ return capsule_run(&context,inputs,outputs); }}
'''
    if interface.get("model_parameters"):
        shim += "int bind_parameters(const void *data,size_t bytes) { return capsule_bind_parameters(&context,data,bytes); }\n"
    if not interface.get("platform_binding"):
        parameter_field = "static capsule_parameters parameters;" if interface.get("model_parameters") else ""
        parameter_argument = "&parameters," if interface.get("model_parameters") else ""
        typed = ""
        pointer_arguments = "inputs,outputs"
        if interface.get("pointer_abi") != "opaque_native_dense_buffers":
            ni, no = len(interface["inputs"]), len(interface["outputs"])
            typed = (f"  const float *typed_inputs[{max(1, ni)}]={{0}}; float *typed_outputs[{max(1, no)}]={{0}};\n"
                     f"  if ((!inputs && {ni}u) || !outputs) return -1;\n"
                     f"  for (size_t i=0;i<{ni}u;++i) typed_inputs[i]=(const float *)inputs[i];\n"
                     f"  for (size_t i=0;i<{no}u;++i) typed_outputs[i]=(float *)outputs[i];\n")
            pointer_arguments = "typed_inputs,typed_outputs"
        shim = f'''#include "{header}"
static capsule_driver_state state;
static _Alignas(64) unsigned char workspace[{interface["workspace_bytes"]}];
static void *contexts[{len(manifest["regions"])}];
static capsule_driver_contexts dispatch={{contexts,{len(manifest["regions"])}}};
{parameter_field}
int run(const void *const *inputs,void *const *outputs) {{
{typed}  return capsule_invoke(&state,workspace,sizeof(workspace),{parameter_argument}{pointer_arguments},&dispatch);
}}
'''
        if interface.get("model_parameters"):
            shim += "int bind_parameters(const void *data,size_t bytes) { return capsule_parameters_bind(&parameters,data,bytes); }\n"
    (package / "shim.c").write_text(shim)
    command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-frounding-math", "-ffp-contract=off", "-shared", "-fPIC",
               interface["source"], "shim.c", "-o", "capsule.so"]
    selection = None
    if execution_host_math is not None:
        from execution_host_math import prepare_execution_host_math
        selection = prepare_execution_host_math(manifest, application / "c-package", package,
            execution_host_math["equivalence"], execution_host_math["archive"], execution_host_math["rv64_output"])
        command += [selection["archive"], *selection["link_flags"]]
    command += ["-lm"]
    if selection is None:
        subprocess.run(command, cwd=package, check=True)
    else:
        from merlin.llvmlower.link_supplier_trace import observe_link_suppliers, verify_link_suppliers
        archive = Path(selection["archive"])
        identity = {"path": str(archive), "sha256": sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size}
        linked = subprocess.run(command, cwd=package, check=True, capture_output=True, text=True)
        if sha256(archive.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("execution-host archive changed during link")
        expected = {symbol: archive for symbol in selection["execution_linkage"][0]["symbols"]}
        observation = observe_link_suppliers(argv=command, input_identities=[identity], expected_suppliers=expected, stderr=linked.stderr)
        verify_link_suppliers(observation, argv=command, input_identities=[identity], expected_suppliers=expected)
        selection.update(validation_only=True, deployment_qualified=False, link_argv=command,
                         link_supplier_observation=observation, library_sha256=sha256((package / "capsule.so").read_bytes()).hexdigest())
        (package / "execution-host-math.json").write_text(json.dumps(selection, indent=2) + "\n")
    return interface

