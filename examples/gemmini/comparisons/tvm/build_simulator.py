#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build and explicitly load the pinned integer Gemmini functional Spike plugin.

Read exact Git blobs and installed Spike headers; never checkout, fetch, install,
or change the reference repositories. The smoke ELF tests plugin loading only.
Numerical execution belongs to the separate primitive-matmul verifier. This is
functional simulation, not an RTL timing or bitstream qualification.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


LIBGEMMINI_REVISION = "ea8f7ed7afd68e001fb06ddccad9a023c990961d"
PARAMS_SHA256 = "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924"
ENV_OVERRIDES = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX",
                 "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")


def allowed(path):
    path = Path(path).absolute()
    if any(name in part.lower() for part in path.parts for name in ("hammer", "vlsi")):
        raise ValueError(f"Restricted path: {path}")
    resolved = path.resolve()
    if any(name in part.lower() for part in resolved.parts for name in ("hammer", "vlsi")):
        raise ValueError(f"Restricted path: {path}")
    return resolved


def sha256(path):
    return hashlib.sha256(allowed(path).read_bytes()).hexdigest()


def clean_environment():
    env = os.environ.copy()
    for name in ENV_OVERRIDES:
        env.pop(name, None)
    return env


def run(command, env, *, timeout=120, check=True, cwd=None):
    result = subprocess.run(list(map(str, command)), env=env, capture_output=True, timeout=timeout, cwd=cwd)
    if check and result.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(map(str, command))}\n{result.stderr.decode(errors='replace')}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libgemmini-repo", type=Path, required=True)
    parser.add_argument("--revision", default=LIBGEMMINI_REVISION)
    parser.add_argument("--spike-prefix", type=Path, required=True, help="installed Spike bin/include/lib prefix")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--runtime-lib-dir", type=Path, required=True, help="explicit compatible libstdc++ directory")
    parser.add_argument("--runtime-bin-dir", type=Path, required=True, help="explicit helper directory containing dtc")
    parser.add_argument("--smoke-elf", type=Path, required=True, help="already verified CPU-only baremetal ELF")
    parser.add_argument("--build-root", type=Path, required=True, help="configured comparison baselines/tvm-gemmini build root")
    parser.add_argument("--output-dir", type=Path, help="fresh child of --build-root")
    args = parser.parse_args()
    if args.revision != LIBGEMMINI_REVISION:
        parser.error("this provisional functional model requires the inspected full libgemmini revision")
    repo, prefix = allowed(args.libgemmini_repo), allowed(args.spike_prefix)
    runtime_lib, runtime_bin = allowed(args.runtime_lib_dir), allowed(args.runtime_bin_dir)
    smoke_elf, build_root = allowed(args.smoke_elf), allowed(args.build_root)
    if build_root.parts[-2:] != ("baselines", "tvm-gemmini"):
        parser.error("--build-root must be the configured baselines/tvm-gemmini build root")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = allowed(args.output_dir or build_root / f"simulator-{timestamp}")
    if output == build_root or not output.is_relative_to(build_root):
        parser.error("--output-dir must be a fresh child of --build-root")
    compiler_name = shutil.which(args.cxx)
    if compiler_name is None:
        parser.error(f"native C++ compiler unavailable: {args.cxx}")
    compiler, spike = allowed(compiler_name), allowed(prefix / "bin/spike")
    runtime_path = f"{runtime_bin}:/usr/bin:/bin"
    dtc_name = shutil.which("dtc", path=runtime_path)
    if dtc_name is None:
        parser.error("dtc unavailable in the explicit runtime helper path")
    dtc = allowed(dtc_name)
    input_paths = {"builder": allowed(__file__), "compiler": compiler, "spike": spike, "smoke_elf": smoke_elf,
                   "dtc": dtc, "runtime_libstdcpp": allowed(runtime_lib / "libstdc++.so.6")}
    input_hashes = {name: sha256(path) for name, path in input_paths.items()}
    env = clean_environment()
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    sources = {}
    for name in ("gemmini.cc", "gemmini.h", "gemmini_params.h", "Makefile"):
        data = run(["git", "-C", repo, "show", f"{args.revision}:{name}"], env).stdout
        (source_dir / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    if sources["gemmini_params.h"] != PARAMS_SHA256:
        raise ValueError("Pinned model parameter header does not match the selected integer ABI")
    plugin = output / "libgemmini.so"
    dependencies_file = output / "plugin.d"
    command = [compiler, "-shared", "-fPIC", "-O3", "-std=c++17", "-I", prefix / "include", "-L", prefix / "lib",
               f"-Wl,-rpath,{prefix / 'lib'}", "-MMD", "-MF", dependencies_file, source_dir / "gemmini.cc", "-o", plugin]
    dependency_command = [compiler, "-shared", "-fPIC", "-O3", "-std=c++17", "-I", prefix / "include",
                          "-MM", "-MF", dependencies_file, source_dir / "gemmini.cc"]
    run(dependency_command, env, cwd=output)
    dependencies = {str(allowed(name)): sha256(name) for name in shlex.split(dependencies_file.read_text().replace("\\\n", " "))[1:]}
    compiled = run(command, env, check=False, cwd=output)
    (output / "build.stdout").write_bytes(compiled.stdout)
    (output / "build.stderr").write_bytes(compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(f"Native plugin compilation failed; inspect {output / 'build.stderr'}")
    resolved_dependencies = {}
    for name in shlex.split(dependencies_file.read_text().replace("\\\n", " "))[1:]:
        path = allowed(name)
        resolved_dependencies[str(path)] = sha256(path)
    if resolved_dependencies != dependencies:
        raise ValueError("Non-system compiler dependencies changed during compilation")
    for name in ("gemmini.cc", "gemmini.h", "gemmini_params.h"):
        if str(source_dir / name) not in dependencies:
            raise ValueError(f"Compiler did not resolve the staged source/header: {name}")
    if not any(Path(name).is_relative_to(prefix / "include") for name in dependencies):
        raise ValueError("Compiler did not record installed Spike headers")

    runtime_env = clean_environment()
    runtime_env["LD_LIBRARY_PATH"] = str(runtime_lib)
    runtime_env["PATH"] = runtime_path
    plugin_hash = sha256(plugin)
    # Explicit absolute dlopen path; a missing path must fail, never substitute
    # an installed plugin with the same extension name.
    arguments = ["--extension=gemmini", "--isa=rv64gc", "-p1", "-m0x80000000:0x1000000", smoke_elf]
    missing = output / "missing-plugin.so"
    bad_command = [spike, f"--extlib={missing}", *arguments]
    bad_load = run(bad_command, runtime_env, timeout=30, check=False, cwd=output)
    (output / "missing_plugin.stdout").write_bytes(bad_load.stdout)
    (output / "missing_plugin.stderr").write_bytes(bad_load.stderr)
    if bad_load.returncode == 0 or str(missing).encode() not in bad_load.stderr:
        raise ValueError("Explicit missing plugin path did not fail for the expected loader reason")
    load_command = [spike, f"--extlib={plugin}", *arguments]
    loaded = run(load_command, runtime_env, timeout=30, check=False, cwd=output)
    (output / "load.stdout").write_bytes(loaded.stdout)
    (output / "load.stderr").write_bytes(loaded.stderr)
    if loaded.returncode:
        raise RuntimeError(f"Explicit plugin load/CPU smoke failed; inspect {output / 'load.stderr'}")
    for name, expected in dependencies.items():
        if sha256(name) != expected:
            raise ValueError(f"Compiler dependency changed during verification: {name}")
    for name, path in input_paths.items():
        if sha256(path) != input_hashes[name]:
            raise ValueError(f"Input changed during verification: {name}")
    if sha256(plugin) != plugin_hash:
        raise ValueError("Built plugin changed during load verification")
    receipt = {
        "status": "built_and_load_verified", "libgemmini_revision": args.revision, "sources": sources,
        "plugin_path": str(plugin), "plugin_sha256": plugin_hash,
        "spike_path": str(spike), "spike_sha256": input_hashes["spike"],
        "compiler": str(compiler), "compiler_sha256": input_hashes["compiler"],
        "compiler_version": run([compiler, "--version"], env).stdout.decode().splitlines()[0],
        "command": list(map(str, command)), "dependency_command": list(map(str, dependency_command)),
        "resolved_compiler_dependencies": dependencies, "compiler_dependency_scope": "non-system dependencies (-MM/-MMD), not the complete SDK closure",
        "builder_path": str(input_paths["builder"]), "builder_sha256": input_hashes["builder"], "dtc_path": str(dtc), "dtc_sha256": input_hashes["dtc"],
        "runtime_libstdcpp_sha256": input_hashes["runtime_libstdcpp"], "inputs_unchanged": True,
        "runtime_environment": {"LD_LIBRARY_PATH": str(runtime_lib), "PATH": runtime_env["PATH"], "LD_PRELOAD": None, "LD_AUDIT": None},
        "negative_load": {"command": list(map(str, bad_command)), "returncode": bad_load.returncode,
                          "stderr_sha256": hashlib.sha256(bad_load.stderr).hexdigest()},
        "load_smoke": {"kind": "CPU-only ELF with explicitly loaded new plugin", "command": list(map(str, load_command)),
                       "returncode": loaded.returncode, "elf_sha256": input_hashes["smoke_elf"],
                       "stdout_sha256": hashlib.sha256(loaded.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(loaded.stderr).hexdigest(),
                       "numerical_gemmini_verified": False},
        "qualification": {"functional_model_only": True, "rtl_bit_exact": False, "timing_verified": False, "deployed_hardware_verified": False},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json"), "plugin": str(plugin)}))


if __name__ == "__main__":
    main()
