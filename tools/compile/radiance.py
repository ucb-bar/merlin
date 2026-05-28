"""Radiance Muon kernel compile path — descriptor.yaml → kernel.radiance.elf.

Extracted from `cli.py`. The `compile_radiance_muon(args)` entry point is
short-circuited from `cli.main()` when `args.target == "radiance_muon"`,
bypassing the standard iree-compile flow.

Two descriptor modes:
- **Phase-1 / inline**: descriptor includes `kernel_body`; Jinja embeds it
  inline in kernel.cpp.
- **Phase-2 / manifest**: descriptor includes `manifest:` + `kernel_entry_symbol`;
  precompile.py emits `<name>.muon.o`, kernel_phase2.cpp.j2 wires the
  `extern "C"` entry, and `./merlin build --profile radiance_muon
  --kernel-body-obj <obj>` links the final ELF.

Schema: `models/radiance_muon/vecadd.yaml` (Phase 1) /
`models/radiance_muon/vecadd_v2.yaml` (Phase 2).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil

import yaml

import utils


def compile_radiance_muon(args: argparse.Namespace) -> int:
    """Compile a Muon kernel descriptor (.yaml) into a kernel.radiance.elf.

    Two modes, distinguished by descriptor schema:

    1. **Phase-1 / inline mode** — descriptor includes a `kernel_body` field.
       The Jinja template (kernel.cpp.j2) embeds the body inline in the
       generated kernel.cpp. No external precompiled object.

    2. **Phase-2 / manifest mode** — descriptor includes a `manifest:` field
       (path to a Radiance kernel manifest) and a `kernel_entry_symbol`. The
       compile path:
         a. Loads the manifest, runs precompile.py for the named kernel
            (target=radiance-muon) to produce `<name>.muon.o`.
         b. Renders kernel_phase2.cpp.j2 (declares the kernel as
            `extern "C"`, calls mu_schedule on it).
         c. Hands off to `./merlin build --profile radiance_muon
            --kernel-body-obj <obj>` which links the body .o + wrapper +
            libmuonrt.a + tohost.S → kernel.radiance.elf.

    Schema: see models/radiance_muon/vecadd.yaml (Phase 1) or
    models/radiance_muon/vecadd_v2.yaml (Phase 2).
    """
    descriptor_p = pathlib.Path(args.input_path).resolve()
    if not descriptor_p.is_file():
        utils.eprint(f"❌ radiance_muon: descriptor not found: {descriptor_p}")
        return 1

    with descriptor_p.open() as f:
        desc = yaml.safe_load(f)

    is_manifest_mode = "manifest" in desc and "kernel_entry_symbol" in desc
    is_mlir_mode = "mlir" in desc and "kernel_entry_symbol" in desc

    if is_mlir_mode:
        required = [
            "kernel_name",
            "num_warps",
            "args_struct_name",
            "args_fields",
            "mlir",
            "kernel_entry_symbol",
        ]
    elif is_manifest_mode:
        required = [
            "kernel_name",
            "num_warps",
            "args_struct_name",
            "args_fields",
            "manifest",
            "kernel_entry_symbol",
        ]
    else:
        required = ["kernel_name", "num_warps", "args_struct_name", "args_fields", "kernel_body"]
    missing = [k for k in required if k not in desc]
    if missing:
        utils.eprint(f"❌ radiance_muon: descriptor missing required keys: {missing}")
        return 1

    kernel_name = desc["kernel_name"]

    # Output directory mirrors compile.py's standard layout.
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir).resolve()
    else:
        output_dir = utils.REPO_ROOT / "build" / "compiled_models" / kernel_name / f"radiance_muon_{kernel_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render kernel.cpp via Jinja2.
    try:
        import jinja2
    except ImportError:
        utils.eprint("❌ radiance_muon: jinja2 not installed (uv sync first).")
        return 1

    tmpl_dir = utils.REPO_ROOT / "build_tools" / "radiance" / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(tmpl_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    import datetime

    ctx = {
        **desc,
        "source_descriptor": str(descriptor_p),
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
    }

    # --- MLIR mode: invoke iree-compile to emit kernel_body.ll, then
    # synthesize a one-entry manifest pointing at it. The rest of the flow
    # is identical to manifest mode. -------------------------------------
    body_obj_path: pathlib.Path | None = None
    if is_mlir_mode:
        mlir_path = (descriptor_p.parent / desc["mlir"]).resolve()
        if not mlir_path.is_file():
            utils.eprint(f"❌ radiance_muon: mlir input not found: {mlir_path}")
            return 1

        kernel_entry = desc["kernel_entry_symbol"]
        ll_path = output_dir / f"{kernel_entry}.ll"
        # Locate iree-compile under build/host-merlin-{release,debug}.
        iree_compile = None
        for cand in (
            "host-merlin-release",
            "host-merlin-debug",
            "host-vanilla-release",
            "host-vanilla-debug",
        ):
            p = utils.REPO_ROOT / "build" / cand / "install" / "bin" / "iree-compile"
            if p.is_file():
                iree_compile = p
                break
        if iree_compile is None:
            utils.eprint(
                "❌ radiance_muon mlir-mode: cannot locate iree-compile.\n"
                "  Run `./merlin build --profile full-plugin --compiler-scope radiance` "
                "first."
            )
            return 1

        cmd = [
            str(iree_compile),
            str(mlir_path),
            "--iree-plugin=radiance",
            "--iree-radiance-enable=true",
            "--iree-radiance-emit-llvm-ir=true",
            f"--iree-radiance-emit-llvm-ir-path={ll_path}",
            f"--iree-radiance-num-warps={desc['num_warps']}",
            # Emit the .ll during input-conversion preprocessing; bail
            # immediately after. We only consume the .ll side-effect.
            "--compile-to=input",
            "-o",
            str(output_dir / f"{kernel_entry}.unused.mlir"),
        ]
        if args.dry_run:
            print("+ " + " ".join(cmd))
        else:
            print(f"  🧪 iree-compile + Radiance plugin -> {ll_path}")
            rc = utils.run(cmd, dry_run=False)
            if rc != 0:
                return rc
        if not ll_path.is_file() and not args.dry_run:
            utils.eprint(f"❌ radiance_muon mlir-mode: iree-compile completed but " f"{ll_path} was not produced.")
            return 1

        # llvm-muon clang is LLVM 18.1; iree-compile is built against
        # LLVM 23. Strip LLVM-23-only keywords from the emitted .ll so
        # the older toolchain accepts it. ALSO rename the body symbol
        # to `<entry>_inner` so the wrapper TU can define the public
        # `<entry>` as a thunk that bridges mu_schedule's 4-arg ABI to
        # the memref-expanded ABI emitted by MLIR.
        inner_symbol = kernel_entry + "_inner"
        if not args.dry_run and ll_path.is_file():
            text = ll_path.read_text()
            import re as _re

            text2 = _re.sub(
                r"\bgetelementptr inbounds nuw ",
                "getelementptr inbounds ",
                text,
            )
            text2 = _re.sub(
                r"\bgetelementptr nuw ",
                "getelementptr ",
                text2,
            )
            # Rename the function symbol so the public `<entry>` is free
            # for the wrapper's thunk. Use a word-boundary match on
            # `@<entry>` to avoid catching substring matches.
            text2 = _re.sub(
                r"@" + _re.escape(kernel_entry) + r"\b",
                "@" + inner_symbol,
                text2,
            )
            if text2 != text:
                ll_path.write_text(text2)
                print(f"  🩹 patched LLVM-23-only keywords + renamed body " f"symbol → {inner_symbol} in {ll_path}")

        # Synthesize a tiny manifest pointing at the emitted .ll.
        synth_manifest = output_dir / "_radiance_manifest.json"
        import json

        synth_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "name": kernel_entry,
                            "source": ll_path.name,
                            "source_lang": "ll",
                            "entry_symbol": kernel_entry,
                            "signature": {"operands": []},
                            "match": {"kind": "named_op", "op_name": f"radiance.embed.{kernel_entry}"},
                            "targets": ["radiance-muon"],
                        }
                    ],
                },
                indent=2,
            )
        )

        from tools.kernels import manifest as _kmanifest
        from tools.kernels import precompile as _kprecompile

        m = _kmanifest.load(synth_manifest)
        cache_dir = output_dir / "kernels_cache"
        artifacts = _kprecompile.precompile(
            m,
            cache_dir,
            targets_filter=("radiance-muon",),
        )
        body_obj_path = artifacts[(kernel_entry, "radiance-muon")].path
        print(f"  📦 Precompiled body .o (mlir mode): {body_obj_path}")

    # --- Manifest mode: precompile the kernel body via the Radiance manifest ---
    if is_manifest_mode:
        manifest_rel = desc["manifest"]
        manifest_p = (descriptor_p.parent / manifest_rel).resolve()
        if not manifest_p.is_file():
            utils.eprint(f"❌ radiance_muon: manifest not found: {manifest_p}")
            return 1

        from tools.kernels import manifest as _kmanifest
        from tools.kernels import precompile as _kprecompile

        m = _kmanifest.load(manifest_p)
        kernel_entry = desc["kernel_entry_symbol"]
        # Find the manifest entry whose entry_symbol matches.
        match = next(
            (k for k in m.kernels if k.entry_symbol == kernel_entry),
            None,
        )
        if match is None:
            utils.eprint(
                f"❌ radiance_muon: manifest at {manifest_p} has no kernel "
                f"with entry_symbol={kernel_entry!r}.\n"
                f"  Available: {[k.entry_symbol for k in m.kernels]}"
            )
            return 1
        if "radiance-muon" not in match.targets:
            utils.eprint(
                f"❌ radiance_muon: manifest kernel {match.name!r} does not "
                f"list 'radiance-muon' in targets={list(match.targets)!r}."
            )
            return 1

        cache_dir = output_dir / "kernels_cache"
        artifacts = _kprecompile.precompile(
            m,
            cache_dir,
            targets_filter=("radiance-muon",),
        )
        body_obj_path = artifacts[(match.name, "radiance-muon")].path
        print(f"  📦 Precompiled body .o: {body_obj_path}")

    # --- MLIR mode only: build thunk metadata for the wrapper ---------------
    if is_mlir_mode:
        ctx["mlir_mode"] = True
        ctx["inner_symbol"] = kernel_entry + "_inner"
        inner_decls: list[str] = []
        thunk_calls: list[str] = []
        for f in desc["args_fields"]:
            ftype = f["type"]
            fname = f["name"]
            is_memref = ("*" in ftype) or ftype.strip().startswith("__global")
            if is_memref:
                # MLIR's memref-to-LLVM expansion: alloc, align, offset,
                # size, stride. We pass `void*` for the two pointer
                # slots and let the linker reconcile addrspaces.
                inner_decls.extend(
                    [
                        f"\tvoid *{fname}_alloc",
                        f"\tvoid *{fname}_align",
                        f"\tint64_t {fname}_offset",
                        f"\tint64_t {fname}_size",
                        f"\tint64_t {fname}_stride",
                    ]
                )
                # Use args->n as the size dimension. Works for vecadd /
                # saxpy and any other 1-D ?xT kernel; multi-dim kernels
                # will need richer field metadata.
                thunk_calls.extend(
                    [
                        f"\t\t(void *)args->{fname}",
                        f"\t\t(void *)args->{fname}",
                        "\t\t(int64_t)0",
                        "\t\t(int64_t)args->n",
                        "\t\t(int64_t)1",
                    ]
                )
            else:
                inner_decls.append(f"\t{ftype} {fname}")
                thunk_calls.append(f"\t\targs->{fname}")
        # Trailing 3 i32: tid, tpt, tbid.
        inner_decls.extend(
            [
                "\tuint32_t tid_in_threadblock",
                "\tuint32_t threads_per_threadblock",
                "\tuint32_t threadblock_id",
            ]
        )
        thunk_calls.extend(
            [
                "\t\ttid_in_threadblock",
                "\t\tthreads_per_threadblock",
                "\t\tthreadblock_id",
            ]
        )
        ctx["inner_args_decl"] = ",\n".join(inner_decls)
        ctx["thunk_call_args"] = ",\n".join(thunk_calls)

    # --- Render wrapper template ---------------------------------------------
    # Phase 2 / MLIR mode: kernel is extern "C", linked from a .o.
    # Phase 1: kernel body inline.
    wrapper_template = "kernel_phase2.cpp.j2" if (is_manifest_mode or is_mlir_mode) else "kernel.cpp.j2"
    kernel_cpp_text = env.get_template(wrapper_template).render(**ctx)
    host_cpp_text = env.get_template("host.cpp.j2").render(**ctx)

    (output_dir / "kernel.cpp").write_text(kernel_cpp_text)
    (output_dir / "host.cpp").write_text(host_cpp_text)

    # Stage the data sidecar next to kernel.cpp.
    data_file_field = desc.get("data_file")
    if data_file_field:
        data_src = (descriptor_p.parent / data_file_field).resolve()
        if not data_src.is_file():
            utils.eprint(f"❌ radiance_muon: data file not found: {data_src}")
            return 1
        shutil.copyfile(data_src, output_dir / "data")
    else:
        utils.eprint(
            "⚠️  radiance_muon: descriptor has no `data_file` field; the "
            'generated kernel.cpp `#include "data"` will fail to resolve.'
        )

    print(f"  📄 Generated: {output_dir / 'kernel.cpp'}")
    if data_file_field:
        print(f"  📄 Staged:    {output_dir / 'data'}")

    if args.compile_to == "cpp":
        # Stop here; useful for inspecting the generated source.
        print("  ✅ --compile-to=cpp: stopping after kernel.cpp emit.")
        return 0

    # Hand off to `./merlin build --profile radiance_muon --kernel-dir`.
    merlin = utils.REPO_ROOT / "merlin"
    cmd = [
        str(merlin),
        "build",
        "--profile",
        "radiance_muon",
        "--config",
        "release",
        "--kernel-dir",
        str(output_dir),
        "--kernel-name",
        kernel_name,
    ]
    if body_obj_path is not None:
        cmd.extend(["--kernel-body-obj", str(body_obj_path)])
    if args.dry_run:
        print("+ " + " ".join(cmd))
        return 0
    rc = utils.run(cmd, dry_run=False)
    if rc != 0:
        return rc

    # Copy the produced ELF into output_dir for downstream consumption.
    src_elf = utils.REPO_ROOT / "build" / "radiance_muon-vanilla-release" / f"{kernel_name}.radiance.elf"
    if src_elf.is_file():
        dst_elf = output_dir / f"{kernel_name}.radiance.elf"
        shutil.copyfile(src_elf, dst_elf)
        os.chmod(dst_elf, 0o755)
        print(f"  ✅ kernel.radiance.elf: {dst_elf}")
    return 0
