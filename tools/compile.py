#!/usr/bin/env python3
# tools/compile.py

import argparse
import sys
import pathlib
import zipfile
import yaml
import utils

def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("input_path", help="Path to the model directory OR specific .mlir/.onnx file")
    parser.add_argument("--target", required=True, help="Target YAML config file name (e.g., spacemit_x60)")
    parser.add_argument("--hw", help="Hardware sub-target from YAML (e.g., RVV, OPU). If omitted, uses default_hw from YAML.")
    parser.add_argument("--quantized", action="store_true", help="Force quantized mode (auto-detected if .q. in filename)")
    parser.add_argument(
        "--output-dir",
        help=("Override output directory (default: build/compiled_models/<model>/<target>_<basename>/). "
              "If set, all generated files/artifacts are written under this directory."),
    )
    
    # NEW: Build Directory Selector
    parser.add_argument("--build-dir", default="host-vanilla-release", help="Which build directory to use for compiler tools (default: host-vanilla-release)")
    
    # Optional Artifacts
    parser.add_argument("--dump-artifacts", action="store_true", help="Dump executable sources, binaries, and configs")
    parser.add_argument("--dump-phases", action="store_true", help="Dump MLIR compilation phases")
    parser.add_argument("--dump-graph", action="store_true", help="Dump the flow dispatch graph (.dot)")
    parser.add_argument("--build-benchmarks", action="store_true", help="Recompile individual dispatch benchmarks and zip them")

def get_iree_tool(tool_name: str, build_dir_name: str) -> pathlib.Path:
    # 1. ALWAYS check the user's requested build directory FIRST
    primary_build_tool = utils.REPO_ROOT / "build" / build_dir_name / "install" / "bin" / tool_name
    if primary_build_tool.exists():
        return primary_build_tool
        
    # 2. Fallback to merlin release if vanilla isn't found
    fallback_build_tool = utils.REPO_ROOT / "build" / "host-merlin-release" / "install" / "bin" / tool_name
    if fallback_build_tool.exists():
        return fallback_build_tool

    # 3. Absolute last resort: the Conda environment
    env_tool = pathlib.Path(sys.executable).parent / tool_name
    if env_tool.exists(): 
        return env_tool
        
    utils.eprint(f"❌ Error: {tool_name} not found in build/{build_dir_name} or environment.")
    sys.exit(1)

def import_onnx(onnx_path: pathlib.Path, mlir_path: pathlib.Path, build_dir: str, dry_run: bool):
    import_tool = get_iree_tool("iree-import-onnx", build_dir)
    print(f"  📥 Importing ONNX to MLIR using {import_tool.parent.parent.parent.name}...")
    cmd = [str(import_tool), str(onnx_path), "--opset-version", "17", "-o", str(mlir_path)]
    if utils.run(cmd, dry_run=dry_run) != 0:
        utils.eprint("❌ ONNX Import failed.")
        sys.exit(1)

def zip_artifacts(zip_path: pathlib.Path, sources_dir: pathlib.Path, vmfb_dir: pathlib.Path):
    print(f"  📦 Zipping benchmark artifacts...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if sources_dir.exists():
            for f in sources_dir.glob("*.mlir"):
                zf.write(f, f.name)
        if vmfb_dir.exists():
            for f in vmfb_dir.glob("*.vmfb"):
                zf.write(f, f.name)
    print(f"  ✅ Created Flattened Archive: {zip_path}")

def main(args: argparse.Namespace) -> int:
    input_p = pathlib.Path(args.input_path).resolve()
    
    is_quantized = args.quantized
    quant_type = "int8"

    if input_p.is_file():
        model_dir = input_p.parent
        model_name = model_dir.name  
        explicit_file = input_p
        
        parts = input_p.name.split('.')
        if "q" in parts:
            is_quantized = True
            q_idx = parts.index("q")
            if q_idx + 1 < len(parts):
                quant_type = parts[q_idx + 1].lower()
                
        basename = input_p.name.replace(".mlir", "").replace(".onnx", "")
    else:
        model_dir = input_p
        model_name = model_dir.name
        explicit_file = None
        suffix = f".q.{quant_type}" if is_quantized else ""
        basename = f"{model_name}{suffix}"

    config_path = utils.REPO_ROOT / "models" / f"{args.target}.yaml"
    if not config_path.exists():
        utils.eprint(f"❌ Error: Config file not found at {config_path}")
        return 1
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hw_choice = args.hw
    if not hw_choice and "default_hw" in cfg:
        hw_choice = cfg["default_hw"]

    if "targets" in cfg:
        if not hw_choice:
            utils.eprint(f"❌ Error: {args.target}.yaml requires a --hw sub-target, but no default_hw is set.")
            return 1
        if hw_choice not in cfg["targets"]:
            utils.eprint(f"❌ Error: Unknown --hw '{hw_choice}'.")
            return 1

    hw_print = f" ({hw_choice})" if hw_choice else ""
    mode_msg = f"Quantized ({quant_type.upper()})" if is_quantized else "Float (FP32)"
    
    print("=" * 80)
    print(f"🚀 Processing Model: {model_name} | Target: {args.target}{hw_print} | Mode: {mode_msg}")
    print("=" * 80)

    hw_suffix = f"_{hw_choice}" if hw_choice else ""
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir).resolve()
    else:
        output_dir = utils.REPO_ROOT / "build" / "compiled_models" / model_name / f"{args.target}{hw_suffix}_{basename}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mlir_file = output_dir / f"{basename}.mlir"
    vmfb_file = output_dir / f"{basename}.vmfb"
    graph_out = output_dir / f"{basename}_dispatch_graph.dot"

    if not mlir_file.exists():
        if explicit_file:
            if explicit_file.suffix == ".onnx":
                import_onnx(explicit_file, mlir_file, args.build_dir, args.dry_run)
            elif explicit_file.suffix == ".mlir":
                print(f"  📄 Using explicit MLIR file: {explicit_file}")
                mlir_file.write_bytes(explicit_file.read_bytes())
            else:
                utils.eprint(f"❌ Error: Unsupported file type: {explicit_file}")
                return 1
        else:
            source_onnx = model_dir / f"{model_name}{'.q.' + quant_type if is_quantized else ''}.onnx"
            source_mlir = model_dir / f"{basename}.mlir"
            if source_onnx.exists():
                import_onnx(source_onnx, mlir_file, args.build_dir, args.dry_run)
            elif source_mlir.exists():
                print(f"  📄 Found Source MLIR: {source_mlir}")
                mlir_file.write_bytes(source_mlir.read_bytes())
            else:
                utils.eprint(f"❌ Error: Could not find ONNX or MLIR in {model_dir}")
                return 1

    # Stack Flags (UPDATED KEYS TO MATCH YOUR YAML)
    static_flags = cfg.get("generic", []) + cfg.get("plugin_flags", [])
    
    if "targets" in cfg and hw_choice:
        static_flags.extend(cfg["targets"][hw_choice])
    
    if is_quantized:
        print(f"  🧊 Applying {quant_type.upper()} quantized flag overrides...")
        quant_flags = cfg.get(f"quantized_{quant_type}", cfg.get("quantized", []))
        static_flags.extend(quant_flags)
        
    model_overrides = cfg.get("models", {}).get(model_name, [])
    if model_overrides:
        print(f"  🎯 Applying model-specific overrides for '{model_name}'...")
        static_flags.extend(model_overrides)

    dynamic_flags = []
    if args.dump_artifacts or args.build_benchmarks:
        dynamic_flags.extend([
            f"--iree-hal-dump-executable-sources-to={output_dir}/sources/",
            f"--iree-hal-dump-executable-files-to={output_dir}/files/",
            f"--iree-hal-dump-executable-binaries-to={output_dir}/binaries/",
            f"--iree-hal-dump-executable-configurations-to={output_dir}/configs/",
            f"--iree-hal-dump-executable-benchmarks-to={output_dir}/benchmarks/",
        ])
        
    if args.dump_phases:
        dynamic_flags.append(f"--dump-compilation-phases-to={output_dir}/phases/")
        
    if args.dump_graph:
        dynamic_flags.extend([
            "--iree-flow-dump-dispatch-graph",
            f"--iree-flow-dump-dispatch-graph-output-file={graph_out}"
        ])

    print(f"  🔨 Compiling main model...")
    iree_compile = get_iree_tool("iree-compile", args.build_dir)
    cmd = [str(iree_compile), str(mlir_file), "-o", str(vmfb_file)] + static_flags + dynamic_flags
    
    if utils.run(cmd, dry_run=args.dry_run) != 0:
        utils.eprint("❌ Main compilation failed.")
        return 1
    print(f"  ✅ Successfully compiled: {vmfb_file}")

    if args.build_benchmarks:
        sources_dir = output_dir / "benchmarks"
        vmfb_out_dir = sources_dir / "vmfb"
        vmfb_out_dir.mkdir(exist_ok=True)
        
        if sources_dir.exists():
            print("  🧩 Compiling individual dispatch sources...")
            for dispatch_mlir in sources_dir.glob("*.mlir"):
                dispatch_vmfb = vmfb_out_dir / f"{dispatch_mlir.stem}.vmfb"
                d_cmd = [str(iree_compile), str(dispatch_mlir), "-o", str(dispatch_vmfb)] + static_flags
                utils.run(d_cmd, dry_run=args.dry_run)

            zip_name = output_dir / f"{basename}_{args.target}{hw_suffix}_benchmarks.zip"
            zip_artifacts(zip_name, sources_dir, vmfb_out_dir)

    print("=" * 80)
    print(f"🎉 Completed {basename} [{args.target}{hw_suffix}]")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))