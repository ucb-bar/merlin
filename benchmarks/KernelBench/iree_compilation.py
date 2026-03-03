#!/usr/bin/env python3

"""
IREE Kernel Compilation Script using iree-turbine

This script automates the compilation of PyTorch models (defined in 'level'
directories) into optimized IREE artifacts (.vmfb) for various configurable
targets (e.g., NVIDIA GPUs, ARM CPUs, RISC-V CPUs).

It performs the following steps:
1.  Parses command-line arguments for input, output, and report directories,
    as well as kernel levels and target names to compile for.
2.  Discovers all '*.py' kernel files in the specified 'level' directories
    (e.g., third_party/KernelBench/KernelBench/level1).
3.  For each kernel file, it dynamically loads the 'Model' class,
    'get_init_inputs()', and 'get_inputs()'.
4.  For each specified target:
    a.  Determines the correct precision (e.g., torch.float16 or torch.float32)
        for the target.
    b.  Instantiates the model and inputs in that precision.
    c.  Exports the PyTorch model to Torch MLIR using iree.turbine.aot.export()
        and saves it as 'kernel.mlir'.
    d.  Compiles 'kernel.mlir' to 'artifact.vmfb' by invoking 'iree-compile'
        with the target-specific flags.
    e.  Dumps intermediate MLIR stages into a 'compilation_phases' subdirectory.
5.  Saves compiled artifacts in a structured output directory
    (e.g., benchmark/KernelBench/results/level1/A100/kernel_name/artifact.vmfb).
6.  Logs all operations to 'iree_compile.log' in the report directory.
7.  Generates detailed 'iree_compile_report.json' and 'iree_compile_report.md'
    summaries in the report directory, including success percentages per-level,
    per-target, and overall.
"""

import sys
import os
import logging
import json
import importlib.util
import subprocess
import tempfile
import time
import argparse
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set, Optional

# Suppress noisy warnings if needed (adjust as necessary)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import torch
    import iree.compiler as ireec
    import iree.runtime as ireert
    import iree.turbine.aot as aot
except ImportError as e:
    print(f"Error: Missing critical dependencies. {e}")
    print("Please ensure 'torch', 'iree-turbine', 'iree-compiler', and 'iree-runtime' are installed.")
    sys.exit(1)

# --- Target Definitions ---
# Defines the compilation flags for each target.
# Users can add new targets here.
#
# Structure:
#   "target_name": {
#       "name": "User-friendly name",
#       "hal_target_backends": List[str], (e.g., ["cuda", "llvm-cpu"])
#       "hal_target_device": Optional[str], (e.g., "cuda", "cpu")
#       "extra_flags": List[str], (Target-specific flags for iree-compile)
#       "export_precision": torch.dtype, (Precision to use for aot.export)
#       "compiler_precision_flags": List[str], (Precision-related compile flags)
#   }
#
TARGET_DEFINITIONS = {
    "LOCAL": {
        "name": "Local Machine (auto-detect)",
        "hal_target_backends": ["llvm-cpu"],
        "hal_target_device": "local",
        "extra_flags": ["--iree-llvmcpu-target-cpu-features=host"],
        "export_precision": torch.float32,
        "compiler_precision_flags": []#"--iree-input-demote-f32-to-f16"],
    },
    "A100": {
        "name": "A100",
        "hal_target_backends": ["cuda"],
        "hal_target_device": "cuda",
        "extra_flags": ["--iree-cuda-target=sm_80"],
        "export_precision": torch.float32,
        "compiler_precision_flags": []#"--iree-input-demote-f32-to-f16"],
    },
    "A6000": {
        "name": "A6000",
        "hal_target_backends": ["cuda"],
        "hal_target_device": "cuda",
        "extra_flags": ["--iree-cuda-target=sm_86"],
        "export_precision": torch.float16,
        "compiler_precision_flags": ["--iree-input-demote-f32-to-f16"],
    },
    "cascadelake": {
        "name": "CascadeLake (x86)",
        "hal_target_backends": ["llvm-cpu"],
        "hal_target_device": "cpu",
        "extra_flags": [
            "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
            # Use "--iree-llvmcpu-target-cpu-features=host" to detect locally
            "--iree-llvmcpu-target-cpu=cascadelake",
            # Add micro-architecture specific flags if needed
        ],
        "export_precision": torch.float32,
        "compiler_precision_flags": [],
    },
    "arm_cortex_a72": {
         "name": "ARM Cortex-A72",
         "hal_target_backends": ["llvm-cpu"],
         "hal_target_device": "cpu",
         "extra_flags": [
             "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
             "--iree-llvmcpu-target-cpu=cortex-a72",
             # Ensure you have the correct sysroot/linker for cross-compilation
         ],
        "export_precision": torch.float32,
        "compiler_precision_flags": [],
    },
    "riscv_sifive_e": {
        "name": "RISC-V SiFive-E (32-bit)",
        "hal_target_backends": ["llvm-cpu"],
        "hal_target_device": "cpu",
        "extra_flags": [
            # These flags are highly dependent on your RISC-V toolchain
            "--iree-llvmcpu-target-triple=riscv32-unknown-elf",
            "--iree-llvmcpu-target-cpu=sifive-e31",
            "--iree-llvmcpu-target-abi=ilp32e",
            "--iree-llvmcpu-target-cpu-features=+m,+a,+c",
            # Add sysroot, linker, etc. as needed for your toolchain
        ],
        "export_precision": torch.float32,
        "compiler_precision_flags": [],
    }
}
# --- End Target Definitions ---


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="IREE Kernel Compilation Benchmark Script")
    
    # Get the script's directory to set defaults relative to it
    script_dir = Path(__file__).parent.resolve() # benchmark/KernelBench
    repo_root = script_dir.parent.parent # Assumes script is at benchmark/KernelBench/iree_compilation.py

    default_input = repo_root / "third_party" / "KernelBench" / "KernelBench"
    default_output = script_dir / "results"
    default_report = script_dir / "report"

    parser.add_argument(
        "--input_base_dir",
        type=Path,
        default=default_input,
        help=f"Base directory containing kernel levels (default: {default_input})"
    )
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        default=default_output,
        help=f"Base directory to save compiled artifacts (default: {default_output})"
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=default_report,
        help=f"Directory to save log and report files (default: {default_report})"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["level1", "level2", "level3", "level4"],
        help="List of kernel levels to process (default: level1 level2 level3 level4)"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["LOCAL"],
        choices=list(TARGET_DEFINITIONS.keys()),
        help=f"List of target names to compile for (default: A100). Choices: {', '.join(TARGET_DEFINITIONS.keys())}"
    )
    
    parser.add_argument(
        "--force_recompile",
        action="store_true",
        help="Force recompilation even if artifact.vmfb already exists (default: skip existing)"
    )
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    args.output_base_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    
    return args


def setup_logging(log_file_path: Path) -> None:
    """Configures logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress excessive logging from libraries if needed
    logging.getLogger("iree").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def load_kernel_module(kernel_file: Path) -> Any:
    """
    Dynamically loads a Python module from a given file path.
    """
    kernel_name = kernel_file.stem
    spec = importlib.util.spec_from_file_location(kernel_name, kernel_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {kernel_file}")

    kernel_module = importlib.util.module_from_spec(spec)

    # Add kernel's directory to sys.path temporarily to handle imports
    kernel_dir = str(kernel_file.parent)
    path_needs_pop = False
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
        path_needs_pop = True

    try:
        spec.loader.exec_module(kernel_module)
    finally:
        if path_needs_pop:
            sys.path.pop(0)

    return kernel_module


def compile_kernel_for_target(
    target_def: Dict[str, Any],
    input_mlir_path: Path,
    vmfb_path: Path,
    mlir_dump_dir: Path,
) -> Tuple[bool, Optional[str]]:
    """
    Invokes iree-compile for a specific target configuration.
    
    Returns:
        (success: bool, error_message: Optional[str])
    """
    
    # Base command
    compile_command = [
        "iree-compile",
        str(input_mlir_path),
        "--iree-input-type=torch",
        "-o", str(vmfb_path),
        f"--dump-compilation-phases-to={mlir_dump_dir}",
        "--iree-opt-level=O3",
    ]
    
    # Add HAL backend targets
    for backend in target_def["hal_target_backends"]:
        compile_command.append(f"--iree-hal-target-backends={backend}")

    # Add HAL device target
    if target_def.get("hal_target_device"):
         compile_command.append(f"--iree-hal-target-device={target_def['hal_target_device']}")
        
    # Add target-specific extra flags
    compile_command.extend(target_def["extra_flags"])
    
    # Add precision-specific flags
    compile_command.extend(target_def["compiler_precision_flags"])

    try:
        logging.debug(f"Executing: {' '.join(compile_command)}")
        process = subprocess.run(
            compile_command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # Ensure consistent encoding
        )
        if process.stderr:
             logging.debug(f"iree-compile stderr:\n{process.stderr}")
        if process.stdout:
            logging.debug(f"iree-compile stdout:\n{process.stdout}")
        
        return True, None

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to compile {vmfb_path.stem} using iree-compile.")
        logging.error(f"Command: {' '.join(map(str, e.cmd))}")
        logging.error(f"Return Code: {e.returncode}")
        error_msg = f"iree-compile failed: {e.stderr.strip().splitlines()[-1] if e.stderr else 'No stderr'}"
        if e.stderr:
             logging.error(f"Stderr:\n{e.stderr}")
        if e.stdout:
             logging.error(f"Stdout:\n{e.stdout}")
        return False, error_msg
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during iree-compile: {e}", exc_info=True)
        return False, str(e)


def generate_reports(
    report: Dict[str, Any],
    total_kernels_by_level: Dict[str, int],
    active_target_names: List[str],
    json_report_path: Path,
    md_report_path: Path,
    start_time: float,
    end_time: float,
) -> int:
    """
    Generates JSON and Markdown summary reports with detailed statistics.
    Returns the total number of failures.
    """
    duration = end_time - start_time
    report_data = report["results"]
    load_failures = report["load_failures"]

    # --- 1. Calculate Statistics ---
    stats = {
        "overall": {"success": 0, "failed": 0, "total": 0},
        "by_level": {},
        "by_level_by_target": {}
    }

    num_targets = len(active_target_names)
    total_kernels_attempted = sum(total_kernels_by_level.values())
    total_compilation_attempts = total_kernels_attempted * num_targets
    
    stats["overall"]["total"] = total_compilation_attempts

    for level, level_total_kernels in total_kernels_by_level.items():
        if level not in report_data:
            continue # Should not happen if initialized correctly

        stats["by_level"][level] = {"success": 0, "failed": 0, "total": 0}
        stats["by_level_by_target"][level] = {}

        level_compilation_attempts = level_total_kernels * num_targets
        stats["by_level"][level]["total"] = level_compilation_attempts
        
        # Count kernel load failures for this level
        level_load_failures_count = len([f for f in load_failures if f["level"] == level])
        
        # Each load failure counts as a failure for *all* targets in that level
        level_failures_from_load = level_load_failures_count * num_targets
        stats["by_level"][level]["failed"] += level_failures_from_load
        
        level_total_success = 0

        for target_name in active_target_names:
            target_results = report_data[level][target_name]
            
            target_success = len(target_results["success"])
            target_compile_export_failures = len(target_results["failed"])
            
            # Total failures for this target = compile/export failures + load failures
            target_total_failed = target_compile_export_failures + level_load_failures_count
            
            # Total attempts for this target = total kernels in the level
            target_total_attempts = level_total_kernels
            
            # Success must be total minus failures
            target_total_success = target_total_attempts - target_total_failed
            # (Sanity check)
            if target_total_success != target_success:
                 logging.warning(f"Stats mismatch for {level}/{target_name}: "
                                 f"{target_total_success} (calc) != {target_success} (list)")
            # This can happen if a kernel is not in load_failures but also not in success/failed
            # Re-calculate success based on total attempts
            target_total_success = target_total_attempts - target_total_failed


            stats["by_level_by_target"][level][target_name] = {
                "success": target_total_success,
                "failed": target_total_failed,
                "total": target_total_attempts
            }
            
            level_total_success += target_total_success
        
        stats["by_level"][level]["success"] = level_total_success
        # Sanity check: Total failed = total attempts - total success
        stats["by_level"][level]["failed"] = level_compilation_attempts - level_total_success
        
        stats["overall"]["success"] += stats["by_level"][level]["success"]
        stats["overall"]["failed"] += stats["by_level"][level]["failed"]

    
    # --- 2. JSON Report ---
    logging.info(f"Writing JSON report to {json_report_path}...")
    report_summary = {
        "summary": {
            "total_compilation_attempts": stats["overall"]["total"],
            "successful_compilations": stats["overall"]["success"],
            "failed_compilations": stats["overall"]["failed"],
            "total_kernels_attempted": total_kernels_attempted,
            "num_targets": num_targets,
            "total_time_seconds": duration,
        },
        "statistics": stats,
        "details": report,
    }
    try:
        with open(json_report_path, "w") as f:
            json.dump(report_summary, f, indent=2, default=str) # Use default=str for Path objects
    except Exception as e:
        logging.error(f"Failed to write JSON report: {e}")

    # --- 3. Markdown Report ---
    logging.info(f"Writing Markdown report to {md_report_path}...")
    try:
        with open(md_report_path, "w") as f:
            f.write("# IREE Compilation Report\n\n")
            f.write("## Overall Summary\n\n")
            
            overall_s = stats['overall']['success']
            overall_t = stats['overall']['total']
            overall_pct = (overall_s / overall_t * 100.0) if overall_t > 0 else 0.0
            
            f.write(f"- **Total Compilation Attempts**: {overall_t} "
                    f"({total_kernels_attempted} kernels x {num_targets} targets)\n")
            f.write(f"- **Successful Compilations**: {overall_s}\n")
            f.write(f"- **Failed Compilations**: {stats['overall']['failed']}\n")
            f.write(f"- **Overall Success Rate**: **{overall_pct:.1f}%**\n")
            f.write(f"- **Total Runtime**: {duration:.2f} seconds\n\n")

            f.write("---")
            f.write("\n## Success Rate by Level\n\n")

            for level in stats["by_level"]:
                level_s = stats['by_level'][level]['success']
                level_t = stats['by_level'][level]['total']
                level_pct = (level_s / level_t * 100.0) if level_t > 0 else 0.0
                
                f.write(f"### Level: {level}\n\n")
                f.write(f"- **Level Total Success**: {level_s} / {level_t} compilations "
                        f"**({level_pct:.1f}%)**\n")
                
                f.write("\n**Success Rate by Target:**\n\n")
                for target in stats["by_level_by_target"][level]:
                    target_s = stats['by_level_by_target'][level][target]['success']
                    target_t = stats['by_level_by_target'][level][target]['total']
                    target_pct = (target_s / target_t * 100.0) if target_t > 0 else 0.0
                    
                    f.write(f"- **{target}**: {target_s} / {target_t} kernels succeeded "
                            f"**({target_pct:.1f}%)**\n")
                f.write("\n")

            f.write("---\n")
            
            # --- Failure Details ---
            if load_failures:
                f.write("## Kernel Loading Failures\n\n")
                f.write("These kernels failed to load and were not attempted for *any* target.\n\n")
                f.write("| Level | Kernel File | Error |\n")
                f.write("|-------|-------------|-------|\n")
                for item in load_failures:
                    error_msg = str(item['error']).strip().split('\n')[-1]
                    f.write(f"| {item['level']} | `{item['kernel_file']}` | `{error_msg}` |\n")
                f.write("\n")

            export_compile_failures = []
            for level, targets in report_data.items():
                for target, results in targets.items():
                    for item in results["failed"]:
                        export_compile_failures.append(item)

            if export_compile_failures:
                f.write("## Export & Compilation Failures\n\n")
                f.write("| Level | Kernel File | Target | Stage | Error |\n")
                f.write("|-------|-------------|--------|-------|-------|\n")
                for item in sorted(export_compile_failures, key=lambda x: (x['level'], x['kernel_file'], x['target'])):
                    try:
                       error_msg = str(item['error']).strip().split('\n')[-1]
                    except:
                       error_msg = str(item['error']).strip()
                    f.write(f"| {item['level']} | `{item['kernel_file']}` | {item['target']} | {item['stage']} | `{error_msg}` |\n")
                f.write("\n")
                
            # --- Success Details (optional, can be very long) ---
            # ... (omitted for brevity, but could be added similar to old script) ...

    except Exception as e:
        logging.error(f"Failed to write Markdown report: {e}", exc_info=True)

    return stats["overall"]["failed"]


def main():
    """Main execution function."""
    start_time_sec = time.time()
    
    args = parse_arguments()
    
    log_file_path = args.report_dir / "iree_compile.log"
    json_report_path = args.report_dir / "iree_compile_report.json"
    md_report_path = args.report_dir / "iree_compile_report.md"

    setup_logging(log_file_path)
    logging.info("Starting IREE kernel compilation process...")
    logging.info(f"Input base directory: {args.input_base_dir}")
    logging.info(f"Output base directory: {args.output_base_dir}")
    logging.info(f"Report directory: {args.report_dir}")
    logging.info(f"Processing levels: {', '.join(args.levels)}")
    logging.info(f"Processing targets: {', '.join(args.targets)}")
    logging.info(f"Force recompile: {args.force_recompile}")

    # Check if iree-compile is available
    try:
        process = subprocess.run(["iree-compile", "--version"], check=True, capture_output=True, text=True)
        logging.info(f"Found iree-compile tool: {process.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logging.error(f"Failed to find or run 'iree-compile': {e}")
        logging.error("Please ensure IREE is correctly installed and in your PATH.")
        sys.exit(1)

    # Filter the target definitions based on user selection
    active_targets = {}
    for t_name in args.targets:
        if t_name in TARGET_DEFINITIONS:
            active_targets[t_name] = TARGET_DEFINITIONS[t_name]
        else:
            logging.warning(f"Target '{t_name}' not found in TARGET_DEFINITIONS. Skipping.")
    
    if not active_targets:
        logging.error("No valid targets selected. Exiting.")
        sys.exit(1)

    logging.info("Active compilation targets:")
    for name, config in active_targets.items():
        logging.info(f"  - {name} (Precision: {config['export_precision']}, Backend: {config['hal_target_backends']})")

    # New report structure
    report = {
        "load_failures": [], # List of {"kernel_file", "level", "error"}
        "results": {}        # Dict[level, Dict[target, {"success": [], "failed": []}]]
    }
    total_kernels_by_level = {}


    for level in args.levels:
        level_dir = args.input_base_dir / level
        if not level_dir.is_dir():
            logging.warning(f"Directory not found, skipping: {level_dir}")
            continue

        logging.info(f"--- Processing Level: {level} ---")
        kernel_files = sorted(level_dir.glob("*.py"))
        total_kernels_by_level[level] = len(kernel_files)
        
        # Initialize report structure for this level
        report["results"][level] = {
            t_name: {"success": [], "failed": []} for t_name in active_targets
        }

        if not kernel_files:
            logging.warning(f"No kernel files (*.py) found in {level_dir}")
            continue

        for kernel_file in kernel_files:
            kernel_name = kernel_file.stem
            # Get relative path from the *repository root* or a sensible base
            try:
                relative_kernel_path = str(kernel_file.relative_to(args.input_base_dir.parent))
            except ValueError:
                relative_kernel_path = str(kernel_file)

            logging.info(f"Processing kernel: {relative_kernel_path}")
            
            model_class = None
            init_args = []
            get_inputs_func = None
            
            # --- 1. Load Module (once per kernel) ---
            try:
                # Set to f32 globally *before* loading
                #torch.set_default_dtype(torch.float32)

                kernel_module = load_kernel_module(kernel_file)

                if not hasattr(kernel_module, 'Model'):
                    raise AttributeError("Kernel file does not define a 'Model' class")
                model_class = kernel_module.Model
                if not (isinstance(model_class, type) and issubclass(model_class, torch.nn.Module)):
                    raise TypeError("'Model' attribute is not a class inheriting from nn.Module")

                if hasattr(kernel_module, 'get_init_inputs'):
                    init_args = kernel_module.get_init_inputs()

                if not hasattr(kernel_module, 'get_inputs'):
                    raise AttributeError("Kernel file does not define a 'get_inputs' function")
                get_inputs_func = kernel_module.get_inputs

            except Exception as e:
                logging.error(f"Failed to load module/variables from {kernel_file}: {e}", exc_info=True)
                report["load_failures"].append({
                    "kernel_file": relative_kernel_path,
                    "level": level,
                    "error": str(e),
                })
                continue # Skip this kernel entirely


            # --- 2. Loop over targets for Export and Compile ---
            for target_name, target_def in active_targets.items():
                logging.info(f"Processing {kernel_name} for {target_name}...")
                
                # --- 2a. Define output paths ---
                kernel_output_dir = args.output_base_dir / level / target_name / kernel_name
                mlir_dump_dir = kernel_output_dir / "compilation_phases" 
                kernel_output_dir.mkdir(parents=True, exist_ok=True)
                mlir_dump_dir.mkdir(parents=True, exist_ok=True)

                vmfb_path = kernel_output_dir / "artifact.vmfb" 
                input_mlir_path = kernel_output_dir / "kernel.mlir" 
                relative_output_path = str(vmfb_path.relative_to(args.output_base_dir.parent))
                
                # --- 2b. Check for existing artifact ---
                if not args.force_recompile and vmfb_path.exists():
                    logging.info(f"Artifact {relative_output_path} already exists. Skipping.")
                    report["results"][level][target_name]["success"].append({
                        "kernel_file": relative_kernel_path,
                        "level": level,
                        "target": target_name,
                        "output_path": relative_output_path,
                        "status": "skipped_exists"
                    })
                    continue # Skip to the next target
                
                # --- 2b. Export to MLIR (target-specific precision) ---
                try:
                    target_precision = target_def["export_precision"]
                    logging.debug(f"Setting default torch dtype to {target_precision}")
                    #torch.set_default_dtype(target_precision)
                    
                    # Re-instantiate model and inputs with the target precision
                    model = model_class(*init_args)
                    model.eval()
                    
                    raw_inputs_list = get_inputs_func()
                    if raw_inputs_list is None:
                        raise ValueError("'get_inputs()' returned None")
                    
                    # Convert tensor inputs to target precision
                    processed_inputs = []
                    for item in raw_inputs_list:
                        if isinstance(item, torch.Tensor) and item.dtype != target_precision:
                            logging.debug(f"Converting input tensor to {target_precision} for {kernel_name}")
                            processed_inputs.append(item.to(target_precision))
                        else:
                            processed_inputs.append(item)
                    
                    processed_inputs_tuple = tuple(processed_inputs)
                    
                    export_output = aot.export(model, args=processed_inputs_tuple)
                    
                    export_output.save_mlir(input_mlir_path)
                    
                    logging.debug(f"Successfully exported Torch MLIR to {input_mlir_path}")

                except Exception as e:
                    logging.error(f"iree.turbine.aot.export failed for {kernel_file} (target: {target_name}): {e}", exc_info=True)
                    report["results"][level][target_name]["failed"].append({
                        "kernel_file": relative_kernel_path,
                        "level": level,
                        "target": target_name,
                        "stage": "export",
                        "error": f"aot.export error: {e}",
                    })
                    
                    continue # Skip to next target
                finally:
                    # ALWAYS reset default dtype
                    #torch.set_default_dtype(torch.float32)
                    DUMMY = 0

                # --- 2c. Compile MLIR (iree-compile) ---
                try:
                    success, error_msg = compile_kernel_for_target(
                        target_def,
                        input_mlir_path, 
                        vmfb_path,
                        mlir_dump_dir
                    )
                    
                    if success:
                        logging.info(f"Successfully compiled. Artifact: {relative_output_path}")
                        report["results"][level][target_name]["success"].append({
                            "kernel_file": relative_kernel_path,
                            "level": level,
                            "target": target_name,
                            "output_path": relative_output_path,
                            "status": "compiled"
                        })
                    else:
                        report["results"][level][target_name]["failed"].append({
                            "kernel_file": relative_kernel_path,
                            "level": level,
                            "target": target_name,
                            "stage": "compile",
                            "error": error_msg,
                        })
                
                except Exception as e:
                    logging.error(f"Unexpected error during compile step for {kernel_name} ({target_name}): {e}", exc_info=True)
                    report["results"][level][target_name]["failed"].append({
                        "kernel_file": relative_kernel_path,
                        "level": level,
                        "target": target_name,
                        "stage": "compile (unexpected)",
                        "error": str(e),
                    })
                

            logging.info(f"Finished processing targets for {kernel_name}.")
            
    end_time_sec = time.time()
    
    logging.info("--- Compilation Process Finished ---")
    
    total_failures = generate_reports(
        report,
        total_kernels_by_level,
        list(active_targets.keys()),
        json_report_path,
        md_report_path,
        start_time_sec,
        end_time_sec
    )
    try:
        # --- Final Summary to Console ---
        with open(json_report_path, "r") as f:
            overall_stats = json.load(f)["summary"]
        print("\n" + "="*30 + " IREE COMPILATION SUMMARY " + "="*30)
        print(f"Total Time: {overall_stats['total_time_seconds']:.2f} seconds")
        print(f"Total Compilation Attempts: {overall_stats['total_compilation_attempts']}")
        print(f"Successful Compilations:    {overall_stats['successful_compilations']}")
        print(f"Failed Compilations:        {overall_stats['failed_compilations']}")
        print(f"\nDetailed log:    {log_file_path}")
        print(f"JSON report:     {json_report_path}")
        print(f"Markdown report: {md_report_path}")
        print("="*82)
    except FileNotFoundError:
        logging.error(f"Failed to read JSON report for final summary: {json_report_path}")
    except Exception as e:
        logging.error(f"An error occurred printing the final summary: {e}")

    if total_failures > 0:
        logging.warning("Compilation finished with one or more failures.")
        sys.exit(1)
    else:
        logging.info("Compilation finished successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
