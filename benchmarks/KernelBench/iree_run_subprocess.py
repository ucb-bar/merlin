#!/usr/bin/env python3

import sys
import os
import logging
import json
import importlib.util
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Any, List


import torch
import numpy as np

# Base directory setup
BASE_DIR = Path.cwd()
KERNELBENCH_BASE_DIR = BASE_DIR / "KernelBench" / "KernelBench"
IREE_ARTIFACT_BASE_DIR = BASE_DIR / "iree"
KERNEL_LEVELS = ["level1", "level2"]
TARGET_NAMES = ["A6000"] 
# TARGET_NAMES = ["A100", "A6000", "H100"] 


# IREE Runtime Configuration
IREE_DEVICE = "cuda" 
ENTRY_FUNCTION_NAME = "main"
IREE_RUN_MODULE_CMD = "iree-run-module" 

# Input dtype configuration for tensor inputs
# All torch tensors will be converted to this dtype before saving for IREE
IREE_INPUT_TENSOR_TORCH_DTYPE = torch.float16

# NCU Configuration
# NCU_PATH = "/path/to/nsight-compute/2024.3.0/ncu"
NCU_PATH = "ncu"
NCU_OUTPUT_DIR = "ncu"

# Log file configuration
LOG_FILE_NAME = "iree_run_subprocess.log"

# Report file names
JSON_REPORT_NAME = "iree_run_subprocess_report.json"
MD_REPORT_NAME = "iree_run_subprocess_report.md"

# --- End Configuration ---


def setup_logging() -> None:
    """Configures logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_NAME, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("iree").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def load_kernel_module(kernel_file: Path) -> Any:
    """
    Dynamically loads a Python module from a given file path.
    (Identical to the compilation script version)
    """
    kernel_name = kernel_file.stem
    spec = importlib.util.spec_from_file_location(kernel_name, kernel_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {kernel_file}")

    kernel_module = importlib.util.module_from_spec(spec)
    
    kernel_dir = str(kernel_file.parent)
    sys.path.insert(0, kernel_dir)
    try:
        spec.loader.exec_module(kernel_module)
    finally:
        sys.path.pop(0)
        
    return kernel_module


def prepare_inputs_for_iree(raw_inputs_list: List[Any]) -> List[np.ndarray]:
    """
    Converts inputs from get_inputs() to NumPy arrays suitable for IREE.
    Handles Tensors and basic Python numeric types.
    """
    iree_inputs = []
    for item in raw_inputs_list:
        if isinstance(item, torch.Tensor):
            iree_inputs.append(
                item.detach().to(IREE_INPUT_TENSOR_TORCH_DTYPE).cpu().numpy()
            )
        elif isinstance(item, (int, float)):
            dtype = np.int64 if isinstance(item, int) else np.float16
            iree_inputs.append(np.array(item, dtype=dtype))
        else:
            raise TypeError(f"Unsupported input type for IREE conversion: {type(item)}")
    return iree_inputs


def generate_reports(report: Dict[str, list], start_time: float, end_time: float) -> None:
    """Generates JSON and Markdown summary reports."""
    total_success = len(report["success"])
    total_failed = len(report["failed"])
    total_attempts = total_success + total_failed
    duration = end_time - start_time

    # --- JSON Report ---
    logging.info(f"Writing JSON report to {JSON_REPORT_NAME}...")
    report_summary = {
        "summary": {
            "total_attempts": total_attempts,
            "successful_runs": total_success,
            "failed_runs": total_failed,
            "total_time_seconds": duration,
        },
        "details": report,
    }
    try:
        with open(JSON_REPORT_NAME, "w") as f:
            json.dump(report_summary, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write JSON report: {e}")

    # --- Markdown Report ---
    logging.info(f"Writing Markdown report to {MD_REPORT_NAME}...")
    try:
        with open(MD_REPORT_NAME, "w") as f:
            f.write("# IREE Execution Report (Subprocess Mode)\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Total Execution Attempts**: {total_attempts}\n")
            f.write(f"- **Successful Runs**: {total_success}\n")
            f.write(f"- **Failed Runs**: {total_failed}\n")
            f.write(f"- **Total Runtime**: {duration:.2f} seconds\n\n")

            if total_failed > 0:
                f.write("## Failures\n\n")
                f.write("| Artifact File | Error |\n") 
                f.write("|---------------|-------|\n")
                for item in report["failed"]:
                    error_msg = str(item['error']).split('\n')[0] 
                    f.write(f"| `{item['artifact_file']}` | `{error_msg}` |\n")
                f.write("\n")

            if total_success > 0:
                f.write("## Successes\n\n")
                f.write("| Artifact File |\n")
                f.write("|---------------|\n")
                for item in report["success"]:
                    f.write(f"| `{item['artifact_file']}` |\n")
    except Exception as e:
        logging.error(f"Failed to write Markdown report: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute IREE compiled kernels with optional NCU profiling"
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Enable NCU profiling for kernel executions",
        default=False,
    )
    parser.add_argument(
        "--ncu-skip-existing",
        action="store_true",
        help="When using --ncu, skip modules whose NCU report already exists in the output directory",
        default=False,
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    import time
    start_time_sec = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    
    setup_logging()
    logging.info("Starting IREE kernel execution process (subprocess mode)...")
    
    if args.ncu:
        logging.info(f"NCU profiling enabled. Reports will be saved to '{NCU_OUTPUT_DIR}/' directory")
        # Create NCU output directory if it doesn't exist
        ncu_dir = BASE_DIR / NCU_OUTPUT_DIR
        ncu_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify NCU is available
        if not os.path.exists(NCU_PATH):
            logging.error(f"NCU not found at {NCU_PATH}")
            logging.error("Please verify the NCU installation path.")
            sys.exit(1)
        logging.info(f"NCU found at {NCU_PATH}")

    # Check if iree-run-module is available
    try:
        result = subprocess.run([IREE_RUN_MODULE_CMD, "--help"], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode != 0:
            raise RuntimeError(f"'{IREE_RUN_MODULE_CMD}' not found or not working")
        logging.info(f"'{IREE_RUN_MODULE_CMD}' command found.")
    except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
        logging.error(f"Failed to verify '{IREE_RUN_MODULE_CMD}' command: {e}")
        logging.error("Please ensure IREE tools are installed and in PATH.")
        sys.exit(1)

    report = {"success": [], "failed": []}

    # --- Iterate through compiled artifacts ---
    for level in KERNEL_LEVELS:
        kernel_source_dir = KERNELBENCH_BASE_DIR / level
        if not kernel_source_dir.is_dir():
            logging.warning(f"Kernel source directory not found, skipping: {kernel_source_dir}")
            continue
            
        logging.info(f"--- Processing Level: {level} ---")
        
        for target_name in TARGET_NAMES:
            artifact_dir = IREE_ARTIFACT_BASE_DIR / level / target_name
            if not artifact_dir.is_dir():
                logging.warning(f"Artifact directory not found, skipping: {artifact_dir}")
                continue

            logging.info(f"--- Processing Target: {target_name} in {level} ---")
            
            vmfb_files = sorted(artifact_dir.glob("*.vmfb"))
            if not vmfb_files:
                logging.warning(f"No artifact files (*.vmfb) found in {artifact_dir}")
                continue

            for vmfb_file in vmfb_files:
                kernel_name = vmfb_file.stem
                relative_artifact_path = str(vmfb_file.relative_to(BASE_DIR))
                logging.info(f"Processing artifact: {relative_artifact_path}")

                # If NCU is enabled and skipping existing reports is requested, skip when report exists
                if args.ncu and args.ncu_skip_existing:
                    ncu_output_file_check = BASE_DIR / NCU_OUTPUT_DIR / f"{kernel_name}_{target_name}.ncu-rep"
                    if ncu_output_file_check.exists():
                        logging.info(
                            f"Skipping profiling for {relative_artifact_path}; NCU report already exists: {ncu_output_file_check}"
                        )
                        continue

                iree_inputs = []
                # --- 1. Find and Load Source Kernel for Inputs ---
                try:
                    source_kernel_file = kernel_source_dir / f"{kernel_name}.py"
                    if not source_kernel_file.exists():
                        raise FileNotFoundError(f"Source file not found: {source_kernel_file}")

                    kernel_module = load_kernel_module(source_kernel_file)

                    if not hasattr(kernel_module, 'get_inputs'):
                        raise AttributeError("Source kernel file does not define 'get_inputs'")
                    
                    raw_inputs_list = kernel_module.get_inputs()
                    if raw_inputs_list is None:
                        raise ValueError("'get_inputs()' returned None")

                    iree_inputs = prepare_inputs_for_iree(raw_inputs_list)
                    logging.debug(f"Prepared {len(iree_inputs)} NumPy inputs.")

                except Exception as e:
                    logging.error(f"Failed to load or prepare inputs for {vmfb_file}: {e}")
                    report["failed"].append({
                        "artifact_file": relative_artifact_path,
                        "error": f"Input loading/preparation error: {e}",
                    })
                    continue

                # --- 2. Save inputs and run IREE Module via subprocess ---
                temp_files = []
                try:
                    # Save each input to a temporary .npy file
                    for idx, input_array in enumerate(iree_inputs):
                        temp_file = tempfile.NamedTemporaryFile(
                            mode='wb', 
                            suffix=f'_input_{idx}.npy', 
                            delete=False,
                            dir=BASE_DIR / "iree"
                        )
                        np.save(temp_file, input_array)
                        print(f"Saving tensor of type {input_array.dtype} to {temp_file.name}")
                        temp_file.close()
                        temp_files.append(temp_file.name)
                        logging.debug(f"Saved input {idx} to {temp_file.name} with shape {input_array.shape}")

                    # Build the iree-run-module command
                    cmd = [
                        IREE_RUN_MODULE_CMD,
                        f"--module={vmfb_file}",
                        f"--device={IREE_DEVICE}",
                        f"--function={ENTRY_FUNCTION_NAME}"
                    ]
                    
                    # Add input arguments
                    for temp_file_path in temp_files:
                        cmd.append(f"--input=@{temp_file_path}")
                    
                    # Wrap with NCU if profiling is enabled
                    if args.ncu:
                        ncu_output_file = BASE_DIR / NCU_OUTPUT_DIR / f"{kernel_name}_{target_name}.ncu-rep"
                        ncu_cmd = [
                            NCU_PATH,
                            "--set", "full",
                            "--export", str(ncu_output_file),
                            "--force-overwrite"
                        ]
                        cmd = ncu_cmd + cmd
                        logging.debug(f"NCU profiling enabled for {kernel_name}, output: {ncu_output_file}")
                    
                    logging.debug(f"Running command: {' '.join(cmd)}")
                    
                    # Execute the command
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300  
                    )
                    
                    if result.returncode == 0:
                        logging.info(f"Successfully executed {relative_artifact_path}")
                        logging.debug(f"Output: {result.stdout}")
                        report["success"].append({
                            "artifact_file": relative_artifact_path,
                        })
                    else:
                        error_msg = result.stderr if result.stderr else result.stdout
                        logging.error(f"Failed to execute {relative_artifact_path}: {error_msg}")
                        report["failed"].append({
                            "artifact_file": relative_artifact_path,
                            "error": f"Execution failed (return code {result.returncode}): {error_msg[:200]}",
                        })

                except subprocess.TimeoutExpired:
                    logging.error(f"Execution timeout for {relative_artifact_path}")
                    report["failed"].append({
                        "artifact_file": relative_artifact_path,
                        "error": "Execution timeout (>300s)",
                    })
                except Exception as e:
                    logging.error(f"Failed to execute IREE module {vmfb_file}: {e}")
                    report["failed"].append({
                        "artifact_file": relative_artifact_path,
                        "error": f"Subprocess error: {e}",
                    })
                finally:
                    # Clean up temporary input files
                    for temp_file_path in temp_files:
                        try:
                            if os.path.exists(temp_file_path):
                                os.unlink(temp_file_path)
                                logging.debug(f"Cleaned up {temp_file_path}")
                        except Exception as e:
                            logging.warning(f"Failed to clean up {temp_file_path}: {e}") 


    end_time_sec = time.time()
    total_duration = end_time_sec - start_time_sec
    
    logging.info("--- Execution Process Finished ---")
    generate_reports(report, 0, total_duration) 

    # --- Final Summary to Console ---
    total_success = len(report["success"])
    total_failed = len(report["failed"])
    print("\n" + "="*30 + " IREE EXECUTION SUMMARY " + "="*30)
    print(f"Total Time: {total_duration:.2f} seconds")
    print(f"Successful Runs: {total_success}")
    print(f"Failed Runs:     {total_failed}")
    print(f"\nDetailed log:    {LOG_FILE_NAME}")
    print(f"JSON report:     {JSON_REPORT_NAME}")
    print(f"Markdown report: {MD_REPORT_NAME}")
    print("="*80) 

if __name__ == "__main__":
    main()