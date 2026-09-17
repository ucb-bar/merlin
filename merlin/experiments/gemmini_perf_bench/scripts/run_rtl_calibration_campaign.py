#!/usr/bin/env python3
"""Execute one exact Gemmini calibration manifest through compiler plus RTL paths."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common.paths import out_dir
from merlin.runtime.backends.base import get_backend
from merlin.targetgen.rtl import mlc_bridge


def _read(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: cannot read exact JSON input {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: input must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _under_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(out_dir().resolve())
    except ValueError:
        return False
    return True


def _module():
    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.gemmini_calibration_execution")


def _no_go(manifest_sha256: str, issue: str) -> dict[str, Any]:
    return {
        "schema": "gemmini_rtl_calibration_execution_v1", "status": "NO_GO",
        "campaign_manifest_sha256": manifest_sha256, "results": [], "empty_runs": [],
        "composition_probe": None, "issues": [issue],
        "partial_execution_is_admissible": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute all content-linked Gemmini RTL calibration requests fail-closed.")
    parser.add_argument("--rtl-facts", required=True)
    parser.add_argument("--harness-capabilities", required=True)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--counter-byte-bindings",
                        help="optional exact proved counter-byte binding artifact")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout <= 0:
        _parser().error("timeout must be positive")
    rtl_path = Path(args.rtl_facts).resolve()
    capabilities_path = Path(args.harness_capabilities).resolve()
    manifest_path = Path(args.campaign_manifest).resolve()
    binding_path = (Path(args.counter_byte_bindings).resolve()
                    if args.counter_byte_bindings else None)
    output = Path(args.output_json).resolve()
    work = Path(args.workdir).resolve()
    outputs = (output, work)
    if not all(_under_output_root(path) for path in outputs):
        _parser().error(f"generated products must be below {out_dir().resolve()}")
    input_paths = {rtl_path, capabilities_path, manifest_path}
    if binding_path is not None:
        input_paths.add(binding_path)
    if output in input_paths or work in input_paths:
        _parser().error("generated output/work paths must be distinct from every exact input path")

    try:
        rtl, rtl_sha256 = _read(rtl_path, "rtl_facts")
        _capabilities, capabilities_sha256 = _read(
            capabilities_path, "harness_capabilities")
        manifest, manifest_sha256 = _read(manifest_path, "campaign_manifest")
    except ValueError as exc:
        artifact = _no_go("UNKNOWN", str(exc))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    inputs = manifest.get("inputs")
    manifest_rtl = inputs.get("rtl_facts") if isinstance(inputs, Mapping) else None
    manifest_capabilities = (inputs.get("harness_capabilities")
                             if isinstance(inputs, Mapping) else None)
    rtl_inputs = rtl.get("inputs")
    circt_sha256 = rtl_inputs.get("core_hw_sha256") if isinstance(rtl_inputs, Mapping) else None
    circt_path = mlc_bridge.core_hw_mlir("gemmini")
    if (not isinstance(manifest_rtl, Mapping) or manifest_rtl.get("sha256") != rtl_sha256
            or not isinstance(manifest_capabilities, Mapping)
            or manifest_capabilities.get("sha256") != capabilities_sha256
            or not isinstance(circt_sha256, str) or len(circt_sha256) != 64
            or circt_path is None or not Path(circt_path).is_file()
            or hashlib.sha256(Path(circt_path).read_bytes()).hexdigest() != circt_sha256):
        artifact = _no_go(
            manifest_sha256,
            "manifest, RTL facts, capabilities, and active CIRCT are not exactly linked")
    else:
        binding = None
        if args.counter_byte_bindings:
            try:
                assert binding_path is not None
                binding, _binding_sha256 = _read(binding_path, "counter_byte_bindings")
            except ValueError as exc:
                artifact = _no_go(manifest_sha256, str(exc))
            else:
                artifact = _module().execute(
                    manifest, rtl, manifest_sha256=manifest_sha256,
                    rtl_facts_sha256=rtl_sha256, capabilities_sha256=capabilities_sha256,
                    circt_hw_sha256=circt_sha256, workdir=work, timeout=args.timeout,
                    counter_binding=binding)
        else:
            artifact = _module().execute(
                manifest, rtl, manifest_sha256=manifest_sha256,
                rtl_facts_sha256=rtl_sha256, capabilities_sha256=capabilities_sha256,
                circt_hw_sha256=circt_sha256, workdir=work, timeout=args.timeout,
                counter_binding=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if artifact.get("status") == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
