"""Shared authentication and seam helpers for unchanged-FX composition."""
from __future__ import annotations

from hashlib import sha256
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def execution_backend_arguments(backend, native_qualification=None, memory_qualification=None, native_cache=None, *, max_instructions=None):
    if backend not in {"functional", "qualified_native"}:
        raise ValueError("unknown validation backend")
    if backend == "qualified_native" and (native_qualification is None or memory_qualification is None):
        raise ValueError("qualified_native requires explicit native and memory qualification receipts")
    if backend == "functional" and any((native_qualification, memory_qualification, native_cache)):
        raise ValueError("native qualification and cache options require the qualified_native backend")
    arguments = ["--backend", backend]
    for flag, path in (("--native-qualification", native_qualification),
                       ("--memory-qualification", memory_qualification), ("--native-cache", native_cache)):
        if path is not None:
            arguments.extend((flag, str(path.resolve())))
    if max_instructions is not None:
        if type(max_instructions) is not int or max_instructions <= 0:
            raise ValueError("functional instruction budget must be a positive integer")
        arguments.extend(("--max-instructions", str(max_instructions)))
    return arguments


def _quality_support(name):
    base = Path(__file__).resolve().parent / "quality"
    if not (base / (name + ".py")).is_file():
        merlin = find_spec("merlin")
        if merlin is None or not merlin.submodule_search_locations:
            raise ValueError("initial input authentication requires the Merlin example source package")
        base = Path(next(iter(merlin.submodule_search_locations))).parents[1] / "examples/atlas/whole-model/unchanged_fx/quality"
    path = base / (name + ".py")
    key = "_atlas_unchanged_fx_quality_" + name
    if key not in sys.modules:
        spec = spec_from_file_location(key, path)
        module = module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def authenticate_initial_inputs(args, manifest, first_capsule):
    options = (args.initial_preparation, args.initial_tokens, args.source_bundle, args.selection_manifest)
    if args.initial_inputs is None:
        if any(value is not None for value in options):
            raise ValueError("initial embedding options require --initial-inputs")
        return None, None
    if args.initial_preparation is None or args.source_bundle is None:
        raise ValueError("initial embedding override requires its preparation and unchanged source bundle")
    _quality_support("atlas_schema").register()
    return _quality_support("initial_embedding").authenticate(
        args.initial_inputs, args.initial_preparation,
        args.initial_tokens or args.initial_preparation.parent / "tokens.npy",
        args.source_bundle, manifest, first_capsule, args.selection_manifest)


def activation_overrides(region, links, previous, actual_by_region):
    """Resolve every declared generated dependency, with no fixture fallback."""
    overrides, seams = {}, []
    for link in links:
        producer = link["region"]
        if producer not in previous:
            raise ValueError(f"generated predecessor has not executed: {producer}")
        actual = actual_by_region[producer]
        value = np.ascontiguousarray(actual[f"output_{link['output_index']}"])
        digest = sha256(value.tobytes()).hexdigest()
        expected = previous[producer]["outputs"][link["output_index"]]["sha256"]
        if digest != expected:
            raise ValueError(f"generated predecessor output changed: {producer}")
        overrides[f"input_{link['input_index']}"] = value
        seams.append({**link, "sha256": digest, "bytes": value.nbytes})
    if region != "layer0" and not seams:
        raise ValueError(f"region has no generated predecessor: {region}")
    return overrides, seams


def require_stateless_manifest(manifest):
    """Cached captures require a runner authenticating generated parent K/V."""
    whole = manifest.get("whole_model", {})
    if ("cache_phase" in manifest or "cache_phase" in whole
            or whole.get("schema") == "atlas.sequential_cache_phase_fx.v1"):
        raise ValueError("cached phase requires authenticated generated parent-cache state")


def verify_fixture_boundaries(sources, links):
    """Only boundaries already present before layer0 may remain capture inputs."""
    fixture_allowlist = set(sources["layer0"]["source_boundaries"])
    for kind, source in sources.items():
        linked = {link["input_index"] for link in links[kind]}
        for index, name in enumerate(source["source_boundaries"]):
            if index not in linked and name not in fixture_allowlist:
                raise ValueError(f"unresolved generated boundary cannot use framework fixture: {kind}:{name}")
