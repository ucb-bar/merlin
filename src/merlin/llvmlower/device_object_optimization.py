"""Expose the exact optimized LLVM used by device object code generation.

The ordinary selected Clang optimization level remains in both commands. The
second command skips repeated LLVM IR optimization, retaining normal backend
optimization. Object/linked instruction inventory checks remain independent.
"""

import hashlib
import json
from pathlib import Path


def _producer_output(pin):
    raw = Path(pin["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("optimized LLVM producer invocation changed")
    outputs = json.loads(raw).get("outputs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0].get("sha256"), str)
        or len(outputs[0]["sha256"]) != 64
    ):
        raise ValueError("optimized LLVM producer selected another output")
    return outputs[0]


def artifacts(custody):
    """Publish original output paths even when multiple kernels share one build."""
    return {
        role: Path(_producer_output(custody[step])["path"])
        for role, step in (("optimized_llvm_ir", "optimization"), ("optimized_llvm_mlir", "optimized_import"))
    }


def commands(compile_command, translator):
    """Split an existing observed LLVM-to-object command without changing flags."""
    argv = list(compile_command)
    if len(argv) < 5 or argv[-4] != "-c" or argv[-2] != "-o":
        raise ValueError("device object command must compile one explicit LLVM input")
    source, output = Path(argv[-3]), Path(argv[-1])
    optimized = source.with_suffix(".optimized.ll")
    imported = source.with_suffix(".optimized.llvm.mlir")
    prefix = argv[:-4]
    return (
        [*prefix, "-S", "-emit-llvm", str(source), "-o", str(optimized)],
        [translator, "--import-llvm", "--mlir-print-op-generic", str(optimized), "-o", str(imported)],
        [*prefix, "-Xclang", "-disable-llvm-passes", "-c", str(optimized), "-o", str(output)],
    )


def observe(compile_command, translator, *, observer, timeout, directory, report):
    """Retain actual source→optimized LLVM→typed import invocation custody."""
    optimize, imported, backend = commands(compile_command, translator)
    result, optimization, elapsed = observer(
        optimize,
        timeout=timeout,
        directory=directory,
        stage="device_llvm_optimization",
        inputs=(Path(optimize[-3]),),
        output=Path(optimize[-1]),
        metrics=report,
        expected_input_sha256=report["analysis"]["sha256"],
        metrics_stage="optimization_s",
    )
    report["stages"]["optimization_s"] = elapsed
    if result.returncode:
        return None, f"clang LLVM optimization: {(result.stderr or '').strip()[:200]}"
    output = _producer_output(optimization)
    if Path(output["path"]) != Path(optimize[-1]).resolve():
        raise ValueError("optimized LLVM producer selected another output path")
    optimized_sha256 = output["sha256"]
    result, import_receipt, elapsed = observer(
        imported,
        timeout=timeout,
        directory=directory,
        stage="device_optimized_llvm_import",
        inputs=(Path(imported[-3]),),
        output=Path(imported[-1]),
        expected_input_sha256=optimized_sha256,
    )
    report["stages"]["optimized_import_s"] = elapsed
    if result.returncode:
        return None, f"mlir-translate optimized LLVM import: {(result.stderr or '').strip()[:200]}"
    return {
        "argv": backend,
        "optimization": optimization,
        "optimized_import": import_receipt,
        "sha256": optimized_sha256,
    }, ""


def _callback_selection(callback):
    from types import FunctionType, MethodType

    function = callback.__func__ if type(callback) is MethodType else callback
    owner = callback.__self__ if type(callback) is MethodType else None
    if type(function) is not FunctionType:
        raise ValueError("device LLVM stages require source-bound original Python callbacks")
    return function, owner, function.__code__


def freeze_transform_chain(chain):
    """Freeze original ordered stages before package callbacks; no effect grant.

    The existing host-named chain is a generic LLVM emission transport. Device
    use retains its original provider-owned semantic verifier and proof closure.
    No instruction effect or qualification is inferred from those data.
    """
    if chain is None:
        return None
    from . import host_transform_chain as H

    if type(chain) is not H.HostLLVMTransformChain:
        raise ValueError("device LLVM requires an original typed transform chain")
    if type(chain.stages) is not tuple or len(chain.stages) > 32:
        raise ValueError("device LLVM transform stage budget exceeded")
    chain.validate()
    stages = tuple(
        (
            stage,
            stage.contract,
            json.dumps(stage.contract.to_dict(), sort_keys=True),
            _callback_selection(stage.transform),
            _callback_selection(stage.verify_emission),
        )
        for stage in chain.stages
    )
    sources = tuple(
        H.HostLLVMArtifact.capture(path)
        for path in (
            Path(__file__),
            Path(H.__file__),
            Path(__file__).with_name("device_build.py"),
            Path(__file__).with_name("device_build_jobs.py"),
        )
    )
    return chain, stages, sources


def recheck_transform_chain(chain, frozen, emission=None):
    """Static reopens only; never invoke the transform or semantic verifier."""
    if frozen is None:
        if chain is not None or emission is not None:
            raise ValueError("device LLVM transform was attached after original selection")
        return
    original, stages, sources = frozen
    if chain is not original or len(chain.stages) != len(stages):
        raise ValueError("original device LLVM transform selection changed")
    chain.validate()
    for source in sources:
        source.validate()
    for actual, (stage, contract, record, transform, verifier) in zip(chain.stages, stages, strict=True):
        callbacks = (_callback_selection(actual.transform), _callback_selection(actual.verify_emission))
        if (
            actual is not stage
            or actual.contract is not contract
            or json.dumps(actual.contract.to_dict(), sort_keys=True) != record
            or any(
                any(left is not right for left, right in zip(current, saved, strict=True))
                for current, saved in zip(callbacks, (transform, verifier), strict=True)
            )
        ):
            raise ValueError("original device LLVM transform callback or contract changed")
    if emission is not None:
        from .host_transform_chain import HostLLVMArtifact, recheck_host_transform_chain

        pin = emission["chain"]
        if HostLLVMArtifact.capture(Path(pin["path"])).to_dict() != pin:
            raise ValueError("device LLVM transform emission receipt changed")
        actual = recheck_host_transform_chain(Path(pin["path"]), expected_chain=chain)
        if (
            actual["source"]["path"] != emission["source_path"]
            or actual["source"]["sha256"] != emission["source_sha256"]
            or actual["selected"]["path"] != emission["selected_path"]
            or actual["selected"]["sha256"] != emission["selected_sha256"]
        ):
            raise ValueError("device LLVM transform emission differs from original binding")


def transform_input(source, directory, chain, frozen):
    """Apply the existing chain to the actual translated LLVM consumed next."""
    from .host_transform_chain import apply_host_transform_chain

    recheck_transform_chain(chain, frozen)
    selected, emission = apply_host_transform_chain(source, directory, chain)
    recheck_transform_chain(chain, frozen, emission)
    return selected, emission


def transform_dependencies(frozen, emission):
    """Bind the actual stage/code/proof products in normal tool invocations."""
    recheck_transform_chain(frozen[0], frozen, emission)
    from .host_transform_chain import recheck_host_transform_chain

    report = recheck_host_transform_chain(Path(emission["chain"]["path"]), expected_chain=frozen[0])
    paths = [Path(emission["chain"]["path"]), *(source.path for source in frozen[2])]
    paths.extend(Path(report[role]["path"]) for role in ("source", "selected"))
    for stage in report["stages"]:
        paths.extend(Path(stage[role]["path"]) for role in ("input", "output"))
        paths.extend(Path(row["path"]) for role in ("sources", "semantic_proofs") for row in stage["contract"][role])
        paths.extend(Path(row["path"]) for row in stage["evidence"])
    return tuple(dict.fromkeys(paths))
