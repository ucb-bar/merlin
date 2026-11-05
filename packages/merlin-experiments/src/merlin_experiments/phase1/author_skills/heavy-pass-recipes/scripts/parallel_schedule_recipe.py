"""Build one counted schedule per geometry, branch real passes, evaluate unique bytes.

The default adapter is neutral: actual LLVM emission and optional object compilation,
not a target kernel or cycle estimate. An author adapter can supply the SAME terminal
emitter and its ordinary numerics/cycle owner through the interface in the reference.
The executor belongs to the caller; callbacks must use its ordinary resource leases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from itertools import zip_longest
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(value if type(value) is bytes else
                          json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_pin(path):
    with path.open("rb") as stream:
        return dict(path=str(path), sha256=hashlib.file_digest(stream, "sha256").hexdigest())


def neutral_binding(shape, caps):
    """Replace this adapter, not the shared pass/evaluation loop, for a target."""
    from merlin.compile.scheduling.dense_panel_pipeline import DenseReadyReadoutAdmission
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_ready_readout
    from merlin.compile.scheduling.dense_segmented_pipeline import batch_dense_segmented_readouts
    from merlin.transforms.pipeline_state import PipelineContext

    counted = load(Path(__file__).with_name("counted_emission_recipe.py"), "parallel_counted_emission")
    S = counted._support()
    p, storage, resources = S.dense_view(shape, caps)
    if p.m * p.k * p.n > 1_000_000:
        raise ValueError("neutral source-check budget exceeded; bind your ordinary numerical owner")
    adapter = S.adapter()
    block = S.pick(S.blocks(p, storage, replace(resources, max_k_blocks=p.tiles[1]-1)),
                   lambda b: b.k < p.tiles[1] and b.m*b.n*p.tile_m <= storage.accumulator.capacity//2
                   and p.tiles[2]//b.n > 1)
    original, _ = S.segmented_pipeline(p, storage, resources, adapter, block, name="source")
    admission = replace(original.admission, **{name: replace(getattr(original.admission, name),
        max_issued_readouts=block.m*block.n, pending_readouts=2,
        readout_configuration="ordered_snapshot_before_store_issue") for name in ("overlap", "reduction")})
    ready = DenseReadyReadoutAdmission(adapter.configuration_identity, adapter.numerical_policy_identity,
        "test-only-ready-completion", "test-only-store-snapshot", block.m*block.n, 2)

    def transform(result, mode):
        if mode in ("original", "identical"):
            return result
        result = batch_dense_segmented_readouts(result, p, block, resources, storage, adapter, admission)
        return result if mode == "batched" else schedule_dense_ready_readout(
            result, p, block, resources, storage, adapter, ready)

    return dict(source=original, transform=transform, emit=counted.emit,
                source_check=lambda result: S.execute_dense(result.kernel, adapter, p),
                shapes=((p.m, p.k), (p.k, p.n)),
                context=PipelineContext(hardware=digest(caps), resources=digest(asdict(resources)),
                    numerical_policy=adapter.numerical_policy_identity,
                    execution="neutral-host-projection-only", library="selected-shared-library"))


def candidates(binding):
    """Register actual shared actions; no emitter rewrite per alternative."""
    from merlin.transforms.pass_exploration import EmittedProgram, PassFamily, explore_candidates, generate_candidates
    from merlin.transforms.pipeline_policy import PolicyFacts, RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec

    def bind(knobs):
        def action(session):
            session.module["result"] = binding["transform"](session.module["result"], knobs["mode"])
            return {"schedule": session.module["result"]}
        return RegisteredPass(PassSpec("readout", 0, produces=("schedule",), ir_level="counted_dense"), action)

    space = generate_candidates((PassFamily("readout", bind,
        (("mode", ("original", "batched", "ready", "identical")),)),), output="schedule")

    def emit(result):
        text, evidence = binding["emit"](result.artifact)
        program = EmittedProgram(text.encode(), "llvm-mlir", result.artifact.kernel.digest())
        binding["emitted"][digest(program.data)] = (result.artifact, evidence)
        return program

    binding["emitted"] = {}
    return explore_candidates({"result": binding["source"]}, space,
        lambda s: PolicyFacts(s.stamp(), shapes=binding["shapes"]), output="schedule", emission=emit,
        context=binding["context"], ir_level="counted_dense", clone=dict,
        fingerprint=lambda state: state["result"].kernel.digest(),
        verify_ir=lambda state: state["result"].revalidate())


def compile_neutral(program, *, directory, translate, compiler, tool_pins, timeout_s=30):
    """Actual selected-tool object build; unresolved neutral effect symbols remain."""
    directory.mkdir(parents=True, exist_ok=False)
    source, lowered, obj = (directory/name for name in ("program.llvm.mlir", "program.ll", "program.o"))
    source.write_bytes(program.data)
    commands = ([str(translate), "--mlir-to-llvmir", str(source), "-o", str(lowered)],
                [str(compiler), "-c", str(lowered), "-o", str(obj)])
    for index, command in enumerate(commands):
        done = subprocess.run(command, capture_output=True, timeout=timeout_s, check=False)
        (directory/f"command-{index}.stdout").write_bytes(done.stdout)
        (directory/f"command-{index}.stderr").write_bytes(done.stderr)
        if done.returncode:
            raise RuntimeError(f"actual object build failed ({done.returncode}): {directory}")
    return dict(commands=commands, object_sha256=digest(obj.read_bytes()), object_bytes=obj.stat().st_size,
                tools=tool_pins,
                qualification="object compilation only; neutral external effects are not a device implementation")


def run(shapes, caps, *, executor, max_in_flight, directory, owner=None, translate=None, compiler=None):
    """Caller-owned executor; shape-interleaved admission and compact publication.

    owner.binding(shape,caps) supplies source/transform/emit/context/shapes plus
    owner.identity(shape,binding) and owner.evaluate(program,receipt,identity).
    evaluate owns target numerics, selected cycles, custody and normal leases.
    Optional owner.hold_inputs(request) pins external blobs through publication.
    """
    from merlin.transforms.evaluated_exploration import (
        CompletedObservation,
        EvaluationRequest,
        ExplorationIdentity,
        evaluate_emitted,
        exploration_context_identity,
    )
    from merlin.transforms.pass_exploration import ObservedMetrics, rank_performance_variants

    if owner is not None and any(not callable(getattr(owner, name, None))
                                 for name in ("binding", "identity", "evaluate")):
        raise ValueError("construct adapter hooks: binding(shape,caps), identity(shape,binding), "
                         "evaluate(program,receipt,identity)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if (translate is None) != (compiler is None):
        raise ValueError("object compilation requires both selected tools")
    tool_pins = [] if translate is None else [file_pin(Path(path)) for path in (translate, compiler)]
    bindings = [(owner.binding(shape, caps) if owner else neutral_binding(shape, caps)) for shape in shapes]
    domains, emitted_rows, records, observations = {}, [], [], {}
    for shape, binding in zip(shapes, bindings, strict=True):
        context = exploration_context_identity(binding["context"])
        if owner:
            identity = owner.identity(shape, binding)
        else:
            # Explicitly a neutral compilation domain, never selected hardware feedback.
            identity = ExplorationIdentity(*(digest(value) for value in (
                dict(shape=shape, semantics="integer source product"), "neutral effect ABI", "host-check inputs",
                "host integer product", caps, Path(__file__).read_bytes(), "compile-only; no cycle provider",
                "neutral function", "no device ROI")), context, "llvm-mlir", "modeled_roi")
        if identity.digest() in domains:
            raise ValueError("independent source geometries require distinct exact input/source domains")
        domains[identity.digest()] = (identity, binding, shape)

    def requests():
        streams = [candidates(binding) for binding in bindings]
        # Round robin over shapes, rather than exhausting the first shape's family.
        for row in zip_longest(*streams):
            for index, candidate in enumerate(row):
                if candidate is None:
                    continue
                summary = candidate.summary()
                summary["shape"] = shapes[index]
                emitted_rows.append(summary)
                if candidate.emission is not None:
                    identity = next(value[0] for value in domains.values() if value[1] is bindings[index])
                    yield EvaluationRequest(candidate.emission, candidate.evaluation.result.receipt, identity)

    def evaluate(program, receipt, identity):
        if owner:
            return owner.evaluate(program, receipt, identity)
        _, binding, shape = domains[identity.digest()]
        result, emitted = binding["emitted"][digest(program.data)]
        feedback = dict(shape=shape, emitted=emitted, source_check=binding["source_check"](result),
                        timing="UNKNOWN: neutral encoder has no ordinary target cycle binding")
        if translate is not None and compiler is not None:
            feedback["compile"] = compile_neutral(program,
                directory=directory/(identity.digest()+"-"+digest(program.data)),
                translate=Path(translate), compiler=Path(compiler), tool_pins=tool_pins)
        feedback_path = directory/(identity.digest()+"-"+digest(program.data)+".json")
        feedback_path.write_text(json.dumps(feedback, indent=2)+"\n")
        return CompletedObservation(identity, digest(program.data), digest(feedback_path.read_bytes()),
                                    "unknown", "unknown", False, None)

    def publish(request, completion):
        # ONLY compact append. Verification/build is in evaluate; maintenance is
        # caller-owned off-path work AFTER a yielded completion releases its pin.
        records.append(completion.record())
        with (directory/"completions.jsonl").open("a") as stream:
            stream.write(json.dumps(records[-1], separators=(",", ":"))+"\n")
        if owner is not None and hasattr(owner, "on_completion"):
            owner.on_completion(request, completion)  # compact append/enqueue only

    for completion in evaluate_emitted(requests(), evaluate, executor=executor, max_in_flight=max_in_flight,
            on_completion=publish, hold_inputs=getattr(owner, "hold_inputs", None)):
        if completion.observation is not None:
            observations.setdefault(completion.identity.digest(), []).append(completion.observation)
    rankings = []
    for domain, (identity, _, shape) in domains.items():
        eligible = tuple((o.program_sha256, ObservedMetrics(identity.basis, domain, (("cycles", o.cycles),)))
                         for o in observations.get(domain, ()) if o.eligible)
        rankings.append(dict(shape=shape, evaluation_identity=domain,
            ranking=rank_performance_variants(eligible) if eligible else None,
            selection="retain incumbent; no eligible cycles" if not eligible else "caller compares exact incumbent"))
    report = dict(schema="merlin.parallel_schedule_recipe.v1", candidates=emitted_rows, completions=records,
        rankings=rankings, schedules_constructed=len(bindings),
        qualifications="neutral projection/object compilation is not target numerics or cycle qualification",
        missing_target_binding=owner is None, performance_qualification=False)
    (directory/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library")
    ap.add_argument("--facts")
    ap.add_argument("--shape", action="append", help="M,K,N; repeat for independent source geometries")
    ap.add_argument("--output", required=True, help="new output directory")
    ap.add_argument("--adapter", help="authored binding/identity/evaluate module; ordinary resource leases required")
    ap.add_argument("--max-in-flight", type=int, default=2)
    ap.add_argument("--compile", action="store_true", help="selected-tool neutral object build, not target timing")
    args = ap.parse_args(argv)
    if args.compile and args.adapter:
        ap.error("--compile is the neutral example; the authored adapter owns its actual compilation")
    support = load(Path(__file__).with_name("recipe_support.py"), "parallel_recipe_support")
    support.use_library(args.library)
    caps = support.capacities(args.facts)
    edge_k, edge = caps["array"]
    width = (caps["accumulator"]["rows"]//(2*edge)+1)*edge
    shapes = [tuple(map(int, s.split(","))) for s in args.shape] if args.shape else [
        (2*edge-1, 3*edge_k, width-1), (2*edge, 3*edge_k, width)]
    if (not 1 <= args.max_in_flight <= 16 or not 1 <= len(shapes) <= 16
            or len(set(shapes)) != len(shapes)
            or any(len(s) != 3 or any(d <= 0 for d in s) for s in shapes)):
        ap.error("use 1..16 distinct positive M,K,N shapes and 1..16 in-flight requests")
    tools = {}
    if args.compile:
        from merlin.llvmlower.toolchain import clang, mlir_translate
        tools = dict(translate=mlir_translate().resolve(strict=True), compiler=clang().resolve(strict=True))
    owner = None if args.adapter is None else load(Path(args.adapter), "authored_schedule_owner")
    # This optional CLI is the caller. It owns this bounded executor, while each
    # external cycle owner still acquires its ordinary shared resource leases.
    with ThreadPoolExecutor(max_workers=args.max_in_flight) as executor:
        result = run(shapes, caps, executor=executor, max_in_flight=args.max_in_flight,
                     directory=args.output, owner=owner, **tools)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
