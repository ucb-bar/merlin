"""Bounded two-revision qualification of an actual internal physical layout copy.

This does not authorize ABI prepacking, execute a model, or infer performance.
The existing host action owns deadlines, sandbox policy and receipt consumption.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from math import isfinite, prod
from pathlib import Path
import tempfile
from time import monotonic

from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.llvmlower.toolchain import mlir_translate
from . import native_host_witness_runner
from .host_source_witness import extract_pointwise_chain, evaluate_pointwise_source
from .physical_transition_evidence import MARKERS, verify_physical_transitions


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _declarations(artifact):
    return artifact.get("command_buffer", {}).get("params", {}).get("global_program_plan", {}).get("physical_transitions", [])


def _edge(row):
    edge = row["source_edge"]
    return tuple(edge[key] for key in ("producer_op_index", "producer_result_index",
                                     "consumer_op_index", "consumer_operand_index"))


def _orientation(layout):
    """Only unambiguous dense permutations are currently qualified across shapes."""
    shape, strides = layout["shape"], layout["strides_elements"]
    if (len(shape) != len(strides) or not shape or any(type(n) is not int or n <= 1 for n in shape)
            or layout["offset_elements"] != 0):
        raise ValueError("copy witness needs non-unit static axes and zero offset")
    order = tuple(sorted(range(len(shape)), key=lambda i: strides[i]))
    step = 1
    for axis in order:
        if strides[axis] != step:
            raise ValueError("copy witness currently supports dense axis permutations only")
        step *= shape[axis]
    return order


def _mechanism(row):
    if row["kind"] != "static_strided_copy":
        raise ValueError("unsupported physical transition mechanism")
    return (row["kind"], row["source"]["dtype"], row["source"]["placement"],
            row["destination"]["placement"], _orientation(row["source_layout"]),
            _orientation(row["destination_layout"]))


def _bounded_native_module(module, *, expected_symbol, argument_count):
    from .host_cfg_activity import analyze_host_cfg_activity
    from xdsl.dialects.llvm import LLVMVoidType
    functions = [op for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks]
    allowed = {"builtin.module", "llvm.func", "llvm.mlir.constant", "llvm.alloca", "llvm.getelementptr",
        "llvm.load", "llvm.store", "llvm.br", "llvm.cond_br", "llvm.return", "llvm.icmp", "llvm.fcmp",
        "llvm.add", "llvm.sub", "llvm.mul", "llvm.sdiv", "llvm.udiv", "llvm.srem", "llvm.urem",
        "llvm.and", "llvm.or", "llvm.xor", "llvm.shl", "llvm.lshr", "llvm.ashr", "llvm.select",
        "llvm.fadd", "llvm.fsub", "llvm.fmul", "llvm.fdiv", "llvm.fneg", "llvm.sext", "llvm.zext",
        "llvm.trunc", "llvm.sitofp", "llvm.uitofp", "llvm.fptosi", "llvm.fptoui"}
    if (len(functions) != 1 or list(module.body.block.ops) != functions
            or any(op.name not in allowed for op in module.walk())):
        raise ValueError("physical transition source witness is not closed native host code")
    if (functions[0].sym_name.data != expected_symbol
            or len(functions[0].body.blocks.first.args) != argument_count
            or not isinstance(functions[0].function_type.output, LLVMVoidType)
            or any(str(arg.type) != "!llvm.ptr" for arg in functions[0].body.blocks.first.args)):
        raise ValueError("short emitted entry symbol/pointer arguments disagree with the host ABI")
    scalar_types = {"!llvm.ptr", "i1", "i8", "i16", "i32", "i64", "f32"}
    values = [value for op in functions[0].walk() for value in (*op.operands, *op.results)]
    values.extend(arg for block in functions[0].body.blocks for arg in block.args)
    if any(str(value.type) not in scalar_types for value in values):
        raise ValueError("physical transition witness only permits bounded scalar values and pointers")
    activity = analyze_host_cfg_activity(functions[0])
    allocation, dynamic = activity["static_allocation_payload_bytes"], activity["dynamic_operations"]
    loads, stores = activity["load_payload_bytes"], activity["store_payload_bytes"]
    if (activity["status"] != "derived" or type(allocation) is not int or allocation > 65536
            or not isinstance(dynamic, Mapping) or any(type(n) is not int or n < 0 for n in dynamic.values())
            or sum(dynamic.values()) > 100_000 or type(loads) is not int or type(stores) is not int
            or loads < 0 or stores < 0 or loads+stores > 1024*1024):
        raise ValueError("actual emitted short witness exceeds allocation/work bound or has unknown work")
    return {"allocation_bytes": allocation, "dynamic_ir_operations": sum(dynamic.values()),
            "dynamic_load_store_bytes": loads+stores,
            "scope": "host safety limits on emitted scalar work, not target facts or cycle prediction"}


def has_physical_transition(artifact):
    """Markers/declarations select a verifier, never establish validity themselves."""
    cb = artifact.get("command_buffer") or {}
    plan = (cb.get("params") or {}).get("global_program_plan") or {}
    if "physical_transitions" in plan and plan["physical_transitions"] != []:
        return True
    return any(marker in artifact.get("lowered_text", "") for marker in MARKERS)


class ChangedRegionQualifierDispatch:
    def __init__(self, *, physical, legacy, lane_migration=None):
        self.physical, self.legacy, self.lane_migration = physical, legacy, lane_migration
        self.abi_provenance = physical.abi_provenance
        if (lane_migration is not None
                and lane_migration.abi_provenance != self.abi_provenance):
            raise ValueError("changed-region providers disagree on host ABI provenance")

    def __call__(self, *, candidate, experiment, timeout_s, portfolio_member=None):
        selected = None
        if callable(getattr(experiment, "selected_changed_portfolio_context", None)):
            selected = experiment.selected_changed_portfolio_context(candidate, portfolio_member)
            artifacts = (selected["previous"]["artifacts"], selected["current"]["artifacts"])
        else:
            if portfolio_member is not None:
                raise ValueError("portfolio member selection requires member-aware experiment accessors")
            artifacts = (experiment.previous_artifacts(candidate), experiment.current_artifacts(candidate))
        if any(has_physical_transition(artifact) for artifact in artifacts):
            # A missing/rejected copy witness must not become an unrelated legacy pass.
            return self.physical(candidate=candidate, experiment=experiment, timeout_s=timeout_s,
                                 portfolio_member=selected["selection"] if selected else None)
        from .lane_migration_qualifier import has_contraction_lane_migration
        if has_contraction_lane_migration(*artifacts):
            if self.lane_migration is None:
                raise ValueError("contraction lane migration has no bounded numerical qualifier")
            return self.lane_migration(candidate=candidate, experiment=experiment, timeout_s=timeout_s,
                                       portfolio_member=selected["selection"] if selected else None)
        return self.legacy(candidate=candidate, experiment=experiment, timeout_s=timeout_s,
                           portfolio_member=selected["selection"] if selected else None)


class HostPhysicalTransitionQualifier:
    def __init__(self, *, native_layout, expected_symbol, abi_provenance, output):
        if not expected_symbol or not abi_provenance:
            raise ValueError("physical copy witness requires a host-derived native ABI")
        self.native_layout, self.expected_symbol = native_layout, expected_symbol
        self.abi_provenance, self.output = dict(abi_provenance), Path(output)

    def __call__(self, *, candidate, experiment, timeout_s, portfolio_member=None):
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("physical transition witness requires a finite positive budget")
        start, deadline = monotonic(), monotonic() + min(timeout_s, 60)
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="physical_", dir=self.output))
        record = {"schema": "host_physical_transition_qualification_v1", "status": "UNKNOWN",
            "proof_scope": "selected internal copy and source pair at reduced extents only",
            "full_model_executed": False, "simulator_executed": False, "timing_measured": False,
            "full_model_correctness": "UNPROVEN", "full_shape_backend_correspondence": "UNPROVEN",
            "prepack_authorized": False, "global_speedup_proven": False, "cycles": None,
            "abi_provenance": self.abi_provenance, "arms": {}}

        def remaining():
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("physical transition witness exhausted its 60-second wall budget")
            return value

        try:
            selected_context = None
            if callable(getattr(experiment, "selected_changed_portfolio_context", None)):
                selected_context = experiment.selected_changed_portfolio_context(
                    candidate, portfolio_member)
                artifacts = {"before": selected_context["previous"]["artifacts"],
                             "after": selected_context["current"]["artifacts"]}
                analyses = {"before": selected_context["previous"]["analysis"],
                            "after": selected_context["current"]["analysis"]}
                member_binding = {
                    "selection": selected_context["selection"],
                    "previous": selected_context["previous"]["member_binding"],
                    "current": selected_context["current"]["member_binding"],
                }
                record["portfolio_member_binding"] = member_binding
            else:
                if portfolio_member is not None:
                    raise ValueError("portfolio member selection requires member-aware experiment accessors")
                artifacts = {"before": experiment.previous_artifacts(candidate),
                             "after": experiment.current_artifacts(candidate)}
                analyses = dict(zip(
                    artifacts, (row["analysis"] for row in experiment.iterations[-2:]), strict=True))
            sources = {}
            full = {}
            for arm, analysis in analyses.items():
                artifact = artifacts[arm]
                plan = analysis["diagnostics"]["verified_global_plan_emission"]
                sources[arm] = Path(artifact["interface"]).read_text()
                if (plan["status"] != "verified" or plan["source_sha256"] != _sha(sources[arm])
                        or plan["candidate_lowered_sha256"] != _sha(artifact["lowered_text"])
                        or artifact["candidate_lowered_sha256"] != plan["candidate_lowered_sha256"]):
                    raise ValueError("physical copy needs exact bound verified full-model artifacts")
                remaining()
                full[arm] = verify_physical_transitions(source_text=sources[arm],
                    lowered_text=artifact["lowered_text"], command_buffer=artifact["command_buffer"])
                if full[arm]["status"] not in {"verified", "not_declared"}:
                    raise ValueError(f"{arm} full-model physical transitions are not verified")
            record["full_artifact_evidence"] = full
            if sources["before"] != sources["after"]:
                raise ValueError("source changed between physical transition revisions")
            by_arm = {arm: {_edge(row): [] for row in _declarations(artifact)} for arm, artifact in artifacts.items()}
            for arm, artifact in artifacts.items():
                for declaration in _declarations(artifact):
                    by_arm[arm][_edge(declaration)].append(_mechanism(declaration))
            changed = sorted(edge for edge in set(by_arm["before"]) | set(by_arm["after"])
                if Counter(by_arm["before"].get(edge, [])) != Counter(by_arm["after"].get(edge, [])))
            if not changed:
                raise ValueError("no changed supported internal copy mechanism")
            edge = changed[0]
            text, extraction = extract_pointwise_chain(sources["after"], [edge[0], edge[2]],
                                                       max_extent=3, max_operations=2)
            if extraction["source_indices"] != [edge[0], edge[2]]:
                raise ValueError("source witness did not clone the declared producer/consumer pair")
            record.update(source_edge=list(edge), extraction=extraction,
                          other_changed_edges_unqualified=[list(value) for value in changed[1:]])
            source = work / "interface.mlir"
            source.write_text(text)
            function = next(op for op in parse_mlir_text(text).body.block.ops if op.name == "func.func")
            ops = [op for op in function.body.block.ops if op.name != "func.return"]
            pair = [i for i, op in enumerate(ops) if op.name == "linalg.generic"]
            if len(pair) != 2:
                raise ValueError("short witness did not retain exactly the two selected scalar regions")
            short_edge = (pair[0], edge[1], pair[1], edge[3])
            compiled = {}
            from xdsl.dialects.llvm import LLVM
            for arm, compile_method in (("before", experiment.compile_previous_probe_candidate),
                                        ("after", experiment.compile_probe_candidate)):
                result = compile_method(candidate, source, work / (arm+"_compile"),
                                        timeout_s=remaining(), emit_command_buffer=True)
                lowered, cb = result["lowered"], result["command_buffer"]
                if (lowered.returncode or result["command_buffer_emission"] is None
                        or result["command_buffer_emission"].returncode or not isinstance(cb, dict)):
                    raise ValueError(f"{arm} short compilation failed: " + (lowered.stderr or "")[-2000:])
                if len(lowered.stdout.encode()) > 2_000_000:
                    raise ValueError("short emitted artifact exceeds byte budget")
                context = make_context()
                context.load_dialect(LLVM)
                module = parse_mlir_text(lowered.stdout, context)
                module.verify()
                bounds = _bounded_native_module(module, expected_symbol=self.expected_symbol,
                    argument_count=len((cb.get("kernel_abi") or {}).get("args", [])))
                proof = verify_physical_transitions(source_text=text, lowered_text=lowered.stdout, command_buffer=cb)
                if proof["status"] not in {"verified", "not_declared"}:
                    raise ValueError(f"{arm} reduced emitted copy failed address verification")
                declarations = _declarations({"command_buffer": cb})
                if any(_edge(row) != short_edge for row in declarations):
                    raise ValueError("reduced copy belongs to a different source edge")
                if Counter(_mechanism(row) for row in declarations) != Counter(by_arm[arm].get(edge, [])):
                    raise ValueError(f"{arm} reduced emission did not reproduce the full-model copy orientation/type")
                compiled[arm] = (lowered.stdout, cb)
                record["arms"][arm] = {"physical_evidence": proof, "source_edge_correspondence": list(short_edge),
                                       "mechanisms": [_mechanism(row) for row in declarations], "emitted_bounds": bounds}
            for arm, (lowered, cb) in compiled.items():
                record["arms"][arm].update(self._native(candidate, experiment, work / arm,
                    lowered, cb, text, extraction, remaining))
            remaining()
            if selected_context is not None:
                selected_after = experiment.selected_changed_portfolio_context(
                    candidate, selected_context["selection"])
                actual_member_binding = {
                    "selection": selected_after["selection"],
                    "previous": selected_after["previous"]["member_binding"],
                    "current": selected_after["current"]["member_binding"],
                }
                if actual_member_binding != member_binding:
                    raise ValueError("portfolio member source, plan, or artifacts changed during qualification")
            record["status"] = "passed"
        except (ValueError, KeyError, TypeError, StopIteration, TimeoutError) as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = monotonic()-start
        receipt = work / "receipt.json"
        receipt.write_text(json.dumps(record, indent=2, sort_keys=True))
        return {**record, "detail_path": str(receipt), "detail_sha256": _sha(receipt.read_bytes())}

    def _native(self, candidate, experiment, work, lowered, cb, source, extraction, remaining):
        import numpy as np
        from merlin.runtime.commandbuffer import validate_command_buffer
        abi = cb.get("kernel_abi") or {}
        if (validate_command_buffer(cb) or cb.get("declined") or abi.get("kind") != "whole_program" or cb.get("commands")
                or "storage_encodings" in (cb.get("params") or {})):
            raise ValueError("internal-copy witness requires unchanged canonical external pointer ABI")
        inputs, outputs = extraction["inputs"], extraction["outputs"]
        if len(abi["args"]) != len(inputs)+len(outputs) or len(abi["outputs"]) != len(outputs):
            raise ValueError("short copy witness has unexpected external ABI arguments")
        work.mkdir()
        lowered_path, ll, library = work / "lowered.mlir", work / "probe.ll", work / "probe.so"
        lowered_path.write_text(lowered)
        translator = mlir_translate()
        for argv in ([str(translator), "--mlir-to-llvmir", str(lowered_path), "-o", str(ll)],
                     [str(translator.with_name("clang")), "-shared", "-fPIC", "-O2", str(ll), "-o", str(library)]):
            result = experiment.run_native_probe(candidate, work, argv, timeout_s=remaining())
            if result.returncode:
                raise ValueError("sandboxed native copy build failed: " + (result.stderr or "")[-2000:])
        runner = work / "runner.py"
        runner.write_bytes(Path(native_host_witness_runner.__file__).read_bytes())
        cases = []
        for seed in range(3):
            values = []
            for spec in inputs:
                if spec["dtype"] not in {"i8", "i16", "i32", "f32"} or prod(spec["shape"]) > 4096:
                    raise ValueError("unsupported or oversized source witness input")
                dtype = np.dtype("float32" if spec["dtype"] == "f32" else "int"+spec["dtype"][1:])
                pool = ([-0., .5, -.5, 1., -1., 2**-24, -2**-24] if spec["dtype"] == "f32"
                        else [-128, -127, -1, 0, 1, 126, 127])
                values.append(np.asarray([pool[(i+seed)%len(pool)] for i in range(prod(spec["shape"]))],
                                         dtype=dtype).reshape(spec["shape"]))
            expected = evaluate_pointwise_source(source, values)
            arguments, ii, oi = [], 0, 0
            for arg in abi["args"]:
                spec = cb["tensors"][arg["tensor"]]
                read = arg["access"] == "read"
                wanted = inputs[ii] if read else outputs[oi]
                if (spec["shape"] != wanted["shape"] or spec["dtype"] != wanted["dtype"]
                        or (not read and (arg["access"] != "write" or arg["tensor"] != abi["outputs"][oi]))):
                    raise ValueError("short external ABI does not match source argument/output order")
                layout = dict(self.native_layout(spec))
                if not (0 < layout["rows"]*layout["cols"] <= 4096
                        and layout["cols"] <= layout["row_stride"]
                        and layout["rows"]*layout["row_stride"] <= layout["storage_elements"] <= 1_000_000):
                    raise ValueError("native layout exceeds short witness budget")
                row = {**layout, "dtype": spec["dtype"], "access": arg["access"]}
                if read:
                    row["values"] = values[ii].reshape(-1).tolist()
                    ii += 1
                else:
                    oi += 1
                arguments.append(row)
            request = work / f"inputs_{seed}.json"
            request.write_text(json.dumps({"library": str(library), "symbol": self.expected_symbol,
                                           "arguments": arguments}))
            result = experiment.run_native_probe(candidate, work, ["python3", str(runner), str(request)], timeout_s=remaining())
            if result.returncode:
                raise ValueError("sandboxed native copy witness failed: " + (result.stderr or "")[-2000:])
            observed = json.loads(result.stdout)
            if observed.get("warmup_calls") != 1 or observed.get("observed_calls") != 1:
                raise ValueError("native witness did not execute warm1/observed1")
            if len(observed["outputs"]) != len(expected):
                raise ValueError("native output count differs from source")
            for got, wanted in zip(observed["outputs"], expected, strict=True):
                if np.asarray(got, dtype=wanted.dtype).reshape(wanted.shape).tobytes() != wanted.tobytes():
                    raise ValueError("native copy/source-pair output differs from independent source bits")
            cases.append({"seed": seed, "exact_output_bits": True, "warmup_calls": 1, "observed_calls": 1,
                          "request_sha256": _sha(request.read_bytes()),
                          "expected_sha256": [_sha(value.tobytes()) for value in expected]})
        return {"cases": cases, "native_binary_sha256": _sha(library.read_bytes()),
                "native_runner_sha256": _sha(runner.read_bytes()),
                "reference_sha256": _sha(Path(__file__).with_name("host_source_witness.py").read_bytes())}
