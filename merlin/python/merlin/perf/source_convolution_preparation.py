"""Prepare a paired complete-source convolution witness, without executing it.

Full-model proofs must already exist in the bounded static analysis. Only the
short emitted artifacts are verified here. No simulator, candidate import, flags,
full-model compilation, or implicit comparison-arm substitution is permitted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from time import monotonic

from .compiler_plan_evidence import verify_compiler_global_plan
from .model_macs import observe_model_macs
from .source_convolution_witness import extract_source_convolution, evaluate_source_convolution
from .source_program_pair import bind_source_program_pair, program_plan as _plan, source_owners as _owners


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _digest(value):
    return _sha(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))


def prepare_source_convolution(*, candidate: Path, experiment, comparison_arm: str,
                               entry: str, output: Path, timeout_s: float = 60) -> dict:
    """Return hash-bound short artifacts/oracle, or UNKNOWN; never launch a runtime.

    ``comparison_arm`` is mandatory: an optimization baseline is not a previous
    iteration. The controller owns deadlines for compiler subprocesses and charges
    the entire action; this helper must not charge preparation a second time.
    """
    if comparison_arm not in {"optimization_baseline","previous"}:
        raise ValueError("select optimization_baseline or previous explicitly")
    if not isinstance(entry,str) or not entry:
        raise ValueError("source entry must be explicitly host-selected")
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
        raise ValueError("source-convolution preparation requires a finite <=60s budget")
    started, deadline = monotonic(), monotonic()+timeout_s
    record = {"schema":"source_convolution_preparation_v1","status":"UNKNOWN",
        "comparison_arm":comparison_arm,"simulator_executed":False,"full_model_executed":False,
        "full_model_recompiled":False,"full_model_proofs_recomputed":False,
        "numerical_qualification":"UNPROVEN","runtime_admitted":False,
        "target_instruction_semantics":"UNVERIFIED","global_speedup_proven":False,
        "scope":"complete reduced source programs prepared; not executed or full-shape qualified","arms":{}}
    work = None

    def remaining():
        left = deadline-monotonic()
        if left <= 0:
            raise TimeoutError("paired preparation exhausted its shared deadline")
        return left

    try:
        pair = bind_source_program_pair(candidate=candidate, experiment=experiment, comparison_arm=comparison_arm)
        source, source_sha = pair.source, pair.source_sha256
        artifacts, owners = pair.artifacts, pair.owners
        binding, compile_before = pair.comparison_binding, pair.compile_before
        choices = []
        for _,shape in observe_model_macs(source,entry=entry):
            index = shape.source_op_index
            if (shape.status == "derived" and owners["before"].get(index,{}).get("kind") == "host"
                    and owners["after"].get(index,{}).get("kind") == "convolution"):
                choices.append((shape.macs,index))
        if not choices:
            raise ValueError("no verified source MAC owner changes from host to convolution")
        record["source_convolution_opportunities"] = [
            {"source_op_index":index,"source_macs":macs,
             "before_owner":owners["before"][index]["task_index"],
             "after_owner":owners["after"][index]["task_index"],
             "scope":"verified source work/declared ownership change; not measured performance"}
            for macs,index in sorted(choices,reverse=True)]
        errors = []
        for _,index in sorted(choices,reverse=True)[:3]:
            remaining()
            try:
                probe,extraction = extract_source_convolution(source,index,entry=entry)
                break
            except ValueError as error:
                errors.append({"source_op_index":index,"reason":str(error)})
        else:
            raise ValueError("first bounded source candidates are unsupported: "+json.dumps(errors))
        output=Path(output)
        output.mkdir(parents=True,exist_ok=True)
        work=Path(tempfile.mkdtemp(prefix="source_convolution_",dir=output))
        probe_path=work/"interface.mlir"
        probe_path.write_text(probe)
        short_mac = next(shape.source_op_index for _,shape in observe_model_macs(probe,entry=entry)
                         if shape.status == "derived")
        record.update(source_sha256=source_sha,extraction=extraction,skipped_source_options=errors,
            comparison_binding=binding,workdir=str(work),
            full_model_source_owner={arm:owners[arm][index] for arm in owners})
        initial_current_binding=experiment.current_probe_binding(candidate)
        for arm,compile_method in (("before",compile_before),("after",experiment.compile_probe_candidate)):
            result=compile_method(candidate,probe_path,work/(arm+"_compile"),timeout_s=remaining(),emit_command_buffer=True)
            lowered,buffer_result,cb=result["lowered"],result["command_buffer_emission"],result["command_buffer"]
            if (lowered.returncode or buffer_result is None or buffer_result.returncode
                    or not isinstance(cb,dict) or cb.get("declined") is not None):
                raise ValueError(arm+" short normal-entrypoint compilation failed: "+(lowered.stderr or "")[-1200:])
            if len(lowered.stdout.encode())>2_000_000:
                raise ValueError("short emitted artifact exceeds bounded verification size")
            cb_sha=_digest(cb)
            remaining()
            proof=verify_compiler_global_plan(source_text=probe,lowered_text=lowered.stdout,command_buffer=cb,
                candidate_sha256=artifacts[arm]["compiler_sha256"],command_buffer_sha256=cb_sha)
            remaining()
            expected_kind="host" if arm=="before" else "convolution"
            task=_owners(_plan(cb)).get(short_mac,{})
            if proof.get("status")!="verified" or task.get("kind")!=expected_kind:
                raise ValueError(arm+" reduced source/task/CFG route does not reproduce full-model ownership")
            plan=_plan(cb)
            if (len(plan["entry_bindings"])!=2 or len(set(plan["entry_bindings"]))!=2
                    or len(plan["output_bindings"])!=1 or (cb.get("kernel_abi") or {}).get("kind")!="whole_program"):
                raise ValueError("short source lacks the explicit complete-program input/output ABI")
            (work/(arm+".llvm.mlir")).write_text(lowered.stdout)
            (work/(arm+".command_buffer.json")).write_text(json.dumps(cb,sort_keys=True,indent=2)+"\n")
            record["arms"][arm]={"compiler_sha256":artifacts[arm]["compiler_sha256"],
                "lowered_sha256":_sha(lowered.stdout),"command_buffer_canonical_sha256":cb_sha,
                "source_task_cfg_proof":proof,"declared_route":expected_kind,"source_owner":task,
                "input_tensors":plan["entry_bindings"],"output_bindings":plan["output_bindings"],
                "actual_target_instruction_semantics":"UNVERIFIED: runtime adapter must check emitted mechanism"}
        if experiment.current_probe_binding(candidate)!=initial_current_binding:
            raise ValueError("current compiler/source/target binding changed during preparation")
        import numpy as np
        cases=[]
        for mode in ("signed_coordinate_pattern","negative_extremes"):
            arrays=[]
            for shape,step in ((extraction["input_shape"],37),(extraction["weight_shape"],19)):
                array=((np.arange(math.prod(shape)).reshape(shape)*step)%256-128).astype(np.int8)
                if mode=="negative_extremes":
                    array.fill(-128 if step==37 else 127)
                arrays.append(array)
            expected=evaluate_source_convolution(extraction,*arrays)
            cases.append({"case":mode,"inputs":[array.tolist() for array in arrays],"expected":expected.tolist(),
                "input_sha256":[hashlib.sha256(array.tobytes()).hexdigest() for array in arrays],
                "expected_i32_le_sha256":hashlib.sha256(expected.astype("<i4").tobytes()).hexdigest()})
        oracle={"schema":"source_convolution_prepared_oracle_v1","probe_source_sha256":_sha(probe),
                "cases":cases,"expected_output_shape":extraction["output_shape"],"input_dtypes":["i8","i8"],"output_dtype":"i32"}
        oracle_path=work/"independent_oracle.json"
        oracle_path.write_text(json.dumps(oracle,indent=2)+"\n")
        remaining()
        record.update(status="prepared",independent_oracle=str(oracle_path),
                      independent_oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest())
    except (ValueError,KeyError,TypeError,AttributeError,RuntimeError,TimeoutError,OSError,
            subprocess.SubprocessError) as error:
        record.update(status="UNKNOWN",reason=f"{type(error).__name__}: {error}")
    record["elapsed_seconds"]=monotonic()-started
    if work is not None:
        (work/"preparation.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    return record
