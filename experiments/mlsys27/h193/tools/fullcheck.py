#!/usr/bin/env python3
"""fullcheck.py RUNDIR FORMS_DIR [WORKERS]  -- run INSIDE p12-unsealed ns.sh with env.sh sourced (P12_TAG=fin6yn: the neutral-flavor release contract + operand-selection sidecar e5f2486b).
For every case dir under RUNDIR that emitted a program (cb.json + kernel.llvm.mlir), apply the GRADER's own two structural checks, unmodified, and write RUNDIR/<case>/full.json:
  (ii)  instruction-trace : rocc.decode.decode_module(parsed kernel.llvm.mlir) -> trace_check.check(trace, capsule.expected, cb=..., address_model='pointer_args', subsumes=..., target=..., lowered_module=...)
        i.e. exactly capsule_runner.py's trace gate (static decode of the lowered LLVM, not a Spike log).  status pass/fail, trace_requirement_status pass/unavailable_*/..., violations.
  (iii) host-compute      : offload_census._source_host_compute(...) = the function apply_host_compute_veto calls (audit_group, Budget 1.0 arithmetic/element, 0.25 value/element, 0.5 payload).
`capsule.expected` comes from FORMS_DIR/<slug>/capsule_meta.json (gen_expected.py: the op-capsule registry's own capsule dict); cases with none (f32 canonical captures) get expected={}.
VERIFICATION INSTRUMENT: never an author input."""
import json, sys, os, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

RUN, FORMS = Path(sys.argv[1]), Path(sys.argv[2]); W = int(sys.argv[3]) if len(sys.argv) > 3 else 6
TARGET = "gemmini"


def one(args):
    case, meta_path = args
    d = RUN / case; out = dict(case=case, target=TARGET)
    try:
        from merlin.targetgen import offload_census as OC, trace_check as TCK
        from merlin.targetgen.rocc import decode as RD
        cb = json.load(open(d / "cb.json")); text = (d / "kernel.llvm.mlir").read_text()
        meta = json.load(open(meta_path)) if meta_path else None
        cap = (meta or {}).get("capsule") or {}
        out["expected_source"] = (meta or {}).get("source", "none (f32 canonical capture: expected={})")
        expected = cap.get("expected") or {}
        out["expected"] = expected
        # --- (iii) host compute: the grader function, same arguments as apply_host_compute_veto
        try:
            row = OC.program_row(case, cb)
            hc = OC._source_host_compute(case, cap, cb, text, row["outcome"], target=TARGET)
            out["host_compute"] = {k: hc.get(k) for k in ("verdict", "why", "symbol", "elements", "host_arithmetic", "host_value_arithmetic", "host_payload_bytes", "host_total_arithmetic", "host_address_arithmetic",
                                                          "arithmetic_per_element", "value_arithmetic_per_element", "payload_ratio", "arithmetic_basis")}
            out["host_compute"]["program_outcome"] = row["outcome"]
        except Exception as e:  # noqa
            out["host_compute"] = dict(verdict="error", why=f"{type(e).__name__}: {e}"[:300])
        # --- (ii) instruction trace: capsule_runner trace gate
        try:
            mod = RD._parse_module(text)
            trace = RD.decode_module(mod, target=TARGET, source=str(d / "kernel.llvm.mlir")) if mod is not None else RD.decode_text(text, source=str(d / "kernel.llvm.mlir"), target=TARGET)
            res = TCK.check(trace, expected, cb=cb, address_model="pointer_args", subsumes=TCK.subsumption_for(TARGET), target=TARGET, lowered_module=mod)
            out["trace"] = dict(status=res.get("status"), trace_requirement_status=res.get("trace_requirement_status"), unavailable_requirements=res.get("unavailable_requirements"),
                                violations=(res.get("violations") or [])[:8], n_instructions=len(trace.get("instructions", [])), drives_accelerator=TCK.drives_accelerator(trace),
                                accelerator_instruction_count=TCK.accelerator_instruction_count(trace),
                                requirements_complete=TCK.requirements_complete(res))
        except Exception as e:  # noqa
            out["trace"] = dict(status="error", trace_requirement_status="error", violations=[f"{type(e).__name__}: {e}"[:300]], tb=traceback.format_exc()[-600:])
    except Exception as e:  # noqa
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    json.dump(out, open(d / "full.json", "w"), indent=1, default=str)
    return case, (out.get("trace") or {}).get("status"), (out.get("trace") or {}).get("trace_requirement_status"), (out.get("host_compute") or {}).get("verdict")


if __name__ == "__main__":
    jobs = []
    for d in sorted(RUN.iterdir()):
        if not d.is_dir() or not (d / "cb.json").is_file() or not (d / "kernel.llvm.mlir").is_file(): continue
        mp = None
        if d.name.startswith("f_"):
            r = json.load(open(d / "result.json")); slug = Path(r["case"]).name
            mp = FORMS / slug / "capsule_meta.json"; mp = mp if mp.is_file() else None
        jobs.append((d.name, mp))
    with ProcessPoolExecutor(W) as ex:
        for r in ex.map(one, jobs): print(*r, flush=True)
    print("FULLCHECK DONE", len(jobs))
