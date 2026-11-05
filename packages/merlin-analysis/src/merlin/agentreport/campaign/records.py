"""Read one campaign's records into a tidy dataset (read-only; every path comes from the manifest).

What is read, per attempt:

* launch time -- the ``cpu_reservation_verified`` line of the runner log;
* CPU accounting -- the attempt's ``Running as unit`` block of a systemd-run log;
* agent threads -- codex rollouts under ``agent_home`` (turns, commands, tool calls, token counters);
* grades -- ``_qa_work/background_snapshot_owners``, ``fruns_<token>/snapshot_progress.json`` and
  ``qa_history/verdict_fast_<token>.json``; per test case L2 outcome timed by its stage record;
* stage timing -- ``_qa_work/**/setup-timing/*.json`` (``merlin.compiler_setup_timing.v1``);
* RTL certification -- the workspace ``.qa_channel`` ``simreq/simresp/simtiming`` files;
* agent process-tree CPU -- ``rounds/*.resource_samples.jsonl``.

Nothing is interpolated: a value that is absent stays absent and the reason is counted.
Record fields named after the corpus' own vocabulary are read as written; every key this module
returns speaks of *test cases*.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import Manifest, cohort_of

BIN_S = 60.0
SCHEMA = "merlin.agentreport.campaign_records.v1"

# ----------------------------------------------------------------------------------------------- io


def jload(path) -> Any:
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8", "replace"), strict=False)
    except Exception:  # noqa: BLE001 - an unreadable record is absent, never fatal
        return None


def jlines(path):
    try:
        fh = open(path, "rb")
    except OSError:
        return
    with fh:
        for raw in fh:
            try:
                yield json.loads(raw.decode("utf-8", "replace"), strict=False)
            except Exception:  # noqa: BLE001
                continue


def iso(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _ms(x) -> float | None:
    return x / 1000.0 if isinstance(x, (int, float)) else None


def _isdir(p) -> bool:
    return bool(p) and os.path.isdir(p)


# ------------------------------------------------------------------------------------ launch, CPU


def launch_time(log) -> float | None:
    if not log or not os.path.isfile(log):
        return None
    for row in jlines(log):
        if isinstance(row, dict) and row.get("event") == "cpu_reservation_verified":
            return iso(row.get("at"))
    return None


def systemd_seconds(text: str) -> float:
    """``'1d 5h 28min 55.000s'`` -> seconds (token suffixes, no patterns)."""
    total = 0.0
    for tok in text.split():
        for unit, mult in (("min", 60.0), ("ms", 0.001), ("d", 86400.0), ("h", 3600.0), ("s", 1.0)):
            if tok.endswith(unit):
                try:
                    total += float(tok[: -len(unit)]) * mult
                except ValueError:
                    pass
                break
    return total


def accounting(log, entry: int) -> dict[str, float] | None:
    """``{runtime_s, cpu_s}`` of one ``Running as unit`` block, or None when not (yet) recorded."""
    if not log or not os.path.isfile(log):
        return None
    blocks: list[dict[str, float | None]] = []
    for line in open(log, encoding="utf-8", errors="replace"):
        s = line.strip()
        if s.startswith("Running as unit:"):
            blocks.append({"runtime_s": None, "cpu_s": None})
        elif blocks and s.startswith("Service runtime:"):
            blocks[-1]["runtime_s"] = systemd_seconds(s.split(":", 1)[1])
        elif blocks and s.startswith("CPU time consumed:"):
            blocks[-1]["cpu_s"] = systemd_seconds(s.split(":", 1)[1])
    try:
        block = blocks[entry]
    except IndexError:
        return None
    return block if block.get("cpu_s") and block.get("runtime_s") else None


# ----------------------------------------------------------------------------------- agent threads

BOOKKEEPING = {"turn_context", "thread_settings_applied", "task_started"}
READERS = ("sed ", "cat ", "grep ", "rg ", "head ", "tail ", "ls", "find ", "wc ", "diff ", "git ", "jq ", "nl ",
           "pwd", "stat ", "du ", "file ", "echo ", "printf ", "test ", "readlink")
CHECK_TOOLS = ("agent_selfcheck.py", "simjob.py", "simjob_physical", "physical_campaign", "resubmit_physical",
               "agent_model_check.py", "isa_tools", "cca.py", "cca_compare", "cca_contract",
               "capability_query", "action_catalog")
HELPER_CALLS = ("wait_agent", "spawn_agent", "followup_task", "send_message", "list_agents", "close_agent")


def activity_of_command(cmd: str) -> str:
    """reading / writing / checks / shell for one shell command (plain substring tests)."""
    c = cmd.strip()
    if any(t in c for t in CHECK_TOOLS) or c.startswith("sleep") or " sleep " in c:
        return "checks"
    if "write_text(" in c or "cat >" in c or ".write(" in c or "sed -i" in c or "apply_patch" in c or " tee " in c:
        return "writing"
    parts = [p.strip() for p in c.replace("&&", ";").replace("||", ";").replace("|", ";").split(";") if p.strip()]
    if parts and all(p.startswith(READERS) for p in parts):
        return "reading"
    if c.startswith("python3 - <<") or "python - <<" in c or "/python - <<" in c:
        if "subprocess" in c or "run(" in c or "check_call" in c or "Popen" in c:
            return "shell"
        return "reading"
    return "shell"


def check_kind(cmd: str) -> str | None:
    """Which check an agent command requested (None for waits/polls and non-check commands)."""
    if "agent_selfcheck.py" in cmd:
        if "--attach" in cmd or "--list" in cmd or "--status" in cmd:
            return None
        return "rtl_check" if "gsim" in cmd else "selfcheck"
    if "simjob.py" in cmd or "physical_campaign" in cmd or "resubmit_physical" in cmd:
        return None if (" collect" in cmd or " poll" in cmd) else "rtl_check"
    if "agent_model_check.py" in cmd:
        return None if (" wait" in cmd or " status" in cmd) else "model_check"
    return None


def _tokens(usage) -> dict[str, float] | None:
    if not isinstance(usage, dict) or not isinstance(usage.get("input_tokens"), (int, float)):
        return None
    cached = (usage.get("cached_input_tokens") or 0) + (usage.get("cache_write_input_tokens") or 0)
    return {
        "fresh": max(0, usage["input_tokens"] - cached),
        "cached": cached,
        "output": usage.get("output_tokens") or 0,
        "reasoning": usage.get("reasoning_output_tokens") or 0,
    }


def read_threads(agent_home) -> list[dict[str, Any]]:
    """One record per agent thread: turns, activity intervals and cumulative token samples.

    A helper's rollout repeats its parent's history up to ``subagent_history_start_ordinal``; those
    rows are the parent's work and are skipped."""
    if not _isdir(agent_home):
        return []
    threads: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(str(agent_home), "**", "rollout-*.jsonl"), recursive=True)):
        th, start_ord, calls = None, None, {}
        for row in jlines(path):
            if not isinstance(row, dict):
                continue
            kind = row.get("type")
            p = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            when = iso(row.get("timestamp"))
            if kind == "session_meta":
                if th is not None or not isinstance(p.get("id"), str):
                    continue
                src = p.get("source") if isinstance(p.get("source"), dict) else {}
                spawn = ((src.get("subagent") or {}).get("thread_spawn") or {}) if src else {}
                parent = p.get("parent_thread_id") or spawn.get("parent_thread_id")
                start_ord = p.get("subagent_history_start_ordinal")
                th = threads.setdefault(p["id"], {
                    "id": p["id"], "role": "helper" if parent else "lead",
                    "path": p.get("agent_path") or spawn.get("agent_path") or ("/root" if not parent else None),
                    "started": iso(p.get("timestamp")) or when, "turns": [], "spans": [], "checks": [],
                    "tokens": [], "_open": None, "_work": None, "_src": None,
                })
                continue
            if th is None:
                continue
            if isinstance(start_ord, int):
                o = row.get("ordinal")
                if not isinstance(o, int) or o < start_ord:
                    continue
            if p.get("thread_id", th["id"]) != th["id"]:
                continue
            if when and kind not in BOOKKEEPING and p.get("type") not in BOOKKEEPING:
                th["_work"] = max(th["_work"] or 0, when)
            if kind == "response_item" and p.get("type") in ("function_call", "custom_tool_call") and when:
                calls[p.get("call_id")] = (when, p.get("name"), str(p.get("input") or p.get("arguments") or "")[:400])
                continue
            if kind == "response_item" and p.get("type") in ("function_call_output", "custom_tool_call_output") and when:
                c = calls.pop(p.get("call_id"), None)
                if c and when > c[0]:
                    name, inp = c[1], c[2]
                    act = None
                    if name == "exec" and "tools.apply_patch" in inp:
                        act = "writing"
                    elif name in HELPER_CALLS or (name == "exec" and "wait_agent" in inp):
                        act = "helpers"
                    elif name == "sleep" or (name == "exec" and "sleep" in inp and "exec_command" not in inp):
                        act = "checks"
                    if act:
                        th["spans"].append([c[0], when, act])
                continue
            if kind == "token_usage_record":
                tok = _tokens(p.get("thread_token_usage"))
                if tok and when:
                    th["_src"] = "record"
                    th["tokens"].append([when, tok])
                continue
            if kind != "event_msg":
                continue
            ev = p.get("type")
            if ev == "token_count":
                info = p.get("info") if isinstance(p.get("info"), dict) else {}
                tok = _tokens(info.get("total_token_usage"))
                if tok and when and th["_src"] != "record":
                    th["_src"] = "count"
                    th["tokens"].append([when, tok])
            elif ev == "task_started":
                if th["_open"] and th["_work"] and th["_work"] >= th["_open"]:
                    th["turns"].append([th["_open"], th["_work"]])
                th["_open"] = iso(p.get("started_at")) or when
            elif ev in ("task_complete", "task_aborted"):
                begin, end = iso(p.get("started_at")) or th["_open"], iso(p.get("completed_at")) or when
                if begin and end and end >= begin:
                    th["turns"].append([begin, end])
                th["_open"] = None
            elif ev == "item_completed":
                item = p.get("item") if isinstance(p.get("item"), dict) else {}
                b, e = _ms(p.get("started_at_ms")), _ms(p.get("completed_at_ms"))
                if item.get("type") == "CommandExecution":
                    cmd = item.get("command")
                    cmd = cmd[-1] if isinstance(cmd, list) and cmd else cmd
                    cmd = cmd if isinstance(cmd, str) else ""
                    e = e or when
                    b = b or e
                    th["spans"].append([b, e, activity_of_command(cmd)])
                    ck = check_kind(cmd)
                    if ck:
                        th["checks"].append([b, ck])
                elif item.get("type") == "CollabAgentToolCall" and b and e and e > b:
                    th["spans"].append([b, e, "helpers"])
    out = []
    for t in threads.values():
        if t["_open"]:
            end = t["_work"]
            if end and end >= t["_open"]:
                t["turns"].append([t["_open"], end])
        for k in ("_open", "_work", "_src"):
            t.pop(k, None)
        t["tokens"].sort(key=lambda r: r[0])
        thin, last = [], None
        for when, tok in t["tokens"]:  # <= one sample per 30 s per thread
            if last is None or when - last >= 30:
                thin.append([when, tok])
                last = when
            else:
                thin[-1] = [when, tok]
        t["tokens"] = thin
        out.append(t)
    out.sort(key=lambda t: (t["role"] != "lead", t["started"] or 0))
    return out


# ------------------------------------------------------------------------------ grader stage timing

#: grader stage -> compute kind. Parents count only their exclusive (non-child) time; waits are dropped.
STAGE_KIND = {
    "compiler_command": "compile",
    "oracle_compile_link_admit": "compile",
    "oracle_adapter.L2": "compile",
    "oracle_simulation_readback.spike": "l2_sim",
    "emission": "other",
    "grade_cpu.tree_hash": "other",
    "grade_cpu.verify_compiler_global_plan": "other",
    "grade_cpu.capture_compiler_source": "other",
    "grade_cpu.finish_compiler_source": "other",
    "oracle_result_retention": "other",
    "result_processing": "other",
    "grade_cpu.lease": None,
    "capsule_execution": None,
    "scheduler_submission_wait": None,
}


def _stage_kind(name: str) -> str | None:
    if name in STAGE_KIND:
        return STAGE_KIND[name]
    return "reference" if name.startswith("grade_cpu.") else "other"


def _subtract(iv, holes):
    a, b = iv
    out, cur = [], a
    for h0, h1 in sorted(holes):
        if h1 <= cur or h0 >= b:
            continue
        if h0 > cur:
            out.append((cur, h0))
        cur = max(cur, h1)
    if cur < b:
        out.append((cur, b))
    return out


class Bins:
    """Job-seconds per 60 s bin per kind."""

    def __init__(self):
        self.data: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, t0, t1, kind, weight=1.0):
        if t0 is None or t1 is None or t1 <= t0 or kind is None:
            return
        for i in range(int(t0 // BIN_S), int(t1 // BIN_S) + 1):
            lo, hi = max(t0, i * BIN_S), min(t1, (i + 1) * BIN_S)
            if hi > lo:
                self.data[i][kind] += (hi - lo) * weight

    def dump(self) -> dict[str, dict[str, float]]:
        return {str(k): dict(v) for k, v in sorted(self.data.items())}


def read_stage_timing(run_dir, bins: Bins) -> tuple[int, dict[tuple[str, str], float]]:
    """Fill ``bins``; return (files read, {(grade token, test case): finish time})."""
    finish: dict[tuple[str, str], float] = {}
    n = 0
    for path in glob.glob(os.path.join(str(run_dir), "_qa_work", "**", "setup-timing", "*.json"), recursive=True):
        token = next((part[len("fruns_"):] for part in Path(path).parts if part.startswith("fruns_")), None)
        grade = "/selfchecks/" not in path
        d = jload(path)
        if not isinstance(d, dict):
            continue
        stages = [s for s in d.get("stages") or [] if isinstance(s, dict) and s.get("started_ns") and s.get("finished_ns")]
        if not stages:
            continue
        n += 1
        children = defaultdict(list)
        for s in stages:
            if s.get("parent_stage_id") is not None:
                children[s["parent_stage_id"]].append((s["started_ns"] / 1e9, s["finished_ns"] / 1e9))
        for s in stages:
            iv = (s["started_ns"] / 1e9, s["finished_ns"] / 1e9)
            name = str(s.get("stage"))
            if name == "capsule_execution" and grade and token and d.get("capsule"):
                key = (token, d["capsule"])
                finish[key] = max(finish.get(key, 0.0), iv[1])
            kind = _stage_kind(name)
            if kind is None:
                continue
            for a, b in _subtract(iv, children.get(s.get("stage_id"), [])):
                bins.add(a, b, kind)
    return n, finish


def read_agent_cpu(run_dir, bins: Bins) -> int:
    """Positive deltas of the agent process tree's user+system CPU (a lower bound)."""
    n = 0
    for path in sorted(glob.glob(os.path.join(str(run_dir), "rounds", "round_*.resource_samples.jsonl"))):
        prev = None
        for row in jlines(path):
            t = iso(row.get("sampled_at")) if isinstance(row, dict) else None
            if t is None:
                continue
            cpu = (row.get("user_cpu_s") or 0) + (row.get("system_cpu_s") or 0)
            if prev is not None and 0 < t - prev[0] < 120 and cpu > prev[1]:
                bins.add(prev[0], t, "other", weight=min((cpu - prev[1]) / (t - prev[0]), 64.0))
            prev = (t, cpu)
            n += 1
    return n


# ------------------------------------------------------------------------------------------ grades


def read_grades(run_dir, finish) -> tuple[list[dict], list[dict]]:
    work, hist = os.path.join(str(run_dir), "_qa_work"), os.path.join(str(run_dir), "qa_history")
    tokens = {Path(p).stem for p in glob.glob(os.path.join(work, "background_snapshot_owners", "*.json"))}
    tokens |= {Path(p).name[len("fruns_"):] for p in glob.glob(os.path.join(work, "fruns_*"))}
    tokens |= {Path(p).stem[len("verdict_fast_"):] for p in glob.glob(os.path.join(hist, "verdict_fast_*.json"))}
    grades, events = [], []
    for tok in sorted(t for t in tokens if t.isdigit()):
        owner = jload(os.path.join(work, "background_snapshot_owners", tok + ".json")) or {}
        prog = jload(os.path.join(work, "fruns_" + tok, "snapshot_progress.json")) or {}
        verdict = jload(os.path.join(hist, "verdict_fast_" + tok + ".json")) or {}
        rows = prog.get("per_capsule") or verdict.get("per_capsule") or []
        fallback = iso(prog.get("updated_at")) or iso(verdict.get("graded_at"))
        n_done = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            tiers = r.get("tiers") if isinstance(r.get("tiers"), dict) else {}
            if r.get("status") in (None, "gated") and not tiers:
                continue
            n_done += 1
            name = r.get("capsule")
            t = finish.get((tok, name))
            events.append({"t": t or fallback, "test_case": name, "tier": "L2",
                           "status": str(tiers.get("L2") or r.get("status")), "source": tok, "time_is_grade_level": t is None})
        grades.append({"token": tok, "started": owner.get("started_at_epoch") or int(tok) / 1e9,
                       "updated": fallback, "n_completed": n_done})
    return grades, events


def read_rtl(workspace, bins: Bins, engine: str) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    """Certification jobs (intervals), L3 outcome events and bound performance rows.

    Performance binding (after the console's perf rules): a row counts only with the configured
    engine identity, positive barrier cycles, a 64-hex compiler source digest and a test-case digest
    equal to the one its structural comparison is bound to. The roofline of the same row is used only
    when useful MACs, peak MACs/cycle and floor cycles are positive and floor <= cycles. Model
    estimates are used only when PROVISIONAL/QUALIFIED with complete event accounting. Everything
    else is counted under ``excluded``, never filled in."""
    ch = os.path.join(str(workspace or ""), ".qa_channel")
    jobs, events, perf = [], [], []
    excluded: dict[str, int] = defaultdict(int)
    for path in glob.glob(os.path.join(ch, "simtiming_*.json")):
        job = Path(path).stem[len("simtiming_"):]
        tm = jload(path) or {}
        req = jload(os.path.join(ch, "simreq_" + job + ".json")) or {}
        resp = jload(os.path.join(ch, "simresp_" + job + ".json")) or {}
        t0, t1 = tm.get("started_at_unix"), tm.get("finished_at_unix")
        if t0 and not t1 and tm.get("worker_runtime_s"):
            t1 = t0 + tm["worker_runtime_s"]
        if t0 and t1:
            bins.add(t0, t1, "l3_sim")
        jobs.append({"job": job, "submitted": req.get("submitted_at"), "started": t0, "finished": t1})
        for r in resp.get("per_capsule") or []:
            if not isinstance(r, dict) or r.get("barrier_tier") != "L3":
                continue
            events.append({"t": t1, "test_case": r.get("capsule"), "tier": "L3", "status": str(r.get("barrier_status")),
                           "source": job, "time_is_grade_level": not t1})
            if r.get("barrier_status") != "pass":
                continue
            row = bind_perf(r, t1, engine, excluded)
            if row:
                row["job"] = job
                perf.append(row)
    return jobs, events, perf, dict(excluded)


def bind_perf(r: dict, t: float | None, engine: str, excluded: dict[str, int]) -> dict | None:
    """One certifying row -> a bound performance record, or None (and the reason counted)."""
    cycles = r.get("barrier_cycles")
    perf = r.get("performance") if isinstance(r.get("performance"), dict) else {}
    source = ((r.get("tier_boundary") or {}).get("common") or {}).get("compiler_sha256")
    digest = perf.get("capsule_sha256")
    bound = (perf.get("structural_comparison_binding") or {}).get("capsule_sha256")
    if (r.get("barrier_timing_identity") or {}).get("engine") != engine or not isinstance(cycles, (int, float)) or cycles <= 0:
        excluded["no_rtl_cycles"] += 1
        return None
    if not (isinstance(source, str) and len(source) == 64) or not digest or digest != bound:
        excluded["unbound_digest"] += 1
        return None
    if not t:
        excluded["no_time"] += 1
        return None
    roof = perf.get("structural_roofline") if isinstance(perf.get("structural_roofline"), dict) else {}
    macs, peak, floor = roof.get("useful_macs"), roof.get("structural_peak_macs_per_cycle"), roof.get("structural_compute_floor_cycles")
    roof_ok = all(isinstance(x, (int, float)) and x > 0 for x in (macs, peak, floor)) and floor <= cycles
    if not roof_ok:
        excluded["roofline_unavailable"] += 1
    model = perf.get("estimated_whole_soc_cycles")
    model_ok = (perf.get("status") in ("PROVISIONAL", "QUALIFIED") and perf.get("complete_event_accounting") is True
                and isinstance(model, (int, float)))
    if not model_ok:
        excluded["model_estimate_unbound"] += 1
    return {"t": t, "test_case": r.get("capsule"), "cycles": cycles, "source": source, "case_digest": digest,
            "useful_macs": macs if roof_ok else None, "peak_macs_per_cycle": peak if roof_ok else None,
            "floor_cycles": floor if roof_ok else None, "model_cycles": model if model_ok else None}


def read_selfcheck_latency(workspace) -> list[dict]:
    ch = os.path.join(str(workspace or ""), ".qa_channel")
    out = []
    for p in glob.glob(os.path.join(ch, "req_*.json")):
        rid = Path(p).stem[len("req_"):]
        req, resp = jload(p) or {}, jload(os.path.join(ch, "resp_" + rid + ".json")) or {}
        t0 = (req.get("requested_at_unix_ns") or 0) / 1e9 or None
        t1 = (resp.get("graded_at_unix_ns") or 0) / 1e9 or None
        if t0 and t1 and t1 > t0:
            out.append({"t0": t0, "t1": t1})
    return out


# ------------------------------------------------------------------------------------- test cases


def _yaml_fields(path) -> dict[str, Any]:
    try:
        import yaml

        d = yaml.safe_load(open(path)) or {}
    except Exception:  # noqa: BLE001
        return {}
    return {"kind": d.get("kind"), "op": (d.get("operation") or {}).get("op"),
            "tiers": list(d.get("required_oracle_tiers") or [])}


def read_test_cases(run_dirs: list) -> tuple[dict[str, dict], set[str]]:
    """Test-case metadata from the first run dir carrying the corpus, plus calibration members."""
    meta: dict[str, dict] = {}
    for rd in run_dirs:
        base = os.path.join(str(rd or ""), "private_corpus_input")
        for sub, cohort in (("public", "functional"), (os.path.join("scalability", "cases"), "scalability")):
            for p in glob.glob(os.path.join(base, sub, "*", "capsule.yaml")):
                name = Path(p).parent.name
                if name not in meta:
                    meta[name] = dict(_yaml_fields(p), cohort=cohort)
        if meta:
            break
    calibration: set[str] = set()
    for rd in run_dirs:
        jc = jload(os.path.join(str(rd or ""), "joint_campaign.json")) or {}
        prof = ((jc.get("policy") or {}).get("native_calibration") or {}).get("path")
        members = (jload(prof) or {}).get("members") if prof else None
        if isinstance(members, dict):
            calibration = set(members)
            break
    return meta, calibration


# --------------------------------------------------------------------------------------- campaign


def read_campaign(m: Manifest) -> dict[str, Any]:
    attempts, threads, grades, events, jobs, perf, latency = [], [], [], [], [], [], []
    core_bins, excluded = {}, {}
    for a in m.attempts:
        bins = Bins()
        n_stage, finish, att_grades, ev, n_cpu = 0, {}, [], [], 0
        if _isdir(a.run_dir):
            n_stage, finish = read_stage_timing(a.run_dir, bins)
            att_grades, ev = read_grades(a.run_dir, finish)
            n_cpu = read_agent_cpu(a.run_dir, bins)
        if _isdir(a.workspace):
            att_jobs, sev, prow, exc = read_rtl(a.workspace, bins, m.rtl_engine)
            lat = read_selfcheck_latency(a.workspace)
        else:
            att_jobs, sev, prow, exc, lat = [], [], [], {}, []
        att_threads = read_threads(a.agent_home)
        for rows in (att_threads, att_grades, ev, sev, att_jobs, prow, lat):
            for r in rows:
                r["attempt"] = a.id
        stamps = [x for t in att_threads for iv in t["turns"] for x in iv]
        stamps += [x for g in att_grades for x in (g["updated"],) if x]
        stamps += [x for j in att_jobs for x in (j["started"], j["finished"]) if x]
        if bins.data:
            stamps += [min(bins.data) * BIN_S, (max(bins.data) + 1) * BIN_S]
        lead = [t for t in att_threads if t["role"] == "lead" and t["started"]]
        attempts.append({
            "id": a.id, "continuation": a.continuation, "live": a.live,
            "launch": launch_time(a.launch_log), "agent_start": min((t["started"] for t in lead), default=None),
            "first_record": min(stamps) if stamps else None, "last_record": max(stamps) if stamps else None,
            "accounting": accounting(a.accounting_log, a.accounting_entry),
            "counts": {"stage_timing_files": n_stage, "grades": len(att_grades), "l2_rows": len(ev), "rtl_jobs": len(att_jobs),
                       "l3_rows": len(sev), "agent_threads": len(att_threads), "cpu_samples": n_cpu},
        })
        core_bins[a.id] = bins.dump()
        excluded[a.id] = exc
        threads += att_threads
        grades += att_grades
        events += ev + sev
        jobs += att_jobs
        perf += prow
        latency += lat
    meta, calibration = read_test_cases([a.run_dir for a in reversed(m.attempts)])
    names = set(meta) | {e["test_case"] for e in events if e.get("test_case")}
    cases = {}
    for name in sorted(names):
        info = meta.get(name, {})
        cases[name] = {"cohort": cohort_of(m.cohorts, name, info, calibration) if m.cohorts else None,
                       "l3_required": "L3" in (info.get("tiers") or ["L3"])}
    return {
        "schema": SCHEMA, "campaign": m.campaign, "target": m.target, "reserved_cores": m.reserved_cores,
        "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(), "bin_s": BIN_S,
        "attempts": attempts, "threads": threads, "grades": grades, "events": events, "rtl_jobs": jobs,
        "perf": perf, "perf_excluded": excluded, "selfcheck_latency": latency, "core_bins": core_bins,
        "labels": dict(m.labels), "test_cases": cases, "cohorts": [{"id": c["id"], "label": c.get("label", c["id"])} for c in m.cohorts],
    }
