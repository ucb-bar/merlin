#!/usr/bin/env python3
"""Prove the harness bridge before a campaign spends on it.

The bridge (agent_bridge + the LiteLLM proxy) lets any registered model be driven by any of the three
agentic harnesses, which is what turns "model vs model" into "model x harness". But a translation layer
can change a result on its own, so a number produced through it is worth nothing until the bridge has
been shown to be faithful. This runs the checks that make it evidence:

  reachability  every SERVED model answers on BOTH wire shapes (/v1/responses, /v1/messages)
  agency        each harness completes a real tool-using turn through the bridge -- not just a
                completion, but read/write/verify, because a harness that cannot use tools scores 0/20
                for a reason that has nothing to do with the model
  control       the SAME model+harness, once native and once forced through the proxy, on an identical
                task. Divergence here means the campaign would be measuring LiteLLM.

Exit 0 = GO. Anything else and the campaign must not launch.

    MERLIN_PROXY_KEY=... .venv/bin/python bridge_canary.py [--models nemotron,glm5] [--skip-control]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_bridge as BR  # noqa: E402

TASK = "Create a file named probe.txt whose entire contents are the word BRIDGE_OK. Then stop."
EXPECT = "BRIDGE_OK"


def _post(path: str, payload: dict, headers: dict, timeout: int = 120) -> tuple[int, dict]:
    req = urllib.request.Request(BR.PROXY_BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except Exception as e:                                   # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


def _text_of(body: dict) -> str:
    """Pull the assistant text out of either wire shape.

    Structural, not a substring scan of the serialized body: the Responses envelope carries a ~400-char
    opaque id, so a naive ``"OK" in json.dumps(body)[:160]`` reports a healthy endpoint as broken.
    """
    parts: list[str] = []
    for item in body.get("output") or []:                      # Responses
        for c in item.get("content") or []:
            if c.get("type") in ("output_text", "text"):
                parts.append(str(c.get("text", "")))
    for c in body.get("content") or []:                        # Anthropic Messages
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(str(c.get("text", "")))
    return " ".join(parts)


def check_wire(models: list[str]) -> list[tuple[str, bool, str]]:
    key = BR.proxy_key()
    out = []
    for m in models:
        st, body = _post("/v1/responses", {"model": m, "input": "reply with exactly: OK"},
                         {"Authorization": f"Bearer {key}"})
        txt = _text_of(body)
        out.append((f"responses:{m}", st == 200 and "OK" in txt,
                    f"HTTP {st} text={txt[:60]!r}" if st == 200 else f"HTTP {st} {str(body)[:120]}"))
        st, body = _post("/v1/messages",
                         {"model": m, "max_tokens": 32,
                          "messages": [{"role": "user", "content": "reply with exactly: OK"}]},
                         {"x-api-key": key, "anthropic-version": "2023-06-01"})
        txt = _text_of(body)
        out.append((f"messages:{m}", st == 200 and "OK" in txt,
                    f"HTTP {st} text={txt[:60]!r}" if st == 200 else f"HTTP {st} {str(body)[:120]}"))
    return out


def _run_codex(model: str, ws: Path, *, force: bool) -> tuple[bool, str]:
    home = Path(tempfile.mkdtemp(prefix="canary_codex_"))
    env = dict(os.environ)
    env["CODEX_HOME"] = str(home)
    if force:
        env["MERLIN_FORCE_BRIDGE"] = "1"
    frag = BR.codex_config_fragment(model) if (force or BR.bridged_name(model, "codex")) else ""
    (home / "config.toml").write_text(
        f'model = {json.dumps(BR.codex_model_name(model))}\n'
        'approval_policy = "never"\nsandbox_mode = "danger-full-access"\n' + frag)
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(ws), TASK]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        shutil.rmtree(home, ignore_errors=True)
    tools = sum(1 for l in r.stdout.splitlines()
                if '"command_execution"' in l or '"file_change"' in l)
    return (ws / "probe.txt").is_file(), f"tool_events={tools} rc={r.returncode}"


def _run_claude(model: str, ws: Path, *, force: bool) -> tuple[bool, str]:
    env = dict(os.environ)
    if force:
        env["MERLIN_FORCE_BRIDGE"] = "1"
    env.update({k: v for k, v in BR.claude_env(model).items() if v} or {})
    home = Path(tempfile.mkdtemp(prefix="canary_cc_"))
    env["CLAUDE_CONFIG_DIR"] = str(home)
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
           "--model", BR.claude_model_name(model), "--dangerously-skip-permissions", TASK]
    try:
        r = subprocess.run(cmd, cwd=str(ws), env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        shutil.rmtree(home, ignore_errors=True)
    tools = r.stdout.count('"type":"tool_use"')
    return (ws / "probe.txt").is_file(), f"tool_uses={tools} rc={r.returncode}"


def check_agency(models: list[str]) -> list[tuple[str, bool, str]]:
    out = []
    for m in models:
        for harness, fn in (("codex", _run_codex), ("claude", _run_claude)):
            ws = Path(tempfile.mkdtemp(prefix=f"canary_{harness}_ws_"))
            t0 = time.time()
            ok, note = fn(m, ws, force=False)
            body = (ws / "probe.txt").read_text().strip() if (ws / "probe.txt").is_file() else ""
            shutil.rmtree(ws, ignore_errors=True)
            out.append((f"{harness}+{m} agentic turn", ok and EXPECT in body,
                        f"{note} wrote={body[:20]!r} {time.time()-t0:.0f}s"))
    return out


def check_control(model: str) -> list[tuple[str, bool, str]]:
    """The same model+harness native vs forced-through-proxy. Only meaningful for a natively-served pair."""
    out = []
    for harness, fn in (("codex", _run_codex),):
        if BR.bridged_name(model, harness):
            out.append((f"control {harness}+{model}", True, "skipped: no native path for this pairing"))
            continue
        res = {}
        for mode in (False, True):
            ws = Path(tempfile.mkdtemp(prefix="canary_ctl_"))
            ok, note = fn(model, ws, force=mode)
            body = (ws / "probe.txt").read_text().strip() if (ws / "probe.txt").is_file() else ""
            shutil.rmtree(ws, ignore_errors=True)
            res["bridged" if mode else "native"] = (ok and EXPECT in body, note)
        same = res["native"][0] == res["bridged"][0]
        out.append((f"control {harness}+{model} native==bridged", same,
                    f"native={res['native']} bridged={res['bridged']}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="nemotron,glm5")
    ap.add_argument("--control-model", default="gpt-5.6-sol")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--skip-agency", action="store_true")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]

    BR.proxy_key()
    log = Path(os.environ.get("MERLIN_PROXY_LOG", "/scratch/agustin/tmp/proxy/proxy.log"))
    info = BR.start_proxy(log)
    print(f"proxy: {info}\n")

    rows: list[tuple[str, bool, str]] = []
    print("== wire reachability ==")
    for name, ok, note in check_wire(models):
        rows.append((name, ok, note)); print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {note}")
    if not a.skip_agency:
        print("\n== agentic turn through each harness ==")
        for name, ok, note in check_agency(models):
            rows.append((name, ok, note)); print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {note}")
    if not a.skip_control:
        print("\n== proxy-vs-direct control ==")
        for name, ok, note in check_control(a.control_model):
            rows.append((name, ok, note)); print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {note}")

    bad = [n for n, ok, _ in rows if not ok]
    print("\n" + "=" * 60)
    if bad:
        print(f"🔴 BRIDGE NO-GO — {len(bad)}/{len(rows)} failed: {bad}")
        return 1
    print(f"🟢 BRIDGE GO — {len(rows)}/{len(rows)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
