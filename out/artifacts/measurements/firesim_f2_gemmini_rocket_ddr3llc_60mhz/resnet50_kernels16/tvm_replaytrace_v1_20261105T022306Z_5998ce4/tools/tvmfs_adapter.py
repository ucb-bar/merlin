#!/usr/bin/env python3
"""gemmini_firesim_probe_request_v4 adapter for tvmfs F2 hosts.

usage: tvmfs_adapter.py --hosts hosts.json --key KEY --request R --result S --slot ID
Streams the ELF to the F2 host's forced command `run <wall> <slot>`, enforces the request's
absolute deadline remotely (timeout on the driver) and locally, and only reports
remote_stopped after probing that no driver is attached to the physical slot.
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            d.update(b)
    return d.hexdigest()


def ssh(a, ip, *cmd, stdin=None, stdout=None, stderr=None):
    argv = ["ssh", "-i", a.key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={Path.home()}/.ssh/f2run_known_hosts", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "LogLevel=ERROR", f"ubuntu@{ip}", " ".join(cmd)]
    return subprocess.Popen(argv, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    for n in ("hosts", "key", "request", "result", "slot"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--test-remote-wall", type=int, default=None, help="qualification only: override the remote wall to exercise the local-stop path")
    ap.add_argument("--test-local-cutoff", type=float, default=1.0, help="qualification only: seconds before the deadline at which the local end gives up")
    a = ap.parse_args()
    req = json.loads(Path(a.request).read_text())
    out = Path(a.result).resolve().parent
    host_id, _, slot_idx = a.slot.partition("s")  # slot ids look like h0s1
    ip = json.loads(Path(a.hosts).read_text())[host_id]["ip"]
    elf = Path(req["elf"]["path"])
    elf_sha = req["elf"]["sha256"]
    res = dict(schema="gemmini_firesim_probe_result_v4", elf_sha256=elf_sha, status="failed", guest_exit=None,
               wall_limit_enforced=False, remote_stopped=False, slot=a.slot, host=ip)
    if sha(elf) != elf_sha:
        res["error"] = "ELF changed after request"
        Path(a.result).write_text(json.dumps(res, indent=2))
        return 2
    remaining = req["deadline_epoch"] - time.time()
    wall = a.test_remote_wall or "@%d" % int(req["deadline_epoch"])  # reserve for upload and teardown inside the request deadline
    t0 = time.monotonic()
    raw, err = out / "transcript.raw", out / "remote.stderr"
    with elf.open("rb") as fin, raw.open("wb") as fo, err.open("wb") as fe:
        p = ssh(a, ip, "run", str(wall), slot_idx, stdin=fin, stdout=fo, stderr=fe)
        try:
            code = p.wait(timeout=max(1, req["deadline_epoch"] - time.time() - a.test_local_cutoff))
            res["wall_limit_enforced"] = True
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            code = 124
            res["status"] = "timeout"
            res["wall_limit_enforced"] = True
    res["occupied_seconds"] = time.monotonic() - t0
    stopped = False  # confirm the physical slot has no driver; force-stop the slot if the local ssh end left one
    for _ in range(3):
        q = ssh(a, ip, "probe", slot_idx, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if q.communicate(timeout=30)[0].decode().strip() == "stopped":
            stopped = True
            break
        ssh(a, ip, "reset", slot_idx, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).communicate(timeout=120)
        time.sleep(2)
    res["remote_stopped"] = stopped
    tr = out / "transcript.txt"
    tr.write_bytes(raw.read_bytes().replace(b"\r\n", b"\n"))
    boot = {}
    for line in err.read_text(errors="replace").splitlines():
        if line.startswith("[boot] "):
            boot.update(kv.split("=", 1) for kv in line[7:].split())
    boot["host_ip"] = ip
    bev = out / "boot_evidence.json"
    bev.write_text(json.dumps(boot, indent=2))
    res.update(guest_exit=code, transcript=dict(path=str(tr), sha256=sha(tr), bytes=tr.stat().st_size),
               boot_evidence=dict(path=str(bev), sha256=sha(bev), bytes=bev.stat().st_size),
               hardware_sha256=req["hardware"].get("sha256"))
    hw_path = req["hardware"].get("path")
    want_agfi = json.loads(Path(hw_path).read_text()).get("agfi") if hw_path else None
    res["hardware_agfi_verified"] = bool(want_agfi) and boot.get("agfi") == want_agfi
    if code == 0 and boot.get("elf_sha256") == elf_sha and stopped and res["hardware_agfi_verified"]:
        res["status"] = "passed"
    elif code in (124, 137):
        res["status"] = "timeout"
    elif res["status"] != "timeout":
        res["status"] = "failed"
    if boot.get("elf_sha256") != elf_sha:
        res["error"] = "boot evidence missing or ELF differs"
    elif not res["hardware_agfi_verified"]:
        res["error"] = "booted AGFI differs from the hardware receipt"
    Path(a.result).write_text(json.dumps(res, indent=2))
    return 0 if stopped else 3


if __name__ == "__main__":
    sys.exit(main())
