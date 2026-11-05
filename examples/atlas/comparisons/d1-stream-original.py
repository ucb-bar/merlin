"""Overlap original47 projection checks with an immutable, still-running native capture."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
MERLIN_ROOT = HERE.parents[2]
ATLAS_ROOT = Path(os.environ["ATLAS_ROOT"]).absolute()
ROOT = Path(os.environ["TINY_WORK_ROOT"]).absolute()
ATLAS_HELPERS = ATLAS_ROOT / "tools/tiny_full"
MERLIN_HELPERS = MERLIN_ROOT / "examples/atlas/comparisons"
MODEL_ROOT = Path(os.environ["ATLAS_NPU_MODEL_ROOT"]).absolute()
spec = importlib.util.spec_from_file_location("d1_stream_worker", MERLIN_HELPERS / "d1-stream-worker.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def discover(root):
    """Only calls/*/record.json is traversed, with names screened before access."""
    calls = w.safe(root / "calls")
    if not calls.exists():
        return [], []
    records, partial = [], []
    with os.scandir(calls) as entries:
        inventory = sorted(entries, key=lambda entry: entry.name)
    for entry in inventory:
        if entry.is_symlink():
            raise ValueError("symbolic link in capture calls inventory")
        if not entry.is_dir(follow_symlinks=False):
            continue
        path = w.safe(Path(entry.path) / "record.json")
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            partial.append(str(path))
            continue
        if not isinstance(record, dict):
            raise ValueError("complete record is not an object: " + str(path))
        if (
            record.get("model_commit") == w.PIN
            and record.get("accelerated") is False
            and record.get("native_dispatch_enabled") is False
        ):
            continue
        if (
            record.get("kind") != "projection"
            or record.get("model_commit") != w.NATIVE
            or record.get("accelerated") is not True
        ):
            raise ValueError("unexpected complete capture record: " + str(path))
        records.append(path)
    return sorted(records), sorted(partial)


def output_path(out, record):
    return w.safe(out / "records" / (record.parent.name + ".json"))


def freeze_sources(out, resume):
    path = w.safe(out / "source-freeze.json")
    current = w.source_state()
    if path.exists():
        if not resume:
            raise ValueError("stream output already initialized; use --resume")
        frozen = json.loads(path.read_bytes())
        w.verify_freeze(frozen)
        if current != frozen:
            raise ValueError("source freeze differs")
        return frozen
    w.save(path, current)
    return current


def map_bundle(args, out, frozen):
    """Use the existing final-bundle audit before admitting any reuse mapping."""
    spec = importlib.util.spec_from_file_location("d1_stream_final_qualify", w.safe(ATLAS_HELPERS / "d1-qualify.py"))
    qualify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualify)
    bundle = qualify.Bundle(args.map_assets, args.smoke)
    model = w.load_original().model
    model_sources, harness_sources = model.source_binding(), qualify.harness()
    candidates = {}
    records, partial = discover(args.root)
    if partial:
        raise ValueError("cannot map final bundle with incomplete capture records")
    for record_path in records:
        result_path = output_path(out, record_path)
        if result_path.exists():
            result = w.audit(args.root, record_path, result_path, frozen)
            key = (result["program_sha256"], result["arena_initial"]["sha256"], result["arena_expected"]["sha256"])
            candidates.setdefault(key, []).append(dict(root=str(args.root), record=str(result_path)))
    # Existing original47 attention and recovered projections remain subject to
    # the identical qualifier provenance audit, rather than being retagged.
    with os.scandir(w.safe(args.root / "calls")) as entries:
        inventory = sorted(entries, key=lambda entry: entry.name)
    for entry in inventory:
        if entry.is_symlink():
            raise ValueError("symbolic link in final capture inventory")
        if not entry.is_dir(follow_symlinks=False):
            continue
        path = w.safe(Path(entry.path) / "record.json")
        if not path.exists():
            continue
        result = json.loads(path.read_bytes())
        if (
            result.get("model_commit") == w.PIN
            and result.get("accelerated") is False
            and result.get("native_dispatch_enabled") is False
        ):
            key = (result["program_sha256"], result["arena_initial"]["sha256"], result["arena_expected"]["sha256"])
            candidates.setdefault(key, []).append(dict(root=str(args.root), record=str(path)))
    mapping = {}
    for item in bundle.launches:
        payloads = bundle.payloads(item)
        key = (w.sha(payloads[1]), w.sha(payloads[2]), w.sha(payloads[4]))
        selected = candidates.get(key, [])
        if selected:
            qualify.from_capture(selected[0], bundle, item, model, model_sources, harness_sources)
            mapping[str(item["index"])] = selected[0]
    bundle.pins.verify()
    w.verify_freeze(frozen)
    w.save(out / "reuse-records.json", mapping)
    print(
        json.dumps(
            dict(
                status="FINAL_BUNDLE_MAPPING_AUDITED",
                mapped=len(mapping),
                required=len(bundle.launches),
                mapping=str(out / "reuse-records.json"),
                overall_acceptance=False,
            )
        ),
        flush=True,
    )
    return 0


def main(args):
    args.root, out = w.safe(args.root), w.safe(args.out)
    if not out.is_relative_to(ROOT / "d1-stream-original") or out == args.root or out.is_relative_to(args.root):
        raise ValueError("outputs must be beneath TINY_WORK_ROOT/d1-stream-original and outside capture")
    if (
        not 1 <= args.workers <= 16
        or args.poll_seconds < 1
        or args.poll_seconds > 30
        or args.expected_native_records < 1
    ):
        raise ValueError("workers must be 1..16 and polling must be 1..30 seconds")
    out.mkdir(parents=True, exist_ok=True)
    w.safe(out / "records").mkdir(exist_ok=True)
    w.safe(out / "logs").mkdir(exist_ok=True)
    frozen = freeze_sources(out, args.resume)
    if args.map_assets:
        return map_bundle(args, out, frozen)
    completed, identities, failed, running = {}, {}, {}, {}
    resumed = []
    discovered, partial = discover(args.root)
    if args.record:
        wanted = w.safe(args.record)
        if wanted not in discovered:
            raise ValueError("--record must identify a completed native projection")
        discovered = [wanted]
    for path in discovered:
        target = output_path(out, path)
        if target.exists():
            if not args.resume:
                raise ValueError("existing execution receipt requires --resume")
            result = w.audit(args.root, path, target, frozen)
            completed[str(path)] = str(target)
            identities[str(path)] = result["source_capture_record_sha256"]
            resumed.append(str(path))
    failures_path = out / "failures.json"
    if failures_path.exists() and args.resume:
        raise ValueError("prior numerical/protocol failure requires investigation and a fresh output directory")

    def progress(status):
        w.save(
            out / "progress.json",
            dict(
                schema="atlas.d1.stream_original_progress.v1",
                status=status,
                speculative_capture_input_numerical_evidence=True,
                final_full_source_capture_replay_closure_required=True,
                overall_acceptance=False,
                model_pin=w.PIN,
                numerics="rtl",
                accelerated=False,
                native_dispatch_enabled=False,
                capture_root=str(args.root),
                completed=completed,
                resumed=resumed,
                failed=failed,
                running=sorted(str(p) for p in running.values()),
                discovered=len(discovered),
                incomplete_records=partial,
                source_freeze=str(out / "source-freeze.json"),
                command=sys.argv,
                updated_unix=time.time(),
            ),
        )

    def execute(path):
        command = [
            sys.executable,
            "-B",
            str(MERLIN_HELPERS / "d1-stream-worker.py"),
            "--root",
            str(args.root),
            "--record",
            str(path),
            "--out",
            str(output_path(out, path)),
            "--freeze",
            str(out / "source-freeze.json"),
        ]
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
            ATLAS_NPU_MODEL_NATIVE="0",
            NPU_MODEL_RTL_NATIVE_MXU0="0",
            TMPDIR=str(out),
        )
        environment.pop("PYTHONPATH", None)
        environment.pop("NPU_MODEL_RTL_NATIVE_BUILD_DIR", None)
        with w.safe(out / "logs" / (path.parent.name + ".log")).open("w") as log:
            status = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            raise ValueError(
                "isolated original47 worker failed; inspect " + str(out / "logs" / (path.parent.name + ".log"))
            )
        return w.audit(args.root, path, output_path(out, path), frozen)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            w.verify_freeze(frozen)
            for path in discovered:
                key = str(path)
                digest = w.sha(path.read_bytes())
                if key in identities and identities[key] != digest:
                    raise ValueError("completed capture record changed: " + key)
                if (
                    key not in completed
                    and key not in failed
                    and path not in running.values()
                    and len(running) < args.workers
                ):
                    identities[key] = digest
                    running[pool.submit(execute, path)] = path
            progress("RUNNING_SPECULATIVE_NUMERICS")
            if running:
                ready, _ = wait(running, timeout=args.poll_seconds, return_when=FIRST_COMPLETED)
                for future in ready:
                    path = running.pop(future)
                    try:
                        future.result()
                        completed[str(path)] = str(output_path(out, path))
                        print(str(path.parent.name) + " PASS original47", flush=True)
                    except Exception as error:
                        failed[str(path)] = dict(error=str(error), traceback=traceback.format_exc())
                        w.save(failures_path, failed)
                if failed:
                    progress("FAILED")
                    # The context manager drains already-running isolated workers;
                    # no numerical failure is hidden or retried automatically.
                    break
            else:
                if args.record or args.once:
                    break
                time.sleep(args.poll_seconds)
            if not args.record and not args.once:
                discovered, partial = discover(args.root)
            if args.once and not running and all(str(p) in completed for p in discovered):
                break
            if len(discovered) > args.expected_native_records:
                raise ValueError("native projection inventory exceeds requested final count")
            if (
                not args.record
                and not args.once
                and len(completed) == args.expected_native_records
                and len(discovered) == args.expected_native_records
                and not running
                and not partial
            ):
                break
    w.verify_freeze(frozen)
    progress("FAILED" if failed else "PASS_DISCOVERED_SPECULATIVE_NUMERICS")
    receipt = json.loads((out / "progress.json").read_bytes())
    w.save(out / "receipt.json", receipt)
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "d1-full-native-capture"))
    parser.add_argument("--out", default=str(ROOT / "d1-stream-original"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument(
        "--expected-native-records",
        type=int,
        default=836,
        help="Stop after this many actual original47 projection receipts, default full capture836",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--once", action="store_true", help="Drain the currently completed capture records, then exit")
    parser.add_argument("--record", help="Execute exactly one completed native projection for a real smoke check")
    parser.add_argument(
        "--map-assets", help="Audit final bundle joins and write --reuse-records mapping without executing"
    )
    parser.add_argument("--smoke", action="store_true", help="Permit a small final bundle only for mapping tests")
    raise SystemExit(main(parser.parse_args()))
