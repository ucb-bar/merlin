"""Qualify every exported launch on unmodified model 47 with strict resume audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
ATLAS_ROOT = HERE.parents[1]
MERLIN_ROOT = Path(os.environ['MERLIN_ROOT']).absolute()
ROOT = Path(os.environ['TINY_WORK_ROOT']).absolute()
ATLAS_HELPERS = ATLAS_ROOT/'tools/tiny_full'
MERLIN_HELPERS = MERLIN_ROOT/'examples/atlas/comparisons'
MODEL_ROOT = Path(os.environ['ATLAS_NPU_MODEL_ROOT']).absolute()
sys.path.insert(0, str(ATLAS_ROOT / 'tools'))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked

MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"


class NotOriginalCapture(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(value):
    path = checked(Path(value).absolute())
    return path


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_model():
    os.environ["ATLAS_NPU_MODEL_ROOT"] = str(MODEL_ROOT)
    os.environ["ATLAS_NPU_MODEL_PIN"] = MODEL_PIN
    os.environ["ATLAS_NPU_MODEL_NATIVE"] = "0"
    os.environ.pop("NPU_MODEL_RTL_NATIVE_MXU0", None)
    os.environ.pop("NPU_MODEL_RTL_NATIVE_BUILD_DIR", None)
    spec = importlib.util.spec_from_file_location("d1_qualification_model", ATLAS_HELPERS / 'd1-model.py')
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    return model


class Pins:
    def __init__(self):
        self.values = {}

    def read(self, path):
        path = local(path)
        raw = path.read_bytes()
        digest = sha(raw)
        if self.values.setdefault(str(path), digest) != digest:
            raise ValueError("input changed during qualification: " + str(path))
        return raw

    def bind(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError("missing source closure")
        for path, digest in values.items():
            if sha(self.read(path)) != digest:
                raise ValueError("source pin differs: " + path)

    def verify(self):
        for path in list(self.values):
            self.read(path)


class Bundle:
    def __init__(self, assets, smoke=False):
        self.root, self.pins = local(assets), Pins()
        self.manifest = self.root / "replay.json"
        self.raw = self.pins.read(self.manifest)
        self.launches = json.loads(self.raw)["launches"]
        if not self.launches or [row["index"] for row in self.launches] != list(range(len(self.launches))):
            raise ValueError("ordered nonempty launch inventory required")
        if not smoke:
            attention = [row for row in self.launches if row["kind"] == "attention"]
            if (len(self.launches) != 906 or len(attention) != 66
                    or sum(row["kind"] == "projection" for row in self.launches) != 840
                    or any(sum(row["step"] == step for row in self.launches) != 302 for step in range(3))
                    or {(row["step"], row["layer"]) for row in attention} != {(s, l) for s in range(3) for l in range(22)}):
                raise ValueError("full 906-launch, 840-projection, 66-attention inventory required")

    def member(self, name):
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("bundle member must be relative")
        return self.pins.read(self.root / member)

    def payloads(self, item):
        source, binary, initial, expected, final = [self.member(item[key]) for key in
                                                   ("source", "original", "initial", "expected", "expected_arena")]
        if (sha(binary) != item["program_sha256"] or sha(initial) != item["initial_sha256"]
                or sha(expected) != item["expected_sha256"] or len(binary) % 4
                or binary[-4:] != bytes.fromhex("73000000")):
            raise ValueError("manifest source, program, or arena payload identity differs")
        offset, extent = item["output_offset"], len(expected)
        dma = item["requested_dma"]
        if (item["dram_base"] != 0x90000000 or set(dma) != {"load_bytes", "store_bytes"}
                or any(type(value) is not int or value <= 0 or value % 32 for value in dma.values())
                or not 0 <= offset <= len(initial) or extent > len(initial) - offset or len(final) != len(initial)):
            raise ValueError("invalid arena or output extent")
        wanted = bytearray(initial)
        cursor = 0
        spans = item.get("written_spans", [dict(offset=0, bytes=extent)])
        for span in spans:
            start, size = span["offset"], span["bytes"]
            if start < cursor or size <= 0 or start + size > extent:
                raise ValueError("invalid output written spans")
            wanted[offset + start:offset + start + size] = expected[start:start + size]
            cursor = start + size
        if (final != bytes(wanted) or final[offset:offset + extent] != expected
                or sum(span["bytes"] for span in spans) != dma["store_bytes"]
                or item.get("total_dram_bytes", len(initial)) != len(initial)):
            raise ValueError("expected full arena changed an input, guard, or strided hole")
        return source, binary, initial, expected, final


def harness():
    paths = {Path(__file__).absolute(), ATLAS_HELPERS / 'd1-model.py'}
    tools = ATLAS_ROOT / 'tools'
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).absolute()
            if path.suffix == ".py" and path.is_relative_to(tools):
                paths.add(local(path))
    return {str(path): sha(local(path).read_bytes()) for path in sorted(paths)}


def identity(item, payloads):
    source, binary, initial, expected, final = payloads
    return dict(index=item["index"], kind=item["kind"], program_source_sha256=sha(source),
                program_sha256=sha(binary), initial_sha256=sha(initial), output_sha256=sha(expected),
                final_arena_sha256=sha(final), output_offset=item["output_offset"], output_bytes=len(expected),
                arena_bytes=len(initial), dram_base=item["dram_base"], requested_dma=item["requested_dma"])


def execution_matches(execution, item, binary):
    return (execution.get("program_sha256") == sha(binary) and execution.get("program_words") == len(binary) // 4
            and execution.get("halt_pc") == len(binary) // 4 - 1 and execution.get("instructions", 0) > 0
            and all(execution.get("observed", {}).get(key) == value for key, value in item["requested_dma"].items()))


def current_report(path, bundle, item, model_sources, harness_sources):
    report = json.loads(local(path).read_text())
    payloads = bundle.payloads(item)
    expected = identity(item, payloads)
    if (report.get("status") != "PASS" or report.get("exact") is not True
            or report.get("full_arena_exact") is not True or report.get("accelerated") is not False
            or report.get("native_dispatch_enabled") is not False or report.get("model_pin") != MODEL_PIN
            or report.get("numerics") != "rtl" or report.get("model_root") != str(MODEL_ROOT)
            or any(report.get(key) != value for key, value in expected.items())
            or report.get("model_sources_before") != model_sources or report.get("model_sources_after") != model_sources
            or report.get("harness_sources_before") != harness_sources or report.get("harness_sources_after") != harness_sources
            or not execution_matches(report.get("execution", {}), item, payloads[1])):
        raise ValueError("resume receipt identity or exactness differs: " + str(path))
    bundle.pins.bind(report.get("reused_capture_harness_sources", harness_sources))
    if report.get("reused_capture_record"):
        if sha(bundle.pins.read(report["reused_capture_record"])) != report["reused_capture_record_sha256"]:
            raise ValueError("reused capture record changed")
        bundle.pins.bind(report.get("reused_capture_payload_sha256", {}))
    return report


def from_capture(mapping, bundle, item, model, model_sources, harness_sources):
    record_path, root = local(mapping["record"]), local(mapping["root"])
    raw = bundle.pins.read(record_path)
    record = json.loads(raw)
    if (record.get('schema') not in {'atlas.d1.stream_original_execution.v1', 'atlas.d1.original_capture.v1'}
            or record.get('exact') is not True or record.get('full_arena_exact') is not True):
        raise NotOriginalCapture('capture schema or exactness is not admitted')
    if (record.get("model_commit") != MODEL_PIN or record.get("accelerated") is not False
            or record.get("native_dispatch_enabled") is not False):
        raise NotOriginalCapture("capture requires original-model execution")
    if (record.get("numerics") != "rtl"
            or record.get("model_sources_before") != model_sources or record.get("model_sources_after") != model_sources
            or record.get("harness_sources_before") != record.get("harness_sources_after")):
        raise ValueError("capture record lacks authentic unmodified-model execution")
    closure = record.get("harness_sources_before", {})
    if closure.get(str(ATLAS_HELPERS / 'd1-model.py')) != harness_sources[str(ATLAS_HELPERS / 'd1-model.py')]:
        raise ValueError("capture model harness differs")
    if str(MERLIN_HELPERS / 'd1-worker.py') not in closure:
        raise ValueError("capture worker source closure missing")
    bundle.pins.bind(closure)
    payloads = bundle.payloads(item)
    captured_payloads = {}
    for key, expected in zip(("program_source", "program_binary", "arena_initial", "output_tiled", "arena_expected"), payloads):
        desc = record[key]
        relative = Path(desc["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("capture payload escapes root")
        actual = bundle.pins.read(root / relative)
        captured_payloads[str(local(root / relative))] = sha(actual)
        if actual != expected or desc["bytes"] != len(actual) or desc["sha256"] != sha(actual):
            raise ValueError("capture source, program, or full arena differs from final bundle")
    source, binary, initial, expected, final = payloads
    program, assembled_sha = model.assemble(source.decode())
    if assembled_sha != sha(binary) or b"".join(word.to_bytes(4, "little") for word in program.assemble()) != binary:
        raise ValueError("capture assembled source differs from final binary")
    if (record["dram_base"] != item["dram_base"] or record["output_offset"] != item["output_offset"]
            or record["output_bytes"] != len(expected) or record["requested_dma"] != item["requested_dma"]
            or not execution_matches(record.get("execution", {}), item, binary)):
        raise ValueError("capture execution DMA, output extent, program, or halt differs")
    return dict(status="PASS", exact=True, full_arena_exact=True, accelerated=False, native_dispatch_enabled=False,
                model_pin=MODEL_PIN, model_root=str(MODEL_ROOT), numerics="rtl", **identity(item, payloads),
                execution=record["execution"], model_sources_before=model_sources, model_sources_after=model_sources,
                harness_sources_before=harness_sources, harness_sources_after=harness_sources,
                reused_capture_harness_sources=closure, reused_capture_record=str(record_path),
                reused_capture_payload_sha256=captured_payloads,
                reused_capture_record_sha256=sha(raw), method="reused_original_capture_execution", new_timing=False)


def worker(args):
    bundle, out, model = Bundle(args.assets, args.smoke), local(args.out), load_model()
    if not 0 <= args.index < len(bundle.launches):
        raise ValueError("launch index outside inventory")
    item = bundle.launches[args.index]
    out.mkdir(parents=True, exist_ok=True)
    report, started = dict(status="FAILED", index=args.index), time.monotonic()
    try:
        before, harness_before = model.source_binding(), harness()
        payloads = bundle.payloads(item)
        source, binary, initial, expected, final = payloads
        program, digest = model.assemble(source.decode())
        if digest != sha(binary) or b"".join(word.to_bytes(4, "little") for word in program.assemble()) != binary:
            raise ValueError("assembled source differs from exact exported binary")
        coverage = getattr(args, 'coverage_only', False)
        if coverage:
            offset = item['output_offset']
            poisoned = bytearray(initial)
            poisoned[offset:offset + len(expected)] = bytes(value ^ 0xff for value in expected)
            wanted = bytearray(poisoned)
            for span in item.get('written_spans', [dict(offset=0, bytes=len(expected))]):
                start, size = span['offset'], span['bytes']
                wanted[offset + start:offset + start + size] = expected[start:start + size]
            initial, final = bytes(poisoned), bytes(wanted)
            expected = final[offset:offset + len(expected)]
        executed = model.execute(source.decode(), initial, item["output_offset"], len(expected), dram_base=item["dram_base"],
                                 max_instructions=20_000_000)
        if executed.arena != final or executed.output != expected:
            differing = [i for i, (a, b) in enumerate(zip(executed.arena, final)) if a != b]
            report["first_differing_bytes"] = differing[:32]
            raise ValueError("original-model complete arena differs from final capture")
        execution = executed.metadata()
        if not execution_matches(execution, item, binary):
            raise ValueError("executed program, requested DMA, or ECALL differs")
        after, harness_after = model.source_binding(), harness()
        if before != after or harness_before != harness_after:
            raise ValueError("model or qualification harness source bookends changed")
        bundle.pins.verify()
        report.update(status="PASS", exact=True, full_arena_exact=True, accelerated=False, native_dispatch_enabled=False,
                      model_pin=MODEL_PIN, model_root=str(MODEL_ROOT), numerics="rtl", **identity(item, payloads),
                      execution=execution, model_sources_before=before, model_sources_after=after,
                      harness_sources_before=harness_before, harness_sources_after=harness_after,
                      method="executed_original_model", new_timing=False)
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    report["wall_seconds"] = time.monotonic() - started
    report['output_coverage'] = getattr(args, 'coverage_only', False)
    prefix = 'coverage' if report['output_coverage'] else 'launch'
    save(out / f"{prefix}-{args.index:03d}.json", report)
    print(json.dumps(dict(index=args.index, status=report["status"], wall_seconds=report["wall_seconds"])), flush=True)
    return 0 if report["status"] == "PASS" else 1


def coverage_reports(args, bundle, out):
    unique = {}
    for item in bundle.launches:
        unique.setdefault(item['program_sha256'], item)
    def one(item):
        index = item['index']
        path = out / f'coverage-{index:03d}.json'
        if not args.audit_only:
            command = [sys.executable, '-B', str(Path(__file__).absolute()), '--assets', str(bundle.root),
                       '--out', str(out), '--index', str(index), '--coverage-only']
            if args.smoke:
                command.append('--smoke')
            with (out / f'coverage-{index:03d}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        report = json.loads(path.read_text())
        if (report.get('status') != 'PASS' or report.get('output_coverage') is not True
                or report.get('program_sha256') != item['program_sha256']
                or report.get('initial_sha256') != item['initial_sha256']
                or report.get('exact') is not True or report.get('full_arena_exact') is not True
                or report.get('model_sources_before') != report.get('model_sources_after')
                or report.get('harness_sources_before') != harness()):
            raise ValueError('missing or stale output coverage proof')
        bundle.pins.bind(report['model_sources_before'])
        bundle.pins.bind(report['harness_sources_before'])
        return report
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        return list(pool.map(one, unique.values()))


def main(args):
    if args.index is not None:
        return worker(args)
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    bundle, out, model = Bundle(args.assets, args.smoke), local(args.out), load_model()
    out.mkdir(parents=True, exist_ok=True)
    model_sources, harness_sources = model.source_binding(), harness()
    mappings = json.loads(bundle.pins.read(local(args.reuse_records))) if args.reuse_records else {}
    completed, failed, resumed, reused, pending = {}, [], [], [], []
    for item in bundle.launches:
        index, path = item["index"], out / f"launch-{item['index']:03d}.json"
        if path.exists() and (args.resume or args.audit_only):
            saved = json.loads(path.read_text())
            if saved.get("status") == "PASS":
                try:
                    completed[index] = current_report(path, bundle, item, model_sources, harness_sources)
                except Exception as error:
                    rejected = dict(status="FAILED", model_pin=MODEL_PIN, native_not_used=True,
                                    verified_indices=sorted(completed), failed_indices=[index],
                                    pending_indices=[row["index"] for row in bundle.launches if row["index"] not in completed and row["index"] != index],
                                    error=str(error), command=sys.argv, updated_unix=time.time())
                    save(out / "progress.json", rejected)
                    save(out / "receipt.json", rejected)
                    raise
                resumed.append(index)
                continue
        if str(index) in mappings and not args.audit_only:
            try:
                report = from_capture(mappings[str(index)], bundle, item, model, model_sources, harness_sources)
                save(path, report)
                completed[index] = current_report(path, bundle, item, model_sources, harness_sources)
                reused.append(index)
            except NotOriginalCapture:
                pending.append(index)
        else:
            pending.append(index)

    def progress():
        save(out / "progress.json", dict(verified_indices=sorted(completed), failed_indices=sorted(failed),
             pending_indices=sorted(set(pending) - completed.keys() - set(failed)), model_pin=MODEL_PIN,
             native_not_used=True, updated_unix=time.time()))

    progress()
    if args.audit_only and pending:
        save(out / "receipt.json", dict(status="FAILED", model_pin=MODEL_PIN, native_not_used=True,
             verified_indices=sorted(completed), failed_indices=pending, missing_or_failed_indices=pending,
             error="audit incomplete: missing or failed launch receipts", command=sys.argv))
        raise ValueError("audit incomplete: missing or failed launch receipts")
    if not args.audit_only:
        def one(index):
            command = [sys.executable, "-B", str(Path(__file__).absolute()), "--assets", str(bundle.root),
                       "--out", str(out), "--index", str(index)]
            if args.smoke:
                command.append("--smoke")
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                               TMPDIR=str(out), ATLAS_NPU_MODEL_NATIVE="0")
            with (out / f"launch-{index:03d}.log").open("w") as log:
                return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode == 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, index): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    if not future.result():
                        raise ValueError("qualification worker failed")
                    completed[index] = current_report(out / f"launch-{index:03d}.json", bundle, bundle.launches[index], model_sources, harness_sources)
                except Exception:
                    failed.append(index)
                progress()
                print(f"launch {index} {'PASS' if index in completed else 'FAILED'}", flush=True)
    bundle.pins.verify()
    if model.source_binding() != model_sources or harness() != harness_sources:
        raise ValueError("final model or qualification harness bookend changed")
    coverage = coverage_reports(args, bundle, out)
    passed = len(completed) == len(bundle.launches) and not failed
    receipt = dict(schema="atlas.d1.all_launch_original.v1", status="PASS" if passed else "FAILED", model_pin=MODEL_PIN,
                   numerics="rtl", accelerated=False, native_not_used=True, all_launches_original_npu_model_rtl_exact=passed,
                   assets_replay_sha256=sha(bundle.raw), launch_count=len(bundle.launches), smoke_scope=args.smoke,
                   verified_indices=sorted(completed), failed_indices=sorted(failed), resumed_indices=resumed,
                   reused_original_capture_indices=reused, executed_indices=[i for i in pending if i in completed],
                   source_and_payload_sha256=bundle.pins.values, model_sources_before=model_sources,
                   model_sources_after=model_sources, harness_sources_before=harness_sources, harness_sources_after=harness_sources,
                   output_coverage=coverage, all_program_output_spans_written=True, new_device_timing=False, launches=[completed[i] for i in sorted(completed)], command=sys.argv)
    from receipt_paths import encode
    receipt = encode(receipt, dict(atlas=ATLAS_ROOT, merlin=MERLIN_ROOT, model=MODEL_ROOT, data=ROOT))
    save(out / "receipt.json", receipt)
    print(json.dumps(dict(status=receipt["status"], verified=len(completed), required=len(bundle.launches), failed=failed)), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--index", type=int)
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-records")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Explicitly admit a small runner-test bundle")
    raise SystemExit(main(parser.parse_args()))
