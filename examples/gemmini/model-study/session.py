#!/usr/bin/env python3
"""Check or run a retained complete-session image with verified timing repeats.

This evaluator may read sealed references. Never expose its inputs or its guest
image to a compiler-construction agent. Saved-console checks establish numerical
transport agreement, not hardware provenance or physical timing fidelity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from merlin.common import invocation_record
from merlin.model_check import chain_image_readback as readback
from merlin.model_check.chain_image_objects import payload, pin, reopen
from merlin.model_check.chain_image_repeats import decode_repeats


def checked_pin(path, expected):
    record = pin(path)
    if record["sha256"] != expected:
        raise ValueError("selected file digest differs")
    return record


def judge(image_pin, console_pin):
    """Reopen the complete image before and after strict all-output decoding."""
    image = readback.verify_image(image_pin)
    console = payload(console_pin)
    report = decode_repeats(image, console)
    if readback.verify_image(image_pin) != image or payload(console_pin) != console:
        raise ValueError("selected image or console changed during checking")
    return report


def run_spike(image_pin, output, timeout):
    """Require the actual simulator command to match the admitted machine map."""
    from merlin.model_check.models import load
    from merlin.runtime.backends import chipyard_rocc_bound, rocc_selection
    from merlin.runtime.backends.base import get_backend

    image = readback.verify_image(image_pin)
    selection = json.loads(payload(image["selection"]))
    target = load(reopen(selection["models"])).target
    machine = json.loads(payload(selection["memory_map"]))["derivation"]["machine"]
    original = get_backend(target)._data()
    contract = rocc_selection.select_model_spike_timeout(original["contract"], timeout)
    engine = chipyard_rocc_bound.bind(target=target, contract=contract, facts=original["facts"])
    selected = engine._engine("spike")
    if (
        selected["binary"]["sha256"] != machine["binary"]["sha256"]
        or contract["runner"]["spike_extension"]["sha256"] != machine["extension"]["sha256"]
    ):
        raise ValueError("selected Spike binary or extension differs from admitted memory map")
    elf = reopen(image["boot_elf"])
    before = set(elf.parent.glob("invocations/*/invocation.json"))
    console = engine.run_declared_model_elf(elf, timeout=timeout, capture_bytes=True)
    destination = output / "console.bin"
    destination.write_bytes(console)
    observed = set(elf.parent.glob("invocations/*/invocation.json")) - before
    if len(observed) != 1:
        raise ValueError("expected one new actual simulator invocation")
    invocation = observed.pop()
    invocation_record.verify(invocation)
    actual = json.loads(invocation.read_bytes())
    expected = [machine["binary"]["path"], *machine["machine_arguments"], str(elf)]
    if actual["argv"] != expected:
        raise ValueError("actual simulator command differs from admitted machine")
    return pin(destination), {
        "actual_invocation": pin(invocation),
        "timeout_s": timeout,
        "counter_kind": "spike_retired_instruction_proxy",
        "execution_identity": selection["execution_identity"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run-spike"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--console", type=Path)
    parser.add_argument("--console-sha256")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "check" and (args.console is None or args.console_sha256 is None):
        parser.error("check requires a pinned console")
    if args.mode == "run-spike" and (args.console is not None or args.console_sha256 is not None):
        parser.error("run-spike records its own console")
    if not 1 <= args.timeout_s <= 14400:
        parser.error("timeout must be within 1..14400 seconds")
    os.umask(0o077)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "merlin.model_study.session.v1", "status": "checking", "mode": args.mode}
    try:
        image_pin = checked_pin(args.image, args.image_sha256)
        result["image"] = image_pin
        if args.mode == "check":
            console_pin = checked_pin(args.console, args.console_sha256)
            result["counter_kind"] = "unattributed_saved_console"
        else:
            console_pin, execution = run_spike(image_pin, output, args.timeout_s)
            result.update(execution)
        result.update(console=console_pin, readback=judge(image_pin, console_pin))
        passed = all(result["readback"][key] == "pass" for key in ("numerical_status", "runtime_status"))
        result["status"] = "pass" if passed else "fail"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result["scope"] = "Complete-session numerical transport. No expert comparison or physical timing qualification."
    result["producer"] = pin(Path(__file__))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "result": str(output / "result.json")}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
