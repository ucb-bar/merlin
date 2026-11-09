"""Original capsule ELF preparation and exact FireSim UART replay for native L5.

The adapter launches no hardware. Native capsule grading still owns numerical,
placement and interface checks. Deployment observations are distinct from a
historical certificate and must not be substituted for actual selected outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import firesim


def adapter(*, target, hardware, observations=None):
    """Use in the native runner's explicit ``oracle_adapters['L5']`` selection.

    With no observation for the exact freshly compiled ELF, write its transport
    input and return honest unavailable. Once the installed wrapper has run that
    image, select its pinned UART/boot evidence and rerun the original numerical
    checker with this adapter; no hardware is rerun to consume a result.
    """
    from merlin.targetgen.capsule_runner import OracleUnavailable
    from merlin.targetgen.contract import compile as compiler

    def run(command_buffer, lowered, workdir, timeout):
        from merlin.runtime.backends import base
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from .booted_elf import _boot_path

        root = Path(workdir)
        elf = compiler.compile_lowered_to_elf(command_buffer, lowered, root, target=target)
        image = firesim.pin(Path(elf).absolute())
        request = {"schema": "merlin.joint_capsule_firesim.image.v1", "elf": image,
                   "hardware": hardware, "numeric_pass": False, "paid_jobs_launched": 0}
        (root / "joint_firesim_image.json").write_text(json.dumps(request, indent=2) + "\n")
        selection = (observations or {}).get(image["sha256"])
        if selection is None:
            raise OracleUnavailable("frozen capsule ELF prepared for final FireSim; exact board observation pending")
        record = firesim.load(firesim.reopen(selection))
        if (record.get("schema") != "merlin.joint_capsule_firesim.observation.v1"
                or record.get("elf_sha256") != image["sha256"] or record.get("hardware") != hardware):
            raise ValueError("capsule FireSim observation differs from the selected image/hardware")
        boot = firesim.load(firesim.reopen(record["booted_elf"]))
        _boot_path(boot, firesim.reopen(record["sim_run"]))
        if (boot.get("agfi") != hardware["agfi"] or boot.get("consumed_elf", {}).get("sha256") != image["sha256"]):
            raise ValueError("actual FireSim boot file/AGFI differs from this capsule")
        invocation = firesim.load(firesim.reopen(record["ssm_invocation"]))
        if (invocation.get("CommandId") != boot.get("ssm_command_id")
                or invocation.get("Status") != "Success" or invocation.get("ResponseCode") != 0):
            raise ValueError("capsule observation lacks its own successful wrapper command")
        console_path = firesim.reopen(record["uart"])
        console = console_path.read_text()
        if "DONE" not in console.splitlines():
            raise OracleUnavailable("FireSim capsule UART lacks complete DONE termination")
        backend = base.get_backend(target)
        outputs, raw = backend.parse_output(console)
        outputs = base.decode_float_readback(outputs, declared_output_dtypes(command_buffer))
        return {"outputs": outputs, "raw_metrics": raw, "cycles": raw.get("cycles"), "console": console,
                "oracle": {"kind": "firesim_observed_uart", "observation": selection,
                           "hardware_correspondence": "separate final campaign obligation",
                           "legacy_certificate_issued": False},
                "timing": {"observation_replay": True, "sim_active_s": 0.0, "oracle_wait_s": 0.0}}

    return run
