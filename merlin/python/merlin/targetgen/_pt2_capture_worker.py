"""Turn a user's exported PyTorch program into a capture bundle. Runs in the m2m venv.

The only way to hand this system a model has been a ``.py`` exposing ``get_model_and_inputs()`` --
runnable code, with its dependencies. That is exactly what blocked most of the declared roster:
smolvla needs ``lerobot`` installed, resnet50 needs a preprocessed ImageNet npz, and neither is a
statement about the compiler. An exported program carries its graph and its example inputs in one
self-contained file, so a user can hand over the application without handing over its environment.

WHAT THIS WORKER RELIES ON, all verified rather than assumed:

* ``torch.export.save`` / ``load`` round-trip an ``ExportedProgram`` (torch 2.10 in the m2m venv);
* the loaded program carries ``example_inputs`` as ``(args, kwargs)``, so the inputs ``m2m`` requires
  come with the model instead of being asked of the user a second time;
* ``.module()`` returns a callable ``GraphModule``, which is what produces the reference golden.

It writes the SAME bundle layout ``m2m.capture.bundle.write_bundle`` already produces -- that is the
whole trick. The conformance requirement, the memory-regime axis, the composition axis and the DSE
registry all read that store, so an application ingested here becomes evidence everywhere with no
further plumbing.

Run in a SUBPROCESS under the m2m interpreter, exactly as ``_m2m_capture_worker`` is: torch lives in
that venv and not in merlin's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _example_inputs(exported):
    """Positional example inputs carried by the program itself.

    ``example_inputs`` is ``(args, kwargs)``. Keyword inputs are refused rather than dropped: the
    bundle writer feeds inputs positionally, so silently discarding them would capture a DIFFERENT
    program from the one the user exported.
    """
    carried = getattr(exported, "example_inputs", None)
    if not carried:
        raise RuntimeError(
            "the exported program carries no example_inputs; re-export with "
            "torch.export.export(model, args) so the inputs travel with the graph")
    args, kwargs = carried
    if kwargs:
        raise RuntimeError(
            f"the exported program carries keyword example inputs {sorted(kwargs)}, which the bundle "
            f"writer cannot feed positionally; re-export with positional args only")
    return tuple(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capture an exported PyTorch program (.pt2) as a bundle.")
    ap.add_argument("--exported", required=True, help="path to a torch.export .pt2 archive")
    ap.add_argument("--out", required=True, help="bundle directory to write")
    ap.add_argument("--m2m-dir", default=os.environ.get("MERLIN_M2M_DIR"))
    ap.add_argument("--quant", default="", help="torchAO scheme name; empty means no quantization")
    a = ap.parse_args(argv)

    if a.m2m_dir and a.m2m_dir not in sys.path:
        sys.path.insert(0, a.m2m_dir)

    import torch

    from m2m.capture.bundle import write_bundle

    src = Path(a.exported)
    if not src.is_file():
        raise RuntimeError(f"no exported program at {src}")

    exported = torch.export.load(str(src))
    inputs = _example_inputs(exported)
    # `.module()` gives the callable GraphModule the golden is taken from. The bundle writer re-runs
    # its own export internally, which is why the module rather than the ExportedProgram is handed
    # over: passing the program would ask torch to export an already-exported graph.
    model = exported.module()
    # MODE IS ALREADY BAKED IN, so the mode switches are made no-ops rather than left to raise.
    # `write_bundle` calls `mdl.eval()` unconditionally (bundle.py:438) because a live nn.Module
    # could be in either mode, and torch deliberately raises `NotImplementedError` on an exported
    # module to say the question no longer applies -- export captured one mode and that is what the
    # graph is. Answering "already in that mode" is the truthful response; the alternative is
    # teaching every consumer that an exported program is a special case.
    model.eval = lambda: model                     # noqa: E731 -- an instance-level no-op, by design
    model.train = lambda mode=True: model          # noqa: E731

    # `quant` is an m2m QuantizationConfig, not a scheme NAME -- passing the string through would be
    # accepted as a truthy object and then fail deep inside the writer, or worse, be ignored.
    quant = None
    if a.quant:
        from m2m.capture.torchao_pipeline import QuantizationConfig
        quant = QuantizationConfig(scheme=a.quant)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = write_bundle(model, inputs, str(out), quant=quant)

    meta = {
        "ok": True,
        "source_pt2": str(src),
        "source_bytes": src.stat().st_size,
        "quant": a.quant or None,
        "input_shapes": [list(getattr(t, "shape", ())) for t in inputs],
        "bundle": written if isinstance(written, dict) else None,
    }
    (out / "ingest_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    # A machine-readable tail line the parent reads, even when warnings precede it.
    print("__PT2_INGEST__ " + json.dumps({"ok": True, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
