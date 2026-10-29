"""Isolated source-bound calls to the selected existing configuration producer.

Run as a fresh Python script. This module does not import the worker checkout's
Merlin package, select a toolchain, qualify instruction semantics or issue proof
authority. Its JSON output is untrusted input to independent preparation.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reference(path):
    path = Path(path).resolve()
    commands = (
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        ["git", "-C", str(path), "ls-tree", "-r", REVISION, "--", "src/merlin"],
    )
    head, tree = [subprocess.run(c, check=True, capture_output=True, text=True, timeout=30).stdout for c in commands]
    if head.strip() != REVISION:
        raise ValueError("configuration producer revision differs from the explicit source selection")
    blobs = {}
    for line in tree.splitlines():
        metadata, name = line.split("\t", 1)
        _, kind, identity = metadata.split()
        if kind == "blob":
            blobs[str((path / name).resolve())] = identity
    return path, blobs


def main():
    request = json.loads(sys.stdin.read(2 * 1024 * 1024 + 1))
    if set(request) - {"mode", "reference", "facts", "software", "target", "source", "qualification_identity"}:
        raise ValueError("unknown configuration producer request fields")
    reference, blobs = _reference(request["reference"])
    source_root = reference / "src"
    loaded = {}
    if any(name == "merlin" or name.startswith("merlin.") for name in sys.modules):
        raise ValueError("configuration producer was not started in a fresh project import context")

    class PinnedSource(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "merlin" and not fullname.startswith("merlin."):
                return None
            locations = (
                [str(source_root)]
                if path is None
                else [p for p in path if Path(p).resolve().is_relative_to(source_root)]
            )
            spec = importlib.machinery.PathFinder.find_spec(fullname, locations, target)
            if spec is None or spec.origin is None:
                raise ModuleNotFoundError("no project fallback for " + fullname)
            source = Path(spec.origin).resolve()
            data = source.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if not source.is_relative_to(source_root) or blobs.get(str(source)) != blob:
                raise ValueError("import differs from the pinned configuration source: " + fullname)
            loaded[fullname] = {
                "module": fullname,
                "path": str(source),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }

            class CurrentSource(importlib.machinery.SourceFileLoader):
                def get_code(self, module_name):
                    # Execute the exact observed project source, not an older
                    # timestamp-valid project .pyc. External Python is trusted.
                    return compile(data, str(source), "exec", dont_inherit=True)

            spec.loader = CurrentSource(fullname, str(source))
            return spec

    sys.meta_path.insert(0, PinnedSource())
    sys.path.insert(0, str(source_root))
    sys.dont_write_bytecode = True
    os.environ["MERLIN_REPO_ROOT"] = str(reference)
    from merlin.compile.scheduling.selected_contracts import _parse

    facts, software = Path(request["facts"]).resolve(), Path(request["software"]).resolve()
    facts_data, software_data = facts.read_bytes(), software.read_bytes()
    if len(facts_data) > 64 << 20 or len(software_data) > 1 << 20:
        raise ValueError("configuration original inputs exceed bounded selection")
    isa, effects, policy, byteorder, *_ = _parse(facts_data, software_data, request["target"])
    output = {
        "schema": "merlin.configuration-elision.producer.v1",
        "mode": request["mode"],
        "revision": REVISION,
        "reference": str(reference),
        "target": request["target"],
        "isa": isa.record(),
        "effects": json.loads(effects),
        "numerical_policy": policy,
        "instruction_byteorder": byteorder,
        "facts": {"path": str(facts), "sha256": hashlib.sha256(facts_data).hexdigest(), "bytes": len(facts_data)},
        "software": {
            "path": str(software),
            "sha256": hashlib.sha256(software_data).hexdigest(),
            "bytes": len(software_data),
        },
        "hardware_model_approved": False,
        "proof_authority": False,
    }
    if request["mode"] == "produce_mlir":
        from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
        from merlin.transforms.certificates import ir_text

        selected = SelectedSchedulingContracts(facts, software, facts_data, software_data, request["target"])
        # This is the existing producer's explicit caller premise label. It is
        # recorded as unapproved and cannot establish the numerical/effect model.
        selection = ConfigurationIRSelection(selected, request["qualification_identity"])
        original = selection.parse(request["source"])
        result = selection.apply(original)
        result.revalidate()
        output.update(source=ir_text(original), final=ir_text(result.module), producer_report=result.report)
    elif request["mode"] == "produce_kernel":
        from merlin.compile.scheduling.configuration import eliminate_redundant_configuration
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
        from merlin.sched.ir import parse_kernel

        selected = SelectedSchedulingContracts(facts, software, facts_data, software_data, request["target"])
        original = parse_kernel(request["source"])
        result = eliminate_redundant_configuration(
            original, selected.configuration_adapter(request["qualification_identity"])
        )
        output.update(source=original.text(), final=result.kernel.text(), producer_report=result.report)
    elif request["mode"] != "resolve":
        raise ValueError("unsupported configuration producer request mode")
    for row in loaded.values():
        data = Path(row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("configuration producer source changed while executing")
    if facts.read_bytes() != facts_data or software.read_bytes() != software_data:
        raise ValueError("configuration original inputs changed during producer execution")
    # Bind the named actual pass/IR adapter even on resolution-only requests;
    # this does not import the producer or trust its deletion plan.
    for relative in ("compile/scheduling/configuration.py", "compile/scheduling/configuration_ir.py"):
        path = source_root / "merlin" / relative
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blobs.get(str(path)) != blob:
            raise ValueError("selected actual configuration source differs from pinned revision")
        name = "merlin." + relative[:-3].replace("/", ".")
        loaded[name] = {
            "module": name,
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    output["source_refs"] = sorted(loaded.values(), key=lambda row: row["path"])
    print(_json(output))


if __name__ == "__main__":
    main()
