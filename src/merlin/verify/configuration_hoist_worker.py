"""Selected new-producer Kernel parser/hoist bridge, source-only phase1.

No Git commands, SDK selection, Lean execution or proof authority. All returned
plans/reports are diagnostics. The parent independently projects complete trees.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import ModuleType

REVISION = "e0484f4886ae0780d0325cdfa998842b61bb647c"
SOURCE_MANIFEST_SHA256 = "bb35450b5374e4fea125d0f766588830eea7a771131de7d1e961d5a4a4ea2626"


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def revision(reference):
    git = reference / ".git"
    if git.is_file():
        prefix, separator, value = git.read_text().strip().partition(": ")
        if prefix != "gitdir" or not separator:
            raise ValueError("explicit reference has no worktree identity")
        git = (reference / value).resolve()
    value = (git / "HEAD").read_text().strip()
    if value != REVISION:
        raise ValueError("new hoist reference must remain detached at the selected e048 revision")
    return value


def tree(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return {"float64_bits": struct.pack(">d", value).hex()}
    if type(value) is tuple:
        return {"tuple": [tree(item) for item in value]}
    name = type(value).__module__ + "." + type(value).__name__
    if dataclasses.is_dataclass(value) and type(value).__module__ in ("merlin.sched.ir.kernel", "merlin.sched.ir.expr"):
        return {
            "class": name,
            "fields": {field.name: tree(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
    if name == "merlin.sched.ir.kernel._Null":
        return {"class": name, "fields": {}}
    raise ValueError("unsupported complete Kernel value type: " + name)


def main():
    request = json.loads(sys.stdin.read(2 * 1024 * 1024 + 1))
    if set(request) - {
        "reference",
        "facts",
        "software",
        "target",
        "source",
        "final",
        "mode",
        "premise_label",
        "source_manifest",
    }:
        raise ValueError("unknown hoist worker request fields")
    reference = Path(request["reference"]).resolve()
    revision(reference)
    manifest_path = Path(request["source_manifest"]).resolve()
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("original root-selected producer Git source manifest changed")
    manifest = json.loads(manifest_bytes)
    if manifest["revision"] != REVISION or manifest["nonregular_python_paths"]:
        raise ValueError("incorrect original producer Git source selection")
    expected = {str(reference / row["path"]): row for row in manifest["sources"]}

    def checked_source(path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("selected source origin became a symlink")
        path = path.resolve()
        data = path.read_bytes()
        row = expected.get(str(path))
        if (
            row is None
            or hashlib.sha256(data).hexdigest() != row["sha256"]
            or len(data) != row["bytes"]
            or hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != row["git_blob"]
        ):
            raise ValueError("used producer/helper source differs from the selected Git blob")
        return {"path": str(path), "sha256": row["sha256"], "bytes": len(data)}

    if any(name == "merlin" or name.startswith("merlin.") for name in sys.modules):
        raise ValueError("hoist bridge requires a fresh project import process")
    sys.path.insert(0, str(reference / "src"))
    sys.dont_write_bytecode = True
    os.environ["MERLIN_REPO_ROOT"] = str(reference)
    critical = [
        reference / "src/merlin" / name
        for name in (
            "compile/scheduling/configuration_loops.py",
            "compile/scheduling/configuration.py",
            "compile/scheduling/selected_contracts.py",
            "sched/ir/kernel.py",
            "sched/ir/parse.py",
            "sched/ir/expr.py",
        )
    ]
    before = [checked_source(path) for path in critical]
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
    from merlin.sched.ir import parse_kernel
    from merlin.sched.ir.kernel import check_structure

    selected = read_scheduling_contracts(Path(request["facts"]), Path(request["software"]), target=request["target"])
    source = parse_kernel(request["source"])
    if check_structure(source):
        raise ValueError("actual source Kernel failed structural validation")
    report = None
    if request["mode"] == "produce":
        from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration

        if type(request["premise_label"]) is not str or not request["premise_label"]:
            raise ValueError("existing producer requires an explicit unapproved effect-premise label")
        adapter = selected.configuration_adapter(request["premise_label"])
        result = hoist_loop_configuration(source, adapter)
        final_text, report = result.kernel.text(), result.report
    elif request["mode"] == "project":
        final_text = request["final"]
    else:
        raise ValueError("unsupported actual hoist bridge mode")
    # Independently parse the actual final string, not a trusted result plan.
    final = parse_kernel(final_text)
    if check_structure(final):
        raise ValueError("actual final Kernel failed structural validation")
    if tree(parse_kernel(source.text())) != tree(source) or tree(parse_kernel(final.text())) != tree(final):
        raise ValueError("actual complete Kernel tree failed text/parse roundtrip")
    selected.revalidate()
    sources = {}
    pending, seen = list(sys.modules.values()), set()
    while pending:
        module = pending.pop()
        if not isinstance(module, ModuleType) or id(module) in seen:
            continue
        seen.add(id(module))
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        if path.is_relative_to(reference / "src"):
            if path.suffix != ".py":
                raise ValueError("selected project helper has no Python source binding")
            sources[str(path)] = checked_source(path)
            # A helper loaded under an alias may live in module globals without
            # registration in sys.modules (the child-owner helper does this).
            pending.extend(value for value in vars(module).values() if isinstance(value, ModuleType))
        elif module.__name__ == "merlin" or module.__name__.startswith(("merlin.", "merlin_")):
            raise ValueError("new hoist producer imported a foreign project source")
    for row in before:
        if checked_source(row["path"]) != row:
            raise ValueError("selected new hoist source changed during execution")
        sources[row["path"]] = row
    revision(reference)
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("original selected source manifest changed during capture")
    output = {
        "schema": "merlin.configuration-hoist.actual-kernel.v1",
        "revision": REVISION,
        "source": request["source"],
        "final": final_text,
        "before": tree(source),
        "after": tree(final),
        "isa": selected.isa.record(),
        "effects": json.loads(selected.effects_json),
        "selection": selected.record(),
        "facts": ref(request["facts"]),
        "software": ref(request["software"]),
        "project_sources": [sources[key] for key in sorted(sources)],
        "producer_report": report,
        "producer_mode": request["mode"],
        "execution_approval": False,
        "proof_authority": False,
        "Python_parser_and_import_delivery_trusted": True,
        "complete_tree_roundtrip_checked": True,
        "source_manifest": ref(manifest_path),
        "used_source_origins_match_root_selected_git_blobs": True,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
