"""Refresh an explicit reviewed source grant; optionally add neutral ISA products.

No campaign, oracle, simulator or cloud process is invoked. Existing import
membership is retained; newly discovered SDK leaves require a separate review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from merlin_experiments.phase1.generated_isa_library import prepare_library
from merlin_experiments.phase2.compiler_library_state import load_selection

from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library
from merlin.targetgen.compiler_library_fork import REVIEW_PREFIX, enabled
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest
from merlin.targetgen.tool_registry import ARM_TOOLS


def prepare(
    *,
    repo,
    source_revision,
    previous_root,
    previous_contract,
    previous_sha256,
    output,
    generated_package=None,
    manifest_sha256=None,
    facts_sha256=None,
    declaration_sha256=None,
):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_revision:
        raise ValueError("source revision must match the selected repository HEAD")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=repo):
        raise ValueError("compiler source release contains uncommitted edits")
    base = load_selection(root=previous_root, contract_path=previous_contract, contract_sha256=previous_sha256)
    if base is None or not enabled(base):
        raise ValueError("explicit reviewed editable generic compiler selection required")
    if any(member.path.startswith("reviewed_isa/") for member in base.contract.members):
        raise ValueError("refresh the generic base before explicitly selecting ISA scaffolding")
    output.mkdir(parents=True, exist_ok=False)
    sdk = output / "base-sdk"
    manifest = prepare_shared_source_release(
        sdk,
        python_source_root=repo / "src",
        license_directory=repo,
        tools=ARM_TOOLS["merlin_rtlchecks"],
        version=8,
        source_binding=True,
    )
    if reviewed_shared_manifest(sdk, version=8) != manifest:
        raise ValueError("shared SDK identity changed during preparation")
    members = {row.path: row for row in base.contract.members}
    for row in manifest["members"]:
        if row["path"] not in members or members[row["path"]].module != row["module"]:
            raise ValueError("new SDK source grant requires explicit review: " + row["path"])
    sdk_paths = {row["path"] for row in manifest["members"]} | set(manifest["review_resources"])
    library = output / "generic-library"
    library.mkdir()
    changed, hashes = [], {}
    for member in base.contract.members:
        source = sdk / member.path
        if member.path.endswith("/__init__.py") and member.path not in sdk_paths:
            source = base.root / member.path
        elif not source.is_file():
            source = repo / "src" / member.path if member.module else base.root / member.path
        if source.is_symlink() or not source.is_file():
            raise ValueError("selected source is absent or linked: " + str(source))
        raw = source.read_bytes()
        hashes[member.path] = hashlib.sha256(raw).hexdigest()
        if hashes[member.path] != member.sha256:
            changed.append(member.path)
        target = library / member.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    contract = freeze_compiler_library(
        library,
        review_id=REVIEW_PREFIX + "coevolution-readiness:" + source_revision[:12],
        public_modules=base.contract.public_modules,
        sources=tuple((member.path, member.module) for member in base.contract.members),
    )
    selection = CompilerLibrarySelection(contract, library)
    (output / "generic-library-contract.json").write_text(
        json.dumps(contract.record(), sort_keys=True, indent=2) + "\n"
    )
    (output / "generic-library-selection.json").write_text(
        json.dumps(selection.record(), sort_keys=True, indent=2) + "\n"
    )
    if generated_package is not None:
        selection = prepare_library(
            selection,
            Path(generated_package),
            output / "compiler-library",
            review_id=REVIEW_PREFIX + "coevolution-generated-isa:" + source_revision[:12],
            manifest_sha256=manifest_sha256,
            facts_sha256=facts_sha256,
            declaration_sha256=declaration_sha256,
        )
    elif any(value is not None for value in (manifest_sha256, facts_sha256, declaration_sha256)):
        raise ValueError("ISA pins require an explicitly selected generated package")
    if not enabled(selection):
        raise ValueError("editable generic compiler authority was lost")
    report = {
        "schema": "merlin.author_library_preparation.v1",
        "source_revision": source_revision,
        "previous_contract_sha256": base.contract.sha256,
        "contract_sha256": selection.contract.sha256,
        "members": len(selection.contract.members),
        "public_modules": len(selection.contract.public_modules),
        "changed_generic_members": changed,
        "editable_generic_compiler": True,
        "generated_isa_selected": generated_package is not None,
        "grading_authority_changed": False,
        "campaign_launched": False,
        "source_member_sha256": hashes,
    }
    for name, value in [
        ("compiler-library-contract.json", selection.contract.record()),
        ("compiler-library-selection.json", selection.record()),
        ("preparation.json", report),
    ]:
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "previous-root", "previous-contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-revision", "previous-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--generated-package", type=Path)
    for name in ("manifest-sha256", "facts-sha256", "declaration-sha256"):
        parser.add_argument("--" + name)
    report = prepare(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in report.items() if key != "source_member_sha256"}, indent=2))


if __name__ == "__main__":
    main()
