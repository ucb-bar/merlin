"""Prepare a target-independent compiler library and optional neutral ISA products.

No campaign, oracle, simulator or cloud process is invoked. Existing import
membership is retained; newly discovered SDK leaves require a separate review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library
from merlin.targetgen.compiler_library_fork import REVIEW_PREFIX, enabled
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest
from merlin.targetgen.tool_registry import ARM_TOOLS
from merlin_experiments.phase1.generated_isa_library import prepare_library
from merlin_experiments.phase2.compiler_library_state import load_selection


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
    reviewed_module=(),
    reviewed_rtl_discovery_dependencies=False,
    reviewed_tensor_views=False,
    library_version=21,
):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    if type(library_version) is not int or library_version not in (8, 16, 17, 18, 19, 20, 21, 22):
        raise ValueError("author library preparation requires explicit inventory version 8 or 16 through 22")
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
        version=library_version,
        source_binding=True,
    )
    if reviewed_shared_manifest(sdk, version=library_version) != manifest:
        raise ValueError("shared SDK identity changed during preparation")
    members = {row.path: row for row in base.contract.members}
    requested = reviewed_discovery_modules(
        reviewed_module, existing=base.contract.public_modules,
        enabled=reviewed_rtl_discovery_dependencies,
    )
    requested = reviewed_view_modules(
        requested, existing=base.contract.public_modules,
        source_modules={member.module for member in base.contract.members},
        enabled=reviewed_tensor_views,
    )
    additions = reviewed_modules(requested, existing=base.contract.public_modules)
    sources = {member.path: member.module for member in base.contract.members}
    sources.update(additions)
    for row in manifest["members"]:
        if row["path"] not in sources or sources[row["path"]] != row["module"]:
            raise ValueError("new SDK source grant requires explicit review: " + row["path"])
    sdk_paths = {row["path"] for row in manifest["members"]} | set(manifest["review_resources"])
    library = output / "generic-library"
    library.mkdir()
    changed, hashes = [], {}
    for path, module in sources.items():
        source = sdk / path
        if path.endswith("/__init__.py") and path not in sdk_paths:
            source = base.root / path
        elif not source.is_file():
            source = repo / "src" / path if module else base.root / path
        if source.is_symlink() or not source.is_file():
            raise ValueError("selected source is absent or linked: " + str(source))
        raw = source.read_bytes()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        if path not in members or hashes[path] != members[path].sha256:
            changed.append(path)
        target = library / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    contract = freeze_compiler_library(
        library,
        review_id=REVIEW_PREFIX + "coevolution-readiness:" + source_revision[:12],
        public_modules=(*base.contract.public_modules, *additions.values()),
        sources=tuple(sources.items()),
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
        "explicitly_reviewed_new_modules": list(additions.values()),
        "editable_generic_compiler": True,
        "generated_isa_selected": generated_package is not None,
        "grading_authority_changed": False,
        "campaign_launched": False,
        "source_member_sha256": hashes,
    }
    if library_version != 8:
        report["library_inventory_version"] = library_version
    for name, value in [
        ("compiler-library-contract.json", selection.contract.record()),
        ("compiler-library-selection.json", selection.record()),
        ("preparation.json", report),
    ]:
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return report


def reviewed_discovery_modules(modules, *, existing, enabled=False):
    """Opt into one reviewed leaf; never merge Python namespace paths broadly.

    The generic common package shadows the base SDK common package. Its RTL
    discovery generators need this dependency even though both releases contain
    the parent namespace. Historical selections are unchanged by default.
    """
    requested = tuple(modules)
    dependency = "merlin.common.facts_view"
    if enabled and dependency not in existing and dependency not in requested:
        requested += (dependency,)
    return requested


def reviewed_view_modules(modules, *, existing, source_modules, enabled=False):
    """Explicit next-library view normalization; never upgrade a frozen SDK.

    Source closure must already include the structural witness owner. Ordinary
    contract freezing verifies its transitive imports and exact member bytes.
    Callable guidance is derived from the selected source AST, not this option.
    """
    if type(enabled) is not bool:
        raise ValueError("reviewed tensor views requires an explicit Boolean")
    requested = tuple(modules)
    if not enabled:
        return requested
    leaf = "merlin.transforms.tensor_views"
    if "merlin.transforms.certificates" not in set(source_modules) | set(requested):
        raise ValueError("reviewed tensor views requires its certificates source owner")
    if leaf not in existing and leaf not in requested:
        requested += (leaf,)
    return requested


def reviewed_modules(modules, *, existing):
    """Select exact reviewed generic leaves, never a package or companion source."""
    selected = {}
    for module in modules:
        if (
            not isinstance(module, str)
            or not module.startswith("merlin.")
            or any(not part.isidentifier() for part in module.split("."))
            or module.endswith(".__init__")
            or module in existing
            or module in selected.values()
        ):
            raise ValueError("reviewed module must name one new generic Merlin leaf")
        selected[module.replace(".", "/") + ".py"] = module
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "previous-root", "previous-contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-revision", "previous-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--generated-package", type=Path)
    parser.add_argument("--library-version", type=int, choices=(8, 16, 17, 18, 19, 20, 21, 22), default=21,
                        help="prospective reviewed source inventory; existing library grants remain explicit")
    parser.add_argument(
        "--reviewed-rtl-discovery-dependencies", action="store_true",
        help="explicitly include reviewed merlin.common.facts_view for public RTL discovery imports",
    )
    parser.add_argument(
        "--reviewed-tensor-views", action="store_true",
        help="explicitly add static tensor-view canonicalization to the next reviewed library",
    )
    parser.add_argument(
        "--reviewed-module",
        action="append",
        default=[],
        help="one explicitly reviewed new generic Merlin leaf (repeat for each; no package grant)",
    )
    for name in ("manifest-sha256", "facts-sha256", "declaration-sha256"):
        parser.add_argument("--" + name)
    report = prepare(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in report.items() if key != "source_member_sha256"}, indent=2))


if __name__ == "__main__":
    main()
