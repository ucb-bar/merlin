"""Explicit, byte-bound RTL extraction inputs and deterministic CIRCT production.

This is source provenance, not hardware/numerical qualification. No runtime agent
chooses semantics, and an explicit selection never falls back to ambient caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

SCHEMA = "merlin.rtl_source_selection.v1"
_ACTIVE = ContextVar("merlin_rtl_source_selection", default=None)
_ENUM_METADATA = frozenset(
    f"chisel3.experimental.EnumAnnotations${kind}"
    for kind in ("EnumComponentAnnotation", "EnumDefAnnotation", "EnumVecAnnotation")
)


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare_firtool_input(text: str, drop_classes: list[str]) -> tuple[str, int]:
    """Remove only explicitly selected Chisel enum metadata, never circuit bodies."""
    if set(drop_classes) - _ENUM_METADATA:
        raise ValueError("only non-functional Chisel enum annotation classes may be removed")
    if not drop_classes:
        return text, 0
    marker = text.find(":%[")
    if marker < 0:
        raise ValueError("FIRRTL circuit annotation section unavailable")
    begin = marker + len(":%[")
    annotations, end = json.JSONDecoder().raw_decode(text[begin:])
    if not isinstance(annotations, list) or text[begin + end] != "]":
        raise ValueError("unparsed FIRRTL annotation section")
    retained = [row for row in annotations if row.get("class") not in drop_classes]
    return text[:begin] + json.dumps(retained, separators=(",", ":")) + text[begin + end :], len(annotations) - len(
        retained
    )


def active_selection(target: str | None = None) -> dict | None:
    selected = _ACTIVE.get()
    if selected is not None and target is not None and selected["target"] != target:
        raise ValueError("RTL source selection belongs to a different target")
    return selected


@contextmanager
def selected_sources(document: dict):
    token = _ACTIVE.set(document)
    try:
        yield document
    finally:
        _ACTIVE.reset(token)


def load_selection(path: str | Path, *, target: str | None = None) -> dict:
    source = Path(path).resolve()
    doc = json.loads(source.read_bytes())
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("target"), str):
        raise ValueError("RTL source selection must declare schema and target")
    if target is not None and target != doc["target"]:
        raise ValueError("RTL source selection target mismatch")
    sources = doc.get("sources")
    if not isinstance(sources, dict) or not {"core_hw", "soc_hw", "firrtl", "hierarchy"} <= set(sources):
        raise ValueError("explicit selection requires core_hw, soc_hw, firrtl and hierarchy")
    for role, member in sources.items():
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise ValueError(f"invalid selected RTL source {role}")
        path_member = Path(member["path"])
        path_member = path_member if path_member.is_absolute() else source.parent / path_member
        if digest(path_member) != member.get("sha256"):
            raise ValueError(f"selected RTL source bytes changed: {role}")
        member["path"] = str(path_member.resolve())
    doc["selection_path"] = str(source)
    doc["selection_sha256"] = digest(source)
    return doc


def _firrtl_instances(firrtl: Path) -> dict:
    modules, current = {}, None
    with firrtl.open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if line.startswith("  ") and not line.startswith("   "):
                head = stripped.split(":", 1)[0].split()
                if head and head[0] in {"module", "extmodule", "intmodule"}:
                    current = head[1]
                    modules[current] = {}
                elif len(head) > 2 and head[:2] == ["public", "module"]:
                    current = head[2]
                    modules[current] = {}
            if current and stripped.startswith("inst "):
                tokens = stripped.split()
                if len(tokens) < 4 or tokens[2] != "of":
                    raise ValueError("unparsed FIRRTL instance")
                name, child = tokens[1], tokens[3]
                if name in modules[current]:
                    raise ValueError("duplicate FIRRTL instance name")
                modules[current][name] = child
    return modules


def circuit_root(firrtl: Path) -> str:
    """Read the circuit declaration, without guessing a target/module alias."""
    roots = []
    with firrtl.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("circuit "):
                head, separator, _ = line.partition(":")
                tokens = head.split()
                if not separator or len(tokens) != 2:
                    raise ValueError("unparsed FIRRTL circuit declaration")
                roots.append(tokens[1])
    if len(roots) != 1:
        raise ValueError("selected FIRRTL must contain exactly one circuit declaration")
    return roots[0]


def derive_hierarchy(firrtl: Path, root: str | None = None) -> dict:
    """Materialize exact FIRRTL instance edges, without dedup/name alias guesses."""
    root = circuit_root(firrtl) if root is None else root
    modules = _firrtl_instances(firrtl)

    def visit(module, instance, ancestors):
        if module not in modules or module in ancestors:
            raise ValueError(f"unresolved or cyclic FIRRTL instance closure at {module}")
        return {
            "instance_name": instance,
            "module_name": module,
            "instances": [visit(child, name, ancestors | {module}) for name, child in modules[module].items()],
        }

    return visit(root, root, set())


def firrtl_hierarchy_audit(firrtl: Path, hierarchy: Path) -> dict:
    """Compare every hierarchy instance to exact declarations in selected FIRRTL.

    This compares structural content, not filenames/configuration labels or dates.
    A hierarchy may cover one top-level subtree of the elaborated circuit.
    """
    modules = _firrtl_instances(firrtl)
    tree = json.loads(hierarchy.read_bytes())
    errors, compared = [], 0
    stack = [tree]
    while stack:
        node = stack.pop()
        name, children = node.get("module_name"), node.get("instances")
        if not isinstance(name, str) or not isinstance(children, list):
            raise ValueError("malformed selected hierarchy")
        observed = {}
        for child in children:
            instance, module = child.get("instance_name"), child.get("module_name")
            if not isinstance(instance, str) or instance in observed or not isinstance(module, str):
                raise ValueError("malformed or duplicate hierarchy child")
            observed[instance] = module
        if name not in modules:
            errors.append({"module": name, "reason": "hierarchy module absent from selected FIRRTL"})
        elif modules[name] != observed:
            errors.append({"module": name, "reason": "instance name/module correspondence differs"})
        compared += 1
        stack.extend(children)
    return {
        "status": "verified" if not errors else "mismatch",
        "basis": "exact FIRRTL declaration/instance names compared to every hierarchy node",
        "firrtl_sha256": digest(firrtl),
        "hierarchy_sha256": digest(hierarchy),
        "compared_instances": compared,
        "declared_modules": len(modules),
        "errors": errors,
        "qualification": "structural correspondence; not a claim of common historical generation",
    }


def production_consistency(doc: dict) -> dict:
    """Check all consumed production edges, without approving hardware semantics."""
    sources, production = doc["sources"], doc.get("production") or {}
    errors = []
    if production.get("kind") != "firrtl_to_hw_then_exact_module_closure" or production.get("returncode") != 0:
        errors.append("no successful declared deterministic FIRRTL producer")
    for role in ("firrtl", "soc_hw", "core_hw"):
        if production.get(f"{role}_sha256") != sources[role]["sha256"]:
            errors.append(f"producer does not bind selected {role} bytes")
    preparation = production.get("input_preparation") or {}
    try:
        expected, removed = prepare_firtool_input(
            Path(sources["firrtl"]["path"]).read_text(), preparation["drop_annotation_classes"]
        )
        if (
            digest(preparation["path"]) != hashlib.sha256(expected.encode()).hexdigest()
            or removed != preparation["removed_annotations"]
        ):
            errors.append("FIRRTL metadata-only preparation does not reproduce")
    except (KeyError, OSError, ValueError):
        errors.append("FIRRTL generation input preparation unavailable")
    tool = production.get("tool") or {}
    try:
        if digest(tool.get("path", "")) != tool.get("sha256"):
            errors.append("producer tool bytes differ")
    except (OSError, ValueError):
        errors.append("producer tool identity unavailable")
    from . import extract_module

    root = production.get("core_root")
    if not isinstance(root, str) or not root:
        errors.append("core root not declared")
    else:
        expected, included, missing = extract_module.extract(Path(sources["soc_hw"]["path"]).read_text(), root)
        if hashlib.sha256(expected.encode()).hexdigest() != sources["core_hw"]["sha256"] or missing:
            errors.append("core is not the exact complete selected SoC module closure")
        if included != production.get("included_modules"):
            errors.append("declared module closure differs")
    audit = firrtl_hierarchy_audit(Path(sources["firrtl"]["path"]), Path(sources["hierarchy"]["path"]))
    if audit["status"] != "verified":
        errors.append("selected hierarchy differs from FIRRTL")
    elaboration = {"status": "not_selected", "qualification": "configuration label has no source-to-FIRRTL receipt"}
    selected_elaboration = production.get("elaboration")
    if selected_elaboration is not None:
        try:
            from .elaboration import verify

            if (
                not isinstance(selected_elaboration, dict)
                or digest(selected_elaboration["path"]) != selected_elaboration["sha256"]
            ):
                raise ValueError("elaboration receipt bytes differ")
            verified = verify(
                Path(selected_elaboration["path"]),
                firrtl=Path(sources["firrtl"]["path"]),
                config=doc.get("config"),
            )
            elaboration = {
                "status": "reproduced_exact_firrtl",
                "receipt_sha256": selected_elaboration["sha256"],
                "source_revision": verified["source"]["revision"],
                "qualification": verified["qualification"],
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"selected elaboration receipt invalid: {exc}")
            elaboration = {"status": "unverified"}
    return {
        "status": "verified" if not errors else "unverified",
        "basis": "recorded FIRRTL-to-HW execution; exact HW closure and FIRRTL/hierarchy structural correspondence",
        "config": doc.get("config"),
        "sources": [{"role": role, **member} for role, member in sorted(sources.items())],
        "production": production,
        **({"elaboration": elaboration} if selected_elaboration is not None else {}),
        "hierarchy_correspondence": audit,
        "errors": errors,
        "qualification": "source consistency only; numerics and compiler conformance require separate receipts",
    }


def produce_selection(
    *,
    target: str,
    firrtl: Path,
    hierarchy: Path | None = None,
    generator: str,
    config: str,
    core_root: str,
    firtool: Path,
    output: Path,
    drop_annotation_classes: list[str] | None = None,
    elaboration_receipt: Path | None = None,
) -> Path:
    """Generate inspectable production artifacts; never rewrite an OOT source tree."""
    from . import extract_module

    verified_elaboration = None
    if elaboration_receipt is not None:
        from .elaboration import verify

        receipt = Path(elaboration_receipt).resolve(strict=True)
        verify(receipt, firrtl=firrtl, config=config)
        verified_elaboration = {"path": str(receipt), "sha256": digest(receipt)}
    output.mkdir(parents=True, exist_ok=False)
    historical_audit = firrtl_hierarchy_audit(firrtl, hierarchy) if hierarchy is not None else None
    generated_hierarchy = output / "hierarchy.json"
    root = json.loads(hierarchy.read_bytes())["module_name"] if hierarchy is not None else circuit_root(firrtl)
    generated_hierarchy.write_text(json.dumps(derive_hierarchy(firrtl, root), indent=2) + "\n")
    sources = {
        role: {"path": str(path.resolve()), "sha256": digest(path)}
        for role, path in (("firrtl", firrtl), ("hierarchy", generated_hierarchy))
    }
    audit = firrtl_hierarchy_audit(firrtl, generated_hierarchy)
    soc, core = output / "soc.hw.mlir", output / "core.hw.mlir"
    prepared = output / "firtool-input.fir"
    drop_classes = drop_annotation_classes or []
    prepared_text, removed = prepare_firtool_input(firrtl.read_text(), drop_classes)
    prepared.write_text(prepared_text)
    command = [
        str(firtool.resolve()),
        str(prepared.resolve()),
        "--ir-hw",
        "--disable-annotation-unknown",
        "-o",
        str(soc.resolve()),
    ]
    tool_sha = digest(firtool)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (output / "firtool.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"firtool failed; inspect {output / 'firtool.log'}")
    text, included, missing = extract_module.extract(soc.read_text(), core_root)
    if missing:
        raise ValueError(f"core module closure has unresolved references: {missing}")
    core.write_text(text)
    for role, path in (("soc_hw", soc), ("core_hw", core)):
        sources[role] = {"path": str(path.resolve()), "sha256": digest(path)}
    if tool_sha != digest(firtool) or any(digest(item["path"]) != item["sha256"] for item in sources.values()):
        raise RuntimeError("production source/tool changed during observation")
    doc = {
        "schema": SCHEMA,
        "target": target,
        "config": config,
        "generator": generator,
        "sources": sources,
        "production": {
            "kind": "firrtl_to_hw_then_exact_module_closure",
            "command": command,
            "returncode": result.returncode,
            "tool": {"path": str(firtool.resolve()), "sha256": tool_sha},
            "core_root": core_root,
            "hierarchy_root": root,
            "hierarchy_derivation": "authored_subtree_root" if hierarchy is not None else "selected_circuit_root",
            "included_modules": included,
            "extractor_sha256": digest(Path(extract_module.__file__)),
            "producer_source": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
            **({"elaboration": verified_elaboration} if verified_elaboration is not None else {}),
            "input_preparation": {
                "path": str(prepared.resolve()),
                "sha256": digest(prepared),
                "drop_annotation_classes": drop_classes,
                "removed_annotations": removed,
            },
            **{f"{role}_sha256": sources[role]["sha256"] for role in ("firrtl", "soc_hw", "core_hw")},
        },
        "hierarchy_correspondence": audit,
        "diagnostics": (
            {
                "authored_hierarchy": {"path": str(hierarchy.resolve()), "sha256": digest(hierarchy)},
                "authored_hierarchy_correspondence": historical_audit,
            }
            if hierarchy is not None
            else {}
        ),
    }
    path = output / "source-selection.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "generator", "config", "core-root", "firrtl", "firtool", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--hierarchy", help="optional authored subtree root; otherwise derive the selected circuit")
    parser.add_argument("--drop-annotation-class", action="append", default=[])
    parser.add_argument("--elaboration-receipt", help="optional reproduced exact source-to-FIRRTL receipt")
    args = parser.parse_args(argv)
    path = produce_selection(
        target=args.target,
        generator=args.generator,
        config=args.config,
        core_root=args.core_root,
        firrtl=Path(args.firrtl),
        hierarchy=Path(args.hierarchy) if args.hierarchy else None,
        firtool=Path(args.firtool),
        output=Path(args.output),
        drop_annotation_classes=args.drop_annotation_class,
        elaboration_receipt=Path(args.elaboration_receipt) if args.elaboration_receipt else None,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
