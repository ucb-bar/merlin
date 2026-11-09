"""Adapt the existing experiment ABI 0.1 process transport to ordinary compilation.

Only a structurally verified merlin_iface module is admitted. This explicitly
selected ordinary service invokes the chosen compiler through its existing
process or scoped executor. It grants no experiment access, numerical verdict,
callable image or permission to execute the produced device program.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin.compiler.artifact import Artifact, Compilation

name = "merlin.package"
_OPTIONS = {"package.root", "package.contract", "package.timeout"}


def _tree(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("package transport requires direct source and contract directories")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("package transport cannot bind symlinks or special source files")
    return {"path": str(root), **hash_tree(root)}


def _formats(tensors):
    from merlin.common import quant_formats

    formats = {}
    for tensor in tensors.values():
        token = tensor.get("dtype")
        if not isinstance(token, str):
            raise ValueError("package transport requires every logical tensor dtype")
        if quant_formats.has(token):
            formats[token] = {"registered": asdict(quant_formats.get(token))}
        else:
            bits = quant_formats.machine_bits(token)
            if bits is None or bits <= 0:
                raise ValueError(f"package transport cannot admit unknown tensor dtype {token!r}")
            formats[token] = {"machine_bits": bits}
    return formats


@dataclass(frozen=True)
class _Selection:
    package: object
    root: Path
    contract: Path
    timeout: int
    declared: dict
    identity: dict

    def current(self):
        tool = self.package.tool
        if (
            self.package.directory != self.root
            or not tool.is_file()
            or tool.is_symlink()
            or not tool.resolve().is_relative_to(self.root)
        ):
            raise ValueError("selected package tool or directory changed")
        return {
            "source": _tree(self.root),
            "contract": _tree(self.contract / "schemas"),
            "tool": {"path": str(tool), "sha256": hashlib.sha256(tool.read_bytes()).hexdigest()},
            "manifest": self.package.manifest,
            "formats": _formats(self.declared["tensors"]),
        }

    def revalidate(self):
        if json.dumps(self.current(), sort_keys=True) != json.dumps(self.identity, sort_keys=True):
            raise ValueError("selected package source, schema or numerical declarations changed")


def _select(request):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, StringAttr, TensorType, UnregisteredOp
    from xdsl.parser import Parser

    from merlin.targetgen.contract import schemas
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir, to_generic_form
    from merlin.targetgen.package_runtime import load_package

    if request.source.kind != "merlin-interface":
        raise ValueError("package transport requires explicit merlin-interface input, not model MLIR")
    options = request.options
    if any(value is not None for value in (options.system, options.platform, options.harts)):
        raise ValueError("package transport does not admit deployment, harts or a callable image")
    values = options.parameters
    if "package.root" not in values or set(values) - _OPTIONS:
        raise ValueError("package transport requires package.root and refuses unknown options")
    timeout = values.get("package.timeout", 600)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("package.timeout must be a positive integer")
    for key in ("package.root", "package.contract"):
        if key in values and (not isinstance(values[key], str) or not values[key].strip()):
            raise ValueError(f"{key} must name an existing directory")
    root = Path(values["package.root"]).resolve(strict=True)
    contract = schemas.contract_dir(values.get("package.contract")).resolve(strict=True)
    for selected in (root, contract):
        if request.output.is_relative_to(selected) or selected.is_relative_to(request.output):
            raise ValueError("package transport output overlaps selected source or schemas")
    # Observe only schema definitions, never the contract directory's optional
    # reference corpus, samples or independent answers.
    source_identity, contract_identity = _tree(root), _tree(contract / "schemas")
    package = load_package(root, contract=contract)
    if package.manifest.get("artifact_type") != "mlir_oot_target_backend":
        raise ValueError("selected package is not an experiment compiler backend")
    if package.manifest.get("abi_version", "0.1") != "0.1":
        raise ValueError("ordinary package transport only admits experiment ABI 0.1")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    module = Parser(context, to_generic_form(request.source.text)).parse_module()
    module.verify()
    if (
        len(module.body.blocks) != 1
        or module.body.block.args
        or module.properties
        or set(module.attributes) != {"merlin_iface.version", "merlin_iface.target", "merlin_iface.abi_version"}
        or not isinstance(module.attributes.get("merlin_iface.version"), StringAttr)
        or module.attributes["merlin_iface.version"].data != "0.1"
        or any(
            not isinstance(op, UnregisteredOp)
            or not isinstance(op.attributes.get("op_name__"), StringAttr)
            or not op.attributes["op_name__"].data.startswith("merlin_iface.")
            or op.regions
            for op in module.body.block.ops
        )
    ):
        raise ValueError("package transport admits only flat merlin_iface operations")
    declared = parse_interface_mlir(request.source.text)
    leaves = [op for op in module.body.block.ops if op.attributes["op_name__"].data == "merlin_iface.tensor"]
    names = [op.attributes.get("name") for op in leaves]
    if any(not isinstance(item, StringAttr) or not item.data for item in names) or len(
        {item.data for item in names}
    ) != len(names):
        raise ValueError("interface tensor declarations need unique nonempty logical identities")
    for op, logical in zip(leaves, names, strict=True):
        if len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
            raise ValueError("interface tensor declarations need one tensor result")
        tensor = op.results[0].type
        record = declared["tensors"].get(logical.data, {})
        if record.get("shape") != list(tensor.get_shape()) or record.get("dtype") != str(tensor.element_type):
            raise ValueError("interface tensor inventory differs from its parsed SSA declaration")
    if len(declared["commands"]) != sum(
        op.attributes["op_name__"].data != "merlin_iface.tensor" for op in module.body.block.ops
    ):
        raise ValueError("interface command inventory differs from the parsed source operations")
    if declared["abi_version"] != "0.1" or not declared["target"] or declared["target"] != package.target:
        raise ValueError("interface ABI and target must match the selected package")
    schemas.validate_command_buffer(declared, contract=contract)
    selection = _Selection(package, root, contract, timeout, declared, {})
    identity = selection.current()
    if identity["source"] != source_identity or identity["contract"] != contract_identity:
        raise ValueError("selected package source or schemas changed during admission")
    # Copy mutable manifest/format data so a scoped executor cannot change both
    # the live package and the saved comparison through the same object.
    selection = _Selection(package, root, contract, timeout, declared, json.loads(json.dumps(identity)))
    selection.revalidate()
    return selection


def validate(request):
    """Validate the explicit transport boundary before writing or invoking code."""
    _select(request)


def compile(request):
    from merlin.targetgen import capsule_common, package_runtime

    selected = _select(request)
    request.output.mkdir(parents=True, exist_ok=False)
    source = request.output / "source.interface.mlir"
    source.write_text(request.source.text, encoding="utf-8")
    stages = request.output / "stages"
    stages.mkdir()

    def check_source():
        selected.revalidate()
        for path in (source, stages / "input.interface.mlir"):
            if not path.is_file() or path.is_symlink() or path.read_text(encoding="utf-8") != request.source.text:
                raise ValueError("selected interface source changed during package compilation")

    def invoke(package, entry, input_mlir, output_json=None, **kwargs):
        check_source()
        result = package_runtime.run_entrypoint(
            package, entry, input_mlir, output_json, invocation_directory=request.output, **kwargs
        )
        check_source()
        return result

    command, _ = capsule_common.lower_interface(
        selected.package,
        source,
        stages,
        contract=selected.contract,
        timeout=selected.timeout,
        invoke=invoke,
        artifact_name="target-artifact.txt",
    )
    check_source()
    if command.get("abi_version") != "0.1" or command.get("target") != selected.package.target:
        raise ValueError("emitted command-buffer ABI or target differs from selected input")
    emitted = command.get("tensors", {})
    for tensor, expected in selected.declared["tensors"].items():
        if tensor not in emitted or emitted[tensor].get("shape") != expected.get("shape"):
            raise ValueError("emitted command buffer omits or reshapes a declared logical tensor")
    formats = _formats(emitted)
    artifact = stages / "target-artifact.txt"
    if not artifact.read_text(encoding="utf-8").strip():
        raise ValueError("selected package emitted no target artifact")
    products = []
    for path in sorted(request.output.rglob("*")):
        if not path.is_file():
            continue
        role = (
            "command-buffer"
            if path.name == "command_buffer.json"
            else "invocation-record"
            if path.name == "invocation.json"
            else "intermediate"
        )
        products.append(Artifact.from_file(path, kind="json" if path.suffix == ".json" else "data", role=role))
    check_source()
    return Compilation(
        request.output,
        name,
        request.source.sha256,
        tuple(products),
        metadata={
            "scope": "experiment ABI 0.1 structural command-buffer transport",
            "selected_package": selected.identity,
            "emitted_formats": formats,
            "artifact_language": "opaque text declared by the package; not assumed to be LLVM",
            "numerical_verification": "not_run",
            "execution_observed": False,
        },
    )
