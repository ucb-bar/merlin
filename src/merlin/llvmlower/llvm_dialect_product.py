"""Retain the actual serial translation input, without semantic authority.

The ordinary native runner prints its post-pass module immediately before the
same module is translated. Both products belong to one recorded subprocess.
Raw translation is retained separately from later normalization and codegen.
This does not prove descriptor preconditions, imported dependencies or effects.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json

SCHEMA = "merlin.serial_llvm_dialect_product.v1"
_TRANSLATE = "f.write(str(llvm.translate_module_to_llvmir(module.operation)))"
_SCOPE = "actual post-pass module and translation custody only; ABI, semantics, effects and runtime unqualified"


def _pin(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or path.resolve() != path or not path.is_file():
        raise ValueError("retained LLVM products require canonical ordinary files")
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _input(pin):
    return {key: pin[key] for key in ("path", "sha256")}


def run_translation(command, *, timeout, audit, retention, source, runner, dependencies, scalar_stage, scalar_carrier):
    """Keep the ordinary transport; observe products only for selected retention."""
    try:
        if retention is not None:
            return retention.run(command, source=source, runner=runner, dependencies=dependencies, timeout=timeout)
        if scalar_stage is None:
            return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        from .source_stage_transport import run_command

        return run_command(
            command,
            directory=scalar_stage,
            callback=scalar_carrier.callback,
            max_source_bytes=scalar_carrier.max_source_bytes,
            max_response_bytes=scalar_carrier.max_response_bytes,
            timeout=timeout,
        )
    except BaseException as exc:
        if audit is not None:
            try:
                audit.collect_views()
            except (OSError, ValueError) as audit_error:
                exc.add_note(f"IR inspection prefix could not be recorded: {type(audit_error).__name__}")
        raise


def select_retention(work, enabled, *, omp, scalar_stage):
    if not enabled:
        return None
    if omp or scalar_stage is not None:
        raise ValueError("LLVM dialect retention supports ordinary serial translation only")
    return SerialLLVMDialectRetention.prepare(work)


@dataclass
class SerialLLVMDialectRetention:
    """Invocation-local product owner, not a proof or reusable authority."""

    directory: Path
    invocation: Path | None = None

    @classmethod
    def prepare(cls, work):
        return cls(Path(tempfile.mkdtemp(prefix="llvm-dialect-", dir=Path(work))).resolve())

    @property
    def module(self):
        return self.directory / "module.mlir"

    @property
    def translation(self):
        return self.directory / "translated.ll"

    def emitter(self, ordinary):
        if ordinary.count(_TRANSLATE) != 1:
            raise ValueError("retention requires the ordinary serial module translation")
        printed = (
            f"_retained = open({str(self.module)!r}, 'x', encoding='utf-8'); "
            "_retained.write(module.operation.get_asm(print_generic_op_form=True)); _retained.close(); "
        )
        return ordinary.replace(_TRANSLATE, printed + _TRANSLATE)

    def run(self, command, *, source, runner, dependencies, timeout):
        environment = dict(os.environ)
        with I.observe(
            self.directory,
            stage="serial_upstream_llvm_translation",
            argv=command,
            inputs=(source, runner),
            outputs=(self.module, self.translation),
            dependencies=(Path(__file__), *dependencies),
            env=environment,
        ) as observed:
            self.invocation = observed.path
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
            observed.complete(result)
            return result

    def returned(self, *, source, runner, llvm_ir):
        if self.invocation is None:
            raise ValueError("retained LLVM product has no actual translation invocation")
        document = {
            "schema": SCHEMA,
            "scope": _SCOPE,
            "source": _pin(source),
            "runner": _pin(runner),
            "llvm_dialect": _pin(self.module),
            "translated_llvm_ir": _pin(self.translation),
            "producer": _pin(Path(__file__)),
            "invocation": _pin(self.invocation),
            "returned_llvm_ir": {
                "sha256": hashlib.sha256(llvm_ir.encode()).hexdigest(),
                "bytes": len(llvm_ir.encode()),
            },
        }
        receipt = self.directory / "product.json"
        with receipt.open("xb") as output:
            output.write(canonical_json(document) + b"\n")
        verify_llvm_dialect_product(receipt, llvm_ir=llvm_ir)
        return _pin(receipt)

    def bind(self, recipe, selection, *, source, runner, llvm_ir):
        product = self.returned(source=source, runner=runner, llvm_ir=llvm_ir)
        recipe.bind_source("retained_llvm_dialect", self.module)
        recipe.bind_source("llvm_dialect_product_receipt", Path(product["path"]))
        if selection is not None:
            selection["llvm_dialect_product"] = product


def verify_llvm_dialect_product(receipt, *, llvm_ir=None):
    """Reopen actual production; returned data cannot grant correctness roles."""
    from merlin.common.strict_json import loads

    receipt = Path(receipt).absolute()
    _pin(receipt)
    document = loads(receipt.read_bytes())
    files = ("source", "runner", "llvm_dialect", "translated_llvm_ir", "producer", "invocation")
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "scope", "returned_llvm_ir", *files}
        or (document["schema"] != SCHEMA or document["scope"] != _SCOPE)
    ):
        raise ValueError("retained LLVM product has unsupported custody fields")
    for name in files:
        if not isinstance(document[name], dict) or set(document[name]) != {"path", "sha256", "bytes"}:
            raise ValueError("retained LLVM product has malformed file identity")
        if canonical_json(_pin(document[name]["path"])) != canonical_json(document[name]):
            raise ValueError("retained LLVM product or original producer changed")
    returned = document["returned_llvm_ir"]
    if (
        not isinstance(returned, dict)
        or set(returned) != {"sha256", "bytes"}
        or type(returned["bytes"]) is not int
        or returned["bytes"] <= 0
        or not isinstance(returned["sha256"], str)
        or len(returned["sha256"]) != 64
        or any(ch not in "0123456789abcdef" for ch in returned["sha256"])
    ):
        raise ValueError("retained LLVM product has malformed returned IR identity")
    if document["producer"] != _pin(Path(__file__)) or any(
        not Path(document[name]["path"]).is_relative_to(receipt.parent)
        for name in ("llvm_dialect", "translated_llvm_ir", "invocation")
    ):
        raise ValueError("retained LLVM product escaped its original producer")
    observed = I.verify(Path(document["invocation"]["path"]))
    if (
        observed["kind"] != "subprocess"
        or observed["stage"] != "serial_upstream_llvm_translation"
        or observed["inputs"]
        != sorted((_input(document[name]) for name in ("source", "runner")), key=lambda p: p["path"])
        or observed["outputs"]
        != sorted((_input(document[name]) for name in ("llvm_dialect", "translated_llvm_ir")), key=lambda p: p["path"])
        or _input(document["producer"]) not in observed["dependencies"]
        or observed["argv"][1:4] != [document[name]["path"] for name in ("runner", "source", "translated_llvm_ir")]
    ):
        raise ValueError("retained LLVM products do not join the same ordinary translation invocation")
    if llvm_ir is not None and canonical_json(document["returned_llvm_ir"]) != canonical_json(
        {"sha256": hashlib.sha256(llvm_ir.encode()).hexdigest(), "bytes": len(llvm_ir.encode())}
    ):
        raise ValueError("retained LLVM product changed its returned IR binding")
    return document
