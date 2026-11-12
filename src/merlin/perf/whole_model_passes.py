"""A package's own MLIR passes over the whole-model interface, run before the builder splits it.

The whole-model statement (:mod:`merlin.llvmlower.whole_program`) turns a captured model into compute
GROUPS and puts each -- or, since :mod:`merlin.llvmlower.region_legality`, a legal run of them -- to
the package. Neither lets a package change what the MODEL ITSELF says before that split happens: a
package that would rather restate its own program (canonicalize a chain of casts, hoist a constant,
whatever transform it can prove preserves the function) has no way to ask for it. This module is that
way, and nothing more: it runs the package's OWN declared passes, over the package's OWN interpreter,
through the exact entrypoint mechanism every other package call already uses
(:func:`merlin.targetgen.package_runtime.run_entrypoint`) -- never by importing or executing package code
in this process.

**Declaration is data, not code.** A package's ``manifest.yaml`` may carry a top-level
``whole_model_passes: [name, ...]`` list; each name must also be one of its own declared ``commands``
(the same manifest section every entrypoint already comes from), run in the ORDER the list gives, each
one's own declared ``argv`` invoked exactly like any other entrypoint (``{tool}``/``{input_mlir}``/
``{output_json}``) with the previous stage's output as the next stage's input. Nothing here executes a
name the manifest omits, and nothing here decides what a pass may compute: verifying that a pass did
not change the model's own function is a SEPARATE, independent check
(:func:`merlin.perf.whole_model_build.verify_passes_preserve_semantics`), because deciding whether to
trust a transform is not this module's job -- running the ones asked for, in order, and recording
exactly what happened, is.

**Fails closed, one pass at a time.** A pass that cannot even be invoked, or whose own reply is not
parseable MLIR text, stops the chain right there: the caller gets back the LAST module every prior
pass in the chain actually produced (the original, if the very first pass failed), together with a
per-pass record naming which ones ran and which one stopped it. A caller that skipped this and used
whatever text happened to be on disk would silently grade one package's chosen edits and call it every
other package's default behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["PASS_LIST_KEY", "declared_passes", "apply_passes"]

#: The manifest's own list of pass command names, in the order they run. A top-level key (the
#: manifest schema's root already allows arbitrary additional properties), never inferred from a
#: naming convention on `commands` -- a sorted-by-name guess is an assumption about intent this
#: module does not make.
PASS_LIST_KEY = "whole_model_passes"


class PassDeclarationError(RuntimeError):
    """The manifest's own pass list cannot be trusted, named by what is wrong with it."""


def declared_passes(package: Any) -> tuple[str, ...]:
    """The package's own ordered pass list, or ``()`` when it declares none.

    Raises :class:`PassDeclarationError` when the list names something that is not one of the
    package's own declared commands: a manifest that cannot even name its passes correctly is not
    trusted to run any of them, rather than running the ones that happen to resolve and silently
    skipping the rest.
    """
    manifest = getattr(package, "manifest", None) or {}
    names = manifest.get(PASS_LIST_KEY)
    if not names:
        return ()
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise PassDeclarationError(f"{PASS_LIST_KEY!r} must be a list of non-empty command names, got {names!r}")
    commands = manifest.get("commands") or {}
    missing = [n for n in names if n not in commands]
    if missing:
        raise PassDeclarationError(
            f"{PASS_LIST_KEY!r} names {missing}, which the package declares no command for "
            f"(it declares {sorted(commands)})"
        )
    return tuple(names)


def apply_passes(
    interface_path: str | Path,
    package: Any,
    *,
    work: str | Path,
    timeout: int = 600,
    run=None,
) -> tuple[Path, list[dict[str, Any]], str]:
    """Run every pass ``package`` declares, in order, chaining each stage's output into the next.

    Returns ``(final_path, records, stopped_why)``: ``final_path`` is the last module any pass
    actually produced (``interface_path`` itself when no pass ran or the first one failed);
    ``records`` is one entry per ATTEMPTED pass (``name``, ``input``, ``output`` paths and digests,
    ``ok``); ``stopped_why`` is ``""`` when every declared pass ran, else the reason the chain
    stopped, naming which pass.
    """
    from merlin.common.provenance import source_digest
    from merlin.targetgen import package_runtime as OR

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    names = declared_passes(package)
    current = Path(interface_path)
    records: list[dict[str, Any]] = []
    for position, name in enumerate(names):
        destination = work / f"pass_{position:02d}_{name}.mlir"
        try:
            done = (run or OR.run_entrypoint)(package, name, current, destination, timeout=timeout)
        except Exception as error:  # noqa: BLE001 -- named and recorded, never raised past this point
            records.append({"pass": name, "ok": False, "why": f"{type(error).__name__}: {error}"})
            return current, records, f"pass {name!r} could not be invoked: {type(error).__name__}: {error}"
        if (
            getattr(done, "returncode", 1) != 0
            or not destination.is_file()
            or not destination.read_text(encoding="utf-8").strip()
        ):
            reason = (getattr(done, "stderr", "") or "the pass wrote no output")[:300]
            records.append({"pass": name, "ok": False, "why": reason})
            return current, records, f"pass {name!r} produced no usable module: {reason}"
        records.append(
            {
                "pass": name,
                "ok": True,
                "input": str(current),
                "input_sha256": source_digest([current]),
                "output": str(destination),
                "output_sha256": source_digest([destination]),
            }
        )
        current = destination
    return current, records, ""
