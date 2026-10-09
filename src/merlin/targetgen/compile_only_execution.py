"""Ordinary package lowering and no-data linking, without numerical execution.

The caller must independently establish original source/ABI provenance. This
transport runs the same package commands and LLVM/object/link services as normal
execution. Link success is not semantic, index, resource or output-store proof.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.digest import sha256_file
from merlin.targetgen.compiler_library import selected_library_record
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor, prepare_linkage
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def _plain(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(owner.is_symlink() for owner in (path, *path.parents)):
        raise ValueError("compile-only source/product owner is indirect")
    return path


def _pin(path):
    path = _plain(path)
    if not path.is_file():
        raise ValueError("compile-only source/product is absent")
    return {"path": str(path), "sha256": sha256_file(path)}


def _tree(root):
    root = _plain(root)
    if not root.is_dir():
        raise ValueError("compile-only source tree is absent")
    members = {}
    for path in sorted(root.rglob("*")):
        _plain(path)
        if path.is_file():
            members[path.relative_to(root).as_posix()] = _pin(path)
        elif not path.is_dir():
            raise ValueError("compile-only source tree contains a special file")
    if not members:
        raise ValueError("compile-only source tree is empty")
    return members


def _retained_entry(*, elf, entry_symbol, readelf, output, timeout):
    from merlin.targetgen.elf_lanes import executable_sections

    process = invocation_record.run(
        [str(readelf), "--wide", "--symbols", str(elf)],
        directory=output,
        stage="compile_only_linked_entry",
        inputs=(elf,),
        dependencies=(Path(__file__),),
        capture_output=True,
        text=True,
        timeout=min(timeout, 30),
    )
    if process.returncode:
        raise ValueError("compile-only linked symbol inspection failed")
    definitions = set()
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == "FUNC" and fields[6] != "UND" and fields[7] == entry_symbol:
            definitions.add((int(fields[1], 16), int(fields[2])))
    sections = executable_sections(elf.read_bytes())
    if len(definitions) != 1:
        raise ValueError("compile-only linker did not retain one actual kernel definition")
    address, size = next(iter(definitions))
    if size <= 0 or not any(start <= address and address + size <= start + extent for _, _, extent, start in sections):
        raise ValueError("compile-only kernel definition is outside the linked executable sections")
    return {"symbol": entry_symbol, "address": address, "size_bytes": size, "scope": "static linked definition"}


def _recipe(service):
    recipe = service.recipe.with_effective_abi()
    return {
        "compiler": _pin(recipe.compiler),
        "include_roots": [str(path) for path in recipe.include_roots],
        "support_sources": [_pin(path) for path in recipe.support_sources],
        "link_script": _pin(recipe.link_script),
        "load_address": recipe.load_address,
        "cflags": list(recipe.cflags),
        "ldflags": list(recipe.ldflags),
        "kernel_stack_frame": recipe.require_kernel_stack_frame().record(),
        "header_dependencies": [_pin(path) for path in recipe.header_dependencies],
    }


def compile_source_only(
    *,
    package_dir,
    source,
    original_abi,
    contract_root,
    target,
    output_root,
    build_service,
    elf_admission,
    readelf,
    timeout_s,
    compiler_library=None,
    compiler_library_root=None,
):
    """Compile through the ordinary package pipeline; retain exact actual products.

    No capsule, generator, tensor input value, golden or execution service is
    accepted. A caller-written declaration does not qualify source provenance.
    """
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import package_runtime as P
    from merlin.targetgen.contract.compile import link_elf, llvm_mlir_to_object

    if (
        type(build_service) is not BuildOnlyService
        or type(elf_admission) is not LinkedElfAdmissionService
        or type(original_abi) is not CompileOnlySourceAbi
        or type(timeout_s) is not int
        or not 0 < timeout_s <= 600
    ):
        raise ValueError("compile-only transport requires exact bounded build/ISA services and original source ABI")
    if P.active_package_executor() is None:
        raise ValueError("compile-only transport requires the ordinary scoped package executor")
    package_dir, source, contract_root, output = map(_plain, (package_dir, source, contract_root, output_root))
    readelf = _plain(readelf)
    if not readelf.is_file() or not os.access(readelf, os.X_OK):
        raise ValueError("compile-only linked inspection requires an explicit executable tool")
    if output.exists() or any(
        output.is_relative_to(owner) or owner.is_relative_to(output)
        for owner in (package_dir, source.parent, contract_root)
    ):
        raise ValueError("compile-only products require a fresh separate private owner")
    frozen = {
        "package_root": str(package_dir),
        "contract_root": str(contract_root),
        "package": _tree(package_dir),
        "source": _pin(source),
        "contract": _tree(contract_root),
        "readelf": _pin(readelf),
        "original_abi": original_abi.record(),
    }
    library = selected_library_record(compiler_library, compiler_library_root)
    library_sources = (
        tuple(compiler_library_root / member.path for member in compiler_library.members) if library else ()
    )
    if library is not None:
        frozen["compiler_library"] = library
    build_service.verify(target)
    admission = elf_admission.verify(target)
    output.mkdir(parents=True, mode=0o700)
    report = {
        "schema": "merlin.compile_only_transport.v1",
        "target": target,
        "compilation_status": "not_completed",
        "execution": "not_attempted",
        "numeric": "not_attempted",
        "static_obligations": {
            name: "UNKNOWN"
            for name in ("semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage")
        },
        "inputs": frozen,
        "build_source_pins": build_service.source_pins,
        "build_recipe": _recipe(build_service),
        "instruction_selection": admission,
        "scope": "ordinary structural compilation and linked instruction policy; no numerical/physical proof",
    }
    deadline = time.monotonic() + timeout_s

    def remaining():
        if selected_library_record(compiler_library, compiler_library_root) != library:
            raise ValueError("compile-only selected compiler library changed")
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("compile-only total transport budget expired")
        return left

    def invoke(*args, **kwargs):
        return P.run_entrypoint(*args, **{**kwargs, "timeout": remaining(), "invocation_directory": generated})

    try:
        package = P.load_package(package_dir, contract=contract_root)
        if package.manifest.get("target") != target:
            raise ValueError("compile-only package differs from the selected target")
        P.integrity_scan(
            package,
            **(
                {"compiler_library": compiler_library, "compiler_library_root": compiler_library_root}
                if library
                else {}
            ),
        )
        P.build_package(package, timeout=remaining())
        generated = output / "generated"
        products = tuple(
            generated / name
            for name in ("input.interface.mlir", "command_buffer.json", "lowered.target.mlir", "lowered.llvm.mlir")
        )
        with invocation_record.observe_call(
            output,
            stage="compile_only_source_lowering",
            function=CC.lower_interface,
            arguments={"target": target, "timeout_s": timeout_s},
            inputs=(source,),
            outputs=products,
            dependencies=(
                *tuple(Path(row["path"]) for kind in ("package", "contract") for row in frozen[kind].values()),
                *library_sources,
            ),
        ) as observation:
            cb, artifact = CC.lower_interface(
                package, source, generated, contract=contract_root, timeout=timeout_s, invoke=invoke
            )
            observation.returned(stdout=artifact)
        entry = build_service.recipe.require_kernel_stack_frame().entry_symbol
        linkage, binding = prepare_linkage(cb=cb, lowered_mlir=artifact, entry_symbol=entry, original_abi=original_abi)
        build = output / "build"
        obj = llvm_mlir_to_object(
            artifact, build, target=target, _build_service=build_service, build_timeout_s=math.ceil(remaining())
        )
        elf = link_elf(
            cb,
            obj,
            build,
            target=target,
            _build_service=build_service,
            _compile_only_linkage=linkage,
            build_timeout_s=math.ceil(remaining()),
        )
        retained = _retained_entry(elf=elf, entry_symbol=entry, readelf=readelf, output=output, timeout=remaining())
        with invocation_record.observe_call(
            output,
            stage="compile_only_whole_elf_policy",
            function=elf_admission.evaluate,
            arguments={"target": target},
            inputs=(elf,),
            dependencies=tuple(Path(path) for path, _ in elf_admission.source_pins),
        ) as observation:
            policy = elf_admission.evaluate(elf=elf, target=target, evidence_root=output / "instruction_policy")
            observation.returned(stdout=json.dumps(policy, sort_keys=True))
        remaining()
        report.update(
            compilation_status="linked",
            original_abi_binding=binding,
            retained_entry=retained,
            elf=_pin(elf),
            instruction_policy=policy,
        )
        if policy["status"] != "accepted":
            report["compilation_status"] = "refused_by_instruction_policy"
        if frozen != {
            "package_root": str(package_dir),
            "contract_root": str(contract_root),
            "package": _tree(package_dir),
            "source": _pin(source),
            "contract": _tree(contract_root),
            "readelf": _pin(readelf),
            "original_abi": original_abi.record(),
            **(
                {"compiler_library": selected_library_record(compiler_library, compiler_library_root)}
                if library
                else {}
            ),
        }:
            raise ValueError("compile-only original sources/tools changed during compilation")
        build_service.verify(target)
        if _recipe(build_service) != report["build_recipe"]:
            raise ValueError("compile-only selected build recipe changed")
        if elf_admission.verify(target) != admission:
            raise ValueError("compile-only linked instruction selection changed")
        report["invocations"] = []
        for path in sorted(output.rglob("invocation.json")):
            observed = invocation_record.verify(path)
            report["invocations"].append({**_pin(path), "stage": observed["stage"]})
        report["products"] = {
            path.relative_to(output).as_posix(): _pin(path) for path in sorted(output.rglob("*")) if path.is_file()
        }
        return report
    except Exception as error:
        report.update(compilation_status="unavailable", failure={"type": type(error).__name__, "detail": str(error)})
        report["retained_unqualified_records"] = [_pin(path) for path in sorted(output.rglob("invocation.json"))]
        raise
    finally:
        (output / "compile_only_result.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def verify_compile_only_report(
    path, *, build_service, elf_admission, compiler_library=None, compiler_library_root=None
):
    """Reopen actual transport evidence; no JSON can issue a semantic qualification."""
    path = _plain(path)
    report = json.loads(path.read_text())
    if report["inputs"].get("compiler_library") != selected_library_record(compiler_library, compiler_library_root):
        raise ValueError("compile-only report differs from the explicit compiler library selection")
    if (
        report.get("schema") != "merlin.compile_only_transport.v1"
        or report.get("compilation_status") != "linked"
        or report.get("execution") != "not_attempted"
        or report.get("numeric") != "not_attempted"
    ):
        raise ValueError("compile-only report has no complete linked observation")
    if report.get("static_obligations") != {
        name: "UNKNOWN"
        for name in ("semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage")
    }:
        raise ValueError("compile-only transport cannot qualify a static semantic obligation")
    target = report["target"]
    build_service.verify(target)
    if _recipe(build_service) != report["build_recipe"]:
        raise ValueError("compile-only selected build recipe changed")
    if build_service.source_pins != tuple(tuple(pin) for pin in report["build_source_pins"]):
        raise ValueError("compile-only build support differs from the original selection")
    selection = report["instruction_selection"]
    original_selection = {**selection, "source_pins": tuple(tuple(pin) for pin in selection["source_pins"])}
    if elf_admission.verify(target) != original_selection:
        raise ValueError("compile-only ISA support differs from the original selection")
    for kind in ("package", "contract"):
        if _tree(report["inputs"][kind + "_root"]) != report["inputs"][kind]:
            raise ValueError("compile-only original source membership changed")
    for kind in ("source", "readelf"):
        if _pin(report["inputs"][kind]["path"]) != report["inputs"][kind]:
            raise ValueError("compile-only original input/tool changed")
    products = {
        member.relative_to(path.parent).as_posix(): _pin(member)
        for member in sorted(path.parent.rglob("*"))
        if member.is_file() and member != path
    }
    if products != report["products"]:
        raise ValueError("compile-only product membership changed")
    records = []
    for member in sorted(path.parent.rglob("invocation.json")):
        observed = invocation_record.verify(member)
        records.append({**_pin(member), "stage": observed["stage"]})
    if records != report["invocations"] or not records:
        raise ValueError("compile-only actual invocation membership changed")
    abi = report["inputs"]["original_abi"]
    original_abi = CompileOnlySourceAbi(
        *(
            tuple(CompileOnlyTensor(row["name"], tuple(row["shape"]), row["dtype"]) for row in abi[role])
            for role in ("inputs", "outputs")
        )
    )
    generated = path.parent / "generated"
    _, binding = prepare_linkage(
        cb=json.loads((generated / "command_buffer.json").read_text()),
        lowered_mlir=(generated / "lowered.llvm.mlir").read_text(),
        entry_symbol=build_service.recipe.require_kernel_stack_frame().entry_symbol,
        original_abi=original_abi,
    )
    if binding != report["original_abi_binding"]:
        raise ValueError("compile-only original ABI binding changed")
    elf_admission.revalidate(elf=Path(report["elf"]["path"]), result=report["instruction_policy"], target=target)
    return report
