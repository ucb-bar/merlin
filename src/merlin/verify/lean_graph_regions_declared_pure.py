"""NEW original-bound captured-literal development controller; stdlib only.

Run directly with Python -I -S -B. Only private source/object copies and exact
reviewed loader invocations are used. This never invokes MLIR, an old profile,
Lake, a downloader, a numerical oracle or graph qualification. New project
objects are development outputs; ORIGINAL runtime/SDK expectations never change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
from pathlib import Path


def save(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reference(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class DevelopmentRefusal(ValueError):
    pass


class Context:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve(strict=True)
        self.destination = args.destination.resolve()
        self.destination.mkdir(parents=True, exist_ok=False)
        self.source = self.destination / "source"
        self.objects = self.destination / "compiled"
        self.logs = self.destination / "commands"
        for folder in (self.source, self.objects, self.logs):
            folder.mkdir()
        self.api = runpy.run_path(
            str(Path(__file__).with_name("lean_graph_regions_declared_loader_plan.py")),
            run_name="_declared_original_pin_api",
        )
        self.trace = runpy.run_path(
            str(Path(__file__).with_name("lean_graph_regions_declared_pure_trace.py")), run_name="_declared_pure_trace"
        )
        self.provenance = self.api["parse_json"](args.provenance.read_bytes())
        self.pins = self.api["Pins"].original_union(self.provenance)
        self.sdk = self.api["parse_json"](self.api["read_original"](self.provenance["original_F_SDK_manifest"]))
        self.selection = None
        if args.selection is not None:
            selector = runpy.run_path(str(Path(__file__).with_name("lean_graph_regions_declared_pure_selection.py")))
            selected_document = self.api["parse_json"](args.selection.read_bytes())
            original_w = self.api["parse_json"](self.api["read_original"](self.provenance["original_W_fixed_context"]))
            self.selection = selector["select"](
                selected_document,
                self.provenance,
                self.sdk,
                original_w,
                strict=self.api["strict_ref"],
                references=self.api["references"],
            )
            selected_pins = self.api["Pins"]()
            for pin in self.selection["files"].values():
                selected_pins.add(pin, dict(source="reviewed ALL653 five-file module families + original11 runtime"))
            # Selection occurs BEFORE any SDK/runtime preflight/file access.
            # The old full-SDK mode/capsules are unchanged and never invoked here.
            self.pins = selected_pins
        if self.sdk["module_count"] != 653 or len(self.sdk["modules"]) != 653 or len(set(self.sdk["modules"])) != 653:
            raise DevelopmentRefusal("unexpected original SDK module inventory")
        self.lean = Path(self.provenance["original_runtime_refs"][0]["path"])
        self.toolchain = self.lean.parent.parent
        self.interpreter = "/usr/lib/ld-linux-x86-64.so.2"
        self.environment = dict(
            LANG="C",
            LC_ALL="C",
            LEAN_SYSROOT=str(self.toolchain),
            LEAN_PATH=str(self.objects) + ":" + str(self.toolchain / "lib/lean"),
            LEAN_SRC_PATH=str(self.source) + ":" + str(self.toolchain / "src/lean"),
            LD_DEBUG="libs,files",
        )
        self.loader = [
            self.interpreter,
            "--inhibit-cache",
            "--glibc-hwcaps-mask",
            "",
            "--library-path",
            ":".join(self.provenance["proposed_runtime_directories"]),
            str(self.lean),
        ]
        baseline = args.baseline.read_text()
        self.misses = self.trace["baseline_misses"](baseline, set(self.pins.files))
        self.finite_review = None
        if args.symbol_review is not None:
            if self.selection is None:
                raise DevelopmentRefusal(
                    "finite optional symbol requires the reviewed original module/runtime selection"
                )
            reviewer = runpy.run_path(str(Path(__file__).with_name("lean_graph_regions_declared_pure_review.py")))
            self.finite_review = reviewer["reviewed_pair"](
                self.root,
                args.symbol_review,
                self.provenance,
                reference(Path(__file__).with_name("lean_graph_regions_declared_pure_trace.py")),
                self.api,
            )
        self.trace["parse"](
            baseline, self.provenance["original_runtime_refs"], str(self.lean), self.interpreter, self.misses
        )
        self.sources, self.bindings, self.graph, self.active = {}, {}, {}, set()
        self.controls = [
            reference(Path(__file__).with_name(name))
            for name in (
                "lean_graph_regions_declared_pure.py",
                "lean_graph_regions_declared_pure_trace.py",
                "lean_graph_regions_declared_loader_plan.py",
                "lean_graph_regions_declared_elf.py",
            )
        ]
        if args.selection is not None:
            self.controls.append(reference(Path(__file__).with_name("lean_graph_regions_declared_pure_selection.py")))
        if args.symbol_review is not None:
            self.controls.append(reference(Path(__file__).with_name("lean_graph_regions_declared_pure_review.py")))
        self.documents = [
            reference(path.resolve(strict=True)) for path in (args.provenance, args.prior_closure, args.baseline)
        ]
        if args.selection is not None:
            self.documents.append(reference(args.selection.resolve(strict=True)))
        if args.symbol_review is not None:
            self.documents.append(reference(args.symbol_review.resolve(strict=True)))
            self.documents.append(self.finite_review["input_ref"])
        self.cache = {} if args.cache is None else self.api["parse_json"](args.cache.read_bytes())
        self.context = dict(
            schema="merlin.graph-regions.pure-development-context.v1",
            original_provenance=self.provenance,
            original_unique_binding_count=len(self.pins.files),
            original_sdk_module_count=653,
            environment=self.environment,
            loader=self.loader,
            baseline_trace=reference(args.baseline.resolve(strict=True)),
            qualification=False,
            old_failed_context_repaired=False,
            formal_OS_default_directory_denial=False,
            arbitrary_future_late_loads_ruled_out=False,
            trust=[
                "OS/kernel/vDSO",
                "pinned loader and Lean module reporting/kernel",
                "original SDK metadata",
                "host transport and stable file association",
            ],
        )
        if self.selection is not None:
            self.context["schema"] = "merlin.graph-regions.original-module-runtime-development.v1"
            self.context["dependency_rule"] = "ALL five ORIGINAL files of EVERY653 Init module + ORIGINAL11 runtime"
            self.context["selection_counts"] = {key: value for key, value in self.selection.items() if key != "files"}
            self.context["selection_document"] = reference(args.selection.resolve(strict=True))
            self.context["failed_full_SDK_context_promoted"] = False
        if self.finite_review is not None:
            self.context["finite_optional_symbol_review"] = self.finite_review
        save(self.destination / "CONTEXT.json", self.context)
        self.freeze_sources()
        save(self.destination / "SOURCES.json", self.sources)
        save(self.destination / "CONTROLLER-SOURCES.json", self.controls)
        save(self.destination / "ORIGINAL-BINDINGS.json", self.pins.files)
        save(self.destination / "DOCUMENTS.json", self.documents)
        self.preflight()

    def freeze_sources(self):
        prior = self.api["parse_json"](self.args.prior_closure.read_bytes())
        for relative, item in prior.items():
            if relative.startswith("merlin/lean/"):
                suffix = relative.removeprefix("merlin/lean/")
            elif relative.startswith("src/merlin/verify/lean/"):
                suffix = relative.removeprefix("src/merlin/verify/lean/")
            else:
                raise DevelopmentRefusal("unrecognized preserved project source root")
            data = self.api["read_original"](item["original"])
            self.freeze(suffix, data, item["original"], "unchanged qualified v1 source; privately built")
        for path in sorted((self.root / "src/merlin/verify/lean/Merlin").glob("GraphRegionsDeclaredCapture*.lean")):
            current = reference(path)
            self.freeze("Merlin/" + path.name, path.read_bytes(), current, "new unqualified owned candidate")

    def freeze(self, suffix, data, origin, status):
        module = ".".join(Path(suffix).with_suffix("").parts)
        if module in self.sources:
            raise DevelopmentRefusal("duplicate project source module")
        target = self.source / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied = reference(target)
        if any(copied[key] != origin[key] for key in ("bytes", "sha256")):
            raise DevelopmentRefusal("source snapshot/original content association differs")
        self.sources[module] = dict(source=copied, original=origin, provenance=status)

    def preflight(self):
        if Path("/etc/ld.so.preload").exists() or Path("/etc/ld.so.preload").is_symlink():
            raise DevelopmentRefusal("system preload file exists; NEW context stopped")
        self.api["read_original"](self.provenance["original_F_SDK_manifest"])
        self.api["read_original"](self.provenance["original_W_fixed_context"])
        for pin in self.pins.files.values():
            self.api["read_original"](pin)
        for item in self.sources.values():
            self.api["read_original"](item["source"])
            self.api["read_original"](item["original"])
        for pin in self.controls:
            self.api["read_original"](pin)
        for pin in self.documents:
            self.api["read_original"](pin)
        for item in self.bindings.values():
            for pin in item["outputs"].values():
                self.api["read_original"](pin)
        return dict(
            original_sdk_runtime_bindings_checked=len(self.pins.files),
            original_manifests_checked=2,
            project_sources_checked=len(self.sources),
            controller_sources_checked=len(self.controls),
            project_modules_with_bound_outputs=len(self.bindings),
            all_bound_bytes_unchanged=True,
            system_preload_absent=True,
            graph_or_native_qualification=False,
        )

    def invoke(self, label, arguments):
        directory = self.logs / f"{len(list(self.logs.iterdir())):04d}-{label}"
        directory.mkdir()
        command = self.loader + ["-j1", *arguments]
        save(directory / "INTENT.json", dict(command=command, environment=self.environment, cwd=str(self.destination)))
        save(directory / "PRECHECK.json", self.preflight())
        with subprocess.Popen(
            command,
            cwd=self.destination,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                (directory / "stdout.txt").write_text(stdout)
                (directory / "stderr.txt").write_text(stderr)
                save(directory / "PROCESS.json", dict(pid=process.pid, timed_out=True, returncode=process.returncode))
                raise DevelopmentRefusal("NEW native command timed out; no unchanged retry") from None
        (directory / "stdout.txt").write_text(stdout)
        (directory / "stderr.txt").write_text(stderr)
        save(directory / "PROCESS.json", dict(pid=process.pid, command=command, returncode=process.returncode))
        save(directory / "POSTCHECK.json", self.preflight())
        misses = self.misses
        if self.finite_review is not None and process.returncode == 0:
            misses = misses | {self.finite_review["pair"]}
        observed = self.trace["parse"](
            stderr,
            self.provenance["original_runtime_refs"],
            str(self.lean),
            self.interpreter,
            misses,
            expected_pid=process.pid,
            project_initializers={"initialize_" + module.replace(".", "_") for module in self.bindings},
        )
        save(directory / "RUNTIME.json", observed)
        if process.returncode < 0:
            raise DevelopmentRefusal("NEW native signal; context stopped without retry")
        if process.returncode != 0:
            raise DevelopmentRefusal(
                "ordinary Lean elaboration/dependency error; first output retained: " + str(directory)
            )
        return stdout, directory

    def standard(self, path):
        if str(path) not in self.pins.files:
            raise DevelopmentRefusal("unbound/affected standard import: " + str(path))
        if path.is_relative_to(self.toolchain / "src/lean"):
            relative = path.relative_to(self.toolchain / "src/lean").with_suffix("")
        elif path.is_relative_to(self.toolchain / "lib/lean"):
            relative = path.relative_to(self.toolchain / "lib/lean").with_suffix("")
        else:
            raise DevelopmentRefusal("standard import lies outside original source/object roots")
        module = ".".join(relative.parts)
        if module not in self.sdk["modules"]:
            raise DevelopmentRefusal("standard import outside original Init metadata inventory: " + module)
        self.api["read_original"](self.pins.files[str(path)])
        return module

    def paths(self, stdout):
        paths = []
        for line in stdout.splitlines():
            path = Path(line)
            if not line or not path.is_absolute():
                raise DevelopmentRefusal("unexpected module-reporter output: " + line)
            paths.append(path.resolve(strict=True))
        return list(dict.fromkeys(paths))

    def visit(self, module):
        if module in self.bindings:
            return
        if module in self.active or module not in self.sources:
            raise DevelopmentRefusal("cycle/missing explicit project source: " + module)
        self.active.add(module)
        source = Path(self.sources[module]["source"]["path"])
        report, log = self.invoke(module + "-source-imports", ["--src-deps", str(source)])
        direct, standard = [], []
        for path in self.paths(report):
            if path.is_relative_to(self.source):
                dependency = ".".join(path.relative_to(self.source).with_suffix("").parts)
                direct.append(dependency)
                self.visit(dependency)
            else:
                standard.append(self.standard(path))
        self.graph[module] = dict(project=direct, standard=standard, source_report=reference(log / "stdout.txt"))
        object_report, object_log = self.invoke(module + "-object-imports", ["--deps", str(source)])
        imported = {}
        for path in self.paths(object_report):
            if path.is_relative_to(self.objects):
                dependency = ".".join(path.relative_to(self.objects).with_suffix("").parts)
                if dependency not in self.bindings or self.bindings[dependency]["outputs"][".olean"] != reference(path):
                    raise DevelopmentRefusal("unbound actual project object import")
                imported[dependency] = self.bindings[dependency]["outputs"]
            else:
                standard.append(self.standard(path))
        inputs = {}
        for dependency in direct:
            inputs.update(self.bindings[dependency]["project_inputs"])
            inputs[dependency] = self.bindings[dependency]["outputs"]
        target = self.objects / (module.replace(".", "/") + ".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = self.cache.get("project", {}).get(module)
        same_source = cached and all(
            cached["source"][key] == self.sources[module]["source"][key] for key in ("bytes", "sha256")
        )

        def identity(value):
            return {
                name: {suffix: {key: pin[key] for key in ("bytes", "sha256")} for suffix, pin in files.items()}
                for name, files in value.items()
            }

        same_inputs = cached and identity(cached["project_inputs"]) == identity(inputs)
        if same_source and same_inputs and self.cache.get("original_provenance") == self.provenance:
            for suffix, pin in cached["outputs"].items():
                target.with_suffix(suffix).write_bytes(self.api["read_original"](pin))
            compilation = dict(kind="exact byte-copy of NEW-context bound development object", source=cached)
        else:
            _, compiled_log = self.invoke(
                module + "-compile", ["--root=" + str(self.source), "-o", str(target), str(source)]
            )
            compilation = dict(kind="private new-context compile", process=reference(compiled_log / "PROCESS.json"))
        outputs = {}
        for suffix in (".olean", ".olean.private", ".olean.server", ".ilean"):
            path = target.with_suffix(suffix)
            if path.is_file():
                outputs[suffix] = reference(path)
        if ".olean" not in outputs:
            raise DevelopmentRefusal("successful compile has no actual output object")
        self.bindings[module] = dict(
            source=self.sources[module]["source"],
            project_inputs=inputs,
            outputs=outputs,
            actual_object_imports=imported,
            object_report=reference(object_log / "stdout.txt"),
            compilation=compilation,
            qualified=False,
        )
        self.active.remove(module)
        self.checkpoint("ACTIVE development")

    def checkpoint(self, status):
        value = dict(
            status=status,
            original_provenance=self.provenance,
            project=self.bindings,
            actual_source_import_graph=self.graph,
            graph_qualification=False,
            original_runtime_expectations_refreshed=False,
        )
        # This live development index is not a frozen receipt. Command records,
        # sources, errors and prior indices remain append-only exact artifacts.
        sequence = len(list(self.destination.glob("INDEX-*.json")))
        save(self.destination / f"INDEX-{sequence:04d}.json", value)
        return value

    def audit(self, modules, roots):
        for name in [*modules, *roots]:
            if not name or any(
                not part or not part.isascii() or not part.replace("_", "a").isalnum() for part in name.split(".")
            ):
                raise DevelopmentRefusal("strict dotted audit/module identifiers required")
        text = (
            "\n".join("import " + module for module in modules)
            + "\n"
            + "\n".join("#print axioms " + root for root in roots)
            + "\n"
        )
        self.freeze(
            "Merlin/GraphRegionsDeclaredDevelopmentAudit.lean",
            text.encode(),
            dict(
                path=str(self.destination / "AUDIT-INPUT.lean"),
                bytes=len(text.encode()),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            ),
            "new exact-root audit source",
        )
        (self.destination / "AUDIT-INPUT.lean").write_text(text)
        save(self.destination / "AUDIT-SOURCE.json", self.sources["Merlin.GraphRegionsDeclaredDevelopmentAudit"])
        source = self.sources["Merlin.GraphRegionsDeclaredDevelopmentAudit"]["source"]["path"]
        stdout, directory = self.invoke("exact-root-audit", [source])
        found = self.trace["parse_axioms"](stdout, roots)
        save(
            self.destination / "AUDIT.json",
            dict(
                roots=found,
                process=reference(directory / "PROCESS.json"),
                transitive_kernel_report=True,
                qualification=False,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prior-closure", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--symbol-review", type=Path)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--audit", nargs="*", default=[])
    args = parser.parse_args()
    context = None
    try:
        context = Context(args)
        for module in args.roots:
            context.visit(module)
        if args.audit:
            context.audit(args.roots, args.audit)
        context.preflight()
        result = context.checkpoint("DEVELOPMENT COMPILED; NOT graph/native qualification")
        save(context.destination / "RESULT.json", result)
    except Exception as error:
        record = dict(
            status="PARTIAL/UNQUALIFIED",
            error_type=type(error).__name__,
            error=str(error),
            details=getattr(error, "details", None),
            graph_qualification=False,
            old_failed_profile_rerun=False,
            original_runtime_hashes_refreshed=False,
        )
        if context is not None:
            context.checkpoint("FIRST ERROR retained; PARTIAL/UNQUALIFIED")
        args.destination.mkdir(parents=True, exist_ok=True)
        save(args.destination / "FIRST-ERROR.json", record)
        print(json.dumps(record))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
