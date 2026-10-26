"""NEW authenticated-archive common proof controller candidate; review before launch.

No old Pure controller, installed SDK, archive, Lake, MLIR or cached proof is
used. Construction reads only bound metadata/project/controller sources. Native
methods are for a separately reviewed/authorized context; source/data tests do
not authorize invoking them. Process-local results are not replayable authority.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

if __package__:
    from . import lean_graph_original_image_context_data as data
    from . import lean_graph_original_image_context_io as io
    from . import archive_selection as archive
else:
    import lean_graph_original_image_context_data as data
    import lean_graph_original_image_context_io as io
    import archive_selection as archive


class ProofRefused(ValueError):
    """Ordinary Lean elaboration failure with a valid observed runtime."""


class InfrastructureStop(RuntimeError):
    """Permanent context hold: first raw evidence retained, never retried."""


CheckedQuery = data.CheckedQuery
diagnostic_failure = data.diagnostic_failure


class Context:
    @classmethod
    def from_review(cls, *, review, project_sources, controller_sources, workspace, compile_audit_roots=None):
        """Prepare source copies only. Each input ref is independently pinned.

        review keys: request, manifest, roster, selection, host, trace (FileRef instances).
        project_sources: exact dotted module -> original .lean FileRef mapping.
        controller_sources: the FOUR current controller/data/IO/archive-selection source refs.
        compile_audit_roots: reviewed exact roots printed by named frozen project
        dependencies only. Generated queries are disjoint and must print nothing.
        Caller/source/context review is an explicit host premise, not a JSON
        boolean granting launch permission or a whole-program proof receipt.
        """
        if set(review) != {"request", "manifest", "roster", "selection", "host", "trace"}:
            raise data.ContextRefusal("closed authenticated-archive review roles required")
        if any(type(ref) is not data.FileRef for ref in [*review.values(), *controller_sources]):
            raise data.ContextRefusal("typed exact references required")
        review = dict(review)
        documents = {name: io.read_bound(ref) for name, ref in review.items()}
        inputs = {name: io.strict_json(documents[name]) for name in review if name != "trace"}
        selection = archive.select(inputs, review)
        expected_controls = {str(Path(module.__file__).absolute()) for module in (data, io, archive)} | {
            str(Path(__file__).absolute())
        }
        if {ref.path for ref in controller_sources} != expected_controls or len(controller_sources) != 4:
            raise data.ContextRefusal("full current controller/data/IO/archive-selection source identity required")
        for ref in controller_sources:
            io.read_bound(ref, selection.forbidden_roots)
        trace = {"__name__": "_reviewed_original_image_trace"}
        exec(compile(documents["trace"], review["trace"].path, "exec"), trace)
        result = cls()
        result.selection, result.trace = selection, trace
        result.workspace = Path(data.absolute(str(workspace)))
        result.workspace.mkdir(parents=True, exist_ok=False)
        result.sources_dir, result.objects_dir = result.workspace / "source", result.workspace / "compiled"
        result.commands_dir = result.workspace / "commands"
        for directory in (result.sources_dir, result.objects_dir, result.commands_dir):
            directory.mkdir()
        result.review, result.controls = dict(review), tuple(controller_sources)
        result.compile_audit_roots = dict(compile_audit_roots or {})
        for module, roots in result.compile_audit_roots.items():
            if module not in project_sources or type(roots) is not tuple or not roots or len(set(roots)) != len(roots):
                raise data.ContextRefusal("explicit original-module compile audit roots required")
            for root in roots:
                data.module_name(root)
        result.sources, result.originals, result.objects, result.graph = {}, {}, {}, {}
        result.active, result._issued, result._key = set(), {}, object()
        result.pending_outputs, result.unpublished = set(), {}
        result.held, result.sequence = False, 0
        for name, ref in sorted(project_sources.items()):
            data.module_name(name)
            if not name.startswith(("Merlin.", "MerlinVerify.")) or type(ref) is not data.FileRef:
                raise data.ContextRefusal("explicit project namespace/source reference required")
            result.originals[name] = ref
            result._freeze(name, io.read_bound(ref, selection.forbidden_roots))
        sdk = Path(selection.toolchain)
        result.environment = {
            "LANG": "C",
            "LC_ALL": "C",
            "LEAN_SYSROOT": str(sdk),
            "LEAN_PATH": str(result.objects_dir) + ":" + str(sdk / "lib/lean"),
            "LEAN_SRC_PATH": str(result.sources_dir) + ":" + str(sdk / "src/lean"),
            "LD_DEBUG": "libs,files",
        }
        result.loader = [
            selection.interpreter,
            "--inhibit-cache",
            "--glibc-hwcaps-mask",
            "",
            "--library-path",
            f"{sdk}/lib/lean:{sdk}/lib:/usr/lib",
            selection.executable,
        ]
        result.definition = {
            "schema": "merlin.graph.fresh-authenticated-archive-context.v1",
            "diagnostic_failure_capture": "first-failed-read/already-read-buffer.bin; no acceptance or hold clearance",
            "import_policy": "trusted-closed-project-header-and-actual-object-report-v1",
            "selection": selection.record(),
            "review": {k: v.record() for k, v in review.items()},
            "controllers": [ref.record() for ref in controller_sources],
            "project_sources": {k: v.record() for k, v in result.originals.items()},
            "compile_audit_roots": dict(result.compile_audit_roots),
            "workspace": str(result.workspace),
            "environment": result.environment,
            "loader": result.loader,
            "old_profile_repaired": False,
            "native_fidelity_proved": False,
            "trust": [
                "official parent/member association and separate host-loader review",
                "OS/kernel/vDSO",
                "loader/module reporting",
                "Lean kernel",
                "Python host association and process-local authority",
                "closed literal project-header decoder (not Lean source reporting)",
            ],
        }
        result.context_id = data.identity(result.definition)
        io.save(result.workspace / "CONTEXT.json", {"context_id": result.context_id, **result.definition})
        result.context_ref = io.output_reference(result.workspace / "CONTEXT.json", result.workspace)
        io.save(result.workspace / "PREPARED.json", {"native_invocations": 0, "qualified": False})
        return result

    def _freeze(self, module, content):
        data.module_name(module)
        if module in self.sources or type(content) is not bytes:
            raise data.ContextRefusal("unique module and exact source bytes required")
        path = self.sources_dir / (module.replace(".", "/") + ".lean")
        io.write_new(path, content)
        ref = data.FileRef(str(path), hashlib.sha256(content).hexdigest(), len(content))
        io.read_bound(ref, self.selection.forbidden_roots)
        self.sources[module] = ref
        return ref

    def _stop(self, error):
        self.held = True
        path = self.workspace / "FIRST-INFRASTRUCTURE-STOP.json"
        if not path.exists():
            io.save(
                path,
                {
                    "status": "UNQUALIFIED_CONTEXT_HELD",
                    "retry": False,
                    "type": type(error).__name__,
                    "reason": str(error),
                    "evidence": getattr(error, "evidence", None),
                },
            )
        raise InfrastructureStop(str(error)) from error

    def _guard(self):
        if self.held or (self.workspace / "FIRST-INFRASTRUCTURE-STOP.json").exists():
            raise InfrastructureStop("context already held; no new reads or launches")
        try:
            if Path("/etc/ld.so.preload").exists() or Path("/etc/ld.so.preload").is_symlink():
                raise data.ContextRefusal("system preload file exists")
            if (
                data.identity(self.definition) != self.context_id
                or self.selection.record() != self.definition["selection"]
                or {k: v.record() for k, v in self.originals.items()} != self.definition["project_sources"]
                or {k: v.record() for k, v in self.review.items()} != self.definition["review"]
                or [ref.record() for ref in self.controls] != self.definition["controllers"]
                or self.compile_audit_roots != self.definition["compile_audit_roots"]
                or self.environment != self.definition["environment"]
                or self.loader != self.definition["loader"]
            ):
                raise data.ContextRefusal("context definition mutated")
            references = [
                self.context_ref,
                *self.selection.images,
                *self.selection.runtime,
                *self.review.values(),
                *self.controls,
                *self.originals.values(),
                *self.sources.values(),
            ]
            references += [ref for group in self.objects.values() for ref in group.values()]
            references += list(self.unpublished.values())
            for directory, allowed in (
                (Path(self.selection.toolchain), {ref.path for ref in self.selection.images}),
                (self.sources_dir, {ref.path for ref in self.sources.values()}),
                (
                    self.objects_dir,
                    {ref.path for group in self.objects.values() for ref in group.values()}
                    | set(self.unpublished)
                    | self.pending_outputs,
                ),
            ):
                for path in directory.rglob("*"):
                    if path.is_symlink() or (path.is_file() and str(path) not in allowed):
                        raise data.ContextRefusal("unbound/symlink file in a selected source/object/runtime directory")
            for ref in set(references):
                io.read_bound(ref, self.selection.forbidden_roots,
                              failure_directory=self.workspace / "first-failed-read")
            return {
                "context_id": self.context_id,
                "authenticated_archive_images": len(self.selection.images),
                "runtime_images": len(self.selection.runtime),
                "bound_files": len(set(references)),
                "installed_sdk_reads": 0,
                "expectations_refreshed": False,
            }
        except InfrastructureStop:
            raise
        except Exception as error:
            self._stop(error)

    def _invoke(self, label, arguments, *, proof_command=False):
        self._guard()
        directory = self.commands_dir / f"{self.sequence:05d}-{label}"
        self.sequence += 1
        directory.mkdir()
        command = [*self.loader, "-j1", *arguments]
        io.save(
            directory / "INTENT.json",
            {
                "command": command,
                "environment": self.environment,
                "cwd": str(self.workspace),
                "context_id": self.context_id,
            },
        )
        io.save(directory / "PRECHECK.json", self._guard())
        try:
            with subprocess.Popen(
                command,
                cwd=self.workspace,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            ) as process:
                timed_out = False
                try:
                    stdout, stderr = process.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()  # This exact owned process only.
                    stdout, stderr = process.communicate()
            io.write_new(directory / "stdout.txt", stdout)
            io.write_new(directory / "stderr.txt", stderr)
            io.save(
                directory / "PROCESS.json",
                {"pid": process.pid, "returncode": process.returncode, "timed_out": timed_out},
            )
            io.save(directory / "POSTCHECK.json", self._guard())
            if timed_out or process.returncode < 0:
                raise data.ContextRefusal("timeout or native signal; not a proof refusal")
            observed = self.trace["parse"](
                stderr.decode(),
                [ref.record() for ref in self.selection.runtime],
                self.selection.executable,
                self.selection.interpreter,
                self.selection.allowed_misses(process.returncode),
                expected_pid=process.pid,
                project_initializers={"initialize_" + name.replace(".", "_") for name in self.objects},
            )
            io.save(directory / "RUNTIME.json", observed)
            if process.returncode:
                if (
                    proof_command
                    and process.returncode == 1
                    and diagnostic_failure(stdout.decode(), {ref.path for ref in self.sources.values()})
                ):
                    io.save(directory / "PROOF-REFUSAL.json", {"status": "NOT_ESTABLISHED", "context_held": False})
                    raise ProofRefused(str(directory))
                raise data.ContextRefusal("nonzero exit without an ordinary bound-source Lean diagnostic")
            return stdout.decode(), directory
        except ProofRefused:
            raise
        except InfrastructureStop:
            raise
        except Exception as error:
            self._stop(error)

    def _reported(self, stdout, kind, *, allow_empty=False):
        known = (
            {ref.path: module for module, ref in self.sources.items()}
            if kind == "source"
            else {group[".olean"].path: module for module, group in self.objects.items()}
        )
        standard = {ref.path for ref in self.selection.images}
        sdk = Path(self.selection.toolchain)
        project, libraries, imported, seen = set(), set(), [], set()
        for line in stdout.splitlines():
            path = data.absolute(line)
            if not line.endswith(".lean" if kind == "source" else ".olean"):
                raise data.ContextRefusal("wrong-kind actual module import report")
            if line in seen:
                continue  # Same validated canonical path; raw report retains every occurrence.
            seen.add(line)
            if line in known:
                project.add(known[line])
                imported.append({"path": line, "role": "bound-project-" + kind, "module": known[line]})
                continue
            prefix = sdk / ("src/lean" if kind == "source" else "lib/lean")
            if not path.is_relative_to(prefix):
                raise data.ContextRefusal("standard reporter path outside private roots; no fallback")
            relative = path.relative_to(prefix).with_suffix("")
            module = data.module_name(".".join(relative.parts))
            family = [
                str(sdk / "lib/lean" / relative) + suffix for suffix in (".olean", ".olean.private", ".olean.server")
            ]
            if any(member not in standard for member in family):
                raise data.ContextRefusal("incomplete ORIGINAL standard compiled-object family")
            if kind == "object" and line not in standard:
                raise data.ContextRefusal("unbound actual standard object")
            role = "pinned-standard-" + kind if line in standard else "compiled-only-standard-name-cross-check"
            libraries.add(module)
            imported.append({"path": line, "module": module, "role": role, "object_family": family})
        if not imported and not allow_empty:
            raise data.ContextRefusal("empty actual import report is not a closure")
        return project, libraries, imported

    def _inspect(self, source):
        decoded = data.literal_imports(io.read_bound(source, self.selection.forbidden_roots))
        direct = set(decoded) & self.sources.keys()
        libraries = set(decoded) - direct
        # Validate standard membership without opening or fabricating .lean files.
        standard_paths = [
            str(Path(self.selection.toolchain) / "lib/lean" / name.replace(".", "/")) + ".olean"
            for name in sorted(libraries)
        ]
        self._reported("\n".join(standard_paths), "object", allow_empty=not libraries)
        for dependency in sorted(direct):
            self._build(dependency)
        stdout, object_log = self._invoke("object-imports", ["--deps", source.path])
        objects, object_libraries, object_paths = self._reported(stdout, "object", allow_empty=not decoded)
        if objects != direct or object_libraries != libraries:
            raise data.ContextRefusal("decoded header/actual object dependency modules differ")
        return {
            "source": source.record(),
            "header_policy": self.definition["import_policy"],
            "host_decoded_modules": list(decoded),
            "object_imports": object_paths,
            "direct_modules": sorted(direct),
            "standard_modules": sorted(libraries),
            "logs": [str(object_log)],
        }

    def _build(self, module):
        if module in self.objects:
            return
        if module in self.active or module not in self.sources:
            raise data.ContextRefusal("cyclic or missing explicit project source")
        self.active.add(module)
        try:
            source = self.sources[module]
            inspection = self._inspect(source)
            target = self.objects_dir / (module.replace(".", "/") + ".olean")
            target.parent.mkdir(parents=True, exist_ok=True)
            suffixes = (".olean", ".olean.private", ".olean.server", ".ilean")
            self.pending_outputs = {str(target.with_suffix(suffix)) for suffix in suffixes}
            try:
                stdout, compile_log = self._invoke(
                    "compile", ["--root=" + str(self.sources_dir), "-o", str(target), source.path], proof_command=True
                )
            finally:
                if not self.held:
                    for path in self.pending_outputs:
                        if Path(path).exists():
                            self.unpublished[path] = io.output_reference(path, self.workspace)
                self.pending_outputs = set()
            if module in self.compile_audit_roots:
                found = self.trace["parse_axioms"](stdout, self.compile_audit_roots[module])
                io.save(compile_log / "COMPILE-AXIOMS.json", found)
            elif stdout.strip():
                raise data.ContextRefusal("unexpected compile output; query must contain only declarations")
            outputs = {}
            for suffix in (".olean", ".olean.private", ".olean.server", ".ilean"):
                path = target.with_suffix(suffix)
                if path.exists():
                    outputs[suffix] = io.output_reference(path, self.workspace)
            if ".olean" not in outputs:
                raise data.ContextRefusal("successful compile missing actual object")
            self.objects[module] = outputs
            self.graph[module] = {
                **inspection,
                "outputs": {k: v.record() for k, v in outputs.items()},
                "compile_log": str(compile_log),
            }
            self._guard()
        finally:
            self.active.discard(module)

    def check_query(self, module, query_bytes, *, final_root, support_roots=()):
        """Compile exact generated bytes; audit exact roots and retain full types.

        This is kernel-check evidence only. The caller's approved generator must
        bind the final root's statement to the complete compiler boundary/claim.
        An arbitrary local theorem never acquires whole-program policy authority.
        """
        roots = (final_root, *support_roots)
        for name in (module, *roots):
            data.module_name(name)
        if len(set(roots)) != len(roots):
            raise data.ContextRefusal("duplicate exact root")
        if any(
            name in self.sources or name in self.compile_audit_roots
            for name in (module, module + "Audit", module + "Types")
        ):
            raise data.ContextRefusal("generated query/audit names must be disjoint from frozen dependencies")
        self._guard()
        query = self._freeze(module, query_bytes)
        try:
            self._build(module)
            audit_module = module + "Audit"
            audit_bytes = (
                "import " + module + "\n" + "\n".join("#print axioms " + name for name in roots) + "\n"
            ).encode()
            audit_source = self._freeze(audit_module, audit_bytes)
            self.graph[audit_module] = self._inspect(audit_source)
            stdout, audit_log = self._invoke("root-audit", [audit_source.path], proof_command=True)
            axioms = self.trace["parse_axioms"](stdout, roots)
            # Separate buffers keep the strict axiom reader exact. pp.all avoids
            # pretty-printer omission; these raw type reports remain inspectable.
            type_source = self._freeze(
                module + "Types",
                (
                    "import " + module + "\nset_option pp.all true\nset_option pp.universes true\n"
                    "set_option pp.fullNames true\n" + "\n".join("#check @" + name for name in roots) + "\n"
                ).encode(),
            )
            self.graph[module + "Types"] = self._inspect(type_source)
            types, type_log = self._invoke("root-types", [type_source.path], proof_command=True)
            if not types.strip():
                raise data.ContextRefusal("missing exact-root full type output")
            import_path = self.workspace / (module + "-IMPORTS.json")
            io.save(import_path, self.graph)
            audit_path = self.workspace / (module + "-AUDIT.json")
            io.save(
                audit_path,
                {
                    "context_id": self.context_id,
                    "query": query.record(),
                    "roots": axioms,
                    "axioms_stdout": io.output_reference(audit_log / "stdout.txt", self.workspace).record(),
                },
            )
            records = tuple(
                io.output_reference(path, self.workspace)
                for path in sorted(self.commands_dir.rglob("*"))
                if path.is_file()
            )
            result = CheckedQuery(
                self.context_id,
                module,
                query,
                self.objects[module][".olean"],
                final_root,
                tuple(support_roots),
                tuple((name, tuple(axioms[name])) for name in roots),
                io.output_reference(audit_path, self.workspace),
                io.output_reference(type_log / "stdout.txt", self.workspace),
                io.output_reference(import_path, self.workspace),
                records,
                self._key,
            )
            self._issued[id(result)] = result
            self.revalidate(result)
            return result
        except ProofRefused:
            raise
        except InfrastructureStop:
            raise
        except Exception as error:
            self._stop(error)

    def revalidate(self, result):
        if (
            type(result) is not CheckedQuery
            or self._issued.get(id(result)) is not result
            or result._key is not self._key
            or result.context_id != self.context_id
        ):
            raise data.ContextRefusal("only this process/context's issued result may be revalidated")
        self._guard()
        try:
            for ref in (result.query, result.object, result.audit, result.types, result.imports, *result.records):
                io.read_bound(ref, self.selection.forbidden_roots,
                              failure_directory=self.workspace / "first-failed-read")
        except Exception as error:
            self._stop(error)
        return result
