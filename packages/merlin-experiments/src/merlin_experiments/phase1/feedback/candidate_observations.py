"""Original public emission adapter for the reusable own-artifact diagnostics."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

from merlin.perf.candidate_diagnostics import observe_candidate_artifacts
from merlin.perf.candidate_source import MAX_TEXT, digest, public_projection
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

_ROLES = {"source": "input.interface.mlir", "emitted": "lowered.llvm.mlir",
          "target_ir": "lowered.target.mlir", "command_buffer": "command_buffer.json",
          "llvm_ir": "kernel.ll", "object": "kernel.o", "elf": PACKAGE_ELF_NAME,
          "stack_usage": "kernel.su"}
_NUMERIC = frozenset({"dtype", "operand_dtype", "accum_dtype", "output_dtype", "atol", "rtol", "bound_lsb"})


def _reopen_product(pin, *, root=None):
    """Reopen an explicit original role; a basename never selects native products."""
    if not isinstance(pin, dict) or not {"path", "sha256"} <= set(pin):
        raise ValueError("original candidate product has no byte pin")
    path = Path(pin["path"])
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file()
            or path.stat().st_nlink != 1 or path.stat().st_size > 64 << 20
            or (root is not None and not path.is_relative_to(root))):
        raise ValueError("original candidate product escaped its owner or is unsupported")
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (any(getattr(before, key) != getattr(after, key) for key in stable) or value.hexdigest() != pin["sha256"]
            or ("size_bytes" in pin and before.st_size != pin["size_bytes"])):
        raise ValueError("original candidate product changed")
    return path


def reopen_native_products(
    root, *, emission, elf_pin, source=None, build_artifacts=None, candidate=None, emission_root=None
):
    """Join actual source roles and the original LLVM -> object -> ELF records.

    The original grade/timing caller supplies its own semantic product pins.
    This is a consumption observation, never a numerical or admission verdict.
    An explicit emission_root separates already pinned emitted leaves from the
    bounded native build namespace. It never expands the invocation scan.
    """
    from merlin.common import invocation_record as I

    root = Path(root)
    if root.resolve(strict=True) != root or root.is_symlink():
        raise ValueError("original native product root is indirect")
    if not isinstance(emission, dict):
        raise ValueError("original native candidate emission is absent")
    if emission_root is None:
        emission_root = root
    else:
        emission_root = Path(emission_root)
        if (emission_root.resolve(strict=True) != emission_root or emission_root.is_symlink()
                or not emission_root.is_dir()):
            raise ValueError("original emitted product root is indirect")
    roles = {role: _reopen_product(emission[name], root=None if role == "source" else emission_root)
             for role, name in (("source", "source_interface"), ("emitted", "lowered_mlir"),
                                ("command_buffer", "command_buffer"))}
    if "capsule_declaration" in emission:
        _reopen_product(emission["capsule_declaration"])
    roles["elf"] = _reopen_product(elf_pin, root=root)
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("original native source roster is absent")
        _reopen_product(source["interface"])
        if source["interface"]["sha256"] != emission["source_interface"]["sha256"]:
            raise ValueError("original native model source differs from candidate emission")
    if candidate is not None:
        cb = json.loads(roles["command_buffer"].read_bytes())
        if (candidate.get("lowered_mlir_sha256") != emission["lowered_mlir"]["sha256"]
                or candidate.get("command_buffer_sha256") != hashlib.sha256(
                    json.dumps(cb, sort_keys=True).encode()).hexdigest()):
            raise ValueError("original native build consumed another emitted program")
    inventory = None
    if build_artifacts is not None:
        if not isinstance(build_artifacts, dict) or not 0 < len(build_artifacts) <= 4096:
            raise ValueError("original native build roster is unavailable or exceeds limit")
        inventory, total = {}, 0
        for pin in build_artifacts.values():
            path = _reopen_product(pin, root=root)
            total += path.stat().st_size
            if total > 128 << 20:
                raise ValueError("original native build exceeds byte budget")
            inventory[(str(path), pin["sha256"])] = path
        if (str(roles["elf"]), elf_pin["sha256"]) not in inventory:
            raise ValueError("original native ELF is outside its actual build roster")
    records = []
    for index, path in enumerate(root.rglob("invocation.json")):
        if index >= 512 or path.is_symlink() or path.resolve() != path or path.stat().st_size > MAX_TEXT:
            raise ValueError("original native invocation roster exceeds limit or is indirect")
        row = json.loads(path.read_bytes())
        if row.get("status") != "completed" or row.get("stage") not in {"llvm_translation", "object", "elf"}:
            continue
        returned = row.get("returncode") == 0 if row.get("kind") == "subprocess" else row.get("outcome") == "returned"
        if returned:
            records.append((path, row))
    def pins(row, role):
        values = row.get(role, ())
        if len(values) > 4096:
            raise ValueError("original native invocation product roster exceeds limit")
        return {(pin["path"], pin["sha256"]) for pin in values}
    translations, objects = {}, {}
    for path, row in records:
        destination = translations if row["stage"] == "llvm_translation" else objects
        if row["stage"] not in {"llvm_translation", "object"}:
            continue
        for pin in pins(row, "outputs"):
            destination.setdefault(pin, []).append((path, row))
    selected = []
    for link_path, link in records:
        if (link.get("stage") != "elf" or link.get("kind") != "subprocess"
                or (str(roles["elf"]), elf_pin["sha256"]) not in pins(link, "outputs")):
            continue
        linked_objects = {path: row for pin in pins(link, "inputs") for path, row in objects.get(pin, ())}
        for object_path, obj in linked_objects.items():
            object_pins = pins(obj, "outputs") & pins(link, "inputs")
            if obj.get("kind") != "subprocess" or len(object_pins) != 1:
                continue
            translated_inputs = {path: row for pin in pins(obj, "inputs")
                                 for path, row in translations.get(pin, ())}
            for translation_path, translation in translated_inputs.items():
                translated = pins(translation, "outputs") & pins(obj, "inputs")
                inputs = [pin for pin in translation.get("inputs", ())
                          if pin["sha256"] == emission["lowered_mlir"]["sha256"]]
                if translation.get("stage") != "llvm_translation" or len(translated) != 1 or len(inputs) != 1:
                    continue
                observed = (translation_path, object_path, link_path)
                for path in observed:
                    I.verify(path)
                own = [*translated, *object_pins, (inputs[0]["path"], inputs[0]["sha256"])]
                for path, sha in own:
                    _reopen_product({"path": path, "sha256": sha}, root=root)
                    if inventory is not None and (path, sha) not in inventory:
                        raise ValueError("native invocation consumed a product outside the actual build roster")
                selected.append((next(iter(translated)), next(iter(object_pins)), observed))
    identities = {(translated, obj) for translated, obj, _ in selected}
    if len(identities) != 1:
        raise ValueError("original native model lacks one joined LLVM/object/ELF producer chain")
    translated, obj = next(iter(identities))
    roles.update(llvm_ir=Path(translated[0]), object=Path(obj[0]))
    return roles, tuple(sorted({path for _, _, paths in selected for path in paths}))


def _numeric_context(context):
    projected = {key: value for key, value in (context or {}).items() if key in _NUMERIC}
    policy = (context or {}).get("numeric_policy")
    if policy is None:
        return projected
    if (not isinstance(policy, dict) or set(policy) != {"schema", "compare", "outputs"}
            or policy["schema"] != "merlin.numeric_policy.per_output.v1" or policy["compare"] != "per_output"
            or not isinstance(policy["outputs"], dict) or len(policy["outputs"]) > 1024):
        projected["numeric_policy"] = {"status": "UNKNOWN"}
        return projected
    for name, row in policy["outputs"].items():
        if (not isinstance(name, str) or not isinstance(row, dict) or not {"compare", "dtype"} <= set(row)
                or set(row) - {"compare", "dtype", "atol", "rtol"}
                or row["compare"] not in {"exact_int", "tolerance_float"}):
            projected["numeric_policy"] = {"status": "UNKNOWN"}
            return projected
    projected["numeric_policy"] = policy
    return projected


def _observed_options(root, artifacts):
    from merlin.common import invocation_record as I

    members, missing = [], False
    hashes = {hashlib.sha256(raw).hexdigest() for raw in artifacts.values()}
    for index, path in enumerate(Path(root).rglob("invocation.json")):
        if index >= 256:
            return {"status": "UNKNOWN", "reason": "invocation collection limit exceeded"}
        try:
            raw = path.read_bytes()
            if path.is_symlink() or len(raw) > MAX_TEXT:
                raise ValueError("invocation record unavailable")
            row = json.loads(raw)
            if row.get("status") != "completed" or not any(p.get("sha256") in hashes for p in row.get("outputs", [])):
                continue
            I.verify(path)
            argv = row.get("argv", [])
            members.append({"stage": row["stage"], "record_sha256": hashlib.sha256(raw).hexdigest(),
                            "argv_sha256": digest(argv), "observed_flags": [s for s in argv if s.startswith("-")],
                            "tool_sha256": (row.get("executable") or {}).get("sha256"),
                            "environment_sha256": (row.get("environment") or {}).get("sha256"),
                            "output_sha256": [p["sha256"] for p in row.get("outputs", [])
                                              if p.get("sha256") in hashes]})
        except (OSError, ValueError, KeyError, TypeError):
            missing = True
    return public_projection({"status": "observed" if members and not missing else "UNKNOWN", "records": members,
                              "scope": "reopened original completed product invocations; not transform semantics"})


def _read_unique(root, name):
    values = {}
    for index, path in enumerate(Path(root).rglob(name)):
        if index >= 256 or path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
            raise ValueError("candidate product roster is unsupported or exceeds limit")
        raw = path.read_bytes()
        values.setdefault(hashlib.sha256(raw).hexdigest(), raw)
    if len(values) > 1:
        raise ValueError("candidate product has multiple distinct definitions")
    return next(iter(values.values()), None)


def _selected_layer_execution(grade):
    """Old emission-only/legacy grades keep their original descriptive reader."""
    from merlin.targetgen.executed_program import SCHEMA_V2

    if not isinstance(grade, dict) or grade.get("kind") != "layer":
        return False
    if grade.get("label") not in {"public", "dev"}:
        return True  # Never substitute emission-only reads for a private layer.
    source = grade.get("compiler_source")
    records = source.get("executed_programs") if isinstance(source, dict) else None
    return isinstance(records, list) and any(
        isinstance(row, dict) and row.get("schema") == SCHEMA_V2 for row in records
    )


def observe_generated_artifacts(root, *, compiler=None, numeric_context=None, entry_symbol=None,
                                candidate_sha256=None, original_grade=None):
    """Reopen original products; identities are public, host paths never leave this adapter."""
    started = time.monotonic()
    artifacts, missing = {}, []
    native_join = "UNKNOWN"
    if original_grade is not None and "candidate_native_execution" in original_grade:
        try:
            native = original_grade["candidate_native_execution"]
            if (not isinstance(native, dict) or not {"source", "candidate", "build_artifacts", "elf"} <= set(native)
                    or any(not isinstance(native[key], dict) for key in ("source", "candidate", "build_artifacts"))):
                raise ValueError("original native candidate receipt is incomplete")
            products, _ = reopen_native_products(
                root, emission=original_grade.get("candidate_emission"), elf_pin=native.get("elf"),
                source=native.get("source"), build_artifacts=native.get("build_artifacts"),
                candidate=native.get("candidate"),
            )
            if sum(path.stat().st_size for path in products.values()) > 128 << 20:
                raise ValueError("candidate product byte budget exhausted")
            artifacts = {role: path.read_bytes() for role, path in products.items()}
            native_join = "reopened_original_native_products"
        except (OSError, ValueError, KeyError, TypeError):
            artifacts, missing = {}, ["original_native_product_join"]
    elif _selected_layer_execution(original_grade):
        try:
            from .candidate_execution_products import reopen_execution_products

            products, execution_join = reopen_execution_products(
                root, original_grade, compiler=compiler, candidate_sha256=candidate_sha256
            )
            if sum(path.stat().st_size for path in products.values()) > 128 << 20:
                raise ValueError("candidate product byte budget exhausted")
            artifacts = {role: path.read_bytes() for role, path in products.items()}
            native_join = "reopened_original_retained_functional_layer_products"
        except (OSError, ValueError, KeyError, TypeError):
            artifacts, missing = {}, ["original_retained_execution_join"]
            execution_join = None
    else:
        for role, name in _ROLES.items():
            try:
                raw = _read_unique(root, name)
                if raw is not None:
                    if sum(len(value) for value in artifacts.values()) + len(raw) > 128 << 20:
                        raise ValueError("candidate product byte budget exhausted")
                    artifacts[role] = raw
            except (OSError, ValueError):
                missing.append(role)
    options = None
    if compiler is not None:
        manifest = Path(compiler) / "manifest.yaml"
        if manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size <= MAX_TEXT:
            import yaml

            raw = manifest.read_bytes()
            try:
                declaration = yaml.safe_load(raw)
            except yaml.YAMLError:
                declaration = None
                missing.append("manifest_options")
            artifacts["manifest"] = raw
            if isinstance(declaration, dict):
                options = {key: declaration[key] for key in ("build", "commands", "optimizations", "build_options")
                           if key in declaration}
    numeric = _numeric_context(numeric_context)
    def text(role):
        return artifacts[role].decode("utf-8") if role in artifacts else None
    try:
        cb = json.loads(text("command_buffer")) if "command_buffer" in artifacts else None
        report = observe_candidate_artifacts(
            source_mlir=text("source"), emitted_mlir=text("emitted"), command_buffer=cb,
            artifacts=artifacts, compiler_options=options, numeric_context=numeric or None,
            elf_bytes=artifacts.get("elf"), stack_usage_text=text("stack_usage"),
            entry_symbol=entry_symbol, candidate_sha256=candidate_sha256,
        )
    except Exception as error:  # noqa: BLE001 -- diagnostics never waive/refuse original grading
        report = {"schema": "merlin.candidate_diagnostics.v1", "status": "UNKNOWN",
                  "reason": "candidate diagnostics unavailable: " + type(error).__name__, "authority": "none"}
    report["product_collection"] = {"unavailable_roles": missing, "wall_seconds": time.monotonic() - started,
                                    "consumption_join": native_join}
    if native_join == "reopened_original_retained_functional_layer_products":
        report["product_collection"]["execution_join"] = execution_join
    report["observed_codegen_options"] = _observed_options(root, artifacts)
    report["product_collection"]["wall_seconds"] = time.monotonic() - started
    report.pop("observation_sha256", None)
    report["observation_sha256"] = digest(report)
    return report


def emit_public_diagnostics(submission, *, target, capsules_root, contract, capsules="all", timeout=300):
    """Use the same public ABI and integrity/sandbox entrypoints, without tensor/oracle evaluation."""
    from merlin.targetgen.capsule_common import make_run_paths, run_entrypoints
    from merlin.targetgen.package_runtime import build_package, integrity_scan, load_package
    from merlin_experiments.phase1.brokers.selfcheck import _submission_digest

    started = time.monotonic()
    deadline = started + timeout
    candidate_pin = _submission_digest(Path(submission))
    public = CR.discover_capsules(str(capsules_root), labels={"public", "dev"}, contract=contract)
    wanted = {name.strip() for name in capsules.split(",") if name.strip()}
    if wanted != {"all"}:
        known = {str(row.get("name") or Path(row["__dir__"]).name) for row in public}
        if not wanted or wanted - known:
            raise ValueError("candidate diagnostics requested unknown public members")
        public = [row for row in public if str(row.get("name") or Path(row["__dir__"]).name) in wanted]
    if len(public) > 256:
        raise ValueError("candidate diagnostics public member limit exceeded")
    rows, package = [], None
    with tempfile.TemporaryDirectory(prefix="candidate_diagnostics_") as temporary:
        for index, capsule in enumerate(public):
            if not capsule.get("interface_mlir"):
                rows.append({"member_index": index, "status": "UNKNOWN", "reason": "public interface unavailable"})
                continue
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                rows.append({"member_index": index, "status": "UNKNOWN", "reason": "collection budget exhausted"})
                continue
            name = str(capsule.get("name") or Path(capsule["__dir__"]).name)
            paths = make_run_paths(Path(temporary), str(index), suite="candidate-diagnostics", target=target,
                                   dtype=str(capsule.get("dtype") or "unknown"), benchmark="public_emission")
            try:
                if package is None:
                    package = load_package(submission, contract=contract)
                    integrity_scan(package)
                    build_package(package, timeout=remaining)
                    remaining = max(1, int(deadline - time.monotonic()))
                package, _, _ = run_entrypoints(package, submission, capsule, paths, contract=contract,
                                                timeout=remaining, fourth_output_name="lowered.llvm.mlir")
                report = observe_generated_artifacts(paths.generated, compiler=submission, numeric_context=capsule,
                                                     candidate_sha256=candidate_pin)
                rows.append({"member_index": index, "public_member": name, "status": "observed", "diagnostics": report})
            except Exception as error:  # noqa: BLE001 -- failures retained without private compiler stderr
                # A declared refusal is not a successful compilation or zero work.
                report = observe_generated_artifacts(paths.generated, compiler=submission, numeric_context=capsule,
                                                     candidate_sha256=candidate_pin)
                rows.append({"member_index": index, "public_member": name, "status": "UNKNOWN_OR_REFUSAL",
                             "reason": type(error).__name__, "diagnostics": report})
    stable = candidate_pin == _submission_digest(Path(submission))
    if not stable:
        for row in rows:
            row["status"] = "UNKNOWN_SOURCE_CHANGED"
            report = row.get("diagnostics")
            if report is not None:
                report["source_emission_join"] = {"status": "UNKNOWN", "reason": "candidate source changed"}
                report["dependency_stages"] = {"status": "UNKNOWN", "stages": []}
                report.pop("observation_sha256", None)
                report["observation_sha256"] = digest(report)
    return public_projection({"schema": "merlin.public_candidate_diagnostics.v1", "authority": "none",
                              "scope": "original public emission only", "members": rows,
                              "candidate_source_sha256": candidate_pin,
                              "source_stability": "unchanged" if stable else "UNKNOWN",
                              "member_count": len(public), "numerical_status": "NOT_RUN",
                              "selected_rtl_status": "NOT_RUN", "wall_seconds": time.monotonic() - started})


def _shape_coverage(sub: Path, out_path: str, *, context, contract: Path | None = None) -> int:
    """The agent-facing shape-coverage report (see ``--shape-coverage``).

    Leaks nothing: the probe shapes are multiples of the target's own derived tile edge, which the agent
    can read off its RTL facts, and the report contains no golden, no corpus capsule and no reference
    value -- only what the agent's own compiler emitted.
    """

    from merlin.targetgen import lowering_coverage as LC

    try:
        cov = LC.sweep(
            sub,
            target=context.target,
            contract=str(contract if contract is not None else context.repo / "merlin/contract"),
        )
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"shape-coverage probe failed: {type(e).__name__}: {e}"}))
        return 2
    cov["note"] = (
        "Each corner is the SAME contraction at a different extent. `emitted_work` is how many "
        "static instructions YOUR compiler emitted for it. A smaller artifact is not evidence of "
        "numerical refusal or profitability; `collapsed` retains the historical static-count observation. "
        "Actual extent and tail coverage checks remain separate from the original numerical gates. "
        "`multi_tile_axes_uncovered` "
        "names an axis your lowering does not loop over; `tail_axes_uncovered` and "
        "`tail_cases_uncovered` cover sub-tile and non-multiple extents. If you truly cannot lower a shape, DECLARE it "
        "(set `declined` on the command buffer) rather than emitting a terminator."
    )
    txt = json.dumps(cov, indent=2)
    print(txt)
    if out_path:
        Path(out_path).write_text(txt)
    return 0 if cov.get("all_covered") else 1



def diagnostics_cli(args, *, context, capsules_root, contract):
    """Format only public own-artifact observations, without selecting a runtime."""
    try:
        result = emit_public_diagnostics(
            Path(args.submission), target=context.target, capsules_root=capsules_root,
            contract=contract if contract is not None else context.repo / "merlin/contract",
            capsules=args.capsules, timeout=args.timeout,
        )
    except Exception as error:  # noqa: BLE001 -- no private compiler/parser messages
        result = {"status": "UNKNOWN", "reason": "candidate diagnostics refused: " + type(error).__name__}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0 if result.get("schema") == "merlin.public_candidate_diagnostics.v1" else 2
