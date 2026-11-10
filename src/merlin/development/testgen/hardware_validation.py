"""Source-bound, narrowly scoped RTL property evidence.

The caller authors a separate HW/Comb reference module. A combinational LEC
query compares that module with a selected RTL cell for every input bit pattern
at its concrete port widths. This does not certify an accelerator operation,
software ABI, memory protocol, compiler, or a sequential design.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from merlin.targetgen.rtl.extract_module import extract


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _digest(path.read_bytes())}


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _query_with_model(query: str) -> str:
    """Ask for input witnesses only after a first SAT decision.

    CIRCT's LEC SMT-LIB export writes one solver scope ending in reset. The
    declarations, not guessed port names, identify the symbolic inputs.
    """
    declarations = []
    for line in query.splitlines():
        line = line.strip()
        if line.startswith("(declare-const "):
            declarations.append(line[len("(declare-const ") :].split(" ", 1)[0])
    if not declarations or query.count("(check-sat)") != 1:
        raise ValueError("LEC query lacks a unique check and symbolic input declarations")
    return query.replace("(check-sat)", "(check-sat)\n(get-value (" + " ".join(declarations) + "))", 1)


def _selected_module_text(source: bytes, module: str) -> tuple[str, list[str]]:
    sliced, included, missing = extract(source.decode(), module)
    if missing:
        raise ValueError(f"RTL cell closure contains unresolved modules: {missing}")
    forbidden = (
        "seq.",
        "sv.alwaysff",
        "hw.module.extern",
        "hw.module.generated",
        "sv.verbatim.module",
        "verif.assume",
        "verif.assert",
        "smt.",
    )
    if any(token in sliced for token in forbidden):
        raise ValueError("combinational proof refuses stateful or opaque selected cell closure")
    if sliced.count("\n}") != 1:
        raise ValueError("unexpected extracted module wrapper")
    # The selected SoC extraction carries global hierarchy paths and macro
    # declarations unrelated to this closed cell. LEC works on the exact
    # extracted HW module bodies, with those global bookkeeping ops removed.
    selected_lines = sliced.splitlines()
    first_module = next(
        (index for index, line in enumerate(selected_lines) if line.lstrip().startswith("hw.module ")),
        None,
    )
    if first_module is None:
        raise ValueError("selected closure has no HW module body")
    return "module {\n" + "\n".join(selected_lines[first_module:-1]) + "\n}\n", included


def _circt_args(circt: Path, root: Path, module: str, reference_module: str) -> list[str]:
    return [
        str(circt),
        str(root / "lec.mlir"),
        "--strip-om",
        "--strip-emit",
        f"--construct-lec=first-module={module} second-module={reference_module} insert-mode=none",
        "--convert-synth-to-comb",
        "--lower-comb",
        "--convert-hw-to-smt",
        "--convert-datapath-to-smt",
        "--convert-comb-to-smt",
        "--convert-verif-to-smt",
        "--canonicalize",
        "-o",
        str(root / "query.mlir"),
    ]


def _combine(selected_text: str, reference_bytes: bytes) -> str:
    reference_text = "\n".join(
        line for line in reference_bytes.decode().splitlines() if not line.lstrip().startswith("//")
    ).strip()
    return selected_text.rsplit("\n}", 1)[0] + "\n  " + reference_text + "\n}\n"


def prove_combinational_property(
    *,
    hw_source: str | Path,
    module: str,
    reference: str | Path,
    reference_module: str,
    circt_opt: str | Path,
    z3: str | Path,
    output: str | Path,
    property_statement: str,
    assumptions: list[str] | None = None,
    timeout_seconds: int = 60,
) -> dict:
    """Prove an exact, finite-width combinational equivalence or record why not.

    This API has no implicit assumptions. A declaration in ``assumptions`` is
    explanatory metadata only; nonempty entries are refused until the property
    language can encode and check them. A separate reference is indispensable:
    comparing the RTL against itself would be a vacuous proof.
    """
    source, ref, circt, solver = (Path(item).resolve(strict=True) for item in (hw_source, reference, circt_opt, z3))
    root = Path(output).absolute()
    if root.exists():
        raise ValueError("hardware property requires a new output directory")
    if not module.isidentifier() or not reference_module.isidentifier() or module == reference_module:
        raise ValueError("property requires distinct simple HW module names")
    if not property_statement.strip():
        raise ValueError("property statement is required")
    if assumptions:
        raise ValueError("declared assumptions require a checked encoding; this proof admits none")
    if timeout_seconds <= 0:
        raise ValueError("positive solver timeout required")
    source_bytes, ref_bytes = source.read_bytes(), ref.read_bytes()
    if source_bytes == ref_bytes:
        raise ValueError("reference must be authored independently of selected hardware")
    selected_text, included = _selected_module_text(source_bytes, module)
    reference_text = "\n".join(
        line for line in ref_bytes.decode().splitlines() if not line.lstrip().startswith("//")
    ).strip()
    if (
        not reference_text.startswith("hw.module @" + reference_module + "(")
        or reference_text.count("hw.module @") != 1
    ):
        raise ValueError("reference must contain exactly the named HW module declaration")
    if any(token in reference_text for token in ("seq.", "sv.", "hw.instance", "verif.", "smt.")):
        raise ValueError("reference contains state, hierarchy, or unencoded assumptions")
    root.mkdir(parents=True)
    (root / "selected.hw.mlir").write_text(selected_text)
    (root / "reference.hw.mlir").write_bytes(ref_bytes)
    combined = _combine(selected_text, ref_bytes)
    (root / "lec.mlir").write_text(combined)
    circt_args = _circt_args(circt, root, module, reference_module)
    receipt: dict = {
        "schema": "merlin.hardware_property.v1",
        "evidence_kind": "unbounded_combinational_equivalence",
        "status": "unknown",
        "scope": "one selected combinational HW module and one authored reference at concrete port widths",
        "domain": "all two-state raw input bit patterns at the selected module's concrete port widths",
        "property": property_statement,
        "assumptions": [],
        "cycle_bound": None,
        "selected_module": module,
        "reference_module": reference_module,
        "included_modules": included,
        "source": _identity(source),
        "reference": _identity(ref),
        "tools": {"circt_opt": _identity(circt), "z3": _identity(solver)},
        "producer": _identity(Path(__file__)),
        "commands": [],
        "whole_operation_support": "not_qualified",
        "compiler_support": "not_qualified",
    }
    try:
        lowered = _run(circt_args, timeout=timeout_seconds)
        (root / "circt.stdout.log").write_text(lowered.stdout)
        (root / "circt.stderr.log").write_text(lowered.stderr)
        receipt["commands"].append({"argv": circt_args, "returncode": lowered.returncode})
        if lowered.returncode:
            raise RuntimeError("CIRCT failed to construct and lower LEC query")
        translator = circt.parent / "circt-translate"
        if not translator.is_file():
            raise FileNotFoundError("matching CIRCT SMT-LIB translator unavailable")
        receipt["tools"]["circt_translate"] = _identity(translator)
        translate_args = [str(translator), str(root / "query.mlir"), "--export-smtlib"]
        translated = _run(translate_args, timeout=timeout_seconds)
        (root / "translate.stderr.log").write_text(translated.stderr)
        receipt["commands"].append({"argv": translate_args, "returncode": translated.returncode})
        if translated.returncode:
            raise RuntimeError("CIRCT failed to export the LEC query to SMT-LIB")
        query = translated.stdout
        _query_with_model(query)  # malformed or vacuous queries are rejected
        (root / "query.smt2").write_text(query)
        solver_args = [str(solver), "-smt2", str(root / "query.smt2")]
        solved = _run(solver_args, timeout=timeout_seconds)
        (root / "z3.stdout.log").write_text(solved.stdout)
        (root / "z3.stderr.log").write_text(solved.stderr)
        receipt["commands"].append({"argv": solver_args, "returncode": solved.returncode})
        verdicts = solved.stdout.strip().splitlines()
        if solved.returncode or len(verdicts) != 1 or verdicts[0] not in ("unsat", "sat", "unknown"):
            raise RuntimeError("solver did not return one recognized LEC verdict")
        receipt["solver_verdict"] = verdicts[0]
        receipt["status"] = {"unsat": "proven", "sat": "refuted", "unknown": "unknown"}[verdicts[0]]
        if verdicts[0] == "sat":
            (root / "counterexample_query.smt2").write_text(_query_with_model(query))
            witness_args = [str(solver), "-smt2", str(root / "counterexample_query.smt2")]
            witness = _run(witness_args, timeout=timeout_seconds)
            (root / "counterexample.stdout.log").write_text(witness.stdout)
            (root / "counterexample.stderr.log").write_text(witness.stderr)
            receipt["commands"].append({"argv": witness_args, "returncode": witness.returncode})
            receipt["counterexample"] = witness.stdout if witness.returncode == 0 else None
        if (
            _digest(source.read_bytes()) != receipt["source"]["sha256"]
            or _digest(ref.read_bytes()) != receipt["reference"]["sha256"]
        ):
            raise RuntimeError("hardware or authored reference changed during proof")
        if any(_digest(Path(row["path"]).read_bytes()) != row["sha256"] for row in receipt["tools"].values()):
            raise RuntimeError("formal tool changed during proof")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        receipt["status"] = "unknown"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["artifacts"] = {
        path.name: {"sha256": _digest(path.read_bytes()), "size_bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    (root / "property.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_property_receipt(
    path: str | Path,
    *,
    hw_source: str | Path,
    replay: bool = True,
    timeout_seconds: int = 60,
) -> dict:
    """Check saved artifact integrity and the formal verdict's exact scope.

    An unsigned receipt does not establish author identity. Replay regenerates
    the SMT-LIB query from the selected source/reference and reruns the solver.
    """
    receipt_path = Path(path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("schema") != "merlin.hardware_property.v1" or receipt.get("status") != "proven":
        raise ValueError("hardware property is not formally proven")
    if receipt.get("evidence_kind") != "unbounded_combinational_equivalence" or receipt.get("assumptions") != []:
        raise ValueError("hardware property scope or assumptions differ")
    if (
        receipt.get("domain") != "all two-state raw input bit patterns at the selected module's concrete port widths"
        or receipt.get("cycle_bound") is not None
    ):
        raise ValueError("hardware property input domain or temporal bound differs")
    source = receipt.get("source", {}).get("sha256")
    selected_source = Path(hw_source).resolve(strict=True)
    if not isinstance(source, str) or len(source) != 64 or _digest(selected_source.read_bytes()) != source:
        raise ValueError("hardware property is not bound to selected source digest")
    if receipt.get("solver_verdict") != "unsat":
        raise ValueError("hardware property lacks UNSAT decision")
    artifacts = receipt.get("artifacts", {})
    for required in ("selected.hw.mlir", "reference.hw.mlir", "lec.mlir", "query.mlir", "query.smt2", "z3.stdout.log"):
        if required not in artifacts:
            raise ValueError(f"missing hardware proof artifact: {required}")
    for name, identity in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("unsafe hardware proof artifact path")
        member = receipt_path.parent / name
        if member.is_symlink() or not member.is_file() or _digest(member.read_bytes()) != identity.get("sha256"):
            raise ValueError(f"hardware proof artifact changed: {name}")
        if member.stat().st_size != identity.get("size_bytes"):
            raise ValueError(f"hardware proof artifact size changed: {name}")
    if (receipt_path.parent / "z3.stdout.log").read_text().strip() != "unsat":
        raise ValueError("hardware solver log does not establish UNSAT")
    selected_text, included = _selected_module_text(selected_source.read_bytes(), receipt["selected_module"])
    if (
        selected_text.encode() != (receipt_path.parent / "selected.hw.mlir").read_bytes()
        or included != receipt["included_modules"]
    ):
        raise ValueError("saved HW module is not the selected source closure")
    ref = receipt.get("reference", {})
    if _digest((receipt_path.parent / "reference.hw.mlir").read_bytes()) != ref.get("sha256"):
        raise ValueError("authored reference is not bound to the receipt")
    reference_path = Path(ref["path"]).resolve(strict=True)
    if _digest(reference_path.read_bytes()) != ref["sha256"]:
        raise ValueError("authored reference changed since proof")
    combined = _combine(selected_text, reference_path.read_bytes())
    if combined.encode() != (receipt_path.parent / "lec.mlir").read_bytes():
        raise ValueError("LEC statement does not combine selected RTL and authored reference")
    if _digest(Path(__file__).read_bytes()) != receipt.get("producer", {}).get("sha256"):
        raise ValueError("hardware proof producer changed since proof")
    if replay:
        root = receipt_path.parent
        tools = receipt.get("tools", {})
        tool_paths = {name: Path(row["path"]).resolve(strict=True) for name, row in tools.items()}
        if any(_digest(tool_paths[name].read_bytes()) != row.get("sha256") for name, row in tools.items()):
            raise ValueError("formal tool binary changed since proof")
        with tempfile.TemporaryDirectory(prefix="hardware-replay-", dir=root) as temporary:
            replay_root = Path(temporary)
            (replay_root / "lec.mlir").write_bytes((root / "lec.mlir").read_bytes())
            circt = _circt_args(
                tool_paths["circt_opt"], replay_root, receipt["selected_module"], receipt["reference_module"]
            )
            lowered = _run(circt, timeout=timeout_seconds)
            if (
                lowered.returncode
                or _digest((replay_root / "query.mlir").read_bytes()) != artifacts["query.mlir"]["sha256"]
            ):
                raise ValueError("replayed CIRCT query differs from saved proof")
            translated = _run(
                [str(tool_paths["circt_translate"]), str(replay_root / "query.mlir"), "--export-smtlib"],
                timeout=timeout_seconds,
            )
            if translated.returncode or translated.stdout.encode() != (root / "query.smt2").read_bytes():
                raise ValueError("replayed SMT-LIB query differs from saved proof")
            (replay_root / "query.smt2").write_text(translated.stdout)
            solved = _run([str(tool_paths["z3"]), "-smt2", str(replay_root / "query.smt2")], timeout=timeout_seconds)
            if solved.returncode or solved.stdout.strip() != "unsat":
                raise ValueError("replayed hardware property is not UNSAT")
    return receipt
