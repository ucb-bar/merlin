"""The independent oracle a whole-model build is graded against, and the LOCAL grade itself.

Split out of :mod:`merlin.analysis.whole_model.build.stages` (which re-exports every name here, so an existing
caller of ``whole_model_build._oracle``/``.grade``/``.memory_map``/... needs no change) purely to
keep that module under this repo's own module-size gate. The split follows the natural seam its own
comments already named: BUILDING a program (statement, binding, compiling, linking) is one concern;
independently recomputing what it should have produced -- from the same weights and the same input,
never trusting the arm under test -- is another.

Nothing here knows a target beyond the parameter every function already takes; the reference oracle
is recomputed the SAME way regardless of which device the program under test targets.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# ---- whole_model_oracle ------------------------------------------------------------------------


def _wmb():
    # The build module imports this one at load; reach back lazily for its error type.
    from merlin.analysis.whole_model.build import stages as whole_model_build

    return whole_model_build


def _entry_array(capsule, domain: Mapping[str, Any], dtype: str):
    """The model argument on the integer grid, laid out the way the buffer declares its entry tensor."""
    import numpy as np

    from merlin.analysis.whole_model.build import stages as WMB

    if len(capsule.inputs) != 1:
        raise WMB.WholeModelBuildError(f"the capsule declares {len(capsule.inputs)} inputs; one program quantizes one")
    if domain.get("scale") is None:
        raise WMB.WholeModelBuildError("the program's input scale is not a compile-time number")
    # The command-buffer dtype token of a signed integer grid is `i<bits>`; anything else is not a grid
    # a scale puts a float argument on, and is refused rather than read as one.
    if not (dtype.startswith("i") and dtype[1:].isdigit()):
        raise WMB.WholeModelBuildError(f"the program's entry tensor is {dtype!r}, not a signed integer grid")
    (value,) = capsule.inputs.values()
    kind = np.dtype(f"int{int(dtype[1:])}")
    info = np.iinfo(kind)
    grid = np.rint(np.asarray(value, dtype=np.float32).reshape(domain["capture_shape"]) / np.float32(domain["scale"]))
    grid = np.clip(grid + int(domain.get("zero_point") or 0), info.min, info.max).astype(kind)
    return np.ascontiguousarray(grid.transpose(domain["permutation"]))


def _reference_leaves(capsule, *, target: str) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """``(reference statement, every leaf's value by name, the entry array)`` -- the model with no package.

    EVERY LEAF BY NAME, OR NOTHING: a leaf nobody supplies is refused, because the reference would
    otherwise fill it with stimulus and compute a different model.
    """
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.ir.xdsl_io import IR_LOCK
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )
    from merlin.compiler.partition.groups import weights as GP

    reference = WMB.state(capsule, target=target)
    route = reference["whole_program"]
    domain = route["input_domain"]
    with IR_LOCK:
        groups = _merlin_formation_compute_groups_1.form_groups(
            mq.parse(capsule.interface.read_text(encoding="utf-8")), target
        )
    packed = GP.prepack(
        [g for g in groups if g.placement != CG.HOST], capsule.weights_manifest, capsule.weights, device_layout=True
    )
    arrays = packed["arrays"]
    supplied: dict[str, Any] = {}
    # EVERY LEAF BY NAME, OR NOTHING. The prepack names the stored tensor each laid-out array came
    # from, which is the name the statement declares it under; a leaf nobody supplies is refused,
    # because the reference would otherwise fill it with stimulus and compute a different model.
    for row in packed["record"].get("groups") or ():
        for slot in ("weight", "bias"):
            entry = row.get(slot)
            if isinstance(entry, Mapping) and entry.get("stored_tensor") and entry.get("array") in arrays:
                supplied[str(entry["stored_tensor"])] = arrays[entry["array"]]
    for name, spec in (route.get("prepacked") or {}).items():
        if spec.get("array") in arrays:
            supplied[str(name)] = arrays[spec["array"]]
    tensors = reference["tensors"]
    entry_array = _entry_array(capsule, domain, str(tensors[domain["tensor"]]["dtype"]))
    supplied[str(domain["tensor"])] = entry_array
    produced = {c["operands"]["dst"] for c in reference["commands"] if "dst" in (c.get("operands") or {})}
    leaves = [n for n in tensors if n not in produced]
    unsupplied = [n for n in leaves if n not in supplied]
    if unsupplied:
        raise WMB.WholeModelBuildError(f"no value for leaf tensor(s) {unsupplied[:8]}; the reference would invent them")
    return reference, {n: supplied[n] for n in leaves}, entry_array


class LocalReference:
    """Each group's reference computed from the inputs a run ACTUALLY gave it -- LOCAL grading.

    A chained oracle digest is exact only for an arm that reproduces every upstream value exactly. A
    group whose declared contract admits a bounded difference (a residual sum rounding each operand)
    legitimately changes every value below it, and a chained comparison then fails every downstream
    group whatever its own arithmetic. So a group is graded on ITS OWN arithmetic: its commands from
    the reference statement, re-run with every upstream-produced operand replaced by the value the
    device held, and every leaf (weight, bias, the model input) by name from the model.

    ``reference`` is the no-package statement (:func:`merlin.analysis.whole_model.build.stages.state` without a
    package); ``leaves`` maps every tensor no command produces to its value.
    """

    def __init__(self, reference: Mapping[str, Any], leaves: Mapping[str, Any]):
        from merlin.analysis.whole_model.build import stages as WMB

        self.reference = reference
        self.leaves = dict(leaves)
        commands = list(reference.get("commands") or ())
        rows = list((reference.get("whole_program") or {}).get("per_group") or ())
        self.slices: dict[int, tuple[list[dict[str, Any]], str]] = {}
        start = 0
        for row in rows:
            count = int(row.get("commands") or 0)
            self.slices[int(row["group"])] = (commands[start : start + count], str(row["operands"]["dst"]))
            start += count
        if start != len(commands):
            raise WMB.WholeModelBuildError(
                f"the statement's groups account for {start} of its {len(commands)} commands; a group's "
                f"commands cannot be told apart, so no group can be graded on its own"
            )
        self.produced = {str(c["operands"]["dst"]) for c in commands if "dst" in (c.get("operands") or {})}
        # THE SHAPE EACH PRODUCED TENSOR IS HELD IN, as the reference engine itself holds it when a
        # consumer reads it -- taken from one run of the whole statement, never re-derived. It is not
        # always the declared shape: a commit reads its accumulator out as [M, N] (a window mean's
        # [1, C] is declared [C, 1]) and a convolution's input is declared rank-4 over a flat [N*H*W, C].
        import numpy as np

        from merlin.runtime.engine import reference as REF

        whole = {k: v for k, v in reference.items() if k != "outputs"}
        held = REF.reference_outputs(whole, {n: np.asarray(v).tolist() for n, v in self.leaves.items()})
        self.committed = {str(n): list(np.asarray(v).shape) for n, v in held.items() if n in self.produced}

    @classmethod
    def from_capsule(cls, model_capsule, *, target: str) -> LocalReference:
        from merlin.analysis.whole_model.build import stages as WMB

        capsule = (
            model_capsule if isinstance(model_capsule, WMB.ModelCapsule) else WMB.load_model_capsule(model_capsule)
        )
        reference, leaves, _entry = _reference_leaves(capsule, target=target)
        return cls(reference, leaves)

    def _operands(self, group: int) -> tuple[list[str], set[str]]:
        commands, _dst = self.slices[group]
        tensors = self.reference["tensors"]
        named = []
        for command in commands:
            for value in (command.get("operands") or {}).values():
                if isinstance(value, str) and value in tensors and value not in named:
                    named.append(value)
        here = {str(c["operands"]["dst"]) for c in commands if "dst" in (c.get("operands") or {})}
        return named, here

    def inputs_of(self, group: int) -> list[str]:
        """The tensors ``group`` reads that ANOTHER group produces -- what a run must hand back."""
        named, here = self._operands(group)
        return [n for n in named if n in self.produced and n not in here]

    def expected(self, group: int, inputs: Mapping[str, Any]):
        """``group``'s reference output (flat) given the device values of its produced inputs."""
        import numpy as np

        from merlin.analysis.whole_model.build import stages as WMB
        from merlin.runtime.engine import reference as REF

        if group not in self.slices:
            raise WMB.WholeModelBuildError(f"the statement has no group {group}")
        commands, dst = self.slices[group]
        named, here = self._operands(group)
        tensors = self.reference["tensors"]
        missing = [n for n in self.inputs_of(group) if n not in inputs]
        if missing:
            raise WMB.WholeModelBuildError(f"group {group} reads {missing}, which the run did not hand back")
        values: dict[str, Any] = {}
        # A VIEWED tensor is declared in the shape a convolution reads it (rank-4 nhwc) while its
        # producer commits it flat ([N*H*W, C]); the same bytes either way. Inside the whole model the
        # consumer receives the committed form, so a group that reads it as a matmul operand is handed
        # that form; a group that reads it as a convolution's activation keeps the declared one.
        conv_reads = {str((c.get("operands") or {}).get("ifm")) for c in commands if c.get("opcode") == "CONV2D"}
        specs = {n: dict(tensors[n]) for n in named}
        for name in named:
            if name in here:
                continue
            source = inputs[name] if name in self.produced else self.leaves.get(name)
            if source is None:
                raise WMB.WholeModelBuildError(f"group {group} reads leaf {name!r}, which the model does not supply")
            if name in self.committed and name not in conv_reads:
                specs[name]["shape"] = list(self.committed[name])
            shape = [int(e) for e in specs[name]["shape"]]
            values[name] = np.asarray(source).reshape(shape).tolist()
        sub = {
            "abi_version": self.reference.get("abi_version"),
            "target": self.reference.get("target"),
            "tensors": specs,
            "commands": commands,
        }
        return np.asarray(REF.reference_outputs(sub, values)[dst]).reshape(-1)


def _oracle(capsule, *, target: str, digest) -> tuple[dict[str, Any], Any]:
    """Each group's expected output digest from the reference statement, and the model's argmax."""
    import numpy as np

    from merlin.runtime.engine import reference as REF

    reference, supplied, entry_array = _reference_leaves(capsule, target=target)
    route = reference["whole_program"]
    leaves = list(supplied)
    asked = {k: v for k, v in reference.items() if k != "outputs"}
    values = REF.reference_outputs(asked, {n: np.asarray(supplied[n]).tolist() for n in leaves})
    groups_out: dict[str, dict[str, Any]] = {}
    for row in route["per_group"]:
        name = row["operands"]["dst"]
        flat = np.asarray(values[name]).reshape(-1)
        expected = {"output": name, "sum": int(flat.sum()), "fnv1a": int(digest(flat))}
        bound = (row.get("entry") or {}).get("bound_lsb")
        if bound is not None:
            # The op DECLARES a tolerance: the reference rounds once and a scaled load rounds each
            # operand, so an exact digest is not this op's grade. The program checks the bound itself.
            expected.update({"compare": "bounded", "bound_lsb": int(bound)})
        else:
            expected["compare"] = "exact"
        groups_out[str(row["group"])] = expected
    readout = route.get("readout") or {}
    final_raw = np.asarray(values[readout["tensor"]]).reshape(-1)
    logits = final_raw.astype(np.float64) * float(readout["dequantize"])
    (golden,) = capsule.outputs.values()
    golden = np.asarray(golden, dtype=np.float64).reshape(-1)
    cosine = float(logits @ golden / (np.linalg.norm(logits) * np.linalg.norm(golden) + 1e-30))
    oracle = {
        "schema": "group_model_oracle_v2",
        "groups": groups_out,
        "argmax": int(logits.argmax()),
        "golden_argmax": int(golden.argmax()),
        # THE WHOLE MODEL'S OWN FINAL COMMITTED INTEGERS, digested as one vector. Never keyed by group
        # index -- a package pass may rewrite HOW MANY groups the model has (see
        # `merlin.analysis.whole_model.build.inputs`), so a per-group comparison across two differently-grouped
        # statements would be comparing two different keys, not two answers to the same question.
        # This digest is the one fact that survives regrouping: it is a function of the model's
        # inputs and weights alone.
        "final_digest": {"sum": int(final_raw.sum()), "fnv1a": int(digest(final_raw))},
        "cosine_to_golden": cosine,
        "exactness": (
            "a group compared 'exact' is exact given EXACT INPUTS. Downstream of a 'bounded' group an arm "
            "that rounds each operand (as a scaled load does) legitimately differs from these digests; "
            "only an arm that rounds as the reference does reproduces them all"
        ),
    }
    return oracle, entry_array


def verify_passes_preserve_semantics(capsule, *, target: str, transformed_interface: str | Path) -> dict[str, Any]:
    """Whether ``transformed_interface`` computes the SAME function as ``capsule``'s own.

    A package's whole-model pass may restate the model however it likes -- fuse a cast, hoist a
    constant, regroup its compute entirely -- but it may not change what the model COMPUTES, and this
    is the one check that says so: the REFERENCE oracle (no package, :func:`_oracle`) is recomputed
    independently from BOTH modules, given the SAME weights and the SAME input, and compared on the
    one fact that survives a pass changing how many groups the model has -- the final committed
    buffer's own digest (``final_digest``) and the model's argmax. Never the per-group digests: a
    pass that fuses two groups into one has, correctly, fewer of them to compare.

    Fails closed: a transformed module that cannot even be STATED (the pass produced something this
    repo cannot form into groups, or its host region is no longer closed) is a disagreement, not an
    inconclusive result -- ``ok`` is ``False`` either way, and the reason says which.
    """
    import dataclasses

    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.runtime.backends import registry as backends

    digest = backends.whole_model_driver(target).program.group_digest
    try:
        original, _entry = _oracle(capsule, target=target, digest=digest)
    except WMB.WholeModelBuildError as error:
        return {"ok": False, "why": f"the ORIGINAL model could not be stated: {error}"}
    transformed_capsule = dataclasses.replace(capsule, interface=Path(transformed_interface))
    try:
        transformed, _entry2 = _oracle(transformed_capsule, target=target, digest=digest)
    except WMB.WholeModelBuildError as error:
        return {"ok": False, "why": f"the transformed model could not be stated: {error}"}
    same_digest = original["final_digest"] == transformed["final_digest"]
    same_argmax = original["argmax"] == transformed["argmax"]
    ok = bool(same_digest and same_argmax)
    return {
        "ok": ok,
        "original_final_digest": original["final_digest"],
        "transformed_final_digest": transformed["final_digest"],
        "original_argmax": original["argmax"],
        "transformed_argmax": transformed["argmax"],
        "why": (
            ""
            if ok
            else "the transformed model's final committed digest or argmax disagrees with the original's, "
            "given the same weights and the same input"
        ),
    }


def grade(uart: str, oracle: Mapping[str, Any]) -> dict[str, Any]:
    """A run's UART against the oracle, gated on LOCAL checks and the argmax.

    Read structurally (split on whitespace and ``=``), never pattern-matched. THE GATE is per-group
    LOCAL: an exact group passes on its ``GM_LOCAL`` line (the program recomputed its reference on the
    core from the inputs it actually held, and zero elements differ), a bounded group on its
    ``GM_BOUND`` line (every element within the op's declared bound of the reference recomputed from
    its actual operands). A group with no local line is ABSENT, never agreed. The end-to-end argmax is
    the second gate. The chained digests (``GM_GROUP`` against the oracle) are reported as
    ``chained`` -- INFORMATION ONLY: below a legitimate bounded difference upstream they differ for a
    correct run, so they cannot be a gate.

    An oracle a PARTIAL build wrote (``only_groups``) is refused: never quotable, with the reason.
    """
    from merlin.analysis.whole_model.build.inputs import PartialBuildRefused, refuse

    try:
        refuse(oracle, reader="the whole-model grade")
    except PartialBuildRefused as exc:
        return {"gate": "local", "refused": str(exc), "agree": [], "disagree": [], "absent": [], "quotable": False}
    printed: dict[str, dict[str, str]] = {}
    bounded: dict[str, dict[str, str]] = {}
    local: dict[str, dict[str, str]] = {}
    argmax: dict[str, str] | None = None
    for line in uart.splitlines():
        parts = line.split()
        if not parts:
            continue
        fields = dict(p.split("=", 1) for p in parts if "=" in p)
        if parts[0] == "GM_GROUP" and len(parts) > 1:
            printed[parts[1]] = fields
        elif parts[0] == "GM_BOUND" and len(parts) > 1:
            bounded[parts[1]] = fields
        elif parts[0] == "GM_LOCAL" and len(parts) > 1:
            local[parts[1]] = fields
        elif parts[0] == "GM_ARGMAX":
            argmax = fields
    agree, disagree, absent = [], [], []
    c_agree, c_disagree, c_absent = [], [], []
    for group, want in sorted((oracle.get("groups") or {}).items(), key=lambda kv: int(kv[0])):
        if want.get("compare") == "bounded":
            got = bounded.get(group)
            if got is None:
                absent.append(group)
            elif got.get("over") == "0":
                agree.append(group)
            else:
                disagree.append({"group": group, "max_abs": got.get("max_abs"), "over": got.get("over")})
        else:
            got = local.get(group)
            if got is None:
                absent.append(group)
            elif got.get("mismatches") == "0":
                agree.append(group)
            else:
                disagree.append(
                    {
                        "group": group,
                        "mismatches": got.get("mismatches"),
                        "of": got.get("of"),
                        "first": got.get("first"),
                    }
                )
        chained = printed.get(group)
        if chained is None:
            c_absent.append(group)
        elif chained.get("fnv1a") == str(want["fnv1a"]) and chained.get("sum") == str(want["sum"]):
            c_agree.append(group)
        else:
            c_disagree.append({"group": group, "want": want["fnv1a"], "got": chained.get("fnv1a")})
    argmax_ok = argmax is not None and argmax.get("got") == str(oracle.get("argmax"))
    return {
        "gate": "local",
        "agree": agree,
        "disagree": disagree,
        "absent": absent,
        "argmax": argmax,
        "argmax_agrees_with_oracle": argmax_ok,
        "quotable": not disagree and not absent and argmax_ok,
        "chained": {
            "note": "information only: exact only for an arm reproducing every upstream value exactly",
            "agree": c_agree,
            "disagree": c_disagree,
            "absent": c_absent,
        },
    }


# ------------------------------------------------------------------------------- oracle as a job


#: Recomputed oracles, by the digest of everything that decides them (:func:`oracle_cache_key`).
ORACLE_CACHE_NAMESPACE = "whole-model-oracle"


def oracle_cache_key(capsule, *, target: str) -> str:
    """The digest of EVERYTHING an oracle is a function of, so a recorded one is served only for it.

    The model (interface, weights, manifest, golden, capsule declaration), the code that states and
    recomputes it (every file of the core ``merlin`` package -- deliberately all of it, since a narrower
    pick is a cache that silently survives the one edit it missed), the target's program driver (it
    owns the digest a group's expectation is), the selected support provider's tree, the capability
    contract the statement reads (an explicitly selected one included) and the RTL facts. Any change
    to any of them is a new key, never a stale hit.
    """
    import hashlib
    import json

    import merlin
    from merlin.common.digest import hash_tree
    from merlin.runtime.backends import registry as backends
    from merlin.target.registry import identity as TR
    from merlin.target.rtl.facts import find_facts

    digest = hashlib.sha256()

    def part(label: str, value: bytes | str) -> None:
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(value if isinstance(value, bytes) else value.encode())
        digest.update(b"\0")

    part("target", target)
    directory = Path(capsule.directory)
    for name in sorted(p.name for p in directory.iterdir() if p.is_file()):
        part(f"capsule/{name}", (directory / name).read_bytes())
    part("interface", Path(capsule.interface).read_bytes())
    part("weights", Path(capsule.weights).read_bytes())
    part("code", str(hash_tree(Path(merlin.__file__).resolve().parent)["sha256"]))
    driver = backends.whole_model_driver(target)
    part("driver", Path(driver.program.__file__).read_bytes())
    resolved = TR.resolve(target)
    part("support", str(hash_tree(Path(resolved.base))["sha256"]))
    part("contract", json.dumps(resolved.load_contract(), sort_keys=True, default=str))
    facts = find_facts(target)
    part("facts", Path(facts).read_bytes() if facts is not None and Path(facts).is_file() else b"<none>")
    return digest.hexdigest()


def _write_oracle(directory: Path, oracle: Mapping[str, Any], raw: bytes, dtype: str, shape: list[int]) -> None:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "entry.bin").write_bytes(raw)
    (directory / "entry.json").write_text(json.dumps({"dtype": dtype, "shape": shape}), encoding="utf-8")
    # Written last: a directory holding oracle.json is a complete record.
    (directory / "oracle.json").write_text(json.dumps(oracle), encoding="utf-8")


def _read_oracle(directory: Path):
    import json

    import numpy as np

    if not (directory / "oracle.json").is_file():
        return None
    meta = json.loads((directory / "entry.json").read_text(encoding="utf-8"))
    entry = np.frombuffer((directory / "entry.bin").read_bytes(), dtype=meta["dtype"]).reshape(meta["shape"])
    return json.loads((directory / "oracle.json").read_text(encoding="utf-8")), entry


def _oracle_job_main(argv: list[str]) -> int:
    """``python -m merlin.analysis.whole_model.build.oracle <capsule dir> <interface> <target> <out dir>``: the
    oracle, recomputed in a process of its own and written to ``<out dir>``."""
    import dataclasses

    from merlin.runtime.backends import registry as backends

    capsule_dir, interface, target, out = argv
    capsule = dataclasses.replace(_wmb().load_model_capsule(capsule_dir), interface=Path(interface))
    oracle, entry = _oracle(capsule, target=target, digest=backends.whole_model_driver(target).program.group_digest)
    _write_oracle(Path(out), oracle, entry.tobytes(), str(entry.dtype), [int(e) for e in entry.shape])
    return 0


class OracleJob:
    """The oracle, recomputed concurrently with the build or read back from its content-keyed cache.

    It depends on nothing the package does, so it starts with the build -- as a separate PROCESS (the
    statement and the recompute are both interpreter-bound, and a plain child needs nothing of the
    caller's own ``__main__``) -- and is joined only when the record needs it.
    """

    def __init__(self, capsule, *, target: str, cache: bool = True):
        import os
        import subprocess
        import sys
        import tempfile

        from merlin.common.artifacts import cache_dir

        self.key = oracle_cache_key(capsule, target=target)
        self.entry_dir = Path(cache_dir(ORACLE_CACHE_NAMESPACE)) / self.key[:2] / self.key if cache else None
        self.state = "hit" if self.entry_dir is not None and _read_oracle(self.entry_dir) is not None else "miss"
        self._process = None
        if self.state == "miss":
            parent = self.entry_dir.parent if self.entry_dir is not None else Path(tempfile.gettempdir())
            parent.mkdir(parents=True, exist_ok=True)
            self._staged = Path(tempfile.mkdtemp(prefix=".oracle.", dir=parent))
            self._log = self._staged.with_suffix(".log")
            argv = [str(capsule.directory), str(capsule.interface), target, str(self._staged)]
            with self._log.open("w", encoding="utf-8") as log:
                self._process = subprocess.Popen(
                    [sys.executable, "-m", __name__, *argv],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=dict(os.environ),
                )

    def result(self) -> tuple[dict[str, Any], Any]:
        import shutil

        if self._process is None:
            return _read_oracle(self.entry_dir)
        if self._process.wait() != 0:
            tail = self._log.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise _wmb().WholeModelBuildError(f"the oracle could not be recomputed: {tail}")
        computed = _read_oracle(self._staged)
        if self.entry_dir is not None:
            try:
                self._staged.rename(self.entry_dir)
            except OSError:  # another build recorded the same oracle first; it is the same value
                shutil.rmtree(self._staged, ignore_errors=True)
        else:
            shutil.rmtree(self._staged, ignore_errors=True)
        self._log.unlink(missing_ok=True)
        return computed


if __name__ == "__main__":
    import sys

    sys.exit(_oracle_job_main(sys.argv[1:]))


# ---- whole_model_reference ---------------------------------------------------------------------
# The REFERENCE ARM an open model's end result is judged against: its own host code, exact devices.
#
# WHY NOT THE ORACLE. A model whose host code quantizes its activations dynamically amplifies a last-bit
# difference: a rounding step flips, and the flip propagates. Measured on SmolVLA (int8 linears,
# per-row dynamic activation scales), no two faithful computations of the model agree within its
# capsule's numeric policy -- the numpy oracle against the capture's golden 1280/1600 elements, an x86
# build of the very host code against the oracle 1244/1600, the RISC-V program against the oracle
# 1250/1600 -- while two packages whose every dispatch is exact print bit-identical outputs. Scaling the
# input by 1 + 1e-7 moves the output by 0.109. An element-wise check against the oracle therefore fails
# every correct package, and loosening it until it passes would pass incorrect ones too.
#
# WHAT IS JUDGED INSTEAD. Every device group is exact by the program's own local check (the primary
# correctness grade, unchanged). The END RESULT is then the model's own host code, compiled and linked
# the same way, run on the same ISA, toolchain, machine and simulator, with every device group computed
# by the target's own library -- the reference arm. Every dispatch is integer-exact, so a correct
# package's output is BIT-IDENTICAL to it, and so is every dispatch's result digest (``GM_WORDS``), which
# replaces the chained activation check as a gate (that check stays a reported diagnostic).
#
# THE CACHE KEY is everything the reference's bytes depend on: the capsule (interface, weights, leaf
# arguments), the host code (the cut module, the model's objects, the runtime and harness sources, the
# flags), the toolchain (compilers, assembler, linker, C libraries), the machine and its ABI header, and
# the simulator (binary and extension). A part nobody can state is ``UNKNOWN`` and the judgement FAILS
# CLOSED -- a reference whose provenance cannot be named is not one.
#
# WHERE BIT-IDENTITY CANNOT HOLD (a group whose declared epilogue is float, which a package may round
# differently), the model's own declaration may state the fallback: ``end_result: {criterion:
# declared_bound, cosine_min: ..., max_abs: ...}``, judged against the SAME reference arm, never the
# oracle. The default is ``bit_identical``, and the result always says which criterion it applied. No
# bound is written here.

SCHEMA = "whole_model_reference_v1"
BIT_IDENTICAL = "bit_identical"
DECLARED_BOUND = "declared_bound"
CRITERIA = (BIT_IDENTICAL, DECLARED_BOUND)
UNKNOWN = "UNKNOWN"
#: What a reference's bytes depend on; each is a digest or UNKNOWN.
KEY_PARTS = ("capsule", "host_code", "toolchain", "machine", "simulator")


class ReferenceUnavailable(RuntimeError):
    """No reference arm can be named or produced for this build; the end result fails closed."""


# ------------------------------------------------------------------------------------ identity


def file_digest(path: str | Path | None) -> str:
    """sha256 of a file's bytes, or ``UNKNOWN`` when there is no such file."""
    if path is None or not Path(path).is_file():
        return UNKNOWN
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(parts: Mapping[str, Any]) -> str:
    """One digest over a mapping of named parts; ``UNKNOWN`` when any part (at any depth) is."""

    def known(value: Any) -> bool:
        if isinstance(value, Mapping):
            return all(known(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(known(v) for v in value)
        return value is not None and value != UNKNOWN

    if not parts or not known(parts):
        return UNKNOWN
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


BUILD_ROOT = "<build>"


def files_identity(
    files: Mapping[str, str | Path | None], flags: Sequence[str] = (), *, root: str | Path | None = None
) -> dict[str, Any]:
    """``{"files": {name: sha}, "flags": [...], "digest": ...}`` -- the digest is UNKNOWN if any file is.

    ``root`` is the build's own directory: a flag that names a path inside it (an include directory the
    build copied its sources into) is recorded relative to it, because the same program built in two
    directories is the same program -- the bytes those directories hold are digested in ``files``."""
    prefix = str(Path(root)) if root is not None else None
    named = {str(name): file_digest(path) for name, path in sorted(files.items())}
    body = {"files": named, "flags": [_build_relative(str(f), prefix) for f in flags]}
    return {**body, "digest": identity_digest(body)}


def _build_relative(flag: str, prefix: str | None) -> str:
    if not prefix:
        return flag
    head, sep, tail = flag.partition(prefix)
    if not sep or (tail and not tail.startswith(os.sep)):
        return flag
    return head + BUILD_ROOT + tail


def toolchain_identity(compiler: str | Path, clang: str | Path, flags: Sequence[str] = ()) -> dict[str, Any]:
    """The toolchain a program was built with, by content: both compilers, the GCC driver's own passes and
    the C libraries it links for ``flags`` (asked of the driver, never assumed)."""

    def asked(option: str) -> str | None:
        try:
            answer = subprocess.run(
                [str(compiler), *map(str, flags), option], capture_output=True, text=True, timeout=60
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        # A driver that does not find a file echoes the bare name back.
        return answer if answer and os.sep in answer else None

    files = {
        "compiler": compiler,
        "clang": clang,
        "cc1": asked("-print-prog-name=cc1"),
        "as": asked("-print-prog-name=as"),
        "ld": asked("-print-prog-name=ld"),
        "libc.a": asked("-print-file-name=libc.a"),
        "libm.a": asked("-print-file-name=libm.a"),
        "libgcc.a": asked("-print-libgcc-file-name"),
    }
    return files_identity(files, flags)


def key_parts(record: Mapping[str, Any], simulator: Mapping[str, Any] | None) -> dict[str, str]:
    """The cache key's parts from an open build record (``reference_identity``) and a simulator identity."""
    identity = record.get("reference_identity") or {}
    return {
        "capsule": str((identity.get("capsule") or {}).get("digest") or UNKNOWN),
        "host_code": str((identity.get("host_code") or {}).get("digest") or UNKNOWN),
        "toolchain": str((identity.get("toolchain") or {}).get("digest") or UNKNOWN),
        "machine": identity_digest(identity.get("machine") or {}),
        "simulator": str((simulator or {}).get("digest") or UNKNOWN),
    }


def reference_key(parts: Mapping[str, str]) -> str:
    """The cache key, or :class:`ReferenceUnavailable` naming every part nobody could state."""
    unknown = [name for name in KEY_PARTS if parts.get(name) in (None, "", UNKNOWN)]
    if unknown:
        raise ReferenceUnavailable(
            f"the reference arm's key has UNKNOWN part(s) {unknown}: a reference whose provenance cannot be "
            "named is not one, so the end result fails closed"
        )
    return identity_digest({name: parts[name] for name in KEY_PARTS})


def simulator_identity(target: str, simulator: str) -> dict[str, Any] | None:
    """The target backend's own statement of which simulator build runs a program (``None``: it states none)."""
    from merlin.runtime.backends import registry as backends

    stated = getattr(backends.get_backend(target), "simulator_identity", None)
    return stated(simulator) if callable(stated) else None


# --------------------------------------------------------------------------------------- cache


def cache_root(target: str) -> Path:
    from merlin.common.artifacts import cache_dir

    return Path(cache_dir("whole-model-reference")) / str(target)


def lookup(key: str, *, target: str, root: Path | None = None) -> dict[str, Any] | None:
    path = (root or cache_root(target)) / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim(key: str, *, target: str, root: Path, wait_s: int) -> dict[str, Any] | None:
    """The cached entry under ``key``, or ``None`` once this process holds its producer lock.

    One producer per key: an exclusive lock file; a waiter polls for the entry, and a lock whose holder
    died is taken over."""
    lock = root / f"{key}.lock"
    deadline = time.monotonic() + wait_s
    while True:
        existing = lookup(key, target=target, root=root)
        if existing is not None:
            return existing
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            return None
        except FileExistsError:
            holder = lock.read_text(encoding="utf-8").strip() if lock.is_file() else ""
            if holder.isdigit() and not _alive(int(holder)):
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise ReferenceUnavailable(f"another process has held the reference lock {lock} past {wait_s} s")
            time.sleep(30)


def _run_and_record(
    built: Mapping[str, Any],
    *,
    target: str,
    simulator: str,
    key: str,
    parts: Mapping[str, str],
    simulator_stated: Mapping[str, Any] | None,
    timeout: int,
    root: Path,
    build_record: Path,
) -> dict[str, Any]:
    """Run a built reference arm on ``simulator`` and cache what it printed under ``key``."""
    from merlin.analysis.whole_model.grading import verdict as V
    from merlin.runtime.backends import registry as backends

    started = time.monotonic()
    console = backends.get_backend(target).run_elf(built["elf"], simulator=simulator, timeout=timeout)
    wall = round(time.monotonic() - started, 1)
    (root / f"{key}.console.txt").write_text(console, encoding="utf-8")
    driver = backends.whole_model_driver(target)
    parsed = V.parse_log(console, getattr(driver.program, "UART", None))
    inexact = sorted(g for g, (mismatches, _of) in parsed.local.items() if mismatches != 0)
    if inexact or not parsed.local:
        raise ReferenceUnavailable(f"the reference arm's own dispatches are not exact: {inexact[:8]}")
    if parsed.output_digest is None:
        raise ReferenceUnavailable("the reference arm printed no output digest")
    entry = {
        "schema": SCHEMA,
        "key": key,
        "parts": dict(parts),
        "target": target,
        "simulator": {"name": simulator, **dict(simulator_stated or {})},
        "output_digest": {"bytes": parsed.output_digest[0], "digest": parsed.output_digest[1]},
        "output_bits": output_bits(console),
        "words": {g: list(v) for g, v in parsed.words.items()},
        "dispatches": len(parsed.local),
        "elf_sha256": built["elf_sha256"],
        "build_record": str(build_record),
        "console": str(root / f"{key}.console.txt"),
        "wall_s": wall,
        "produced_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "provenance": built.get("provenance"),
    }
    (root / f"{key}.json").write_text(json.dumps(entry, indent=1, default=str) + "\n", encoding="utf-8")
    # The image is regenerable; its digest, record and console are what the entry cites.
    Path(built["elf"]).unlink(missing_ok=True)
    return entry


def produce(
    record: Mapping[str, Any],
    *,
    target: str,
    simulator: str,
    key: str,
    parts: Mapping[str, str],
    simulator_stated: Mapping[str, Any] | None,
    timeout: int,
    jobs: int | None = None,
    root: Path | None = None,
    wait_s: int = 6 * 3600,
) -> dict[str, Any]:
    """Build and run the reference arm of ``record``'s model and cache what it printed.

    The reference is the SAME capsule, machine and header built with no package -- the target's
    library answers every group -- and its host code must digest equal to the candidate's, or it is not
    this model's reference and is refused."""
    from merlin.analysis.whole_model.build import open_driver as _merlin_open_driver_2
    from merlin.analysis.whole_model.build import open_program as WO

    root = root or cache_root(target)
    root.mkdir(parents=True, exist_ok=True)
    existing = _claim(key, target=target, root=root, wait_s=wait_s)
    if existing is not None:
        return existing
    try:
        work = root / key
        header = (record.get("program") or {}).get("abi_header") or {}
        extra = ((record.get("arguments") or {}).get("extra") or {}).get("path")
        # The candidate's host code is the reference's only if it is cut into the same chunks: a chunked
        # candidate's main.mlir never digests equal to an unchunked reference's, and the unchunked
        # reference is the 2-hour compile the chunking exists to avoid.
        chunk_ops = (record.get("forward_chunks") or {}).get("chunk_ops")
        # ...and run on the same harts: a candidate whose host code runs on a vector hart is compiled
        # for that hart's ISA, so a one-hart reference would never digest equal to it.
        host_hart = ((record.get("program") or {}).get("harts") or {}).get("host")
        built = _merlin_open_driver_2.build(
            None,
            str((record.get("capsule") or {}).get("directory")),
            target=target,
            machine=str(record.get("machine")),
            header=str(header.get("header")),
            header_sha256=None if header.get("status") == "registry_declared" else header.get("sha256"),
            extra=extra,
            out=work / "build",
            verify="local",
            jobs=jobs,
            **({"chunk_ops": int(chunk_ops)} if chunk_ops is not None else {}),
            **({"host_hart": int(host_hart)} if host_hart is not None else {}),
        )
        mine = (record.get("reference_identity") or {}).get("host_code") or {}
        theirs = (built.get("reference_identity") or {}).get("host_code") or {}
        if mine.get("digest") in (None, UNKNOWN) or mine.get("digest") != theirs.get("digest"):
            differ = sorted(
                name
                for name in set(mine.get("files") or {}) | set(theirs.get("files") or {})
                if (mine.get("files") or {}).get(name) != (theirs.get("files") or {}).get(name)
            )
            flags = sorted(set(mine.get("flags") or ()) ^ set(theirs.get("flags") or ()))
            raise ReferenceUnavailable(
                f"the reference arm's host code does not digest equal to the candidate's (differs in "
                f"{differ[:8]}, flags {flags[:4]}); "
                "it would not be this program's reference"
            )
        return _run_and_record(
            built,
            target=target,
            simulator=simulator,
            key=key,
            parts=parts,
            simulator_stated=simulator_stated,
            timeout=timeout,
            root=root,
            build_record=work / "build" / "whole_model_open_build.json",
        )
    finally:
        (root / f"{key}.lock").unlink(missing_ok=True)


def prepare(
    model_capsule: str | Path,
    *,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    simulator: str = "spike",
    timeout: int = 7200,
    jobs: int | None = None,
    root: Path | None = None,
    wait_s: int = 6 * 3600,
    chunk_ops: int | None = None,
) -> dict[str, Any]:
    """Produce a model's reference arm AHEAD of any candidate, so a gate finds it cached.

    ``chunk_ops`` must be the size the candidates are built with (see :func:`produce`); ``None`` is the
    unchunked program.

    The reference's key is a property of the model, the machine, the toolchain and the simulator, never
    of a package, so it can be built and run while a candidate is still building: its own build with no
    package states the same host code a candidate's does (``produce`` refuses one that does not). The
    build is kept under ``<cache>/prepared/``; a key some other process already cached or is producing
    is returned or waited for, never produced twice."""
    from merlin.analysis.whole_model.build import open_driver as _merlin_open_driver_3
    from merlin.analysis.whole_model.build import open_program as WO

    root = root or cache_root(target)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = root / "prepared" / f"{Path(model_capsule).name}_{machine}_{stamp}_{os.getpid()}"
    try:
        built = _merlin_open_driver_3.build(
            None,
            str(model_capsule),
            target=target,
            machine=machine,
            header=str(header),
            header_sha256=header_sha256,
            out=work,
            verify="local",
            jobs=jobs,
            **({"chunk_ops": chunk_ops} if chunk_ops is not None else {}),
            # The harts a candidate built through the service runs on (its default: the machine's
            # vector hart, when it declares one), so the prepared reference is the candidate's.
            host_hart=WO.vector_host_hart(machine),
        )
    except BaseException:
        # A refused or failed build leaves nothing an entry could cite.
        shutil.rmtree(work, ignore_errors=True)
        raise
    stated = simulator_identity(target, simulator)
    parts = key_parts(built, stated)
    key = reference_key(parts)
    existing = _claim(key, target=target, root=root, wait_s=wait_s)
    if existing is not None:
        return existing
    try:
        return _run_and_record(
            built,
            target=target,
            simulator=simulator,
            key=key,
            parts=parts,
            simulator_stated=stated,
            timeout=timeout,
            root=root,
            build_record=work / "whole_model_open_build.json",
        )
    finally:
        (root / f"{key}.lock").unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------- judge


def output_bits(console: str) -> list[int] | None:
    """The ``OUT <n> <bits...>`` values a program printed (its first ``n`` output elements)."""
    for line in console.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == "OUT" and parts[1].isdigit():
            values = parts[2 : 2 + int(parts[1])]
            return [int(v) for v in values] if all(v.isdigit() for v in values) else None
    return None


def _floats(bits: Sequence[int]):
    import numpy as np

    return np.asarray([int(b) & 0xFFFFFFFF for b in bits], np.uint32).view(np.float32).astype(np.float64)


def judge(
    console: str,
    reference: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any] | None = None,
    templates: Mapping[str, str] | None = None,
    elements: int | None = None,
) -> dict[str, Any]:
    """A candidate's console against a reference entry, under the model's declared criterion."""
    import numpy as np

    from merlin.analysis.whole_model.grading import verdict as V

    declaration = dict(declaration or {})
    criterion = str(declaration.get("criterion") or BIT_IDENTICAL)
    parsed = V.parse_log(console, templates)
    mine = parsed.output_digest
    theirs = reference.get("output_digest") or {}
    result: dict[str, Any] = {
        "basis": "reference arm",
        "end_result_criterion": criterion,
        "reference": {
            "key": reference.get("key"),
            "output_digest": theirs,
            "elf_sha256": reference.get("elf_sha256"),
            "simulator": reference.get("simulator"),
            "provenance": reference.get("provenance"),
        },
        "output_digest": None if mine is None else {"bytes": mine[0], "digest": mine[1]},
    }
    ref_words = {str(g): tuple(v) for g, v in (reference.get("words") or {}).items()}
    differ = sorted(
        (g for g in set(ref_words) | set(parsed.words) if ref_words.get(g) != parsed.words.get(g)),
        key=lambda g: int(g) if g.isdigit() else 10**9,
    )
    result["words"] = {"dispatches": len(ref_words), "agree": len(ref_words) - len(differ), "differ": differ[:24]}
    identical = mine is not None and [mine[0], mine[1]] == [theirs.get("bytes"), theirs.get("digest")]
    result["bit_identical"] = identical
    if criterion == BIT_IDENTICAL:
        result["passed"] = identical and not differ
    elif criterion == DECLARED_BOUND:
        cosine_min, max_abs = declaration.get("cosine_min"), declaration.get("max_abs")
        got, want = output_bits(console), reference.get("output_bits")
        if cosine_min is None or max_abs is None:
            result.update(passed=False, note="the declared bound states no cosine_min and max_abs")
        elif not got or not want or len(got) != len(want) or (elements is not None and len(got) != int(elements)):
            result.update(
                passed=False,
                note="a declared bound is judged over the whole output; the program printed only part of it",
            )
        else:
            a, b = _floats(got), _floats(want)
            cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
            worst = float(np.abs(a - b).max())
            result["bound"] = {"cosine": cosine, "max_abs": worst, "cosine_min": cosine_min, "max_abs_bound": max_abs}
            result["passed"] = cosine >= float(cosine_min) and worst <= float(max_abs)
    else:
        result.update(passed=False, note=f"unknown end-result criterion {criterion!r}; declared one of {CRITERIA}")
    return result


def end_result(
    console: str,
    record: Mapping[str, Any],
    *,
    target: str,
    simulator: str,
    declaration: Mapping[str, Any] | None = None,
    templates: Mapping[str, str] | None = None,
    timeout: int = 7200,
    jobs: int | None = None,
    root: Path | None = None,
    produce_missing: bool = True,
    elements: int | None = None,
) -> dict[str, Any]:
    """The end result of an open model's run against its reference arm; never raises -- a reference that
    cannot be named or produced is a failed end result with the reason."""
    try:
        stated = simulator_identity(target, simulator)
        parts = key_parts(record, stated)
        key = reference_key(parts)
        reference = lookup(key, target=target, root=root)
        if reference is None:
            if not produce_missing:
                raise ReferenceUnavailable(f"no reference arm is cached under {key[:16]} and none was produced")
            reference = produce(
                record,
                target=target,
                simulator=simulator,
                key=key,
                parts=parts,
                simulator_stated=stated,
                timeout=timeout,
                jobs=jobs,
                root=root,
            )
    except ReferenceUnavailable as why:
        return {
            "passed": False,
            "basis": "reference arm",
            "end_result_criterion": str((declaration or {}).get("criterion") or BIT_IDENTICAL),
            "note": str(why),
        }
    return judge(console, reference, declaration=declaration, templates=templates, elements=elements)


__all__ = [
    "LocalReference",
    "OracleJob",
    "grade",
    "oracle_cache_key",
    "verify_passes_preserve_semantics",
]
