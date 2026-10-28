"""Pre-apply immutable weight layouts for the generated inference caller.

The existing optional feature rewrites one source bundle into an owned cache.
Lowering and runtime ABI generation consume the same transformed IR, manifest
and bit-preserving tensor bytes. Explicit current-source geometry and runtime
ownership checks precede publication; unknown effects and mutable state refuse.
Content identities and actual inverse layout checks reject stale cached products.
This does not qualify target instructions, arbitrary entry clients or latency.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from merlin.common.digest import sha256_file

#: the compiler-feature name a package/fork lists in `compiler_features`
FEATURE = "prepack_weight_layout"

#: bump when the produced bundle's BYTES would change, so cached bundles from an older rewrite are
#: not reused under the new semantics
REWRITE_VERSION = "3"

#: Required bundle inputs; all original flat sidecars are content-pinned as well.
_KEY_FILES = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json")
_IDENTITY_FILE = "weight-prepack-identity.json"


class PrepackRefused(RuntimeError):
    """The weight layout of this bundle cannot be pre-applied soundly. Never downgraded to a
    warning: a build that quietly kept the stock bundle would report the lever as applied."""


@dataclass(frozen=True)
class PrepackPlan:
    """What pre-applying `bundle`'s weight layout would do, without doing it."""

    bundle: str
    hoistable: int
    blocked: int
    #: weight bytes the model re-lays-out per inference today, and would stop moving
    bytes_per_inference: int
    #: why it cannot be done; empty means it can
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.hoistable > 0 and not self.problems


def plan(src: Path | str, func_name: str = "forward", *, immutable_arguments=None) -> PrepackPlan:
    """The hoistable/blocked split for `src` plus the storage-level safety checks. Writes nothing.

    Shares its analysis and its checks with :func:`prepacked_bundle`, so a plan and a build can never
    disagree about what is hoistable or why it was refused.
    """
    from ..capture.rewrite import hoist_safety_problems
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..xdsl_dialects.lowering.weight_layout import weight_layout_report

    src = Path(src)
    with IR_LOCK:
        report = weight_layout_report(mq.parse((src / "model.mlir").read_text()), func_name)

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    weight_args = {int(k) for k, entry in man.items() if k.isdigit() and "weight" in entry}
    # `weight_layout_report` deliberately knows only IR structure, so it also reports transposed
    # runtime inputs.  At the bundle seam we can distinguish them: only manifest-declared weights
    # are candidates for an offline storage rewrite.  A non-weight transpose stays in the graph; it
    # is not a soundness failure for the weights that *can* be hoisted.
    eligible = [
        r
        for r in report.hoistable
        if r.arg in weight_args and (immutable_arguments is None or r.arg in immutable_arguments)
    ]
    problems: list[str] = [
        problem for problem in report.unpriceable if any(problem.startswith(f"arg {arg}:") for arg in weight_args)
    ]
    want: dict[str, int] = {}
    for r in eligible:
        entry = man.get(str(r.arg))
        assert entry is not None and "weight" in entry
        want[entry["weight"]] = r.arg
    if want:
        problems.extend(
            hoist_safety_problems(
                src,
                man,
                want,
                {r.arg for r in eligible},
                relayouts=tuple(eligible),
                func_name=func_name,
                immutable_arguments=immutable_arguments,
            )
        )
    return PrepackPlan(
        bundle=src.name,
        hoistable=len(eligible),
        blocked=len(report.blocked) + len(report.hoistable) - len(eligible),
        bytes_per_inference=sum(r.bytes_moved for r in eligible),
        problems=tuple(problems),
    )


def _file_identity(path):
    return dict(path=str(path.resolve()), sha256=sha256_file(path), bytes=path.stat().st_size)


def _implementation_paths():
    from merlin.capture import rewrite
    from merlin.xdsl_dialects.lowering import weight_layout

    from .weight_prepack_runtime import implementation_paths

    return (Path(__file__), Path(rewrite.__file__), Path(weight_layout.__file__), *implementation_paths())


def _input_identity(src, func_name, immutable_arguments):
    if immutable_arguments is not None and (
        type(immutable_arguments) is not tuple
        or any(type(i) is not int or i < 0 for i in immutable_arguments)
        or len(set(immutable_arguments)) != len(immutable_arguments)
    ):
        raise PrepackRefused("prepacking requires an explicit immutable original argument tuple")
    src = Path(src).resolve()
    for name in _KEY_FILES:
        if not (src / name).is_file():
            raise PrepackRefused("weight prepacking input is absent: " + name)
    if (src / _IDENTITY_FILE).exists():
        raise PrepackRefused("source bundle already contains a weight-preparation cache identity")
    files = {
        p.name: _file_identity(p)
        for p in sorted(src.iterdir())
        if p.is_file() and not p.name.startswith("model.prepared")
    }
    return dict(
        schema="merlin.weight_prepack_inputs.v1",
        feature=FEATURE,
        version=REWRITE_VERSION,
        source=str(src),
        entry=func_name,
        immutable_arguments=list(immutable_arguments) if immutable_arguments is not None else None,
        files=files,
        absent_session=not (src / "session_contract.yaml").exists(),
        implementations=[_file_identity(p) for p in _implementation_paths()],
    )


def _identity_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def cache_key(src: Path | str, func_name="forward", *, immutable_arguments=None) -> str:
    """Content identity, including entry, original ownership and implementation.

    Weight hashes stream in bounded chunks. This deliberately pays the read cost
    rather than accepting same-size, same-timestamp replacements as current.
    """
    return _identity_key(_input_identity(src, func_name, immutable_arguments))


def _prepared_files(directory):
    return {p.name: _file_identity(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != _IDENTITY_FILE}


def _validate_cached(src, dst, identity):
    path = dst / _IDENTITY_FILE
    if path.is_symlink() or not path.is_file():
        raise PrepackRefused("prepacked cache lacks its original content identity; use a fresh cache")
    try:
        record = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise PrepackRefused("prepacked cache identity is unreadable") from error
    if (
        type(record) is not dict
        or set(record) != {"schema", "inputs", "outputs"}
        or record.get("schema") != "merlin.weight_prepack_result.v1"
        or record.get("inputs") != identity
        or record.get("outputs") != _prepared_files(dst)
        or _input_identity(
            src,
            identity["entry"],
            tuple(identity["immutable_arguments"]) if identity["immutable_arguments"] is not None else None,
        )
        != identity
    ):
        raise PrepackRefused("prepacked source, implementation or realized files changed; replan")
    _verify_cached_contents(str(src.resolve()), str(dst.resolve()), json.dumps(record, sort_keys=True))
    return record


@lru_cache(maxsize=16)
def _verify_cached_contents(source: str, destination: str, record_json: str) -> None:
    """Reuse a successful semantic check only within this process and exact content.

    The caller rehashes all source, implementation and realized files before every
    lookup. A cache writer's new endorsement therefore cannot bypass the rewrite
    check. This avoids repeatedly parsing a full model during a single build.
    Failed checks are not retained, and another process starts with no authority.
    """
    from merlin.capture.rewrite import RewriteRefused, verify_hoisted_bundle

    src, dst = Path(source), Path(destination)
    record = json.loads(record_json)
    identity = record["inputs"]
    immutable = tuple(identity["immutable_arguments"]) if identity["immutable_arguments"] is not None else None

    try:
        verify_hoisted_bundle(
            src,
            dst,
            identity["entry"],
            immutable_arguments=immutable,
        )
    except RewriteRefused as error:
        raise PrepackRefused(str(error)) from error
    # Do not memoize a proof over files that changed while it was being checked.
    if _input_identity(src, identity["entry"], immutable) != identity or _prepared_files(dst) != record["outputs"]:
        raise PrepackRefused("prepacked files changed during semantic validation; replan")


def prepacked_bundle(
    src: Path | str, *, cache_root: Path | str | None = None, func_name: str = "forward", immutable_arguments=None
) -> tuple[Path, dict]:
    """`(bundle_dir, effect)` for a bundle storing `src`'s weights in their consumers' layout.

    NEVER mutates `src`: the recapture tree is shared by every other session and every other
    measurement made from it. The result is a separate, cached directory keyed by
    :func:`cache_key`, published by an atomic rename so two concurrent builds cannot observe a
    half-written bundle (the loser reuses the winner's).
    """
    from ..capture.rewrite import RewriteRefused, hoist_weight_transposes, read_rewrites

    src = Path(src).resolve()
    identity = _input_identity(src, func_name, immutable_arguments)
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"{src.name}__{_identity_key(identity)}"
    if dst.is_dir():
        _validate_cached(src, dst, identity)
        recs = [r for r in read_rewrites(dst) if r.name == "hoist_weight_transposes"]
        if recs:
            return dst, {"cached": True, **recs[-1].effect}
        raise PrepackRefused("prepacked cache lacks its rewrite record; use a fresh cache")

    p = plan(src, func_name, immutable_arguments=immutable_arguments)
    if not p.ok:
        raise PrepackRefused(f"{src.name}: {'; '.join(p.problems) if p.problems else 'no hoistable weight transposes'}")

    from ..capture.rewrite import retarget_weights_file

    with tempfile.TemporaryDirectory(dir=root, prefix=f".tmp-{src.name}-") as staging:
        tmp = Path(staging) / "bundle"
        try:
            rec = hoist_weight_transposes(src, tmp, func_name, immutable_arguments=immutable_arguments)
        except RewriteRefused as error:
            raise PrepackRefused(str(error)) from error
        # Publish no intermediate path in the input handed to the compiler.
        text, retargeted = retarget_weights_file((tmp / "model.mlir").read_text(), dst / "weights.safetensors")
        if retargeted:
            (tmp / "model.mlir").write_text(text)
        if _input_identity(src, func_name, immutable_arguments) != identity:
            raise PrepackRefused("prepacking original inputs changed during rewrite; replan")
        outputs = _prepared_files(tmp)
        outputs = {name: {**value, "path": str((dst / name).resolve())} for name, value in outputs.items()}
        (tmp / _IDENTITY_FILE).write_text(
            json.dumps(
                dict(schema="merlin.weight_prepack_result.v1", inputs=identity, outputs=outputs),
                sort_keys=True,
            )
        )
        try:
            os.replace(tmp, dst)
        except OSError:
            if not dst.is_dir():
                raise
            _validate_cached(src, dst, identity)
            return dst, {"cached": True, **rec.effect}
    _validate_cached(src, dst, identity)
    return dst, {"cached": False, **rec.effect}


def _default_cache_root() -> Path:
    from ..common.artifacts import cache_dir

    return cache_dir("weight_prepack")


def prepare_build_bundle(
    src: Path | str,
    work: Path | str,
    features: frozenset[str] | None,
    *,
    compilation=None,
) -> Path:
    """Select ONE bundle for both lowering and runtime ABI generation.

    Backends call this before reading model.mlir or selecting default inputs. No feature means no
    parsing, copying or receipt. The receipt records the deletion, not a predicted speedup.
    """
    from .impr_features import normalize

    ensure_registered()
    src = Path(src).resolve()
    if FEATURE not in normalize(features or ()):
        return src
    from .weight_prepack_runtime import fixed_arguments

    manifest = json.loads((src / "weights.safetensors.manifest.json").read_text())
    try:
        immutable = fixed_arguments(src, manifest)
    except ValueError as error:
        raise PrepackRefused("generated caller cannot establish immutable weight ownership: " + str(error)) from error
    destination, effect = prepacked_bundle(src, immutable_arguments=immutable)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    receipt = work / "bundle_preparation.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "merlin.bundle-preparation.v2",
                "feature": FEATURE,
                "source_bundle": str(src),
                "prepared_bundle": str(destination),
                "effect": effect,
                "immutable_arguments": list(immutable),
                "identity_sha256": sha256_file(destination / _IDENTITY_FILE),
                "scope": "pure tensor source and unchanged stored arguments under the generated inference caller",
                "consumers": ["model_lowering", "runtime_abi_and_weights"],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    if compilation is not None:
        compilation.bind_preparation("weight_prepack", receipt)
        identity = json.loads((destination / _IDENTITY_FILE).read_text())
        dependencies = [
            Path(item["path"])
            for item in (
                *identity["inputs"]["files"].values(),
                *identity["inputs"]["implementations"],
                *identity["outputs"].values(),
            )
        ]
        dependencies.append(destination / _IDENTITY_FILE)
        for index, path in enumerate(dict.fromkeys(dependencies)):
            compilation.bind_preparation("weight_prepack_dependency_" + str(index), path)
    revalidate_build_preparation(receipt)
    return destination


def revalidate_build_preparation(receipt):
    """Reopen original source/session, actual cached bytes and caller premises."""
    from .weight_prepack_runtime import fixed_arguments

    record = json.loads(Path(receipt).read_text())
    if (
        record.get("schema") != "merlin.bundle-preparation.v2"
        or record.get("feature") != FEATURE
        or record.get("consumers") != ["model_lowering", "runtime_abi_and_weights"]
    ):
        raise PrepackRefused("weight preparation is not bound to the generated caller")
    src, dst = Path(record["source_bundle"]), Path(record["prepared_bundle"])
    if sha256_file(dst / _IDENTITY_FILE) != record["identity_sha256"]:
        raise PrepackRefused("original weight preparation identity changed")
    immutable = fixed_arguments(src, json.loads((src / "weights.safetensors.manifest.json").read_text()))
    if list(immutable) != record["immutable_arguments"]:
        raise PrepackRefused("original weight ownership or session changed; replan")
    _validate_cached(src, dst, _input_identity(src, "forward", immutable))


def _feature():
    from .impr_features import ImprFeature

    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "pre-apply exact sole-use static weight layout chains at build time. "
            "The generated caller requires pure tensor source semantics and "
            "unchanged stored arguments; state, streams and unknown effects "
            "remain barriers. Lowering and runtime consume one verified rewritten "
            "bundle. Content and realized-byte checks reject stale caches. "
            "This is a movement/storage optimization, not a latency guarantee."
        ),
    )


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent, so importing from several entry points
    is safe, and necessary in EVERY process that normalizes a feature set -- `impr_features.normalize`
    rejects an unregistered name, and `wholemodel_proposer._composes` swallows that rejection as
    "does not compose", which would make the lever silently unproposable rather than reported."""
    from .impr_features import known, register

    if FEATURE not in known():
        register(_feature())
    return FEATURE
