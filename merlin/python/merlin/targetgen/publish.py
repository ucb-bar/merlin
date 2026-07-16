"""WS-E: the "target becomes its own repo" publishing bridge.

Merlin tracks *every* codegen experiment internally under
``out/artifacts/targets/<target>/<package_id>/``. This module exports the certified **champion**
package for a target as a standalone, cloneable, buildable git repository — the LAYERED publish
model the team settled on:

  * the buildable out-of-tree (OOT) tree **is** the repo content (``git clone`` + ``cmake`` build),
  * the package manifest rides along as a release/provenance record under ``.merlin/``.

History on the target repo is the provenance trail (one commit per promotion, its message embeds
the champion package_id + internal run + Merlin git sha + certification summary); ``HEAD`` is the
current champion; a ``v<version>-<package_id>`` tag names each release. Idempotency is a
``sha256(package_id + merlin_sha + cert_run_id)`` fingerprint carried in the commit trailer + tag.

Canonical repo layout (same skeleton for every target; ``family`` decides which dirs are populated)::

    <target>-mlir/
      README.md                 # generated: what / how-to-build / provenance summary
      CMakeLists.txt            # buildable OOT tree at ROOT
      include/<Dialect>/        #   (rvv: placeholder + .gitkeep; gemmini: hoisted from mlir_oot/)
      lib/
      tools/<target>-opt/
      test/
      manifest.yaml             # rewritten contract manifest (repo root == {package})
      payload/                  # family assets:
                                #   rvv     -> schedule.mlir + knobs.yaml + baseline_runs/
                                #   gemmini -> dialect.py, lowering.yaml, contracts/, inputs/
      .merlin/                  # committed METADATA layer:
        manifest.yaml           #   identical to the root manifest (provenance copy)
        provenance.yaml         #   lineage / source package / run refs / merlin sha
        certification.yaml      #   recorded certification (oot_runner.certify / rvv spike gate)
        CHAMPION                #   one line: <package_id> <sha7> <cert_run_id>

The bridge NEVER hardcodes a remote (see :func:`resolve_remote`) and — by design of the WS-E task —
is verified only against a LOCAL bare remote (``file://…``); it is never pushed to GitHub by the
harness. It reuses, and does not re-implement, the shared machinery:
:mod:`merlin.common.artifacts` (``utc_stamp`` / ``git_sha7`` / ``new_product``),
:mod:`merlin.targetgen.oot_runner` (``load_package`` / ``build_package`` — the fresh-clone build
verify), :mod:`merlin.rvvgen.registry` (``load_rvv_package`` — the rvv payload),
:mod:`merlin.common.paths`, and ``targetgen.contract.schemas.validate_manifest``.

CLI (``merlin-target-publish``)::

    merlin-target-publish publish  --target rvv [--dry-run] [--execute] [--remote URL] [--no-gate] [--champion ID]
    merlin-target-publish promote  --target rvv --champion <package_id> [--no-gate]
    merlin-target-publish inspect  --target rvv [--champion <package_id>]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import paths
from ..common.artifacts import git_sha7, new_product, utc_stamp
from ..common.yaml import dump_yaml, load_yaml, write_yaml
from .contract import schemas

LAYOUT_VERSION = "1.0"

# family -> layout kind. RVV is a transform SCHEDULE (no dialect); everything resident-accelerator
# shaped (tensor_resident / mlir_oot_target_backend) uses the hoisted OOT-tree layout.
_VECTOR_SCHEDULE_FAMILIES = frozenset({"vector_schedule"})

# status ranking for automatic champion selection (lower is better).
_STATUS_RANK = {"rtl_certified": 0, "spike_verified": 1}
_DEFAULT_STATUS_RANK = 2


class PublishError(RuntimeError):
    """A publish/promote step could not proceed (selection, gate, config, or git)."""


# ---------------------------------------------------------------------------- selection


@dataclass
class ChampionSelection:
    """The chosen champion package for a target + the facts the publish flow needs about it."""

    target: str
    package_id: str
    package_dir: Path
    manifest: dict[str, Any]
    family: str
    layout_kind: str            # "vector_schedule" | "mlir_oot"
    status: str
    cert_status: str | None     # publication.certification (e.g. "pass") if recorded
    cert_run: str | None        # publication.certified_by_run if recorded
    oracle_cycles: int | None
    version: int
    lineage_depth: int
    timestamp: str

    @property
    def dialect_name(self) -> str:
        return "".join(part.capitalize() for part in self.target.replace("-", "_").split("_"))

    @property
    def tool_name(self) -> str:
        return f"{self.target}-opt"


def _layout_kind(family: str) -> str:
    return "vector_schedule" if family in _VECTOR_SCHEDULE_FAMILIES else "mlir_oot"


def _oracle_cycles(manifest: dict[str, Any]) -> int | None:
    """Best-effort recorded oracle cycle count (lower == faster == better champion)."""
    for holder in (manifest.get("certification"), manifest.get("oracle"), manifest.get("publication")):
        if isinstance(holder, dict) and isinstance(holder.get("cycles"), int):
            return holder["cycles"]
    if isinstance(manifest.get("cycles"), int):
        return manifest["cycles"]
    return None


def _build_selection(target: str, pkg_dir: Path, manifest: dict[str, Any]) -> ChampionSelection:
    family = str(manifest.get("family", ""))
    pub = manifest.get("publication") if isinstance(manifest.get("publication"), dict) else {}
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    return ChampionSelection(
        target=target,
        package_id=str(manifest.get("package_id", pkg_dir.name)),
        package_dir=pkg_dir,
        manifest=manifest,
        family=family,
        layout_kind=_layout_kind(family),
        status=str(manifest.get("status", "")),
        cert_status=pub.get("certification"),
        cert_run=pub.get("certified_by_run"),
        oracle_cycles=_oracle_cycles(manifest),
        version=int(manifest.get("version", 0) or 0),
        lineage_depth=int(lineage.get("depth", 0) or 0),
        timestamp=str(manifest.get("timestamp", "")),
    )


def _rank_key(pkg_dir: Path, manifest: dict[str, Any]) -> tuple:
    """Deterministic ranking key (best sorts first): status, oracle cycles, version/depth, ts."""
    status = str(manifest.get("status", ""))
    cycles = _oracle_cycles(manifest)
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    return (
        _STATUS_RANK.get(status, _DEFAULT_STATUS_RANK),
        cycles if cycles is not None else float("inf"),
        -int(manifest.get("version", 0) or 0),
        -int(lineage.get("depth", 0) or 0),
        # newer timestamp preferred -> invert lexical order via a descending sort on the raw string
        _InvStr(str(manifest.get("timestamp", ""))),
        pkg_dir.name,  # final stable tie-break
    )


@dataclass(frozen=True)
class _InvStr:
    """Wrap a string so that larger (later) strings sort FIRST — for 'newest timestamp wins'."""

    value: str

    def __lt__(self, other: "_InvStr") -> bool:
        return self.value > other.value


def _targets_root(artifacts_root: str | Path | None) -> Path:
    base = Path(artifacts_root) if artifacts_root else paths.artifacts_dir()
    return base / "targets"


def select_champion(target: str, *, artifacts_root: str | Path | None = None,
                    package_id: str | None = None) -> ChampionSelection:
    """Pick the champion package for ``target`` under ``out/artifacts/targets/<target>/``.

    If ``package_id`` is given, that package is selected. Otherwise, if exactly one package is
    flagged ``publication.champion: true`` it wins; failing that, packages are ranked
    deterministically: ``rtl_certified`` > ``spike_verified`` > other, then fewer oracle cycles,
    then higher lineage version/depth, then newer timestamp, then package_id.
    """
    tdir = _targets_root(artifacts_root) / target
    if not tdir.is_dir():
        raise PublishError(f"no target dir for {target!r}: {tdir}")

    packages: list[tuple[Path, dict[str, Any]]] = []
    for man_path in sorted(tdir.glob("*/manifest.yaml")):
        man = load_yaml(man_path)
        if isinstance(man, dict):
            packages.append((man_path.parent, man))
    if not packages:
        raise PublishError(f"no packages with a manifest.yaml under {tdir}")

    if package_id is not None:
        for pkg_dir, man in packages:
            if str(man.get("package_id", pkg_dir.name)) == package_id or pkg_dir.name == package_id:
                return _build_selection(target, pkg_dir, man)
        raise PublishError(f"package_id {package_id!r} not found under {tdir}")

    champs = [(d, m) for d, m in packages
              if isinstance(m.get("publication"), dict) and m["publication"].get("champion") is True]
    if len(champs) == 1:
        return _build_selection(target, champs[0][0], champs[0][1])

    ranked = sorted(packages, key=lambda dm: _rank_key(dm[0], dm[1]))
    return _build_selection(target, ranked[0][0], ranked[0][1])


# ---------------------------------------------------------------------------- remote resolution


def resolve_remote(target: str, *, config: str | Path | None = None,
                   override: str | None = None) -> str:
    """Resolve the git remote for ``target``. Precedence: ``override`` (``--remote``) >
    env ``MERLIN_PUBLISH_REMOTE_<TARGET>`` (via :func:`merlin.common.paths.env`, honoring ``.env``)
    > ``merlin/targets/publish.yaml``. Never hardcoded."""
    if override:
        return override
    env_val = paths.env(f"MERLIN_PUBLISH_REMOTE_{target.upper()}")
    if env_val:
        return env_val
    cfg_path = Path(config) if config else paths.targets_dir() / "publish.yaml"
    if not cfg_path.is_file():
        raise PublishError(f"publish config not found: {cfg_path}")
    data = load_yaml(cfg_path) or {}
    remote = (data.get("targets") or {}).get(target)
    if not remote:
        raise PublishError(f"no remote configured for {target!r} in {cfg_path} "
                           f"(and no --remote / MERLIN_PUBLISH_REMOTE_{target.upper()})")
    return str(remote)


# ---------------------------------------------------------------------------- tree assembly


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _gitkeep(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitkeep").write_text("", encoding="utf-8")


_SKELETON_IGNORE = {"manifest.yaml", ".merlin", "build", "__pycache__", ".git", "mlir_oot"}


def _cmakelists(sel: ChampionSelection) -> str:
    return (
        "cmake_minimum_required(VERSION 3.13)\n"
        f"project({sel.target}_mlir CXX)\n\n"
        "# Buildable OOT tree emitted by merlin-target-publish (WS-E). The thin driver below makes\n"
        "# the repo contract-shaped so oot_runner can build build/bin/{tool} on a fresh clone; a\n"
        "# real champion with a full mlir_oot/ tree overwrites this file with its own CMakeLists.\n"
        "set(CMAKE_CXX_STANDARD 17)\n"
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)\n"
        f"add_executable({sel.tool_name} tools/{sel.tool_name}/main.cpp)\n"
    )


def _driver_cpp(sel: ChampionSelection) -> str:
    t = sel.tool_name
    return (
        f"// Thin {t} driver emitted by merlin-target-publish (WS-E).\n"
        "//\n"
        "// The buildable OOT tree at the repo root IS the published target repo; the real codegen\n"
        "// payload rides under payload/. This driver accepts the experiment-ABI entrypoint flags so\n"
        "// the repo is contract-shaped and merlin.targetgen.oot_runner can build it on a fresh clone.\n"
        "#include <cstdio>\n"
        "#include <cstring>\n\n"
        "int main(int argc, char** argv) {\n"
        "  for (int i = 1; i < argc; ++i) {\n"
        "    if (std::strcmp(argv[i], \"--version\") == 0) {\n"
        f"      std::printf(\"{t} (merlin publish skeleton)\\n\");\n"
        "      return 0;\n"
        "    }\n"
        "  }\n"
        f"  std::fprintf(stderr, \"{t}: thin publish-skeleton driver; \"\n"
        "               \"see payload/ for the codegen artifact.\\n\");\n"
        "  return 0;\n"
        "}\n"
    )


def _readme(sel: ChampionSelection, manifest: dict[str, Any]) -> str:
    merlin_sha = git_sha7()
    pub = manifest.get("publication") or {}
    lines = [
        f"# {sel.target}-mlir",
        "",
        f"Standalone, buildable out-of-tree Merlin codegen backend for **{sel.target}** "
        f"(family `{sel.family or 'unknown'}`).",
        "",
        "This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the "
        "certified champion codegen package for the target, exported as its own repo. The buildable "
        "tree at the repo root *is* the content; the package manifest + provenance ride along under "
        "`.merlin/`.",
        "",
        "## What",
        "",
        f"- Champion package: `{sel.package_id}`",
        f"- Family: `{sel.family or 'unknown'}`",
        f"- Recorded status: `{sel.status or 'unknown'}`",
        f"- Merlin git sha (this export): `{merlin_sha}`",
        "",
        "## How to build",
        "",
        "```sh",
        f"git clone <this-repo> {sel.target}-mlir",
        f"cd {sel.target}-mlir",
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
        "cmake --build build",
        f"./build/bin/{sel.tool_name} --version",
        "```",
        "",
        "The codegen payload (schedule/knobs for rvv; dialect/lowering/contracts for gemmini) lives "
        "under `payload/`.",
        "",
        "## Provenance",
        "",
        f"- Certification: `{pub.get('certification', sel.cert_status or 'recorded:' + (sel.status or 'unknown'))}`",
        f"- Certified by run: `{pub.get('certified_by_run', sel.cert_run or 'n/a')}`",
        f"- Fingerprint: `{pub.get('fingerprint', 'n/a')}`",
        "",
        "See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each "
        "commit on this repo is one promotion; the history is the provenance trail.",
        "",
    ]
    return "\n".join(lines)


def _rewrite_build_paths(build: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a hoisted-gemmini build block so repo-root == {package}: ``{package}/mlir_oot`` ->
    ``{package}`` and ``mlir_oot/build`` -> ``build`` (structured string replacement, no regex)."""
    out: dict[str, Any] = {}
    for key, val in build.items():
        if key in ("configure", "command") and isinstance(val, list):
            out[key] = [str(tok).replace("{package}/mlir_oot", "{package}").replace("mlir_oot/build", "build")
                        for tok in val]
        elif key == "tool_output" and isinstance(val, str):
            out[key] = val.replace("mlir_oot/build", "build").replace("{package}/mlir_oot", "{package}")
        else:
            out[key] = val
    return out


def _default_commands(sel: ChampionSelection) -> dict[str, Any]:
    t = sel.target
    return {
        "parse": {"argv": ["{tool}", "--verify-diagnostics", "{input_mlir}"]},
        "lower_interface_to_target": {"argv": ["{tool}", f"--convert-iface-to-{t}", "{input_mlir}"]},
        "emit_command_buffer": {"argv": ["{tool}", f"--convert-iface-to-{t}",
                                         "--emit-command-buffer={output_json}", "{input_mlir}"]},
        "lower_target_to_llvm": {"argv": ["{tool}", f"--convert-{t}-to-llvm", "{input_mlir}"]},
    }


def _rewrite_manifest(sel: ChampionSelection, *, layout_version: str,
                      hoisted_tree: bool) -> dict[str, Any]:
    """Build the contract manifest for the exported repo (repo root == {package}).

    If the source manifest is already contract-shaped, it is reused and its build paths rewritten;
    otherwise a contract manifest is synthesized around the generated buildable skeleton. Either way
    the result validates against the contract manifest schema and drives ``oot_runner.build_package``
    verbatim on a fresh clone.
    """
    src = sel.manifest
    tool_out = f"build/bin/{sel.tool_name}"
    contract_shaped = isinstance(src.get("entrypoints"), dict) and isinstance(src.get("commands"), dict)

    if contract_shaped:
        man = copy.deepcopy(src)
        if hoisted_tree and isinstance(man.get("build"), dict):
            man["build"] = _rewrite_build_paths(man["build"])
        else:
            man["build"] = {
                "configure": ["cmake", "-S", "{package}", "-B", "{package}/build",
                              "-DCMAKE_BUILD_TYPE=Release"],
                "command": ["cmake", "--build", "{package}/build"],
                "tool_output": tool_out,
            }
        man.setdefault("entrypoints", {})["tool"] = man["build"]["tool_output"]
    else:
        man = {
            "artifact_type": "mlir_oot_target_backend",
            "target": sel.target,
            "language": "cpp",
            "authoring": src.get("authoring")
            if isinstance(src.get("authoring"), dict)
            else {"mode": "deterministic_generated_from_spec",
                  "author": "merlin-target-publish", "generated_by_agent": False},
            "integrity_exempt": bool(src.get("integrity_exempt", False)),
            "build": {
                "configure": ["cmake", "-S", "{package}", "-B", "{package}/build",
                              "-DCMAKE_BUILD_TYPE=Release"],
                "command": ["cmake", "--build", "{package}/build"],
                "tool_output": tool_out,
            },
            "entrypoints": {"tool": tool_out},
            "commands": _default_commands(sel),
        }

    man["artifact_type"] = "mlir_oot_target_backend"
    man["target"] = sel.target
    man["package_id"] = sel.package_id
    man["family"] = sel.family
    man["layout_version"] = layout_version
    if isinstance(src.get("publication"), dict):
        man["publication"] = copy.deepcopy(src["publication"])

    schemas.validate_manifest(man)
    return man


def assemble_repo_tree(sel: ChampionSelection, dest: str | Path, *, layout_version: str) -> dict[str, Any]:
    """Build the canonical target-repo skeleton for ``sel`` in the staging dir ``dest``.

    Returns the rewritten contract manifest (also written to ``dest/manifest.yaml``). The buildable
    OOT tree lives at the repo root; the family payload under ``payload/``; the metadata layer is
    written separately by :func:`embed_provenance`.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # canonical buildable skeleton (same for every target)
    _write(dest / "CMakeLists.txt", _cmakelists(sel))
    _write(dest / "tools" / sel.tool_name / "main.cpp", _driver_cpp(sel))
    _gitkeep(dest / "include" / sel.dialect_name)
    _gitkeep(dest / "lib")
    _gitkeep(dest / "test")

    payload = dest / "payload"
    payload.mkdir(parents=True, exist_ok=True)

    hoisted_tree = False
    if sel.layout_kind == "vector_schedule":
        _populate_vector_schedule_payload(sel, payload)
    else:
        hoisted_tree = _populate_mlir_oot(sel, dest, payload)

    manifest = _rewrite_manifest(sel, layout_version=layout_version, hoisted_tree=hoisted_tree)
    _write(dest / "manifest.yaml", dump_yaml(manifest))
    _write(dest / "README.md", _readme(sel, manifest))
    return manifest


def _populate_vector_schedule_payload(sel: ChampionSelection, payload: Path) -> None:
    """rvv family: payload/schedule.mlir + knobs.yaml + baseline_runs/ (via load_rvv_package)."""
    from ..rvvgen.registry import load_rvv_package

    pkg = load_rvv_package(sel.package_dir)
    _write(payload / "schedule.mlir", pkg.schedule_text)
    shutil.copyfile(sel.package_dir / "knobs.yaml", payload / "knobs.yaml")
    src_runs = sel.package_dir / "baseline_runs"
    if src_runs.is_dir():
        shutil.copytree(src_runs, payload / "baseline_runs")


def _populate_mlir_oot(sel: ChampionSelection, dest: Path, payload: Path) -> bool:
    """gemmini family: move dialect.py/lowering.yaml/contracts/inputs into payload/; hoist any
    mlir_oot/ tree to the repo root. Returns True iff an mlir_oot/ tree was hoisted."""
    src = sel.package_dir
    for name in ("dialect.py", "lowering.yaml"):
        p = src / name
        if p.is_file():
            shutil.copyfile(p, payload / name)
    for name in ("contracts", "inputs"):
        d = src / name
        if d.is_dir():
            shutil.copytree(d, payload / name)

    hoisted = False
    mlir_oot = src / "mlir_oot"
    if mlir_oot.is_dir():
        hoisted = True
        for entry in sorted(mlir_oot.iterdir()):
            if entry.name in ("build", "__pycache__", ".git"):
                continue
            target = dest / entry.name
            if entry.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(entry, target)
            else:
                shutil.copyfile(entry, target)
    return hoisted


# ---------------------------------------------------------------------------- provenance layer


def _cert_run_id(sel: ChampionSelection) -> str:
    return sel.cert_run or f"recorded-{sel.status or 'unknown'}"


def _fingerprint(package_id: str, merlin_sha: str, cert_run_id: str) -> str:
    payload = f"{package_id}\n{merlin_sha}\n{cert_run_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha_full(root: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root or paths.repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def embed_provenance(dest: str | Path, sel: ChampionSelection) -> None:
    """Write the committed metadata layer under ``dest/.merlin/``: a copy of the manifest, the
    provenance lineage, the recorded certification, and the one-line CHAMPION marker."""
    dest = Path(dest)
    meta = dest / ".merlin"
    meta.mkdir(parents=True, exist_ok=True)

    manifest = load_yaml(dest / "manifest.yaml")
    merlin_sha = _git_sha_full()
    sha7 = git_sha7()
    cert_run = _cert_run_id(sel)
    fp = _fingerprint(sel.package_id, merlin_sha, cert_run)

    # 1) manifest copy (provenance record; identical to the root build manifest)
    _write(meta / "manifest.yaml", dump_yaml(manifest))

    # 2) provenance lineage
    try:
        src_rel = str(sel.package_dir.relative_to(paths.repo_root()))
    except ValueError:
        src_rel = str(sel.package_dir)
    provenance = {
        "layout_version": LAYOUT_VERSION,
        "generator": "merlin-target-publish",
        "generated_at": utc_stamp(),
        "target": sel.target,
        "package_id": sel.package_id,
        "family": sel.family,
        "source_package": src_rel,
        "merlin_git_sha": merlin_sha,
        "merlin_git_sha7": sha7,
        "source_status": sel.status,
        "version": sel.version,
        "lineage_depth": sel.lineage_depth,
        "run_refs": [cert_run],
        "fingerprint": fp,
    }
    _write(meta / "provenance.yaml", dump_yaml(provenance))

    # 3) recorded certification (oot_runner.certify for gemmini / rvv spike gate)
    gate_ok, gate_detail = _check_gate(sel)
    certification = {
        "target": sel.target,
        "package_id": sel.package_id,
        "layout_kind": sel.layout_kind,
        "gate": "oot_runner.certify" if sel.layout_kind == "mlir_oot" else "rvv_spike_verified",
        "status": "pass" if gate_ok else "unverified",
        "recorded_status": sel.status,
        "certification": sel.cert_status,
        "certified_by_run": cert_run,
        "oracle_cycles": sel.oracle_cycles,
        "detail": gate_detail,
    }
    _write(meta / "certification.yaml", dump_yaml(certification))

    # 4) CHAMPION marker: one line
    _write(meta / "CHAMPION", f"{sel.package_id} {sha7} {cert_run}\n")


# ---------------------------------------------------------------------------- gate


def _check_gate(sel: ChampionSelection) -> tuple[bool, str]:
    """Certification gate. rvv: the recorded spike_verified (or better) gate. gemmini/mlir_oot:
    an ``oot_runner.certify`` pass, surfaced as ``rtl_certified`` status or a recorded
    ``publication.certification == 'pass'``."""
    if sel.layout_kind == "vector_schedule":
        ok = sel.status in ("spike_verified", "rtl_certified")
        return ok, (f"rvv gate: status={sel.status!r} "
                    f"(need spike_verified or rtl_certified)")
    ok = sel.status == "rtl_certified" or sel.cert_status == "pass"
    return ok, (f"mlir_oot gate: status={sel.status!r} certification={sel.cert_status!r} "
                f"(need rtl_certified or oot_runner.certify pass)")


# ---------------------------------------------------------------------------- promotion


def promote(target: str, package_id: str, *, gate: bool = True,
            artifacts_root: str | Path | None = None) -> None:
    """Promote ``package_id`` to be the single champion for ``target`` (manifest-only edit in out/).

    Verifies the certification gate (unless ``gate=False``), clears any prior champion, and sets
    ``publication.champion: true`` + promotion metadata + fingerprint on the chosen package. The
    single-champion invariant is enforced: exactly one package per target ends up flagged."""
    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    ok, detail = _check_gate(sel)
    if gate and not ok:
        raise PublishError(f"promote gate refused for {target}/{package_id}: {detail}")
    if not ok:
        sys.stderr.write(f"WARNING: --no-gate promoting UNCERTIFIED {target}/{package_id}: {detail}\n")

    tdir = _targets_root(artifacts_root) / target
    # clear prior champions (single-champion invariant)
    for man_path in sorted(tdir.glob("*/manifest.yaml")):
        if man_path.parent == sel.package_dir:
            continue
        man = load_yaml(man_path)
        pub = man.get("publication")
        if isinstance(pub, dict) and pub.get("champion"):
            pub["champion"] = False
            man["publication"] = pub
            write_yaml(man_path, man)

    merlin_sha = _git_sha_full()
    cert_run = _cert_run_id(sel)
    man = load_yaml(sel.package_dir / "manifest.yaml")
    pub = man.get("publication") if isinstance(man.get("publication"), dict) else {}
    pub.update({
        "champion": True,
        "certification": "pass" if ok else "unverified",
        "certified_by_run": cert_run,
        "promoted_at": utc_stamp(),
        "promoted_by": "merlin-target-publish",
        "fingerprint": _fingerprint(sel.package_id, merlin_sha, cert_run),
    })
    man["publication"] = pub
    write_yaml(sel.package_dir / "manifest.yaml", man)


# ---------------------------------------------------------------------------- git mechanics


def _git(args: list[str], cwd: Path | None = None, *, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=300)
    if check and proc.returncode != 0:
        raise PublishError(f"git {' '.join(args)} failed (rc={proc.returncode}):\n{proc.stderr}")
    return proc


def _head_fingerprint(clone_dir: Path) -> str | None:
    """Parse the ``Merlin-Publish-Fingerprint:`` trailer from the clone HEAD commit, if any."""
    proc = _git(["-C", str(clone_dir), "log", "-1", "--format=%B"], check=False)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        key, sep, val = line.partition(":")
        if sep and key.strip() == "Merlin-Publish-Fingerprint":
            return val.strip()
    return None


def _sync_tree(clone_dir: Path, repo_dir: Path) -> None:
    """Replace the clone working tree (except .git) with the assembled repo tree."""
    for entry in clone_dir.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for entry in repo_dir.iterdir():
        target = clone_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copyfile(entry, target)


def _git_publish(remote: str, repo_dir: Path, sel: ChampionSelection, manifest: dict[str, Any],
                 fingerprint: str, cert_run: str, stage_root: Path) -> tuple[str, str, bool]:
    """Clone the remote, replace the tree with the assembled repo, commit + tag, push. Idempotent:
    a matching HEAD fingerprint or existing tag is a no-op. Returns (commit_sha, tag, noop)."""
    clone_dir = stage_root / "clone"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    _git(["clone", remote, str(clone_dir)])

    version = int(manifest.get("version", sel.version) or 0)
    tag = f"v{version}-{sel.package_id}"

    existing_tags = set(_git(["-C", str(clone_dir), "tag", "--list"]).stdout.split())
    if tag in existing_tags or _head_fingerprint(clone_dir) == fingerprint:
        head = _git(["-C", str(clone_dir), "rev-parse", "HEAD"], check=False)
        return (head.stdout.strip() or "unknown", tag, True)

    _sync_tree(clone_dir, repo_dir)
    _git(["-C", str(clone_dir), "add", "--", "."])

    merlin_sha = _git_sha_full()
    subject = f"publish({sel.target}): champion {sel.package_id}"
    body = (
        f"Champion: {sel.package_id}\n"
        f"Target: {sel.target}\n"
        f"Family: {sel.family}\n"
        f"Merlin-Sha: {merlin_sha}\n"
        f"Internal-Run: {cert_run}\n"
        f"Cert: {manifest.get('publication', {}).get('certification', sel.cert_status or 'recorded:' + sel.status)}\n"
        f"Layout-Version: {manifest.get('layout_version', LAYOUT_VERSION)}\n"
        "\n"
        f"Merlin-Publish-Fingerprint: {fingerprint}\n"
    )
    _git(["-C", str(clone_dir),
          "-c", "user.name=merlin-target-publish", "-c", "user.email=publish@merlin.local",
          "commit", "-m", subject, "-m", body])
    # annotated tag (carries the fingerprint; also robust to git configs that force annotation)
    _git(["-C", str(clone_dir),
          "-c", "user.name=merlin-target-publish", "-c", "user.email=publish@merlin.local",
          "tag", "-a", tag, "-m",
          f"{sel.target} champion {sel.package_id}\nMerlin-Publish-Fingerprint: {fingerprint}\n"])
    commit_sha = _git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()

    _git(["-C", str(clone_dir), "push", "origin", "HEAD"])
    _git(["-C", str(clone_dir), "push", "origin", "--tags"])
    # NOTE: `gh release` is intentionally skipped for file:// bare remotes (the verification path).
    return commit_sha, tag, False


# ---------------------------------------------------------------------------- publish


@dataclass
class PublishResult:
    """Outcome of a :func:`publish` invocation (dry-run or real)."""

    target: str
    package_id: str
    remote: str
    dry_run: bool
    gate_ok: bool
    gate_detail: str
    fingerprint: str
    tag: str
    repo_dir: Path
    committed: bool = False
    noop: bool = False
    commit_sha: str | None = None
    product_dir: Path | None = None
    actions: list[str] = field(default_factory=list)


def publish(target: str, *, dry_run: bool = True, remote: str | None = None, gate: bool = True,
            verify_build: bool = True, package_id: str | None = None,
            artifacts_root: str | Path | None = None, config: str | Path | None = None) -> PublishResult:
    """Publish the champion of ``target`` as its own repo. Dry-run by default (no git/network).

    The gate refuses an uncertified champion unless ``gate=False`` (a loud warning is emitted).
    A real publish clones the resolved remote, replaces its tree with the assembled repo, commits
    (message = provenance), tags ``v<version>-<package_id>``, and pushes — idempotently. Each real
    publish event is recorded via :func:`merlin.common.artifacts.new_product`."""
    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    resolved_remote = resolve_remote(target, config=config, override=remote)
    gate_ok, gate_detail = _check_gate(sel)
    if gate and not gate_ok:
        raise PublishError(f"publish gate refused for {target}/{sel.package_id}: {gate_detail}")
    if not gate and not gate_ok:
        sys.stderr.write(f"WARNING: --no-gate publishing UNCERTIFIED {target}/{sel.package_id}: "
                         f"{gate_detail}\n")

    ts = utc_stamp()
    stage_root = paths.build_dir() / "publish" / target / ts
    repo_dir = stage_root / "repo"
    manifest = assemble_repo_tree(sel, repo_dir, layout_version=LAYOUT_VERSION)
    embed_provenance(repo_dir, sel)

    merlin_sha = _git_sha_full()
    cert_run = _cert_run_id(sel)
    fingerprint = _fingerprint(sel.package_id, merlin_sha, cert_run)
    version = int(manifest.get("version", sel.version) or 0)
    tag = f"v{version}-{sel.package_id}"

    result = PublishResult(
        target=target, package_id=sel.package_id, remote=resolved_remote, dry_run=dry_run,
        gate_ok=gate_ok, gate_detail=gate_detail, fingerprint=fingerprint, tag=tag, repo_dir=repo_dir,
    )
    result.actions = [
        f"select champion {sel.package_id} (family={sel.family}, status={sel.status})",
        f"assemble {sel.layout_kind} repo tree at {repo_dir}",
        f"gate: {'OK' if gate_ok else 'FAILED'} ({gate_detail})",
        f"remote: {resolved_remote}",
        f"tag: {tag}",
        f"fingerprint: {fingerprint}",
    ]

    if dry_run:
        result.actions.insert(0, "DRY-RUN (no clone/commit/push)")
        return result

    commit_sha, published_tag, noop = _git_publish(
        resolved_remote, repo_dir, sel, manifest, fingerprint, cert_run, stage_root)
    result.committed = not noop
    result.noop = noop
    result.commit_sha = commit_sha
    result.tag = published_tag
    result.actions.append("no-op (fingerprint/tag already published)" if noop
                          else f"committed {commit_sha} + tag {published_tag}, pushed to remote")

    # record the publish event as a versioned product
    prod = new_product("publish", version=1, target=target,
                       notes=f"publish {sel.package_id} -> {resolved_remote} ({'noop' if noop else 'committed'})")
    event = {
        "target": target, "package_id": sel.package_id, "remote": resolved_remote,
        "commit_sha": commit_sha, "tag": published_tag, "noop": noop,
        "fingerprint": fingerprint, "merlin_git_sha": merlin_sha, "cert_run": cert_run,
        "gate_ok": gate_ok, "gate_detail": gate_detail, "actions": result.actions,
    }
    out = prod.add_artifact("publish_event.yaml")
    out.write_text(dump_yaml(event), encoding="utf-8")
    prod.write_manifest()
    result.product_dir = prod.path
    return result


# ---------------------------------------------------------------------------- CLI


def _print_result(res: PublishResult) -> None:
    print(f"target={res.target} package={res.package_id}")
    print(f"remote={res.remote}")
    print(f"dry_run={res.dry_run} committed={res.committed} noop={res.noop}")
    if res.commit_sha:
        print(f"commit={res.commit_sha} tag={res.tag}")
    print(f"fingerprint={res.fingerprint}")
    for a in res.actions:
        print(f"  - {a}")
    if res.product_dir:
        print(f"event recorded at {res.product_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-target-publish",
                                 description="Publish a target's certified champion as its own repo (WS-E).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pub = sub.add_parser("publish", help="publish the champion as a standalone repo")
    p_pub.add_argument("--target", required=True)
    p_pub.add_argument("--champion", help="explicit package_id (else the selection rules apply)")
    p_pub.add_argument("--remote", help="override the resolved remote")
    p_pub.add_argument("--config", help="override the publish.yaml config path")
    p_pub.add_argument("--artifacts-root", help="override the out/artifacts root (targets under it)")
    p_pub.add_argument("--dry-run", action="store_true", help="plan only (default)")
    p_pub.add_argument("--execute", action="store_true", help="actually clone/commit/push")
    p_pub.add_argument("--no-gate", action="store_true", help="publish even if uncertified (LOUD warning)")

    p_prom = sub.add_parser("promote", help="mark a package the single champion for a target")
    p_prom.add_argument("--target", required=True)
    p_prom.add_argument("--champion", "--package", dest="champion", required=True)
    p_prom.add_argument("--artifacts-root")
    p_prom.add_argument("--no-gate", action="store_true")

    p_ins = sub.add_parser("inspect", help="show the selected champion + plan (no git)")
    p_ins.add_argument("--target", required=True)
    p_ins.add_argument("--champion")
    p_ins.add_argument("--config")
    p_ins.add_argument("--artifacts-root")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "publish":
            res = publish(args.target, dry_run=not args.execute, remote=args.remote,
                          gate=not args.no_gate, package_id=args.champion,
                          artifacts_root=args.artifacts_root, config=args.config)
            _print_result(res)
            return 0
        if args.cmd == "promote":
            promote(args.target, args.champion, gate=not args.no_gate,
                    artifacts_root=args.artifacts_root)
            print(f"promoted {args.target}/{args.champion} to champion")
            return 0
        if args.cmd == "inspect":
            sel = select_champion(args.target, artifacts_root=args.artifacts_root,
                                  package_id=args.champion)
            remote = resolve_remote(args.target, config=args.config)
            ok, detail = _check_gate(sel)
            print(f"target={sel.target} champion={sel.package_id}")
            print(f"family={sel.family} layout={sel.layout_kind} status={sel.status}")
            print(f"remote={remote}")
            print(f"gate={'OK' if ok else 'FAILED'} ({detail})")
            print(f"cert_run={_cert_run_id(sel)}")
            return 0
    except PublishError as e:
        sys.stderr.write(f"merlin-target-publish: {e}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
