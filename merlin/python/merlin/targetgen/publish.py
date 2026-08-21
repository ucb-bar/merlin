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
#
# `k1_verified` must appear here. `_check_gate` already treats it as a certification at least as
# strong as spike for a physical RVV target ("measured correct AND faster on the real SpacemiT
# board"), but with no rank entry it fell to _DEFAULT_STATUS_RANK and sorted BELOW every
# spike_verified package — so a board-verified champion passed the gate and was then never
# selected, leaving the frozen hand baseline as the default. Ordering: RTL certification first
# (cycle-accurate on our own SoC), then real silicon, then the functional simulator.
_STATUS_RANK = {"rtl_certified": 0, "k1_verified": 1, "spike_verified": 2}
_DEFAULT_STATUS_RANK = 3

# The certified-status vocabulary is the SAME single source as the ranking above: every status that
# carries a rank is a recognized certification tier (RTL cycle-cert / real-silicon / functional sim),
# so the gate accepts exactly the ranked set. A new substrate becomes "certified" the moment it appends
# its status to `_STATUS_RANK` (with its tier) — there is no second literal list to keep in sync here.
# (This is still a fixed enum keyed on substrate-named statuses; the deeper fix would model an explicit
# substrate-agnostic certification TIER on the selection record and gate on that.)
CERTIFIED_STATUSES = frozenset(_STATUS_RANK)

# The FROZEN, hand-authored, UNoptimized controls (BB0 / C5). These publish to a single shared
# `baseline` branch so the before->after is externally visible; every certified champion publishes
# to its own `stable/<package_id>` branch. A package can also opt into the baseline branch via a
# manifest `publication.role: baseline` flag (so a renamed control still resolves correctly).
_BASELINE_PACKAGE_IDS = frozenset({"hand_v0", "hand_v0_int8"})
BASELINE_BRANCH = "baseline"


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


def package_dtype(pkg_dir: Path) -> str:
    """The package's ``dtype_strategy`` knob (``fp32`` when it declares none).

    Needed because a target's packages are NOT interchangeable across datatypes: an int8
    (W8A8) workload built with an fp32 package's schedule silently emits the wrong datapath.
    Read from knobs.yaml, which is where the strategy lives (the manifest does not carry it).
    """
    try:
        knobs = load_yaml(pkg_dir / "knobs.yaml")
    except Exception:  # noqa: BLE001
        return "fp32"
    if not isinstance(knobs, dict):
        return "fp32"
    return str(knobs.get("dtype_strategy", "fp32"))


def select_champion(target: str, *, artifacts_root: str | Path | None = None,
                    package_id: str | None = None,
                    dtype_strategy: str | None = None) -> ChampionSelection:
    """Pick the champion package for ``target`` under ``out/artifacts/targets/<target>/``.

    If ``package_id`` is given, that package is selected. Otherwise, if exactly one package is
    flagged ``publication.champion: true`` it wins; failing that, packages are ranked
    deterministically: ``rtl_certified`` > ``spike_verified`` > other, then fewer oracle cycles,
    then higher lineage version/depth, then newer timestamp, then package_id.

    ``dtype_strategy`` (e.g. ``"int8_w8a8"``) restricts the candidates to packages carrying that
    knob. Without it, an int8 caller can be handed the globally best package even when that
    package is fp32 — which builds a silently wrong datapath rather than failing.
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

    if dtype_strategy is not None:
        packages = [(d, m) for d, m in packages if package_dtype(d) == dtype_strategy]
        if not packages:
            raise PublishError(
                f"no {target!r} package with dtype_strategy={dtype_strategy!r} under {tdir}")

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


def _is_baseline(sel: "ChampionSelection") -> bool:
    """The frozen unoptimized control (by known package_id or a manifest opt-in flag)."""
    if sel.package_id in _BASELINE_PACKAGE_IDS:
        return True
    pub = sel.manifest.get("publication")
    return isinstance(pub, dict) and pub.get("role") == "baseline"


def resolve_branch(sel: "ChampionSelection", *, override: str | None = None,
                   config: str | Path | None = None) -> str:
    """Resolve the publish BRANCH for a package (BB0 branch-per-version). Precedence, highest first:
    ``override`` (``--branch``) > env ``MERLIN_PUBLISH_BRANCH_<TARGET>`` > ``publish.yaml``
    ``branches.<target>.<package_id>`` > the default policy.

    Default policy (C5): the FROZEN unoptimized baseline publishes to the shared ``baseline`` branch
    (one control, published FIRST so before->after is externally visible); every certified champion
    publishes to its own ``stable/<package_id>`` branch. This replaces the single-champion-to-HEAD
    model with one-champion-per-branch."""
    if override:
        return override
    env_val = paths.env(f"MERLIN_PUBLISH_BRANCH_{sel.target.upper()}")
    if env_val:
        return env_val
    cfg_path = Path(config) if config else paths.targets_dir() / "publish.yaml"
    if cfg_path.is_file():
        data = load_yaml(cfg_path) or {}
        per_target = (data.get("branches") or {}).get(sel.target)
        if isinstance(per_target, dict):
            mapped = per_target.get(sel.package_id)
            if mapped:
                return str(mapped)
    if not _is_baseline(sel):
        return f"stable/{sel.package_id}"
    # The baseline branch is per-DATATYPE. `_BASELINE_PACKAGE_IDS` holds one frozen control per
    # datatype (hand_v0 for fp32, hand_v0_int8 for int8_w8a8), and mapping them all to the single
    # `baseline` branch made them overwrite each other: whichever published last became "the"
    # control, so a speedup claimed against `baseline` could silently be measured against the
    # wrong datatype's schedule. fp32 keeps the historical bare name so already-published
    # branches and any external reference to them stay valid.
    dtype = package_dtype(sel.package_dir)
    return BASELINE_BRANCH if dtype in ("", "fp32") else f"{BASELINE_BRANCH}-{dtype}"


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


def _index_readme(target: str, entries: list[dict[str, Any]]) -> str:
    """README for the repo's DEFAULT branch — the landing page a fresh clone lands on.

    Every champion publishes to its own ``stable/<package_id>`` branch (and the frozen control
    to ``baseline``), which means the default branch is otherwise EMPTY: `git clone` gives a
    repo with nothing in it and no hint that the content is on other branches. That is the
    whole problem this fixes. The page is a directory, not a package: it names each published
    branch, what it is for, and how to consume it.
    """
    lines = [
        f"# {target}-mlir",
        "",
        f"Merlin's published codegen packages for the **{target}** target.",
        "",
        "This repository is **generated** by Merlin's `merlin-target-publish` bridge. It uses "
        "**branch-per-version** publishing, so *this* branch is only a directory — the packages "
        "themselves live on the branches below. Check one out to get a standalone, buildable "
        "out-of-tree tree plus its provenance under `.merlin/`.",
        "",
        "## Published packages",
        "",
        "| branch | package | dtype | status | what it is |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| `{e['branch']}` | `{e['package_id']}` | `{e['dtype']}` | `{e['status']}` | "
            f"{e['role']} |")
    lines += [
        "",
        "## Using a package",
        "",
        "```sh",
        f"git clone -b <branch> <this-repo> {target}-mlir",
        f"cd {target}-mlir",
        "```",
        "",
    ]
    if target == "rvv":
        lines += [
            "An `rvv` package is a **vector schedule**, not a dialect: the payload is a "
            "transform-dialect schedule plus the codegen knobs that go with it.",
            "",
            "- `payload/schedule.mlir` — the transform-dialect schedule (tiling + vectorization "
            "of the contractions)",
            "- `payload/knobs.yaml` — `cflags`, `dtype_strategy`, `op_match` tile/vector sizes, "
            "`lmul_policy`, and the `expected_instructions` the emitted code must contain",
            "- `payload/baseline_runs/` — the recorded reference runs",
            "",
            "Merlin consumes it through `merlin.rvvgen.registry.load_rvv_package(<dir>)` and "
            "applies it with `merlin.rvvgen.apply.apply_rvv_package(...)`; the schedule and "
            "cflags are the only things that change, so the rest of the pipeline is untouched.",
            "",
            "The `baseline` branch is the FROZEN, hand-authored, unoptimized control. It exists "
            "so a speedup claim can be reproduced against the same before/after this repo "
            "published, not against a moving target.",
            "",
        ]
    lines += [
        "## Provenance",
        "",
        "Each commit on a package branch is one promotion, and its message embeds the champion "
        "package id, the internal run id, the Merlin git sha and the certification summary. "
        "History is the provenance trail; the branch tip is the current champion.",
        "",
        f"Generated from Merlin `{git_sha7()}`.",
        "",
    ]
    return "\n".join(lines)


def index_entries(target: str, *, artifacts_root: str | Path | None = None
                  ) -> list[dict[str, Any]]:
    """Describe every package that WOULD be published for ``target``, for the landing page.

    Derived from the same selection/branch rules the publish path uses, so the index cannot
    drift from what is actually on the remote.
    """
    tdir = _targets_root(artifacts_root) / target
    out: list[dict[str, Any]] = []
    for man_path in sorted(tdir.glob("*/manifest.yaml")):
        man = load_yaml(man_path)
        if not isinstance(man, dict):
            continue
        sel = _build_selection(target, man_path.parent, man)
        gate_ok, _ = _check_gate(sel)
        if not gate_ok:
            continue          # only certified packages are published, so only they are listed
        is_base = _is_baseline(sel)
        out.append({
            "branch": resolve_branch(sel),
            "package_id": sel.package_id,
            "dtype": package_dtype(man_path.parent),
            "status": sel.status or "unknown",
            "role": ("frozen unoptimized control (the before/after reference)" if is_base
                     else "certified champion"),
        })
    return sorted(out, key=lambda e: (e["dtype"], e["branch"]))


def assemble_index_tree(target: str, dest: str | Path, *,
                        artifacts_root: str | Path | None = None,
                        only_branches: "set[str] | None" = None) -> dict[str, Any]:
    """Assemble the default-branch landing page (README + LICENSE) into ``dest``.

    Deliberately NOT a package tree: the default branch must not look like one champion, or a
    consumer would build the wrong thing. Reuses the repo's LICENSE so the published repo
    carries the same terms as Merlin.

    ``only_branches`` restricts the listing to branches that ACTUALLY exist on the remote (the
    publish path passes what the clone reports). Without it the page would advertise every
    locally-certified package, including ones never pushed — a fresh clone would then be sent
    to branches that do not exist, which is worse than the empty default branch this replaces.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    entries = index_entries(target, artifacts_root=artifacts_root)
    if only_branches is not None:
        entries = [e for e in entries if e["branch"] in only_branches]
    _write(dest / "README.md", _index_readme(target, entries))
    lic = paths.repo_root() / "LICENSE"
    if lic.is_file():
        _write(dest / "LICENSE", lic.read_text(encoding="utf-8"))
    return {"target": target, "entries": entries, "dest": str(dest)}


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
        # k1_verified = measured correct AND faster on the real SpacemiT K1 board (the beam's
        # promote_champion stamp). For a physical RVV target that is a STRONGER certification than
        # the spike simulator, not a weaker one — so accept it alongside spike/rtl. A beam champion
        # then publishes without --no-gate, because its certification IS real silicon.
        ok = sel.status in CERTIFIED_STATUSES
        accepted = sorted(CERTIFIED_STATUSES, key=lambda s: _STATUS_RANK[s])
        return ok, (f"rvv gate: status={sel.status!r} "
                    f"(need one of {accepted})")
    ok = sel.status == "rtl_certified" or sel.cert_status == "pass"
    return ok, (f"mlir_oot gate: status={sel.status!r} certification={sel.cert_status!r} "
                f"(need rtl_certified or oot_runner.certify pass)")


# ---------------------------------------------------------------------------- promotion


class MaterializeRefused(PublishError):
    """The submission or its score is not fit to install as a target's compiler."""


def _score_is_honest(score: dict) -> tuple[bool, str]:
    """Whether a capsule score may be used to justify installing a compiler.

    Mirrors the suite-level vacuous-pass guard: an empty or ungradeable run must never read as evidence.
    A row with no ``tiers`` is the specific shape that let four whole-model capsules report ``pass``
    without executing, so it is refused here too rather than trusted a second time.
    """
    if score.get("integrity_status") not in (None, "clean"):
        return False, f"integrity_status={score.get('integrity_status')!r} (want 'clean')"
    if score.get("gradeable") is False:
        return False, "the run reported gradeable=false"
    rows = score.get("per_capsule") or []
    if not rows:
        return False, "no per-capsule rows — nothing was graded"
    passed = [r for r in rows if r.get("status") == "pass"]
    if not passed:
        return False, "no capsule passed"
    hollow = [r.get("capsule") for r in passed if not (r.get("tiers") or {})]
    if hollow:
        return False, (f"{len(hollow)} capsule(s) report pass with no tier evidence "
                       f"({', '.join(str(h) for h in hollow[:4])}) — a pass with an empty tier map is "
                       f"not evidence a compiler ran")
    return True, f"{len(passed)}/{len(rows)} passed with tier evidence"


def materialize_package(target: str, source: str | Path, *, package_id: str = "agent_spec_v1_mlir_oot",
                        certified_by_run: str = "", score_path: str | Path | None = None,
                        artifacts_root: str | Path | None = None, force: bool = False) -> Path:
    """Install a run's submission as ``out/artifacts/targets/<target>/<package_id>/`` and return its path.

    Deliberately does NOT set ``status:``. ``_STATUS_RANK`` knows only rtl_certified / k1_verified /
    spike_verified; a functional-tier verdict is none of those, and claiming one would both overstate the
    evidence and silently win :func:`select_champion`. ``publication.champion`` is written ``false`` for
    the same reason -- ranking ties break on directory name, so a new package would otherwise displace an
    existing champion just by sorting earlier.
    """
    import shutil
    from ..common import provenance as PROV

    src_dir = Path(source)
    if not (src_dir / "manifest.yaml").is_file():
        raise MaterializeRefused(f"{src_dir} carries no manifest.yaml — not an OOT backend package")
    if score_path:
        import json as _json
        score = _json.loads(Path(score_path).read_text(encoding="utf-8"))
        ok, detail = _score_is_honest(score)
        if not ok:
            raise MaterializeRefused(f"refusing to install {target} compiler from {src_dir}: {detail}")
        evidence = {"n_passed": score.get("n_passed"), "n_capsules": score.get("n_capsules"),
                    "labels_graded": score.get("labels_graded"), "detail": detail}
    else:
        evidence = {"detail": "no score supplied — installed unverified"}

    dst = _targets_root(artifacts_root) / target / package_id
    if dst.exists():
        if not force:
            raise MaterializeRefused(f"{dst} already exists — pass force=True to replace it")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst, ignore=shutil.ignore_patterns("__pycache__", ".git", "build"))

    man = load_yaml(dst / "manifest.yaml")
    source_package_id = man.get("package_id")
    man["package_id"] = package_id
    pub = man.get("publication") if isinstance(man.get("publication"), dict) else {}
    pub.update(champion=False, certification="functional_tier", certified_by_run=certified_by_run)
    man["publication"] = pub
    man["provenance"] = PROV.record(sources=[str(src_dir)],
                                    extra={"installed_from_run": certified_by_run})
    man["promotion"] = {"from": str(src_dir), "source_package_id": source_package_id,
                        "promoted_by": "merlin-target-publish materialize",
                        "score": str(score_path) if score_path else None, "evidence": evidence}
    write_yaml(dst / "manifest.yaml", man)   # the module's own writer, not a raw dump
    return dst


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


def _checkout_branch(clone_dir: Path, branch: str) -> bool:
    """Check out ``branch`` in the clone. If it exists on the remote, track it (so its tip is the
    per-branch idempotency reference); otherwise start it as an orphan (a fresh, unrelated history —
    each published branch is a standalone repo view, not a commit off the default branch). Returns
    True iff the branch pre-existed on the remote."""
    remote_branches = _git(["-C", str(clone_dir), "branch", "-r"], check=False).stdout.split()
    if f"origin/{branch}" in remote_branches:
        _git(["-C", str(clone_dir), "checkout", "-B", branch, "--track", f"origin/{branch}"])
        return True
    _git(["-C", str(clone_dir), "checkout", "--orphan", branch])
    # an orphan checkout keeps the old tree staged; clear it so _sync_tree writes a clean state.
    _git(["-C", str(clone_dir), "rm", "-rf", "--quiet", "."], check=False)
    return False


def _git_publish(remote: str, repo_dir: Path, sel: ChampionSelection, manifest: dict[str, Any],
                 fingerprint: str, cert_run: str, stage_root: Path,
                 branch: str) -> tuple[str, str, bool]:
    """Clone the remote, check out ``branch``, replace the tree with the assembled repo, commit +
    tag, push to ``refs/heads/<branch>``. Idempotent PER BRANCH on the **fingerprint**: a matching
    branch-tip fingerprint is a no-op. Returns (commit_sha, tag, noop).

    An existing TAG is deliberately not a veto. The tag names a package *version*, but the
    fingerprint covers ``package_id + merlin_sha + cert_run_id`` — so re-certifying an unchanged
    payload (``spike_verified`` -> ``k1_verified`` after a board campaign) produces new provenance
    at the same version. Letting the tag veto that meant the published ``.merlin/certification.yaml``
    kept the WEAKER status forever, silently understating the certification to every consumer. The
    branch moves; the existing tag is left exactly where it is, because consumers may have pinned
    it and rewriting a published tag is worse than not adding one."""
    clone_dir = stage_root / "clone"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    _git(["clone", remote, str(clone_dir)])
    branch_existed = _checkout_branch(clone_dir, branch)

    version = int(manifest.get("version", sel.version) or 0)
    tag = f"v{version}-{sel.package_id}"

    existing_tags = set(_git(["-C", str(clone_dir), "tag", "--list"]).stdout.split())
    # per-branch idempotency: the fingerprint is read from THIS branch's tip (not the default HEAD),
    # so publishing the baseline branch never masks a stale champion-branch tip and vice-versa.
    if branch_existed and _head_fingerprint(clone_dir) == fingerprint:
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
    # annotated tag (carries the fingerprint; also robust to git configs that force annotation).
    # Never re-point an existing tag: a consumer may have pinned it, and this commit is a
    # re-certification of the SAME version, not a new release.
    if tag not in existing_tags:
        _git(["-C", str(clone_dir),
              "-c", "user.name=merlin-target-publish", "-c", "user.email=publish@merlin.local",
              "tag", "-a", tag, "-m",
              f"{sel.target} champion {sel.package_id}\nMerlin-Publish-Fingerprint: {fingerprint}\n"])
    commit_sha = _git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()

    _git(["-C", str(clone_dir), "push", "origin", f"HEAD:refs/heads/{branch}"])
    _git(["-C", str(clone_dir), "push", "origin", "--tags"])
    # NOTE: `gh release` is intentionally skipped for file:// bare remotes (the verification path).
    return commit_sha, tag, False


def publish_index(target: str, *, dry_run: bool = True, remote: str | None = None,
                  config: str | Path | None = None,
                  artifacts_root: str | Path | None = None,
                  branch: str = "main", confirm_push: str | None = None) -> dict[str, Any]:
    """Publish the landing page to the repo's DEFAULT branch.

    Branch-per-version publishing leaves that branch empty, so `git clone` with no `-b` gives a
    repo containing nothing — the state ucb-bar/rvv-mlir was actually in. This writes a
    directory page listing the branches that exist ON THE REMOTE (read from the clone, so the
    page cannot advertise a branch that is not there) and how to consume a package.

    Same safety posture as :func:`publish`: dry-run by default, and a non-local remote needs an
    explicit ``confirm_push`` fingerprint.
    """
    resolved = remote or resolve_remote(target, config=config)
    stage_root = paths.build_dir() / "publish" / target / f"{utc_stamp()}_index"
    repo_dir = stage_root / "repo"
    actions = [f"remote: {resolved}", f"branch: {branch}"]

    clone_dir = stage_root / "clone"
    remote_branches: set[str] | None = None
    if not dry_run:
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        _git(["clone", resolved, str(clone_dir)])
        remote_branches = {
            b.strip().removeprefix("origin/")
            for b in _git(["-C", str(clone_dir), "branch", "-r"], check=False).stdout.split()
            if b.strip().startswith("origin/") and "->" not in b
        }
        actions.append(f"remote branches: {sorted(remote_branches)}")

    info = assemble_index_tree(target, repo_dir, artifacts_root=artifacts_root,
                               only_branches=remote_branches)
    actions.append(f"assembled index listing {len(info['entries'])} package(s) at {repo_dir}")
    res: dict[str, Any] = {"target": target, "remote": resolved, "branch": branch,
                           "entries": info["entries"], "repo_dir": str(repo_dir),
                           "dry_run": dry_run, "actions": actions, "noop": False}
    if dry_run:
        actions.append("dry-run: nothing cloned, committed or pushed")
        return res

    fingerprint = _fingerprint(f"{target}-index", _git_sha_full(),
                               ",".join(e["branch"] for e in info["entries"]))
    res["fingerprint"] = fingerprint
    if _needs_push_confirmation(resolved) and confirm_push != fingerprint:
        actions.append(f"REFUSED push to non-local remote; re-run with --confirm-push {fingerprint}")
        res["noop"] = True
        return res

    _checkout_branch(clone_dir, branch)
    if _head_fingerprint(clone_dir) == fingerprint:
        actions.append("no-op: remote index already matches")
        res["noop"] = True
        return res
    _sync_tree(clone_dir, repo_dir)
    _git(["-C", str(clone_dir), "add", "--", "."])
    _git(["-C", str(clone_dir),
          "-c", "user.name=merlin-target-publish", "-c", "user.email=publish@merlin.local",
          "commit", "-m", f"docs({target}): landing page for the published package branches",
          "-m", (f"Lists the branches present on this remote and how to consume a package.\n"
                 f"Merlin-Sha: {_git_sha_full()}\n\n"
                 f"Merlin-Publish-Fingerprint: {fingerprint}\n")])
    _git(["-C", str(clone_dir), "push", "origin", f"HEAD:refs/heads/{branch}"])
    res["commit_sha"] = _git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()
    actions.append(f"pushed {res['commit_sha']} to {branch}")
    return res


# ---------------------------------------------------------------------------- publish


def _needs_push_confirmation(remote: str) -> bool:
    """True for a NON-LOCAL remote (a real network push — git@…, https://…, ssh://…). Local/file
    remotes (``file://…``, an absolute/relative path, an existing bare dir — the verification + test
    path) never need confirmation, so those flows are unchanged."""
    r = remote.strip()
    if r.startswith("file://") or r.startswith(("/", "./", "../", "~")):
        return False
    try:
        if Path(r).exists():
            return False
    except OSError:
        pass
    return r.startswith(("git@", "ssh://", "https://", "http://")) or ":" in r


def _require_push_confirmation(remote: str, repo_dir: Path, branch: str, fingerprint: str,
                               confirm_push: str | None) -> None:
    """Human gate before a real GitHub/network push: refuse unless ``confirm_push`` equals THIS publish's
    content fingerprint. Because the fingerprint is content-derived, a blind constant cannot pass — the
    operator must have seen the assembled artifact. On refusal, print the assembled repo tree (what would
    be pushed) so it can be inspected, then raise. Local/file remotes are exempt (see
    :func:`_needs_push_confirmation`)."""
    if not _needs_push_confirmation(remote) or confirm_push == fingerprint:
        return
    files = sorted(str(p.relative_to(repo_dir)) for p in repo_dir.rglob("*") if p.is_file())
    tree = "\n".join(f"    {f}" for f in files) or "    (empty)"
    raise PublishError(
        f"push to non-local remote {remote} (branch {branch}) REFUSED without confirmation.\n"
        f"  Assembled repo tree that WOULD be pushed (inspect at {repo_dir}):\n{tree}\n"
        f"  Re-run with --confirm-push {fingerprint} (CLI) / confirm_push={fingerprint!r} (API) to push.\n"
        f"  The token must equal this publish's content fingerprint, so it cannot be passed blindly.")


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
    branch: str = ""
    committed: bool = False
    noop: bool = False
    commit_sha: str | None = None
    product_dir: Path | None = None
    actions: list[str] = field(default_factory=list)


def publish(target: str, *, dry_run: bool = True, remote: str | None = None, gate: bool = True,
            verify_build: bool = True, package_id: str | None = None,
            artifacts_root: str | Path | None = None, config: str | Path | None = None,
            branch: str | None = None, confirm_push: str | None = None) -> PublishResult:
    """Publish the champion of ``target`` as its own repo. Dry-run by default (no git/network).

    The gate refuses an uncertified champion unless ``gate=False`` (a loud warning is emitted).
    A real publish clones the resolved remote, checks out the resolved BRANCH (the frozen baseline ->
    ``baseline``; a champion -> ``stable/<package_id>``; overridable), replaces its tree with the
    assembled repo, commits (message = provenance), tags ``v<version>-<package_id>``, and pushes to
    ``refs/heads/<branch>`` — idempotently PER BRANCH. Each real publish event is recorded via
    :func:`merlin.common.artifacts.new_product`."""
    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    resolved_remote = resolve_remote(target, config=config, override=remote)
    resolved_branch = resolve_branch(sel, override=branch, config=config)
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
    result.branch = resolved_branch
    result.actions = [
        f"select champion {sel.package_id} (family={sel.family}, status={sel.status})",
        f"assemble {sel.layout_kind} repo tree at {repo_dir}",
        f"gate: {'OK' if gate_ok else 'FAILED'} ({gate_detail})",
        f"remote: {resolved_remote}",
        f"branch: {resolved_branch}",
        f"tag: {tag}",
        f"fingerprint: {fingerprint}",
    ]

    if dry_run:
        result.actions.insert(0, "DRY-RUN (no clone/commit/push)")
        return result

    # human diff-confirm gate before any real network push (local/file remotes are exempt).
    _require_push_confirmation(resolved_remote, repo_dir, resolved_branch, fingerprint, confirm_push)
    result.actions.append(f"push confirmed for non-local remote (fingerprint {fingerprint})"
                          if _needs_push_confirmation(resolved_remote) else "local remote (no confirm)")

    commit_sha, published_tag, noop = _git_publish(
        resolved_remote, repo_dir, sel, manifest, fingerprint, cert_run, stage_root,
        resolved_branch)
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
        "branch": resolved_branch,
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


def _print_result(res: "PublishResult | dict[str, Any]") -> None:
    # `index` returns a plain dict (it publishes a landing page, not a champion package), so it
    # has no package_id/tag. Print the fields it does carry instead of crashing on the ones it
    # does not -- the push had already succeeded when this raised, which is the worst kind of
    # failure to report: a real error message about work that actually completed.
    if isinstance(res, dict):
        print(f"target={res.get('target')} branch={res.get('branch')}")
        print(f"remote={res.get('remote')}")
        print(f"dry_run={res.get('dry_run')} noop={res.get('noop')}")
        if res.get("commit_sha"):
            print(f"commit={res['commit_sha']}")
        print(f"listed {len(res.get('entries') or [])} branch(es):")
        for e in res.get("entries") or []:
            print(f"  - {e['branch']}  ({e['package_id']}, {e['dtype']}, {e['status']})")
        for a in res.get("actions") or []:
            print(f"  - {a}")
        return
    print(f"target={res.target} package={res.package_id}")
    print(f"remote={res.remote} branch={res.branch}")
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
    p_pub.add_argument("--branch", help="override the resolved branch (default: baseline | stable/<pkg>)")
    p_pub.add_argument("--config", help="override the publish.yaml config path")
    p_pub.add_argument("--artifacts-root", help="override the out/artifacts root (targets under it)")
    p_pub.add_argument("--dry-run", action="store_true", help="plan only (default)")
    p_pub.add_argument("--execute", action="store_true", help="actually clone/commit/push")
    p_pub.add_argument("--no-gate", action="store_true", help="publish even if uncertified (LOUD warning)")
    p_pub.add_argument("--confirm-push", help="content fingerprint confirming a real push to a non-local "
                       "remote (printed when refused); required for a GitHub/network push")

    p_prom = sub.add_parser("promote", help="mark a package the single champion for a target")
    p_prom.add_argument("--target", required=True)
    p_prom.add_argument("--champion", "--package", dest="champion", required=True)
    p_prom.add_argument("--artifacts-root")
    p_prom.add_argument("--no-gate", action="store_true")

    p_mat = sub.add_parser("materialize",
                           help="install a run's submission as the target's OOT backend package")
    p_mat.add_argument("--target", required=True)
    p_mat.add_argument("--from", dest="source", required=True,
                       help="the run's submission/ directory (manifest.yaml + the backend tree)")
    p_mat.add_argument("--package-id", default="agent_spec_v1_mlir_oot")
    p_mat.add_argument("--certified-by-run", default="")
    p_mat.add_argument("--score", help="score_capsule.json justifying the install (checked, not trusted)")
    p_mat.add_argument("--artifacts-root")
    p_mat.add_argument("--force", action="store_true")

    p_idx = sub.add_parser("index", help="publish the landing page to the repo's default branch")
    p_idx.add_argument("--target", required=True)
    p_idx.add_argument("--remote", help="override the resolved remote")
    p_idx.add_argument("--branch", default="main", help="default branch to write (default: main)")
    p_idx.add_argument("--config", help="override the publish.yaml config path")
    p_idx.add_argument("--artifacts-root")
    p_idx.add_argument("--dry-run", action="store_true", help="plan only (default)")
    p_idx.add_argument("--execute", action="store_true", help="actually clone/commit/push")
    p_idx.add_argument("--confirm-push", help="fingerprint confirming a real push to a non-local remote")

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
                          artifacts_root=args.artifacts_root, config=args.config,
                          branch=args.branch, confirm_push=args.confirm_push)
            _print_result(res)
            return 0
        if args.cmd == "index":
            res = publish_index(args.target, dry_run=not args.execute, remote=args.remote,
                                config=args.config, artifacts_root=args.artifacts_root,
                                branch=args.branch, confirm_push=args.confirm_push)
            _print_result(res)
            return 0
        if args.cmd == "materialize":
            dst = materialize_package(args.target, args.source, package_id=args.package_id,
                                      certified_by_run=args.certified_by_run, score_path=args.score,
                                      artifacts_root=args.artifacts_root, force=args.force)
            print(f"installed {args.target} backend -> {dst}")
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
            branch = resolve_branch(sel, config=args.config)
            ok, detail = _check_gate(sel)
            print(f"target={sel.target} champion={sel.package_id}")
            print(f"family={sel.family} layout={sel.layout_kind} status={sel.status}")
            print(f"remote={remote} branch={branch}")
            print(f"gate={'OK' if ok else 'FAILED'} ({detail})")
            print(f"cert_run={_cert_run_id(sel)}")
            return 0
    except PublishError as e:
        sys.stderr.write(f"merlin-target-publish: {e}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
