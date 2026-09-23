"""Publish preserved target payloads with separate, explicitly scoped provenance.

Candidate compiler ABI/tool/build declarations must already exist. Host schedules
remain data; support packages and unknown roles are not compiler publications.
No compiler wrappers, commands, manifests or build recipes are synthesized.

Source files stay byte-identical at the export root. Only reserved .merlin/
metadata and MERLIN_PUBLICATION.md are added. Exact source and exported inventories
are recorded separately; source certification never certifies the expanded export.
Build verification uses a retained copy and establishes only build-artifact presence.
Remotes are explicitly resolved; default invocation is non-executing dry-run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import paths
from ..common.artifacts import git_sha7, new_product, utc_stamp
from ..common.jsonio import write_pretty_json
from ..common.yaml import dump_yaml, load_yaml
from . import package_records

LAYOUT_VERSION = "2.0"

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
    layout_kind: str  # Preserved payload layout; retained for provenance API compatibility.
    status: str
    cert_status: str | None  # publication.certification (e.g. "pass") if recorded
    cert_run: str | None  # publication.certified_by_run if recorded
    oracle_cycles: int | None
    version: int
    lineage_depth: int
    timestamp: str
    repo_name: str = ""
    publication_record: dict[str, Any] | None = None

    @property
    def dialect_name(self) -> str:
        return "".join(part.capitalize() for part in self.target.replace("-", "_").split("_"))

    @property
    def tool_name(self) -> str:
        return f"{self.target}-opt"


def _oracle_cycles(manifest: dict[str, Any]) -> int | None:
    """Best-effort recorded oracle cycle count (lower == faster == better champion)."""
    for holder in (manifest.get("certification"), manifest.get("oracle"), manifest.get("publication")):
        if isinstance(holder, dict) and isinstance(holder.get("cycles"), int):
            return holder["cycles"]
    if isinstance(manifest.get("cycles"), int):
        return manifest["cycles"]
    return None


def _build_selection(target: str, pkg_dir: Path, manifest: dict[str, Any]) -> ChampionSelection:
    try:
        record = package_records.read_record(pkg_dir)
    except (ValueError, OSError) as exc:
        raise PublishError(str(exc)) from exc
    manifest = copy.deepcopy(manifest)
    historical = {"status": manifest.get("status"), "publication": manifest.get("publication")}
    manifest["historical_publication"] = historical
    legacy_pub = manifest.get("publication")
    hints = {"role": legacy_pub["role"]} if isinstance(legacy_pub, dict) and "role" in legacy_pub else {}
    manifest["publication"] = record.get("publication", {}) if record else hints
    family = str(manifest.get("family", ""))
    pub = manifest.get("publication") if isinstance(manifest.get("publication"), dict) else {}
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    return ChampionSelection(
        target=target,
        package_id=pkg_dir.name,
        package_dir=pkg_dir,
        manifest=manifest,
        family=family,
        layout_kind="preserved_payload",
        status=str(manifest.get("status", "")),
        cert_status=pub.get("certification"),
        cert_run=pub.get("certified_by_run"),
        oracle_cycles=_oracle_cycles(manifest),
        version=int(manifest.get("version", 0) or 0),
        lineage_depth=int(lineage.get("depth", 0) or 0),
        timestamp=str(manifest.get("timestamp", "")),
        repo_name=resolve_repo_name(target),
        publication_record=record,
    )


def _rank_key(pkg_dir: Path, manifest: dict[str, Any]) -> tuple:
    """Deterministic ranking key (best sorts first): status, oracle cycles, version/depth, ts."""
    status = str(manifest.get("status", ""))
    cycles = _oracle_cycles(manifest)
    lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
    # An explicit `champion: false` demotes. Only `true` was ever consulted, so a package that declared
    # itself NOT the champion still competed on the ordinary ranking -- whose last tie-break is the
    # directory name, which is no basis for redirecting every consumer to a different compiler.
    pub = manifest.get("publication") if isinstance(manifest.get("publication"), dict) else {}
    declined = 1 if pub.get("champion") is False else 0
    return (
        declined,
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


def select_champion(
    target: str,
    *,
    artifacts_root: str | Path | None = None,
    package_id: str | None = None,
    dtype_strategy: str | None = None,
) -> ChampionSelection:
    """Pick the champion package for ``target`` under ``out/artifacts/targets/<target>/``.

    If ``package_id`` is given, that package is selected. Otherwise, if exactly one package is
    flagged ``publication.champion: true`` it wins; failing that, packages are ranked
    deterministically: a package declaring ``publication.champion: false`` sorts last, then
    ``rtl_certified`` > ``spike_verified`` > other, then fewer oracle cycles, then higher lineage
    version/depth, then newer timestamp, then package_id.

    ``dtype_strategy`` (e.g. ``"int8_w8a8"``) restricts the candidates to packages carrying that
    knob. Without it, an int8 caller can be handed the globally best package even when that
    package is fp32 — which builds a silently wrong datapath rather than failing.
    """
    tdir = _targets_root(artifacts_root) / target
    try:
        package_records.component(target)
        if package_id is not None:
            package_records.component(package_id)
        package_records.safe_path(tdir)
    except ValueError as exc:
        raise PublishError(str(exc)) from exc
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
            raise PublishError(f"no {target!r} package with dtype_strategy={dtype_strategy!r} under {tdir}")

    selections = {d: _build_selection(target, d, m) for d, m in packages}
    packages = [(d, selections[d].manifest) for d, _ in packages]
    champs = [
        (d, m)
        for d, m in packages
        if isinstance(m.get("publication"), dict) and m["publication"].get("champion") is True
    ]
    if len(champs) == 1:
        return selections[champs[0][0]]

    ranked = sorted(packages, key=lambda dm: _rank_key(dm[0], dm[1]))
    return selections[ranked[0][0]]


# ---------------------------------------------------------------------------- remote resolution


def resolve_repo_name(target: str, *, config: str | Path | None = None, override: str | None = None) -> str:
    """Resolve the PUBLIC repo name for ``target``. Precedence matches :func:`resolve_remote`:
    ``override`` > env ``MERLIN_PUBLISH_REPO_NAME_<TARGET>`` > ``publish.yaml``'s ``repo_names`` >
    the default ``<target>-mlir``.

    The repo name is deliberately NOT the target key. A target key names the thing we generate code
    for; the repo name is what the public sees, and the two do not have to agree -- the host target
    is keyed ``rvv`` because its payload is a vector schedule, but the repo holds all host codegen,
    scalar included, so it publishes as ``host-mlir``. Keeping them separate also leaves the build
    contract alone: the tool is still ``<target>-opt``, which is what oot_runner builds.
    """
    if override:
        return override
    env_val = paths.env(f"MERLIN_PUBLISH_REPO_NAME_{target.upper()}")
    if env_val:
        return env_val
    cfg_path = Path(config) if config else paths.targets_dir() / "publish.yaml"
    if cfg_path.is_file():
        data = load_yaml(cfg_path) or {}
        name = (data.get("repo_names") or {}).get(target)
        if name:
            return str(name)
    return f"{target}-mlir"


def resolve_remote(target: str, *, config: str | Path | None = None, override: str | None = None) -> str:
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
        raise PublishError(
            f"no remote configured for {target!r} in {cfg_path} "
            f"(and no --remote / MERLIN_PUBLISH_REMOTE_{target.upper()})"
        )
    return str(remote)


def _is_baseline(sel: "ChampionSelection") -> bool:
    """The frozen unoptimized control (by known package_id or a manifest opt-in flag)."""
    if sel.package_id in _BASELINE_PACKAGE_IDS:
        return True
    pub = sel.manifest.get("publication")
    return isinstance(pub, dict) and pub.get("role") == "baseline"


def resolve_branch(sel: "ChampionSelection", *, override: str | None = None, config: str | Path | None = None) -> str:
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


def _staging_location(dest: str | Path, *, sources: tuple[Path, ...]) -> Path:
    """Validate location before writes.

    Reject symlink components as well as canonical source overlap. This protects
    ordinary host mistakes, not concurrent hostile filesystem replacement.
    """
    dest = Path(dest).absolute()
    if any(part.is_symlink() for part in (dest, *dest.parents)):
        raise PublishError(f"staging destination has a symlink component: {dest}")
    resolved = dest.resolve()
    for source in sources:
        source = source.resolve()
        if resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise PublishError(f"staging destination overlaps source {source}: {dest}")
    return resolved


def _fresh_destination(dest: str | Path, *, sources: tuple[Path, ...]) -> Path:
    dest = _staging_location(dest, sources=sources)
    if dest.exists():
        raise PublishError(f"staging destination must be fresh: {dest}")
    return dest


def _new_stage_root(target: str, *, sources: tuple[Path, ...], index: bool = False) -> Path:
    """Allocate a unique retained parent; assembly receives its absent repo leaf."""
    if not target or target in (".", "..") or Path(target).name != target:
        raise PublishError(f"target must be a single staging path component: {target!r}")
    parent = _staging_location(paths.build_dir() / "publish" / target, sources=sources)
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{utc_stamp()}{'_index' if index else ''}_"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tier_phrase(tier: Any) -> str:
    """Say in words what a recorded certification is a certification OF.

    `pass` alone invites the reader to assume the strongest thing it could mean. Naming the oracle
    -- and saying plainly when the tier was never recorded -- keeps the repo's own README from
    overclaiming on behalf of a run nobody here can see.
    """
    if not isinstance(tier, dict) or not tier:
        return "**tier not recorded** \u2014 do not read this as an RTL result"
    oracles = ", ".join(f"`{o}`" for o in (tier.get("oracles") or [])) or "unnamed oracle"
    if tier.get("derived_from_rtl") and tier.get("cycle_accurate"):
        return f"cycle-accurate RTL ({oracles})"
    if tier.get("derived_from_rtl"):
        return f"RTL-derived, not cycle-accurate ({oracles})"
    return f"a functional simulator ({oracles}) \u2014 numerically correct; **not** an RTL or timing result"


def _readme(sel: ChampionSelection, manifest: dict[str, Any]) -> str:
    """The published repo's landing page.

    A package published with ``--no-gate`` says so HERE, at the top, in the reader's first
    paragraph. Suppressing the refusal to stderr and shipping the ordinary "certified champion"
    wording is how an uncertified package gets cited as a certified one: the warning is seen by
    the operator who already knows, and never by the person who clones the repo.
    """
    merlin_sha = git_sha7()
    pub = sel.manifest.get("publication") or {}
    gate_ok, gate_detail = _check_gate(sel)
    lines = [
        f"# {sel.repo_name}",
        "",
        f"Out-of-tree Merlin codegen export for **{sel.target}** (family `{sel.family or 'unknown'}`).",
        "",
    ]
    if not gate_ok:
        lines += [
            "> ## \u26a0 NOT CERTIFIED \u2014 published with `--no-gate`",
            ">",
            "> This package did **not** pass Merlin's publication certification gate, and was "
            "exported anyway with `--no-gate`. It is **not a champion** and it is **not the "
            "baseline**.",
            ">",
            f"> Gate refusal, verbatim: `{gate_detail}`",
            ">",
            f"> Recorded status: `{sel.status or 'unknown'}`. Whatever this package earned is "
            "recorded under `.merlin/certification.yaml` and in the `grading:` block of "
            "`manifest.yaml` \u2014 read those before citing any number from it. A certification "
            "gate is not a formality here: a functional pass, a graded pass and a cycle-accurate "
            "RTL certification are three different claims.",
            "",
        ]
    lines += [
        (
            "This repository is **generated** by Merlin's `merlin-target-publish` bridge. The "
            "original payload stays at the repo root; additive provenance rides under `.merlin/`."
            if not gate_ok
            else "This repository is **generated** by Merlin's `merlin-target-publish` bridge: it is the "
            "transformed export of an observed package-payload certification. Only the exact original "
            "package inputs were tested. This exported tree is **NOT CERTIFIED**; external dependency "
            "closure is **not-attested**, and portable buildability is not established. "
            "Original input evidence rides along under `.merlin/`."
        ),
        "",
        "## What",
        "",
        f"- {'Package' if not gate_ok else 'Champion package'}: `{sel.package_id}`",
        f"- Family: `{sel.family or 'unknown'}`",
        f"- Recorded status: `{sel.status or 'unknown'}`",
        f"- Merlin git sha (this export): `{merlin_sha}`",
        "",
    ]
    # Only describe commands the original payload actually declares.
    tool_rel = str((manifest.get("entrypoints") or {}).get("tool") or sel.tool_name)
    role = _export_role(sel)
    if role.value == "host_schedule":
        lines += [
            "## Host schedule data",
            "",
            "This is a host schedule, not a standalone compiler. "
            "Use its original manifest and target-owned host consumer. Merlin did not synthesize "
            "a compiler driver, ABI commands or a build recipe.",
            "",
        ]
    elif not manifest.get("build"):
        lines += [
            "## How to run it",
            "",
            f"No declared build step. The manifest names `{tool_rel}`; use its declared command argv "
            "and required interpreter/environment, not an invented help flag.",
            "",
            "```sh",
            f"git clone <this-repo> {sel.repo_name}",
            f"cd {sel.repo_name}",
            "```",
            "",
            "`manifest.yaml` declares the entrypoint and the argv of every command the "
            "experiment ABI expects; run those, not a build.",
            "",
            "```yaml",
            dump_yaml({"commands": manifest["commands"]}).rstrip(),
            "```",
            "",
        ]
    else:
        lines += [
            "## How to build",
            "",
            "```sh",
            f"git clone <this-repo> {sel.repo_name}",
            f"cd {sel.repo_name}",
            "# Follow the original manifest.yaml build.configure/build.command declarations.",
            "```",
            "",
            "Declared build recipe (verbatim data; use the payload's documented environment):",
            "```yaml",
            dump_yaml({"build": manifest["build"]}).rstrip(),
            "```",
            "",
        ]
    lines += [
        "## Provenance",
        "",
        f"- Original package-input certification: `{pub.get('certification', sel.cert_status or 'unverified')}`",
        "- Scope: `package-payload`; external dependency closure: `not-attested`",
        "- Transformed exported payload certification: `unverified`",
        f"- {'Graded' if not gate_ok else 'Certified'} by run: `{pub.get('certified_by_run', sel.cert_run or 'n/a')}`",
        # An uncertified package's tier block describes the GRADING oracle, not a certification, so
        # it must not be introduced with the word "certified".
        f"- {'Oracle behind that tier' if not gate_ok else 'Certified against'}: "
        f"{_tier_phrase(pub.get('certification_tier'))}",
        f"- Fingerprint: `{pub.get('fingerprint', 'n/a')}`",
        "",
        "See `.merlin/provenance.yaml` and `.merlin/certification.yaml` for the full lineage. Each "
        "commit on this repo is one promotion; the history is the provenance trail.",
        "",
    ]
    return "\n".join(lines)


def _index_readme(target: str, entries: list[dict[str, Any]]) -> str:
    """Describe actual published roles without inventing a universal execution path."""
    lines = [
        f"# {resolve_repo_name(target)}",
        "",
        f"Published artifact branches for **{target}**.",
        "",
        "This is a **branch-per-version** directory, not a compiler package. "
        "Each listed branch preserves its original payload and adds scoped provenance.",
        "",
        "## Published packages",
        "",
        "| branch | package slot | dtype | source status | role |",
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['branch']}` | `{entry['package_id']}` | `{entry['dtype']}` | "
            f"`{entry['status']}` | {entry['role']} |"
        )
    lines += [
        "",
        "## Using a branch",
        "",
        "```sh",
        f"git clone -b <branch> <this-repo> {resolve_repo_name(target)}",
        f"cd {resolve_repo_name(target)}",
        "```",
        "",
        "Read the original manifest and author documentation, then MERLIN_PUBLICATION.md "
        "and .merlin/provenance.yaml for role and evidence scope.",
        "",
        "- **candidate_compiler**: use its actual declared ABI commands and build recipe "
        "with the required environment. Tool presence/build observations do not establish "
        "portable buildability or numerical correctness.",
        "- **host_schedule**: use the target-owned host consumer named by the artifact's "
        "documentation. Schedule data is not a standalone compiler.",
        "- Support packages have a separate target-support workflow and are not compiler releases "
        "listed by this exporter. Cloning a branch does not register it as target support.",
        "",
        "There is no universal model-compilation command for all these roles. "
        "No generated compiler wrapper or inferred ISA/toolchain is supplied.",
        "",
        "## Provenance",
        "",
        "Source observations have scope package-payload and external dependency closure "
        "not-attested. Added publication files and portable execution are not certified by "
        "source evidence. See .merlin/certification.yaml before citing a result. "
        "Historical manifest status alone grants no new qualification.",
        "",
        "Commit/tag messages bind the package slot, run, Merlin revision and exact source "
        "payload inventory. Existing tags remain immutable.",
        "",
        f"Generated from Merlin `{git_sha7()}`.",
        "",
    ]
    return "\n".join(lines)


def _cert_phrase(manifest: dict[str, Any]) -> str:
    """The tier a package EARNED, for a package whose flat ``status`` field is empty.

    A landing page that prints `unknown` beside a package certified on cycle-accurate RTL
    understates it, and a reader cannot tell that from a package with no evidence at all -- which
    is the failure the "quote the tier, never a bare score" rule exists to prevent. The tier is
    already recorded per rung by ``record_certification``; this reads it back rather than inventing
    a status. Returns "" when there is genuinely nothing recorded, so `unknown` still means unknown.
    """
    pub = manifest.get("publication")
    if not isinstance(pub, dict):
        return ""
    tier = pub.get("certification_tier")
    if not isinstance(tier, dict):
        return ""
    oracles = [str(o) for o in (tier.get("oracles") or []) if o]
    if not oracles:
        return ""
    # The weakest qualifier wins: one rung on a functional model does not make the package
    # cycle-accurate, and overstating that is exactly the citation error to avoid.
    rungs = [r for r in (pub.get("certified_rungs") or []) if isinstance(r, dict)]
    accurate = bool(tier.get("cycle_accurate")) and all(r.get("cycle_accurate") for r in rungs)
    from_rtl = bool(tier.get("derived_from_rtl")) and all(r.get("derived_from_rtl") for r in rungs)
    qualifier = "cycle-accurate RTL" if accurate and from_rtl else "RTL-derived" if from_rtl else "functional"
    n = len(rungs) or len(oracles)
    return f"certified ({qualifier}, {n} rung{'s' if n != 1 else ''}, {'/'.join(sorted(set(oracles)))})"


def index_entries(target: str, *, artifacts_root: str | Path | None = None) -> list[dict[str, Any]]:
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
            continue  # only certified packages are published, so only they are listed
        try:
            provider_role = _export_role(sel)
        except PublishError:
            continue
        is_base = _is_baseline(sel)
        out.append(
            {
                "branch": resolve_branch(sel),
                "package_id": sel.package_id,
                "dtype": package_dtype(man_path.parent),
                "status": sel.status or _cert_phrase(man) or "unknown",
                "role": (
                    f"{provider_role.value}: "
                    + ("frozen unoptimized control" if is_base else "observed package inputs")
                ),
            }
        )
    return sorted(out, key=lambda e: (e["dtype"], e["branch"]))


def assemble_index_tree(
    target: str, dest: str | Path, *, artifacts_root: str | Path | None = None, only_branches: "set[str] | None" = None
) -> dict[str, Any]:
    """Assemble the default-branch landing page (README + LICENSE) into ``dest``.

    Destination must be absent and disjoint from source inputs; existing directories
    (even empty ones) are caller-owned and are never reused.

    Deliberately NOT a package tree: the default branch must not look like one champion, or a
    consumer would build the wrong thing. Reuses the repo's LICENSE so the published repo
    carries the same terms as Merlin.

    ``only_branches`` restricts the listing to branches that ACTUALLY exist on the remote (the
    publish path passes what the clone reports). Without it the page would advertise every
    locally-certified package, including ones never pushed — a fresh clone would then be sent
    to branches that do not exist, which is worse than the empty default branch this replaces.
    """
    dest = _fresh_destination(dest, sources=(_targets_root(artifacts_root) / target, paths.repo_root() / "LICENSE"))
    entries = index_entries(target, artifacts_root=artifacts_root)
    if only_branches is not None:
        entries = [e for e in entries if e["branch"] in only_branches]
    dest.mkdir(parents=True)
    _write(dest / "README.md", _index_readme(target, entries))
    lic = paths.repo_root() / "LICENSE"
    if lic.is_file():
        _write(dest / "LICENSE", lic.read_text(encoding="utf-8"))
    return {"target": target, "entries": entries, "dest": str(dest)}


def _export_role(sel: ChampionSelection):
    """Read declared ownership; never turn support/schedule data into a compiler."""
    from .providers import ProviderError, ProviderRole, read_provider

    try:
        provider = read_provider(sel.package_dir)
    except ProviderError as exc:
        raise PublishError(str(exc)) from exc
    if provider is None or provider.role == ProviderRole.SUPPORT:
        raise PublishError("publication requires a candidate compiler or host schedule, not support/unknown data")
    if provider.target != sel.target:
        raise PublishError("provider target does not match selected publication target")
    if provider.role == ProviderRole.CANDIDATE_COMPILER:
        from .package_runtime import load_package

        try:
            pkg = load_package(sel.package_dir)
        except Exception as exc:
            raise PublishError(f"candidate compiler ABI unavailable: {exc}") from exc
        if not pkg.tool.resolve().is_relative_to(sel.package_dir.resolve()):
            raise PublishError("candidate compiler tool must be inside its payload")
        if not pkg.manifest.get("build") and not pkg.tool.is_file():
            raise PublishError("candidate compiler has neither an existing tool nor a build recipe")
    return provider.role


def assemble_repo_tree(sel: ChampionSelection, dest: str | Path, *, layout_version: str) -> dict[str, Any]:
    """Preserve original payload paths/bytes; append explicitly scoped publication metadata."""
    dest = _fresh_destination(dest, sources=(sel.package_dir,))
    _export_role(sel)
    for reserved in (".merlin", "MERLIN_PUBLICATION.md", ".git"):
        if (sel.package_dir / reserved).exists():
            raise PublishError(f"payload collides with reserved publication path: {reserved}")
    identity = package_records.payload_inventory(sel.package_dir)
    shutil.copytree(sel.package_dir, dest)
    if (
        package_records.payload_inventory(dest) != identity
        or package_records.payload_inventory(sel.package_dir) != identity
    ):
        raise PublishError("payload changed during export copy")
    manifest = load_yaml(dest / "manifest.yaml")
    _write(dest / "MERLIN_PUBLICATION.md", _readme(sel, manifest))
    return manifest


# ---------------------------------------------------------------------------- provenance layer


def _cert_run_id(sel: ChampionSelection) -> str:
    return sel.cert_run or f"recorded-{sel.status or 'unknown'}"


def _fingerprint(package_id: str, merlin_sha: str, cert_run_id: str, payload_sha256: str) -> str:
    payload = f"{package_id}\n{merlin_sha}\n{cert_run_id}\n{payload_sha256}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha_full(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root or paths.repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
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
    fp = _fingerprint(
        sel.package_id, merlin_sha, cert_run, package_records.payload_inventory(sel.package_dir)["sha256"]
    )

    # 1) manifest copy (provenance record; identical to the root build manifest)
    shutil.copy2(dest / "manifest.yaml", meta / "manifest.yaml")

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
        "package_slot": sel.package_id,
        "compiler_package_id": manifest.get("package_id"),
        "family": sel.family,
        "provider_role": _export_role(sel).value,
        "source_package": src_rel,
        "merlin_git_sha": merlin_sha,
        "merlin_git_sha7": sha7,
        "source_status": sel.status,
        "version": sel.version,
        "lineage_depth": sel.lineage_depth,
        "run_refs": [cert_run],
        "fingerprint": fp,
        "certification_scope": "package-payload",
        "external_dependency_closure": "not-attested",
        "exported_payload_certification": "unverified",
        "source_payload": sel.publication_record.get("payload") if sel.publication_record else None,
    }
    _write(meta / "provenance.yaml", dump_yaml(provenance))

    # 3) recorded certification (oot_runner.certify for gemmini / rvv spike gate)
    gate_ok, gate_detail = _check_gate(sel)
    certification = {
        "target": sel.target,
        "package_id": sel.package_id,
        "layout_kind": sel.layout_kind,
        "gate": "package-payload-observation",
        "status": "unverified",
        "source_input_status": "pass" if gate_ok else "unverified",
        "scope": "package-payload",
        "external_dependency_closure": "not-attested",
        "transformed_export_status": "unverified",
        "recorded_status": sel.status,
        "certification": "unverified",
        "source_recorded_certification": sel.cert_status,
        "certified_by_run": cert_run,
        "oracle_cycles": sel.oracle_cycles,
        "detail": gate_detail,
        # The TIER the pass was earned at. The gate treats a functional-simulator pass and a
        # cycle-accurate RTL pass alike, but they are different claims and a reader of this repo
        # must be able to tell them apart without rerunning anything. Absent = never recorded.
        "tier": (sel.manifest.get("publication") or {}).get("certification_tier") or "UNKNOWN",
        "rungs": (sel.manifest.get("publication") or {}).get("certified_rungs") or [],
    }
    _write(meta / "certification.yaml", dump_yaml(certification))

    # 4) CHAMPION marker: one line
    _write(meta / "CHAMPION", f"{sel.package_id} {sha7} {cert_run}\n")


# ---------------------------------------------------------------------------- gate


def _check_gate(sel: ChampionSelection) -> tuple[bool, str]:
    """Admit only scoped observed tests of the selected exact package input bytes."""
    try:
        record = package_records.read_record(sel.package_dir)
    except (ValueError, OSError) as exc:
        return False, str(exc)
    if (
        record
        and record["publication"].get("certification") == "pass"
        and record["publication"].get("oracle_metadata_identified") is True
    ):
        rungs = record["publication"].get("certified_rungs")
        if (
            isinstance(rungs, list)
            and rungs
            and all(
                isinstance(r, dict)
                and r.get("status") == "pass"
                and package_records.bound_inputs(
                    r.get("package_input_identity"), record["payload"], record.get("compiler_package_id")
                )
                for r in rungs
            )
        ):
            return True, (
                "observed certification of exact package-payload inputs; external dependency closure "
                "not-attested; transformed exports and standalone portability are not certified"
            )
    return False, "producer-bound certification input identity is unavailable; publication is unverified"


# ---------------------------------------------------------------------------- promotion


def record_certification(
    target: str, package_id: str, results: "list[str | Path]", *, artifacts_root: str | Path | None = None
) -> dict[str, Any]:
    """Record historical verdicts outside the immutable package payload.

    A certify run writes ``results.yaml`` into its own run dir and stops there. Nothing ever carried
    that verdict back to the package, so ``publication.certification`` could only be written by
    :func:`promote` -- which asks :func:`_check_gate`, which asks for the certification. Nothing can
    satisfy that loop, which is why a package carrying a real out-of-tree dialect could never be
    promoted and the only publishable champion was the hand baseline, whose repo builds a stub.

    The verdict keeps its TIER. A pass on the functional simulator and a pass on cycle-accurate RTL
    are both ``pass`` to the gate, but they are emphatically not the same claim: one says the
    lowering computes the right numbers, the other says the hardware does. Both travel here so the
    published ``certification.yaml`` states which one it is, and a reader can never mistake a
    functional pass for an RTL one.

    Any non-passing rung records failure. Missing target, run or typed successful
    oracle metadata records an unverified result, never a passing certification.
    Explicit target mismatch is refused before any record write. These identity
    checks do not yet bind the results to package input bytes.
    """
    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    rungs: list[dict[str, Any]] = []
    identified = []
    bound = []
    payload = package_records.payload_inventory(sel.package_dir)
    for r in results:
        rp = Path(r)
        if rp.is_dir():
            rp = rp / "results.yaml"
        if not rp.is_file():
            raise PublishError(f"no certify results at {rp}")
        data = load_yaml(rp) or {}
        if not isinstance(data, dict):
            raise PublishError(f"certify results must be an object: {rp}")
        if data.get("target") is not None and data["target"] != target:
            raise PublishError(f"certify target {data['target']!r} does not match {target!r}: {rp}")
        oracle = data.get("oracle") if isinstance(data.get("oracle"), dict) else {}
        kind = oracle.get("kind")
        bound.append(
            package_records.bound_inputs(data.get("package_input_identity"), payload, sel.manifest.get("package_id"))
        )
        identified.append(
            data.get("target") == target
            and isinstance(data.get("run_id"), str)
            and bool(data["run_id"].strip())
            and isinstance(kind, str)
            and bool(kind.strip())
            and kind.strip().lower() not in {"none", "unknown", "skipped"}
            and oracle.get("result") == "pass"
            and type(oracle.get("derived_from_rtl")) is bool
            and type(oracle.get("cycle_accurate")) is bool
        )
        rungs.append(
            {
                "rung": str(data.get("rung", rp.parent.name)),
                "run_id": str(data.get("run_id", "")),
                "target": data.get("target"),
                "status": str(data.get("status", "UNKNOWN")),
                "oracle": str(oracle.get("kind", "UNKNOWN")),
                # `is True` on purpose: a missing key must not read as False, which would silently
                # downgrade an RTL pass to a functional one (or vice versa) on a malformed file.
                "derived_from_rtl": oracle.get("derived_from_rtl") is True,
                "cycle_accurate": oracle.get("cycle_accurate") is True,
                "cycles": oracle.get("cycles"),
                "package_input_identity": data.get("package_input_identity"),
            }
        )
    if not rungs:
        raise PublishError("no certify results given")

    passed = all(r["status"] == "pass" for r in rungs)
    certification = "fail" if not passed else "pass" if all(identified) and all(bound) else "unverified"
    pub = dict(sel.manifest.get("publication") or {})
    pub.update(
        {
            "certification": certification,
            "certified_by_run": rungs[0]["run_id"] or None,
            "certified_at": utc_stamp(),
            "certified_by": "merlin.targetgen.oot_runner.certify",
            # The tier is the weakest rung's, not the strongest: a package is only as certified as its
            # least-certified covered rung, and quoting the best one is how a headline outruns its
            # evidence.
            "certification_tier": {
                "derived_from_rtl": all(r["derived_from_rtl"] for r in rungs),
                "cycle_accurate": all(r["cycle_accurate"] for r in rungs),
                "oracles": sorted({r["oracle"] for r in rungs}),
            },
            "certified_rungs": rungs,
            "oracle_metadata_identified": all(identified),
            "input_binding": "package-payload" if all(bound) else "unverified",
            "external_dependency_closure": "not-attested",
        }
    )
    package_records.write_record(sel.package_dir, load_yaml(sel.package_dir / "manifest.yaml"), pub)
    return pub


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
        return False, (
            f"{len(hollow)} capsule(s) report pass with no tier evidence "
            f"({', '.join(str(h) for h in hollow[:4])}) — a pass with an empty tier map is "
            f"not evidence a compiler ran"
        )
    return True, f"{len(passed)}/{len(rows)} passed with tier evidence"


def materialize_package(
    target: str,
    source: str | Path,
    *,
    package_id: str = "agent_spec_v1_mlir_oot",
    certified_by_run: str = "",
    score_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Install a run's submission as ``out/artifacts/targets/<target>/<package_id>/`` and return its path.

    Installation preserves payload bytes without inheriting publication authority. A structurally
    non-vacuous score is descriptive evidence, not a binding to the copied payload.
    Source claims remain historical payload/record metadata only. ``certified_by_run``
    is retained as a compatibility argument naming the installation's source run,
    not a certification. ``publication.champion`` is false so ordinary ranking
    does not displace an existing champion merely by sorting earlier.
    """
    import shutil

    src_dir = Path(source)
    for name, component in (("target", target), ("package_id", package_id)):
        try:
            package_records.component(component)
        except ValueError as exc:
            raise MaterializeRefused(f"{name} must be a non-reserved single path component: {component!r}") from exc
    try:
        sources = (src_dir, Path(score_path)) if score_path is not None else (src_dir,)
        dst = _staging_location(_targets_root(artifacts_root) / target / package_id, sources=sources)
        _staging_location(package_records.record_path(dst), sources=sources)
    except PublishError as exc:
        raise MaterializeRefused(str(exc)) from exc
    if dst.exists():
        if not force:
            raise MaterializeRefused(f"{dst} already exists — pass force=True to replace it")
        if not dst.is_dir():
            raise MaterializeRefused(f"existing package is not a directory: {dst}")
    if not (src_dir / "manifest.yaml").is_file():
        raise MaterializeRefused(f"{src_dir} carries no manifest.yaml — not an OOT backend package")
    if score_path:
        import json as _json

        score = _json.loads(Path(score_path).read_text(encoding="utf-8"))
        ok, detail = _score_is_honest(score)
        if not ok:
            raise MaterializeRefused(f"refusing to install {target} compiler from {src_dir}: {detail}")
        evidence = {
            "n_passed": score.get("n_passed"),
            "n_capsules": score.get("n_capsules"),
            "labels_graded": score.get("labels_graded"),
            "detail": detail,
        }
    else:
        evidence = {"detail": "no score supplied — installed unverified"}
    evidence["payload_binding"] = "unverified"

    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".materialize-", dir=dst.parent))
    prepared = stage / "package"
    previous = stage / "previous" if dst.exists() else None
    expected_payload = package_records.payload_inventory(src_dir)
    shutil.copytree(src_dir, prepared)
    if (
        package_records.payload_inventory(prepared) != expected_payload
        or package_records.payload_inventory(src_dir) != expected_payload
    ):
        raise MaterializeRefused("source payload changed during installation")

    man = load_yaml(prepared / "manifest.yaml")
    source_package_id = man.get("package_id")
    source_status = man.get("status")
    source_publication = man.get("publication")
    promotion = {
        "from": str(src_dir),
        "source_package_id": source_package_id,
        "source_status": source_status,
        "source_publication": source_publication,
        "installed_from_run": certified_by_run,
        "promoted_by": "merlin-target-publish materialize",
        "score": str(score_path) if score_path else None,
        "evidence": evidence,
        "previous_package": str(previous) if previous is not None else None,
    }
    old_record = package_records.record_path(dst)
    if old_record.exists():
        shutil.copy2(old_record, stage / "previous-publication.json")
    # Prepare completely before moving the existing tree. Retain its exact bytes
    # for recovery; neither successful replacement nor failure recursively deletes
    # operator data. This assumes a stable trusted host filesystem, not hostile
    # concurrent path replacement, and is not a crash-atomic multi-rename commit.
    if previous is not None:
        dst.rename(previous)
    try:
        prepared.rename(dst)
        package_records.write_record(dst, man, {"champion": False, "certification": "unverified"}, promotion=promotion)
    except (OSError, ValueError):
        if dst.exists():
            dst.rename(stage / "failed-package")
        if previous is not None:
            previous.rename(dst)
        raise
    if previous is None:
        try:
            stage.rmdir()
        except OSError:
            # Installation already succeeded. An empty-container cleanup failure
            # must not report a failed or rolled-back installation.
            sys.stderr.write(f"WARNING: package installed at {dst}; staging cleanup failed at {stage}\n")
    return dst


def promote(target: str, package_id: str, *, gate: bool = True, artifacts_root: str | Path | None = None) -> None:
    """Promote ``package_id`` using external records, never editing its payload manifest.

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
        other = _build_selection(target, man_path.parent, load_yaml(man_path))
        pub = other.manifest.get("publication")
        if isinstance(pub, dict) and pub.get("champion"):
            pub["champion"] = False
            package_records.write_record(man_path.parent, load_yaml(man_path), pub)

    merlin_sha = _git_sha_full()
    cert_run = _cert_run_id(sel)
    man = load_yaml(sel.package_dir / "manifest.yaml")
    pub = dict(sel.manifest.get("publication") or {})
    pub.update(
        {
            "champion": True,
            "certification": "pass" if ok else "unverified",
            "certified_by_run": cert_run,
            "promoted_at": utc_stamp(),
            "promoted_by": "merlin-target-publish",
            "fingerprint": _fingerprint(
                sel.package_id, merlin_sha, cert_run, package_records.payload_inventory(sel.package_dir)["sha256"]
            ),
        }
    )
    package_records.write_record(sel.package_dir, man, pub)


# ---------------------------------------------------------------------------- git mechanics


def _git(args: list[str], cwd: Path | None = None, *, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=300)
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
            # copy2 preserves mode. `copyfile` here is why the published entrypoint arrived at
            # 100644: the assembled tree had it executable and this last hop into the clone stripped
            # it, so git recorded a script a fresh clone could not run.
            shutil.copy2(entry, target)


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


def _git_publish(
    remote: str,
    repo_dir: Path,
    sel: ChampionSelection,
    manifest: dict[str, Any],
    fingerprint: str,
    cert_run: str,
    stage_root: Path,
    branch: str,
) -> tuple[str, str, bool]:
    """Clone the remote, check out ``branch``, replace the tree with the assembled repo, commit +
    tag, push to ``refs/heads/<branch>``. Idempotent PER BRANCH on the **fingerprint**: a matching
    branch-tip fingerprint is a no-op. Returns (commit_sha, tag, noop).

    An existing TAG is deliberately not a veto. The tag names a package *version*, but the
    fingerprint covers package ID, Merlin revision, cert run and source payload inventory — so re-certifying an unchanged
    payload (``spike_verified`` -> ``k1_verified`` after a board campaign) produces new provenance
    at the same version. Letting the tag veto that meant the published ``.merlin/certification.yaml``
    kept the WEAKER status forever, silently understating the certification to every consumer. The
    branch moves; the existing tag is left exactly where it is, because consumers may have pinned
    it and rewriting a published tag is worse than not adding one."""
    clone_dir = _fresh_destination(stage_root / "clone", sources=(repo_dir, sel.package_dir))
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
    gate_ok, gate_detail = _check_gate(sel)
    # A --no-gate publish names itself in the SUBJECT. The history of a target repo is its
    # provenance trail; a commit that says "champion" for a package the gate refused makes that
    # trail assert the one thing that is not true about it.
    subject = (
        f"publish({sel.target}): export of tested package inputs {sel.package_id}"
        if gate_ok
        else f"publish({sel.target}): UNCERTIFIED package {sel.package_id}"
    )
    warning = (
        ""
        if gate_ok
        else (
            f"WARNING: published with --no-gate. This package did NOT pass the certification gate.\n"
            f"Gate-Refusal: {gate_detail}\n"
            f"Not-A-Champion: true\n"
        )
    )
    body = (
        warning + f"Package: {sel.package_id}\n"
        f"Target: {sel.target}\n"
        f"Family: {sel.family}\n"
        f"Merlin-Sha: {merlin_sha}\n"
        f"Internal-Run: {cert_run}\n"
        "Cert: unverified (exported tree)\n"
        f"Source-Input-Certification: {'pass' if gate_ok else 'unverified'}\n"
        "Certification-Scope: package-payload\n"
        "External-Dependency-Closure: not-attested\n"
        "Transformed-Export-Certification: unverified\n"
        f"Layout-Version: {manifest.get('layout_version', LAYOUT_VERSION)}\n"
        "\n"
        f"Merlin-Publish-Fingerprint: {fingerprint}\n"
    )
    _git(
        [
            "-C",
            str(clone_dir),
            "-c",
            "user.name=merlin-target-publish",
            "-c",
            "user.email=publish@merlin.local",
            "commit",
            "-m",
            subject,
            "-m",
            body,
        ]
    )
    # annotated tag (carries the fingerprint; also robust to git configs that force annotation).
    # Never re-point an existing tag: a consumer may have pinned it, and this commit is a
    # re-certification of the SAME version, not a new release.
    if tag not in existing_tags:
        _git(
            [
                "-C",
                str(clone_dir),
                "-c",
                "user.name=merlin-target-publish",
                "-c",
                "user.email=publish@merlin.local",
                "tag",
                "-a",
                tag,
                "-m",
                (
                    f"{sel.target} export of tested package inputs {sel.package_id}"
                    if gate_ok
                    else f"{sel.target} UNCERTIFIED package {sel.package_id} (published --no-gate)"
                )
                + f"\nSource-Input-Certification: {'pass' if gate_ok else 'unverified'}\n"
                "Certification-Scope: package-payload\n"
                "External-Dependency-Closure: not-attested\n"
                "Transformed-Export-Certification: unverified\n"
                f"Merlin-Publish-Fingerprint: {fingerprint}\n",
            ]
        )
    commit_sha = _git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()

    _git(["-C", str(clone_dir), "push", "origin", f"HEAD:refs/heads/{branch}"])
    _git(["-C", str(clone_dir), "push", "origin", "--tags"])
    # NOTE: `gh release` is intentionally skipped for file:// bare remotes (the verification path).
    return commit_sha, tag, False


def publish_index(
    target: str,
    *,
    dry_run: bool = True,
    remote: str | None = None,
    config: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    branch: str = "main",
    confirm_push: str | None = None,
) -> dict[str, Any]:
    """Publish the landing page to the repo's DEFAULT branch.

    Branch-per-version publishing leaves that branch empty, so `git clone` with no `-b` gives a
    repo containing nothing — the state ucb-bar/rvv-mlir was actually in. This writes a
    directory page listing the branches that exist ON THE REMOTE (read from the clone, so the
    page cannot advertise a branch that is not there) and how to consume a package.

    Same safety posture as :func:`publish`: dry-run by default, and a non-local remote needs an
    explicit ``confirm_push`` fingerprint.
    """
    resolved = remote or resolve_remote(target, config=config)
    stage_root = _new_stage_root(
        target, sources=(_targets_root(artifacts_root) / target, paths.repo_root() / "LICENSE"), index=True
    )
    repo_dir = stage_root / "repo"
    actions = [f"remote: {resolved}", f"branch: {branch}"]

    clone_dir = stage_root / "clone"
    remote_branches: set[str] | None = None
    if not dry_run:
        clone_dir = _fresh_destination(clone_dir, sources=(_targets_root(artifacts_root) / target,))
        _git(["clone", resolved, str(clone_dir)])
        remote_branches = {
            b.strip().removeprefix("origin/")
            for b in _git(["-C", str(clone_dir), "branch", "-r"], check=False).stdout.split()
            if b.strip().startswith("origin/") and "->" not in b
        }
        actions.append(f"remote branches: {sorted(remote_branches)}")

    info = assemble_index_tree(target, repo_dir, artifacts_root=artifacts_root, only_branches=remote_branches)
    actions.append(f"assembled index listing {len(info['entries'])} package(s) at {repo_dir}")
    res: dict[str, Any] = {
        "target": target,
        "remote": resolved,
        "branch": branch,
        "entries": info["entries"],
        "repo_dir": str(repo_dir),
        "dry_run": dry_run,
        "actions": actions,
        "noop": False,
    }
    if dry_run:
        actions.append("dry-run: nothing cloned, committed or pushed")
        return res

    fingerprint = _fingerprint(
        f"{target}-index",
        _git_sha_full(),
        ",".join(e["branch"] for e in info["entries"]),
        package_records.payload_inventory(repo_dir)["sha256"],
    )
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
    _git(
        [
            "-C",
            str(clone_dir),
            "-c",
            "user.name=merlin-target-publish",
            "-c",
            "user.email=publish@merlin.local",
            "commit",
            "-m",
            f"docs({target}): landing page for the published package branches",
            "-m",
            (
                f"Lists the branches present on this remote and how to consume a package.\n"
                f"Merlin-Sha: {_git_sha_full()}\n\n"
                f"Merlin-Publish-Fingerprint: {fingerprint}\n"
            ),
        ]
    )
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


def _require_push_confirmation(
    remote: str, repo_dir: Path, branch: str, fingerprint: str, confirm_push: str | None
) -> None:
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
        f"  The token must equal this publish's content fingerprint, so it cannot be passed blindly."
    )


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
    export_identity_path: Path | None = None
    build_verification_path: Path | None = None
    actions: list[str] = field(default_factory=list)


def publish(
    target: str,
    *,
    dry_run: bool = True,
    remote: str | None = None,
    gate: bool = True,
    verify_build: bool = True,
    build_timeout: int = 1800,
    package_id: str | None = None,
    artifacts_root: str | Path | None = None,
    config: str | Path | None = None,
    branch: str | None = None,
    confirm_push: str | None = None,
) -> PublishResult:
    """Publish the champion of ``target`` as its own repo. Dry-run by default (no git/network).

    The gate refuses an uncertified champion unless ``gate=False`` (a loud warning is emitted).
    A real publish clones the resolved remote, checks out the resolved BRANCH (the frozen baseline ->
    ``baseline``; a champion -> ``stable/<package_id>``; overridable), replaces its tree with the
    assembled repo, commits (message = provenance), tags ``v<version>-<package_id>``, and pushes to
    ``refs/heads/<branch>`` — idempotently PER BRANCH. Each real publish event is recorded via
    :func:`merlin.common.artifacts.new_product`."""
    if type(build_timeout) is not int or build_timeout <= 0:
        raise PublishError("build_timeout must be a positive integer")
    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    resolved_remote = resolve_remote(target, config=config, override=remote)
    resolved_branch = resolve_branch(sel, override=branch, config=config)
    gate_ok, gate_detail = _check_gate(sel)
    if gate and not gate_ok:
        raise PublishError(f"publish gate refused for {target}/{sel.package_id}: {gate_detail}")
    if not gate and not gate_ok:
        sys.stderr.write(f"WARNING: --no-gate publishing UNCERTIFIED {target}/{sel.package_id}: {gate_detail}\n")

    stage_root = _new_stage_root(target, sources=(sel.package_dir,))
    repo_dir = stage_root / "repo"
    source_payload = package_records.payload_inventory(sel.package_dir)
    manifest = assemble_repo_tree(sel, repo_dir, layout_version=LAYOUT_VERSION)
    embed_provenance(repo_dir, sel)
    if package_records.payload_inventory(sel.package_dir) != source_payload:
        raise PublishError("package inputs changed during export")
    if gate and not _check_gate(sel)[0]:
        raise PublishError("source certification changed during export")
    export_identity_path = stage_root / "export_identity.json"
    write_pretty_json(
        export_identity_path,
        {
            "version": 1,
            "target": target,
            "package_slot": sel.package_id,
            "compiler_package_id": manifest.get("package_id"),
            "source_payload": source_payload,
            "source_input_certification": "pass" if gate_ok else "unverified",
            "scope": "package-payload",
            "external_dependency_closure": "not-attested",
            "exported_payload": package_records.payload_inventory(repo_dir),
            "transformation": {
                "layout_version": LAYOUT_VERSION,
                "kind": "additive-publication-metadata",
                "provider_role": _export_role(sel).value,
                "preserved_source_layout": True,
            },
            "exported_payload_certification": "unverified",
        },
    )

    merlin_sha = _git_sha_full()
    cert_run = _cert_run_id(sel)
    fingerprint = _fingerprint(sel.package_id, merlin_sha, cert_run, source_payload["sha256"])
    version = int(manifest.get("version", sel.version) or 0)
    tag = f"v{version}-{sel.package_id}"

    result = PublishResult(
        target=target,
        package_id=sel.package_id,
        remote=resolved_remote,
        dry_run=dry_run,
        gate_ok=gate_ok,
        gate_detail=gate_detail,
        fingerprint=fingerprint,
        tag=tag,
        repo_dir=repo_dir,
        export_identity_path=export_identity_path,
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
        write_pretty_json(
            stage_root / "build_verification.json",
            {
                "version": 1,
                "status": "not-run",
                "reason": "dry-run",
                "requested": verify_build,
            },
        )
        result.build_verification_path = stage_root / "build_verification.json"
        result.actions.insert(0, "DRY-RUN (no clone/commit/push)")
        return result

    result.build_verification_path = stage_root / "build_verification.json"
    if _export_role(sel).value == "candidate_compiler" and verify_build:
        from .publication_verification import verify_export

        verification = verify_export(repo_dir, stage_root, timeout=build_timeout)
        if verification["status"] != "pass":
            raise PublishError(f"build verification failed; evidence: {result.build_verification_path}")
        result.actions.append(
            "build-artifact presence verified on retained copy; numerical certification not performed"
        )
    else:
        write_pretty_json(
            result.build_verification_path,
            {
                "version": 1,
                "status": "not-run",
                "reason": "not-requested" if not verify_build else "host-schedule",
            },
        )

    # human diff-confirm gate before any real network push (local/file remotes are exempt).
    _require_push_confirmation(resolved_remote, repo_dir, resolved_branch, fingerprint, confirm_push)
    result.actions.append(
        f"push confirmed for non-local remote (fingerprint {fingerprint})"
        if _needs_push_confirmation(resolved_remote)
        else "local remote (no confirm)"
    )

    commit_sha, published_tag, noop = _git_publish(
        resolved_remote, repo_dir, sel, manifest, fingerprint, cert_run, stage_root, resolved_branch
    )
    result.committed = not noop
    result.noop = noop
    result.commit_sha = commit_sha
    result.tag = published_tag
    result.actions.append(
        "no-op (remote branch fingerprint already matches)"
        if noop
        else f"committed {commit_sha} + tag {published_tag}, pushed to remote"
    )

    # record the publish event as a versioned product
    prod = new_product(
        "publish",
        version=1,
        target=target,
        notes=f"publish {sel.package_id} -> {resolved_remote} ({'noop' if noop else 'committed'})",
    )
    event = {
        "target": target,
        "package_id": sel.package_id,
        "remote": resolved_remote,
        "branch": resolved_branch,
        "commit_sha": commit_sha,
        "tag": published_tag,
        "noop": noop,
        "fingerprint": fingerprint,
        "merlin_git_sha": merlin_sha,
        "cert_run": cert_run,
        "gate_ok": gate_ok,
        "gate_detail": gate_detail,
        "actions": result.actions,
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
    if res.export_identity_path:
        print(f"export identity: {res.export_identity_path}")
    if res.build_verification_path:
        print(f"build verification: {res.build_verification_path}")
    if res.product_dir:
        print(f"event recorded at {res.product_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-target-publish", description="Publish a target's certified champion as its own repo (WS-E)."
    )
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
    p_pub.add_argument("--no-verify-build", action="store_true", help="explicitly skip build-artifact verification")
    p_pub.add_argument("--build-timeout", type=int, default=1800, help="timeout in seconds per declared build step")
    p_pub.add_argument(
        "--confirm-push",
        help="content fingerprint confirming a real push to a non-local "
        "remote (printed when refused); required for a GitHub/network push",
    )

    p_prom = sub.add_parser("promote", help="mark a package the single champion for a target")
    p_prom.add_argument("--target", required=True)
    p_prom.add_argument("--champion", "--package", dest="champion", required=True)
    p_prom.add_argument("--artifacts-root")
    p_prom.add_argument("--no-gate", action="store_true")

    p_cert = sub.add_parser("record-cert", help="record certification verdicts outside the package payload")
    p_cert.add_argument("--target", required=True)
    p_cert.add_argument("--champion", "--package", dest="champion", required=True)
    p_cert.add_argument(
        "--results", nargs="+", required=True, help="one or more certify run dirs (or results.yaml paths)"
    )
    p_cert.add_argument("--artifacts-root")
    p_mat = sub.add_parser("materialize", help="install a run's submission as the target's OOT backend package")
    p_mat.add_argument("--target", required=True)
    p_mat.add_argument(
        "--from",
        dest="source",
        required=True,
        help="the run's submission/ directory (manifest.yaml + the backend tree)",
    )
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
            res = publish(
                args.target,
                dry_run=not args.execute,
                remote=args.remote,
                gate=not args.no_gate,
                package_id=args.champion,
                artifacts_root=args.artifacts_root,
                config=args.config,
                branch=args.branch,
                confirm_push=args.confirm_push,
                verify_build=not args.no_verify_build,
                build_timeout=args.build_timeout,
            )
            _print_result(res)
            return 0
        if args.cmd == "index":
            res = publish_index(
                args.target,
                dry_run=not args.execute,
                remote=args.remote,
                config=args.config,
                artifacts_root=args.artifacts_root,
                branch=args.branch,
                confirm_push=args.confirm_push,
            )
            _print_result(res)
            return 0
        if args.cmd == "record-cert":
            pub = record_certification(args.target, args.champion, args.results, artifacts_root=args.artifacts_root)
            tier = pub.get("certification_tier") or {}
            print(
                f"certification={pub.get('certification')} "
                f"rungs={len(pub.get('certified_rungs') or [])} "
                f"derived_from_rtl={tier.get('derived_from_rtl')} "
                f"cycle_accurate={tier.get('cycle_accurate')} "
                f"oracles={','.join(tier.get('oracles') or [])}"
            )
            return 0 if pub.get("certification") == "pass" else 1

        if args.cmd == "materialize":
            dst = materialize_package(
                args.target,
                args.source,
                package_id=args.package_id,
                certified_by_run=args.certified_by_run,
                score_path=args.score,
                artifacts_root=args.artifacts_root,
                force=args.force,
            )
            print(f"installed {args.target} backend -> {dst}")
            return 0
        if args.cmd == "promote":
            promote(args.target, args.champion, gate=not args.no_gate, artifacts_root=args.artifacts_root)
            print(f"promoted {args.target}/{args.champion} to champion")
            return 0
        if args.cmd == "inspect":
            sel = select_champion(args.target, artifacts_root=args.artifacts_root, package_id=args.champion)
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
