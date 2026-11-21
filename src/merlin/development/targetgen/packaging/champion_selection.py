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

from merlin.common import paths
from merlin.common.artifacts import git_sha7, new_product, utc_stamp
from merlin.common.digest import canonical_sha256, write_pretty_json
from merlin.common.record_formats import dump_yaml, load_yaml
from merlin.development.targetgen.packaging import records as package_records

# ---- publish -----------------------------------------------------------------------------------

LAYOUT_VERSION = "3.0"


_STATUS_RANK = {"rtl_certified": 0, "k1_verified": 1, "spike_verified": 2}


_DEFAULT_STATUS_RANK = 3


CERTIFIED_STATUSES = frozenset(_STATUS_RANK)


_BASELINE_PACKAGE_IDS = frozenset({"hand_v0", "hand_v0_int8"})


BASELINE_BRANCH = "baseline"


class PublishError(RuntimeError):
    """A publish/promote step could not proceed (selection, gate, config, or git)."""


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

    def __lt__(self, other: _InvStr) -> bool:
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
    > ``experiments/inputs/target-inputs/publish.yaml``. Never hardcoded."""
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


def _is_baseline(sel: ChampionSelection) -> bool:
    """The frozen unoptimized control (by known package_id or a manifest opt-in flag)."""
    if sel.package_id in _BASELINE_PACKAGE_IDS:
        return True
    pub = sel.manifest.get("publication")
    return isinstance(pub, dict) and pub.get("role") == "baseline"


def resolve_branch(sel: ChampionSelection, *, override: str | None = None, config: str | Path | None = None) -> str:
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


def _readme(sel: ChampionSelection, manifest: dict[str, Any], champion: dict[str, Any] | None = None) -> str:
    """The published repo's landing page.

    A package published with ``--no-gate`` says so HERE, at the top, in the reader's first
    paragraph. Suppressing the refusal to stderr and shipping the ordinary "certified champion"
    wording is how an uncertified package gets cited as a certified one: the warning is seen by
    the operator who already knows, and never by the person who clones the repo.

    ``champion`` is what :func:`merlin.development.targetgen.packaging.champions.export_champion` passes once a champion's
    records have passed its phase's evidence profile: ``{phase, profile, evidence}``. The page then
    leads with that evidence instead of the payload gate's verdict. That gate (``oot_runner.certify``
    rungs bound to the payload) is a different, optional certification, and the page still names it.
    """
    from merlin.development.targetgen.packaging.publication_trees import _check_gate, _export_role

    merlin_sha = git_sha7()
    pub = sel.manifest.get("publication") or {}
    gate_ok, gate_detail = _check_gate(sel)
    lines = [
        f"# {sel.repo_name}",
        "",
        f"Out-of-tree Merlin codegen export for **{sel.target}** (family `{sel.family or 'unknown'}`).",
        "",
    ]
    if champion is not None:
        lines += [
            f"> **Phase-{champion['phase']} champion.** Merlin's champion export checked this package against "
            f"its phase-{champion['phase']} evidence profile, and it passed:",
            ">",
            *(f"> - {item}" for item in champion["evidence"]),
            ">",
            "> The evidence is in `.merlin/provenance.json`, `certification.json`, `measurements.json` and "
            "`isa_prohibition.json`.",
            "",
        ]
    elif not gate_ok:
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
            "This repository is **generated** by Merlin's champion export through the `merlin-target-publish` "
            "bridge. The package payload at the root is byte-identical to the bytes the evidence above was "
            "taken on; Merlin adds only this page and `.merlin/`. External dependency closure is "
            "**not-attested**: the evidence is about these bytes in the environment its records name."
            if champion is not None
            else "This repository is **generated** by Merlin's `merlin-target-publish` bridge. The "
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
        f"- {'Champion package' if gate_ok or champion is not None else 'Package'}: `{sel.package_id}`",
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
    if champion is not None:
        lines += [
            "## Provenance",
            "",
            f"- Champion evidence profile: `{champion['profile']}`, passed (the evidence is listed above)",
            "- Package-payload certification by `oot_runner.certify` rungs (a separate gate): "
            f"`{pub.get('certification', sel.cert_status or 'not recorded')}`",
            "- External dependency closure: `not-attested`",
            f"- Fingerprint: `{pub.get('fingerprint', 'n/a')}`",
            "",
            "See `.merlin/provenance.json` for the full lineage and `.merlin/provenance.yaml` and "
            "`.merlin/certification.yaml` for the publish layer's own records.",
            "",
        ]
        return "\n".join(lines)
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
    from merlin.development.targetgen.packaging.publication_trees import _check_gate, _export_role

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
