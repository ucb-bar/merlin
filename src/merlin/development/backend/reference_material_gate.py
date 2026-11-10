"""Read-only reference-material gate over one rendered author workspace, and a cross-arm diff.

Checks, each reported separately:

* ``allow_list``: every file under ``reference-kernels/`` is a pinned original (the reviewed
  reference manifest, or a bank arm's own ``bank.json`` raw list), besides its two staging records;
  ``cca-evidence/`` exists only for a pinned bank arm; no other reference-named input is rendered.
* ``provenance``: derived reference artifacts carry ``generated_by`` (tool, version, input hashes).
  A ``merlin.mining.reference_subsets`` bank records the equivalent in ``bank.json`` (its schema
  names the producer and version, ``raw[].sha256`` the inputs, ``files`` pins every output).
* ``profile_record``: AUTHOR_PROFILE.json's hashes (``raw_reference_sha256``, or its pinned bank's
  raw list) cover every granted reference file.
* ``tripwire``: no rendered input is a historical reference map (by bytes or hash mention), carries
  the reference-map schema, a ``techniques`` field in reference-derived JSON, or the map's
  ``weight reuse:`` / ``COMPUTE_AND_STAY`` annotations outside the raw originals themselves.

Rendered inputs are the workspace's top-level entries apart from author-owned outputs. In a used
workspace, entries changed after the earliest sandbox launch record beside it are session work
(including the broker clients a session stages).
The content checks read ``reference-kernels/``, ``cca-evidence/`` and the top-level rendered files;
other rendered directories are listed (and diffed) but read only with ``deep=True``.
Nothing is written into a workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from merlin.development.backend import author_profiles as A
from merlin.development.backend import fresh10_profiles as F10

SCHEMA = "merlin.reference_material_gate.v1"
RAW = "reference-kernels"
BANK = A.BANK_GRANT
RECORDS = A._REFERENCE_RECORDS
BANK_SCHEMA = "merlin.reference_subset_bank.v1"
#: The historical map's per-entry annotations ("weight reuse: N of M computes are COMPUTE_AND_STAY").
#: A bare COMPUTE_AND_STAY is an ordinary instruction name of the selected RTL facts, target contract
#: and a bank's decoded records, so only its annotation phrasing trips.
TRIPWIRE_TEXT = (b"weight reuse:", b"computes are COMPUTE_AND_STAY")
#: Top-level names an author or the runtime writes; never rendered reference inputs.
AUTHOR_OWNED = frozenset(
    (
        "submission",
        "artifacts",
        "variants",
        "selfcheck_out",
        "qa",
        ".git",
        ".compiler-git",
        ".codex",
        ".codex_out",
        ".qa_channel",
        ".isa_channel",
        ".model_check_channel",
        ".aws",
    )
)
#: Reference containers by name (the author-profile owner's set) other than the two governed grants.
_REFERENCE_NAMES = frozenset(A._EXPERT_REFERENCE_COMPONENTS - {RAW, BANK, "seed_performance"})
#: Staged compiler source copies inside rendered inputs (a seed's variants); compiler sources, not
#: reference material, so the tripwire walk leaves them out and reports how many it left out.
COMPILER_TREES = frozenset(("compiler_library", ".git", "__pycache__"))
_CONTENT_LIMIT = 64 << 20
#: Rendered inputs a capability flag explains when only some arms have them (cross-arm diff).
EXPLAINED = {
    BANK: "reference_analysis",
    RAW: "raw_kernel_references",
    **dict.fromkeys(
        (
            "OPTIMIZATION_API_SELECTION.json",
            "OPTIMIZATION_BY_OPERATION.json",
            "OPTIMIZATION_CAPABILITIES.json",
            "OPTIMIZATION_CATALOG.json",
            "OPTIMIZATION_SUGGESTIONS.json",
            "SELECTED_LIBRARY_GUIDANCE.md",
            "optimization_suggest.py",
        ),
        "transform_library",
    ),
    **dict.fromkeys(
        (
            "SEED_EVIDENCE/",
            "SEED_EXPERIMENTS/",
            "SEED_INDEX.md",
            "SEED_NOTES/",
            "SEED_PERFORMANCE/",
            "SEED_VARIANTS/",
            "PREVIOUS_SESSION.md",
        ),
        "seeded",
    ),
}


def _digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _launch_cutoff(workspace):
    """mtime of the earliest sandbox launch record beside the workspace; None before any launch.

    Rendering finishes before the first sandbox starts; later sessions rewrite their records, so
    the newest one would count earlier author work as input.
    """
    times = [path.stat().st_mtime for path in workspace.parent.glob(".workspace.bwrap-args.*") if path.is_file()]
    return min(times) if times else None


def _rendered(workspace, cutoff):
    """Top-level rendered inputs: name -> kind; plus the author-written names left out."""
    rendered, later = {}, []
    for entry in sorted(workspace.iterdir()):
        if entry.name in AUTHOR_OWNED:
            continue
        link, directory = entry.is_symlink(), entry.is_dir()
        if cutoff is not None and not link:
            times = [entry.lstat().st_mtime]
            if directory:
                times += [child.lstat().st_mtime for child in entry.iterdir()]
            if min(times) > cutoff:
                later.append(entry.name)
                continue
        rendered[entry.name] = "link" if link else "dir" if directory else "file"
    return rendered, later


def _files(root, cutoff=None, prune=None):
    """Regular files under ``root`` (not following links); indirect members are returned apart.

    ``prune`` (a list) collects the staged compiler source trees left out of the walk.
    """
    found, indirect = [], []
    for directory, names, files in os.walk(root):
        if prune is not None:
            for name in [name for name in names if name in COMPILER_TREES]:
                names.remove(name)
                prune.append(Path(directory) / name)
        for name in [*names, *files]:
            path = Path(directory) / name
            if path.is_symlink():
                indirect.append(path)
            elif name in files and (cutoff is None or path.lstat().st_mtime <= cutoff):
                found.append(path)
    return sorted(found), sorted(indirect)


def _manifest_hashes(document):
    if document.get("schema") == "operator.allowed_reference_sources.v2":
        return [member["sha256"] for origin in document.get("origins", ()) for member in origin.get("files", ())]
    return [member["sha256"] for member in document.get("members", ()) if "sha256" in member]


def _keys(document):
    stack = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            yield from value
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def _profile(workspace, failures):
    path = workspace / A.RECORD
    if not path.is_file():
        return None, None
    record = json.loads(path.read_bytes())
    try:
        return record, A.resolve(record.get("name"))
    except ValueError:
        failures.append(dict(check="profile_record", path=A.RECORD, reason="unknown_author_profile"))
        return record, None


class _Gate:
    def __init__(self, workspace, *, manifest=None, bank=None, launched=True, deep=False):
        self.workspace, self.launched, self.deep = Path(workspace).absolute(), launched, deep
        self.manifest, self.bank = (
            (None if manifest is None else Path(manifest)),
            (None if bank is None else Path(bank)),
        )
        self.failures, self.provenance, self.scanned, self.not_scanned, self.pruned = [], [], 0, [], []

    def fail(self, check, path, reason, **details):
        self.failures.append(dict(check=check, path=str(path), reason=reason, **details))

    def run(self):
        ws = self.workspace
        self.record, self.profile = _profile(ws, self.failures)
        self.cutoff = _launch_cutoff(ws) if self.launched else None
        self.rendered, later = _rendered(ws, self.cutoff)
        granted = self._allow_list()
        self._bank_provenance()
        self._profile_record(granted)
        self._tripwire()
        flags = dict((self.record or {}).get("capabilities") or {}, seeded=bool((self.record or {}).get("seeded")))
        return dict(
            schema=SCHEMA,
            workspace=str(ws),
            status="fail" if self.failures else "pass",
            author_profile=None if self.record is None else self.record.get("name"),
            capabilities=flags,
            allowed_reference_source=self.source,
            granted_reference_files=sum(granted.values()),
            launch_cutoff=self.cutoff,
            author_written_top_level=sorted(later),
            rendered_inputs=sorted(
                name + ("/" if kind != "file" and (ws / name).is_dir() else "") for name, kind in self.rendered.items()
            ),
            scanned_files=self.scanned,
            deep=self.deep,
            directories_not_content_scanned=sorted(self.not_scanned),
            compiler_trees_left_out=len(self.pruned),
            provenance=self.provenance,
            failures=self.failures,
        )

    # (a) allow-list -----------------------------------------------------------------------------
    def _bank_root(self):
        staged = self.workspace / BANK
        pin = None if self.profile is None else self.profile.bank_sha256
        if self.bank is not None:
            selected = _digest(self.bank / "bank.json")
            if pin is not None and selected != pin:
                self.fail("allow_list", self.bank / "bank.json", "selected_bank_differs_from_profile_pin")
            pin = selected
        if not staged.exists():
            if pin is not None:
                self.fail("allow_list", BANK, "pinned_bank_not_rendered")
            return None
        if pin is None:
            self.fail("allow_list", BANK, "bank_rendered_without_a_pinned_bank_arm")
            return None
        if not (staged / "bank.json").is_file() or _digest(staged / "bank.json") != pin:
            self.fail("allow_list", BANK + "/bank.json", "bank_differs_from_its_pin", pin=pin)
            return None
        return staged.resolve()

    def _allow_list(self):
        ws, profile = self.workspace, self.profile
        for name in self.rendered:
            folded = name.casefold()
            if name not in (RAW, BANK) and ("reference" in folded or folded in _REFERENCE_NAMES):
                self.fail("allow_list", name, "reference_derived_input")
        self.bank_root = self._bank_root()
        pinned = None
        if self.manifest is not None:
            pinned = Counter(_manifest_hashes(json.loads(self.manifest.read_bytes())))
        self.source, expected = "none", None
        if self.bank_root is not None:
            try:
                expected = Counter(A.bank_raw_sha256(self.bank_root))
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.fail("allow_list", BANK + "/bank.json", "bank_raw_list_invalid", error=str(error))
                expected = Counter()
            self.source = "bank raw list"
            if pinned is not None and expected - pinned:
                self.fail("allow_list", BANK + "/bank.json", "bank_raw_list_outside_pinned_manifest")
            if profile is not None and profile.raw_originals and expected != Counter(F10.COVERAGE_RAW_REFERENCE_SHA256):
                self.fail("allow_list", BANK + "/bank.json", "raw_originals_bank_is_not_all_originals")
        elif pinned is not None:
            expected, self.source = pinned, "pinned manifest " + str(self.manifest)
        elif profile is not None and (profile.coverage or profile.raw_originals):
            expected, self.source = Counter(F10.COVERAGE_RAW_REFERENCE_SHA256), "profile pinned originals"
        elif profile is not None and not profile.raw_kernel_references:
            expected, self.source = Counter(), "no-reference profile"
        root = ws / RAW
        if not root.exists():
            if expected:
                self.fail("allow_list", RAW, "granted_references_not_rendered")
            self.originals = frozenset()
            return Counter()
        found, indirect = _files(root.resolve())
        for path in indirect:
            self.fail("allow_list", path, "indirect_reference_member")
        records = {name: root.resolve() / name for name in RECORDS}
        granted = Counter()
        hashes = {}
        for path in found:
            if path in records.values():
                continue
            hashes[path] = digest = _digest(path)
            granted[digest] += 1
        if expected is None:
            staged = records["reference-manifest.json"]
            expected = Counter(_manifest_hashes(json.loads(staged.read_bytes()))) if staged.is_file() else Counter()
            self.source = "staged reference-manifest.json (self-consistency only)"
        for path, digest in hashes.items():
            if granted[digest] > expected[digest]:
                self.fail(
                    "allow_list", path.relative_to(root.resolve()).as_posix(), "unlisted_reference_file", sha256=digest
                )
        if expected - granted:
            self.fail("allow_list", RAW, "pinned_reference_file_missing", missing=sorted(expected - granted))
        staged = records["reference-manifest.json"]
        if not staged.is_file() or Counter(_manifest_hashes(json.loads(staged.read_bytes()))) != granted:
            self.fail("allow_list", RAW + "/reference-manifest.json", "staged_manifest_differs_from_granted_files")
        self.originals = frozenset(digest for digest in granted if expected[digest])
        return granted

    # (b) provenance -----------------------------------------------------------------------------
    def _bank_provenance(self):
        if self.bank_root is None:
            return
        bank = json.loads((self.bank_root / "bank.json").read_bytes())
        row = dict(artifact=BANK + "/bank.json", generated_by=bank.get("generated_by"))
        try:
            from merlin.mining.reference_subsets import verify_bank

            verify_bank(self.bank_root)
        except ImportError:
            self.fail("provenance", BANK, "bank_verification_requires_merlin_mining")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.fail("provenance", BANK, "bank_pins_do_not_verify", error=str(error))
        raw = bank.get("raw") or []
        if row["generated_by"] is None:
            # The bank's own equivalent: producer schema/version, input hashes and pinned outputs.
            row["equivalent"] = dict(
                field="bank.json schema/raw/files",
                tool=bank.get("schema"),
                version=str(bank.get("schema", "")).rpartition(".")[2],
                inputs=len(raw),
                outputs=len(bank.get("files") or {}),
            )
            if (
                bank.get("schema") != BANK_SCHEMA
                or not raw
                or any(not F10._sha256((item or {}).get("sha256")) for item in raw)
            ):
                self.fail("provenance", BANK + "/bank.json", "bank_without_generator_record")
        else:
            self._generated_by(BANK + "/bank.json", bank)
        self.provenance.append(row)

    def _generated_by(self, path, document):
        record = document.get("generated_by")
        inputs = record.get("inputs", record.get("input_sha256")) if isinstance(record, dict) else None
        hashes = list(inputs.values()) if isinstance(inputs, dict) else inputs
        if (
            not isinstance(record, dict)
            or not record.get("tool")
            or not record.get("version")
            or not isinstance(hashes, list)
            or not hashes
            or not all(map(F10._sha256, hashes))
        ):
            self.fail("provenance", path, "hand_authored_derived_artifact")
            return False
        self.provenance.append(dict(artifact=str(path), generated_by=record))
        return True

    # (c) profile record ---------------------------------------------------------------------------
    def _profile_record(self, granted):
        record = self.record
        if record is None:
            if granted:
                self.fail("profile_record", A.RECORD, "no_profile_record_for_granted_references")
            return
        covered = Counter(record.get("raw_reference_sha256") or ())
        if record.get("cca_bank_sha256"):
            if self.bank_root is None or _digest(self.bank_root / "bank.json") != record["cca_bank_sha256"]:
                self.fail("profile_record", A.RECORD, "recorded_bank_is_not_the_rendered_bank")
            else:
                try:
                    covered |= Counter(A.bank_raw_sha256(self.bank_root))
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # Already reported by the allow-list.
        if not (record.get("capabilities") or {}).get("raw_kernel_references", True) and granted:
            self.fail("profile_record", A.RECORD, "no_reference_profile_with_granted_references")
        uncovered = granted - covered
        if uncovered:
            self.fail(
                "profile_record",
                A.RECORD,
                "record_omits_granted_reference_files",
                count=sum(uncovered.values()),
                examples=sorted(uncovered)[:4],
            )
        if self.profile is not None and self.profile.record() != {
            key: value for key, value in record.items() if key != "source_selection"
        }:
            self.provenance.append(dict(artifact=A.RECORD, note="differs from the current profile record"))

    # (e) tripwire ---------------------------------------------------------------------------------
    def _tripwire(self):
        ws = self.workspace
        for name, kind in sorted(self.rendered.items()):
            path = ws / name
            mount = name in (RAW, BANK)
            if kind == "file" or (kind == "link" and path.is_file()):
                self._scan(path, name, mount=False, structural=False)
                continue
            if not path.is_dir():
                continue
            if not mount and not self.deep:
                # Default scope: the reference grants and top-level rendered files only; other
                # rendered directories (corpus, seed copies, toolchains) are listed, not read.
                self.not_scanned.append(name)
                continue
            root = path.resolve() if mount or kind == "link" else path
            files, _indirect = _files(
                root, None if mount or kind == "link" else self.cutoff, None if mount else self.pruned
            )
            if kind == "link" and not mount:
                self.not_scanned.append(name)
            for member in files:
                relative = name + "/" + member.relative_to(root).as_posix()
                self._scan(member, relative, mount=mount, structural=kind == "link" and not mount)

    def _scan(self, path, relative, *, mount, structural):
        """``structural``: an operator-corpus link is checked for map JSON only, not hashed whole."""
        if structural and path.suffix != ".json":
            return
        size = path.stat().st_size
        if size > _CONTENT_LIMIT:
            if mount and relative.startswith(RAW + "/"):
                return  # A pinned large original was hashed by the allow-list.
            if _digest(path) in F10.WITHHELD_REFERENCE_MAP_SHA256:
                self.fail("tripwire", relative, "withheld_reference_map")
            return
        data = path.read_bytes()
        self.scanned += 1
        digest = hashlib.sha256(data).hexdigest()
        if digest in F10.WITHHELD_REFERENCE_MAP_SHA256:
            self.fail("tripwire", relative, "withheld_reference_map", sha256=digest)
        mentioned = sorted(pin for pin in F10.WITHHELD_REFERENCE_MAP_SHA256 if pin.encode() in data)
        if mentioned and digest not in F10.WITHHELD_REFERENCE_MAP_SHA256:
            self.fail("tripwire", relative, "mentions_withheld_reference_map", sha256=mentioned)
        original = mount and relative.startswith(RAW + "/") and digest in self.originals
        if not original:
            hits = [text.decode() for text in TRIPWIRE_TEXT if text in data]
            if hits:
                self.fail("tripwire", relative, "reference_map_annotation", text=hits)
        if original or (path.suffix != ".json" and data.lstrip()[:1] not in (b"{", b"[")):
            return
        try:
            document = json.loads(data)
        except (UnicodeDecodeError, ValueError):
            return
        if not isinstance(document, (dict, list)):
            return
        from merlin.development.backend.feedback.reference_map import SCHEMA as MAP_SCHEMA

        schema = document.get("schema") if isinstance(document, dict) else None
        if schema == MAP_SCHEMA:
            self.fail("tripwire", relative, "reference_map_schema")
        derived = (
            mount
            or "reference" in relative.casefold()
            or (isinstance(schema, str) and "reference" in schema.casefold())
            or (isinstance(document, dict) and "derived_from" in document)
        )
        if derived and "techniques" in set(_keys(document)):
            self.fail("tripwire", relative, "techniques_field_in_reference_derived_json")
        # A generated map or reference analysis must name its generator; bank members are pinned
        # by bank.json and the two staging records are the reviewed manifest and its selection.
        generated = isinstance(document, dict) and (
            schema == MAP_SCHEMA or "derived_from" in document or relative.startswith(RAW + "/")
        )
        if generated and Path(relative).name not in RECORDS:
            self._generated_by(relative, document)


def check_workspace(workspace, *, manifest=None, bank=None, launched=True, deep=False):
    """Gate one rendered workspace read-only; returns the report (``status`` pass/fail).

    ``launched=False`` (preflight) treats every top-level entry as rendered input. ``deep`` also
    reads every other rendered directory (operator-corpus links for map JSON only).
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        raise ValueError("reference material gate requires a workspace directory: " + str(workspace))
    return _Gate(workspace, manifest=manifest, bank=bank, launched=launched, deep=deep).run()


def diff_arms(reports):
    """Rendered top-level inputs per arm; names only some arms have that no capability flag explains."""
    arms = {report["workspace"]: report for report in reports}
    names = sorted({name for report in reports for name in report["rendered_inputs"]})
    flagged, explained = [], []
    for name in names:
        present = {arm: name in report["rendered_inputs"] for arm, report in arms.items()}
        if all(present.values()):
            continue
        flag = EXPLAINED.get(name) or EXPLAINED.get(name.rstrip("/"))
        values = {arm: None if flag is None else bool(report["capabilities"].get(flag)) for arm, report in arms.items()}
        row = dict(input=name, present=present, flag=flag, flag_values=values)
        (explained if flag is not None and values == present else flagged).append(row)
    return dict(
        schema=SCHEMA + "/cross-arm",
        arms={arm: report["rendered_inputs"] for arm, report in arms.items()},
        explained=explained,
        unexplained=flagged,
        status="fail" if flagged else "pass",
    )


def check_prepared(prepared):
    """Preflight continuation: gate the freshly rendered workspace of a versioned author profile."""
    if (getattr(prepared, "bundle", None) or {}).get("author_profile") is None:
        return None
    report = check_workspace(prepared.workspace, launched=False)  # Nothing is author work yet.
    (Path(prepared.run_dir) / "reference_material_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        reasons = sorted({row["reason"] + " " + row["path"] for row in report["failures"]})
        raise ValueError("reference material gate failed: " + "; ".join(reasons[:8]))
    return dict(
        status=report["status"],
        granted_reference_files=report["granted_reference_files"],
        allowed_reference_source=report["allowed_reference_source"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("workspace", nargs="?", type=Path)
    parser.add_argument("--manifest", type=Path, help="pinned reference-manifest.json (v2)")
    parser.add_argument("--bank", type=Path, help="the bank arm's pinned bank directory (holding bank.json)")
    parser.add_argument("--deep", action="store_true", help="also read every other rendered directory")
    parser.add_argument(
        "--diff", type=Path, nargs="+", metavar="REPORT", help="cross-arm diff of earlier per-workspace gate reports"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if (args.workspace is None) == (args.diff is None):
        parser.error("give one workspace or --diff REPORT...")
    if args.diff:
        report = diff_arms([json.loads(path.read_bytes()) for path in args.diff])
    else:
        report = check_workspace(args.workspace, manifest=args.manifest, bank=args.bank, deep=args.deep)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(text)
    summary = {
        key: report[key]
        for key in ("status", "workspace", "allowed_reference_source", "granted_reference_files")
        if key in report
    }
    summary["failures"] = sorted({row["check"] + ":" + row["reason"] for row in report.get("failures", ())})
    if "unexplained" in report:
        summary["unexplained"] = [row["input"] for row in report["unexplained"]]
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
