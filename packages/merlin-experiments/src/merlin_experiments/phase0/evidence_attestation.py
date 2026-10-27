"""ATTESTED, not proven: operator-authorized reuse of shards across commits.

Identity controls bind the retained shards and current stamp. Trace summaries and
AST importers are informational only; they cannot establish producer independence
or numerical equivalence. No import closure or trace completeness gate is used.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.phase0.evidence_attestation.v1"
TRACE_SCHEMA = "merlin.phase0.runtime_trace.v1"
RECEIPT_SCHEMA = "merlin.phase0.evidence_attestation.receipt.v1"
RUNNER = "build_tools/scripts/run_bounded_core_aten_suite.py"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
COMMIT_LABELS = {"merlin_commit", "merlin_source_label", "merlin"}
RESIDUAL_RISKS = [
    "A matching trace SHA does not prove the trace is complete: the trace is operator-authored and is not bound to P, the saved provenance, the suite, the selected cases or child processes.",
    "Script and sibling imports escape module resolution: the stamped capture worker inserts its own directory into sys.path and imports helpers by bare name, which static resolution does not model.",
    "Literal import aliases and builtin wildcard imports (importlib.import_module aliases, builtins.__import__, __import__ with fromlist ['*']) are not followed.",
    "Dependencies other than imports are unchecked: a traced producer that reads an allowed file as data (for example source bytes hashed for capture-cache selection) can change shard production undetected.",
    "Files outside the repository are excluded from any checks recorded here.",
    "A lazy import chain through code paths none of the traced runs exercised is not excluded.",
    "The sealed m2m capture subprocess has no runtime trace; only its entry file is statically rooted.",
    "The receipt records that shards produced at P are reused at C; it does not establish numerical equivalence of C.",
]


class AttestationRefused(ValueError):
    """An attestation failed an identity, input or byte-integrity control."""


def require(condition, reason):
    if not condition:
        raise AttestationRefused(reason)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique_object(pairs):
    document = {}
    for key, value in pairs:
        require(key not in document, f"duplicate attestation key: {key}")
        document[key] = value
    return document


def _commit(value):
    return isinstance(value, str) and len(value) == 40 and all(c.lower() in "0123456789abcdef" for c in value)


def _relative(value):
    require(isinstance(value, str) and bool(value), "allow paths must be nonempty strings")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and all(part not in {"", ".", ".."} for part in value.split("/"))
        and not any(c in value for c in "*?[]\\:") and not any(ord(c) < 32 for c in value),
        f"invalid repository-relative literal path: {value}",
    )
    return value


def load_attestation(path):
    raw = Path(path).read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object)
    require(isinstance(document, dict), "attestation must be an object")
    required = {"schema", "shard_commit", "allow", "attested_by", "submitted_by", "statement", "evidence"}
    require(required <= document.keys(), "attestation is missing required keys")
    require(set(document) == required, "attestation has unknown keys")
    require(document["schema"] == SCHEMA, f"attestation schema must be {SCHEMA}")
    require(_commit(document["shard_commit"]), "shard_commit must be a 40-hex commit")
    allow = document["allow"]
    require(isinstance(allow, list) and bool(allow), "attestation allow must be a nonempty list")
    for path in allow:
        _relative(path)
    require(len(allow) == len(set(allow)), "attestation allow contains duplicate paths")
    for key in ("attested_by", "submitted_by", "statement"):
        require(isinstance(document[key], str) and bool(document[key].strip()), f"{key} is empty")
    evidence = document["evidence"]
    require(isinstance(evidence, dict) and set(evidence) == {"runtime_trace", "notes"},
            "evidence must contain exactly runtime_trace and notes")
    _trace_reference(evidence["runtime_trace"])
    require(isinstance(evidence["notes"], list) and all(isinstance(note, str) for note in evidence["notes"]),
            "evidence notes must be a list of strings")
    return document, sha256(raw)


def _trace_reference(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
            "runtime_trace must contain exactly path and sha256")
    path, digest = reference["path"], reference["sha256"]
    require(isinstance(path, str) and Path(path).is_absolute(), "runtime_trace path must be absolute")
    require(isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest), "runtime_trace sha256 must be 64-hex")
    return Path(path), digest


def _summarize_trace(raw, repo, selected):
    summary = {"schema": TRACE_SCHEMA, "runs": {"capture": 0, "execute": 0},
               "distinct_py_file_count": 0, "repo_file_count": 0, "allowed_or_changed_files": [],
               "errors": []}
    files, repo_files = set(), set()
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object)
        require(isinstance(document, dict) and document.get("schema") == TRACE_SCHEMA,
                f"runtime_trace schema must be {TRACE_SCHEMA}")
        require(isinstance(document.get("runs"), list), "runtime_trace runs must be a list")
        for run in document["runs"]:
            require(isinstance(run, dict) and run.get("phase") in summary["runs"], "invalid trace phase")
            summary["runs"][run["phase"]] += 1
            require(isinstance(run.get("py_files"), list), "runtime_trace py_files must be a list")
            for source in run["py_files"]:
                require(isinstance(source, str) and Path(source).is_absolute() and source.endswith(".py"),
                        "runtime_trace entries must be absolute .py paths")
                path = Path(source)
                files.add(source)
                # Count lexical and resolved repository paths; neither is a dependency proof.
                for candidate in (path, path.resolve()):
                    if candidate.is_relative_to(repo):
                        repo_files.add(candidate.relative_to(repo).as_posix())
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        summary["errors"].append(str(exc))
    summary.update(distinct_py_file_count=len(files), repo_file_count=len(repo_files),
                   allowed_or_changed_files=sorted(repo_files & selected))
    return summary


class _Git:
    def __init__(self, repo, runner):
        self.repo = Path(repo).resolve()
        self.runner = runner or subprocess.run

    def read(self, *args):
        result = self.runner(
            ["git", "-C", str(self.repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.decode().strip()}")
        return result.stdout

    def paths(self, *args):
        return [p.decode("utf-8") for p in self.read(*args).split(b"\0") if p]


def _check_stamps(saved, current, attestation, git):
    require(isinstance(saved, dict) and isinstance(current, dict), "stamps must be objects")
    previous = attestation["shard_commit"]
    head = git.read("rev-parse", "HEAD").decode().strip()
    require(saved.get("merlin_commit") == previous, "saved commit differs from attestation shard_commit")
    require(current.get("merlin_commit") == head, "current stamp commit differs from HEAD")
    require(previous != head, "shard_commit equals current commit")
    git.read("merge-base", "--is-ancestor", previous, head)
    require(not git.read("status", "--porcelain"), "current git worktree is dirty")
    for name, stamp in (("saved", saved), ("current", current)):
        require(stamp.get("merlin_source_kind") == "git_worktree", f"{name} stamp is not git_worktree")
        require(stamp.get("merlin_diff_sha256") == EMPTY_SHA256, f"{name} stamp diff is not clean")
        nested = stamp.get("merlin")
        require(isinstance(nested, dict) and set(nested) == {"commit", "source_label", "dirty_files"},
                f"{name} nested merlin stamp is malformed")
        require(type(nested["dirty_files"]) is int and nested["dirty_files"] == 0,
                f"{name} stamp dirty_files is not zero")
        require(nested["commit"] == nested["source_label"] == stamp.get("merlin_source_label")
                == stamp["merlin_commit"], f"{name} commit labels are inconsistent")
    require({k: v for k, v in saved.items() if k not in COMMIT_LABELS}
            == {k: v for k, v in current.items() if k not in COMMIT_LABELS},
            "stamps differ outside commit labels")
    return previous, head


def _source_roots(paths):
    roots = {"src"}
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) >= 4 and parts[0] == "packages" and parts[2] == "src":
            roots.add("/".join(parts[:3]))
    return sorted(roots)


class _DirectImportScanner:
    """Best-effort direct import observations, never an admission gate."""

    def __init__(self, git, commit, *, module_level_only=False):
        self.git, self.commit = git, commit
        self.module_level_only = module_level_only
        entries = git.read("ls-tree", "-r", "-z", commit).split(b"\0")
        self.modes = {}
        for entry in entries:
            if entry:
                metadata, path = entry.split(b"\t", 1)
                self.modes[path.decode("utf-8")] = metadata.split()[0]
        self.paths = set(self.modes)
        self.modules = {}
        self.names = {}
        self.pending = []
        for root in _source_roots(self.paths):
            for path in self.paths:
                if path.startswith(root + "/") and path.endswith(".py"):
                    name = path[len(root) + 1:-3].replace("/", ".")
                    if name.endswith(".__init__"):
                        name = name[:-9]
                    self.modules.setdefault(name, set()).add(path)
                    self.names[path] = name

    def add_module(self, name):
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            self.pending.extend(self.modules.get(".".join(parts[:count]), ()))

    def _package(self, path):
        name = self.names.get(path, "")
        return name if path.endswith("/__init__.py") else name.rpartition(".")[0]

    def _from_base(self, node, path):
        if not node.level:
            return node.module or ""
        package = self._package(path).split(".") if self._package(path) else []
        require(node.level <= len(package), f"unresolved relative import in {path}")
        base = package[:len(package) - node.level + 1]
        return ".".join(base + ([node.module] if node.module else []))

    def _relative_literal(self, name, package, path):
        level = len(name) - len(name.lstrip("."))
        require(isinstance(package, str) and bool(package), f"unresolved literal import in {path}")
        parts = package.split(".")
        require(level <= len(parts), f"unresolved literal import in {path}")
        return ".".join(parts[:len(parts) - level + 1] + ([name[level:]] if name[level:] else []))

    def _literal_import(self, node, path, aliases):
        function = node.func
        named = isinstance(function, ast.Name) and function.id in aliases
        imported = isinstance(function, ast.Attribute) and function.attr == "import_module"
        if not (named or imported):
            return
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        argument = node.args[0] if node.args else keywords.get("name")
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            return
        name = argument.value
        builtin = isinstance(function, ast.Name) and function.id == "__import__"
        if builtin:
            level = node.args[4] if len(node.args) > 4 else keywords.get("level", ast.Constant(0))
            require(isinstance(level, ast.Constant) and type(level.value) is int and level.value >= 0,
                    f"unresolved literal import level in {path}")
            name = "." * level.value + name
        if name.startswith("."):
            package = self._package(path)
            if not builtin:
                argument = node.args[1] if len(node.args) > 1 else keywords.get("package")
                if isinstance(argument, ast.Name) and argument.id == "__package__":
                    package = self._package(path)
                elif argument is not None:
                    require(isinstance(argument, ast.Constant), f"unresolved literal import package in {path}")
                    package = argument.value
            name = self._relative_literal(name, package, path)
        self.add_module(name)
        # __import__('pkg', fromlist=['mod']) can load pkg.mod.
        fromlist = node.args[3] if builtin and len(node.args) > 3 else keywords.get("fromlist")
        if isinstance(fromlist, (ast.List, ast.Tuple)):
            for item in fromlist.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    self.add_module(name + "." + item.value)

    def _nodes(self, tree):
        if not self.module_level_only:
            yield from ast.walk(tree)
            return
        pending = [tree]
        while pending:
            node = pending.pop()
            yield node
            # Defaults, decorators and annotations execute when a function is defined;
            # its body (and a lambda's body) does not. Class bodies remain visible.
            for field, value in ast.iter_fields(node):
                if field == "body" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                if isinstance(value, ast.AST):
                    pending.append(value)
                elif isinstance(value, list):
                    pending.extend(child for child in value if isinstance(child, ast.AST))

    def _visit(self, path, raw):
        tree = ast.parse(raw, filename=path)
        aliases = {"__import__", "import_module"}
        nodes = list(self._nodes(tree))
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = self._from_base(node, path)
                self.add_module(base)
                for alias in node.names:
                    if alias.name == "*":
                        # Any submodule may be re-exported through computed __all__.
                        for name in self.modules:
                            if name.startswith(base + "."):
                                self.add_module(name)
                    else:
                        self.add_module(base + "." + alias.name)
            elif isinstance(node, ast.Call):
                self._literal_import(node, path, aliases)

    def report(self, selected):
        importers, errors = [], []
        roots = _source_roots(self.paths) + ["build_tools/scripts"]
        candidates = sorted(path for path in self.paths if path.endswith(".py")
                            and any(path.startswith(root + "/") for root in roots))
        for path in candidates:
            self.pending = []
            try:
                require(self.modes[path] in {b"100644", b"100755"}, "not a regular tracked file")
                self._visit(path, self.git.read("show", f"{self.commit}:{path}"))
                imported = sorted(set(self.pending) & selected)
                if imported:
                    importers.append({"path": path, "imports": imported})
            except (OSError, ValueError, TypeError, KeyError, SyntaxError, RuntimeError, subprocess.SubprocessError) as exc:
                errors.append({"path": path, "error": str(exc)})
        return {"importers": importers, "errors": errors, "scanned_file_count": len(candidates)}


def _importer_summary(git, previous, head, selected):
    summary = {}
    for label, revision in (("P", previous), ("C", head)):
        summary[label] = {}
        for mode, module_only in (("module_level", True), ("all_bodies", False)):
            try:
                scanner = _DirectImportScanner(git, revision, module_level_only=module_only)
                summary[label][mode] = scanner.report(selected)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                summary[label][mode] = {"importers": [], "errors": [{"error": str(exc)}]}
    return summary


def _changed_hashes(git, previous, head, changed):
    inventories = {revision: set(git.paths("ls-tree", "-r", "--name-only", "-z", revision))
                   for revision in (previous, head)}
    def digest_at(revision, path):
        return sha256(git.read("show", f"{revision}:{path}")) if path in inventories[revision] else None
    return [{"path": path, "sha256_at_P": digest_at(previous, path), "sha256_at_C": digest_at(head, path)}
            for path in changed]


def _decide(saved, current, *, repo, attestation, git_runner):
    policy, attestation_sha = load_attestation(attestation)
    git = _Git(repo, git_runner)
    previous, head = _check_stamps(saved, current, policy, git)
    changed = git.paths("diff", "--no-renames", "--name-only", "-z", previous, head)
    require(bool(changed), "commit diff is empty")
    outside = sorted(set(changed) - set(policy["allow"]))
    require(not outside, f"changed paths outside attestation allow: {outside}")
    trace_path, trace_sha = _trace_reference(policy["evidence"]["runtime_trace"])
    trace_raw = trace_path.read_bytes()
    require(sha256(trace_raw) == trace_sha, "runtime_trace sha256 mismatch")
    selected = set(policy["allow"]) | set(changed)
    evidence = {"runtime_trace": {"path": str(trace_path), "sha256": trace_sha,
                                  **_summarize_trace(trace_raw, git.repo, selected)},
                "direct_importers": _importer_summary(git, previous, head, selected),
                "notes": policy["evidence"]["notes"]}
    hashes = _changed_hashes(git, previous, head, changed)
    require(git.read("rev-parse", "HEAD").decode().strip() == head, "HEAD changed during attestation decision")
    require(not git.read("status", "--porcelain"), "current git worktree changed during attestation decision")
    return {
        "schema": RECEIPT_SCHEMA, "status": "ATTESTED_NOT_PROVEN", "decision": "attested_reuse_of_shards",
        "proof": "none: operator attestation; see residual_risks",
        "shard_commit": previous, "current_commit": head, "allow": policy["allow"], "changed": hashes,
        "attested_by": policy["attested_by"], "submitted_by": policy["submitted_by"],
        "statement": policy["statement"], "evidence": evidence, "attestation_sha256": attestation_sha,
        "saved_stamp_sha256": sha256(canonical(saved)), "current_stamp_sha256": sha256(canonical(current)),
        "residual_risks": list(RESIDUAL_RISKS), "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def decide(saved, current, *, repo, attestation, git_runner=None):
    """Return an ATTESTED, not proven receipt after identity and byte-integrity checks."""
    try:
        return _decide(saved, current, repo=repo, attestation=attestation, git_runner=git_runner)
    except AttestationRefused:
        raise
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise AttestationRefused(str(exc)) from exc
