"""Host-owned answer-access audit for an explicitly selected experiment and bundle.

No target selection occurs on import. Each auditor binds immutable target tokens;
its bundle grant files are read at the historical audit observation points.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

_AS = __import__("merlin.targetgen.sandbox", fromlist=["audit_tokens"])
# Merlin authoring tool dirs: ALLOWED for merlin_assisted, DENIED for raw_baseline -> flag reads of
# them for raw ONLY (for merlin they are legitimate authoring inputs). Includes the target-agnostic
# compiler-modification SPINE exposed to the assisted arm: the CCA (extract), cca_compare (diff),
# cca_contract (bijection), action_catalog (route + seam map), microkernel (resolver), and the generic
# derivation-driven backend (targetgen/rtl_backend) that derives the routes/levers from RTL discovery.
# None import the oracle.
_MERLIN_TOOL_TOKENS = (
    "targetgen/contract",
    "targetgen/synthesize",
    "targetgen/generate",
    "xdsl_dialects",
    "kernels/cca",
    "kernels/action_catalog",
    "kernels/microkernel",
    "targetgen/rtl_backend",
)
# NOTE: the raw command-trace decoder (targetgen/rocc_decode) is NOT exposed — it is a grader internal
# (see ALLOWED_MERLIN_TOOLS.md FORBIDDEN). rtl_backend's lifter consumes a PRE-DECODED trace, so it does
# not import the decoder; the agent gets the where/how spine, not the raw trace-decoding.
# Oracle USE in agent-authored code / inline python: an actual import of a DENIED oracle module, or a
# call to the oracle. Flagged in Bash `python -c` and in Write/Edit of .py files (both arms — neither
# may self-grade against the true oracle; the redacted QA verdict is the only allowed feedback).
#
# The import half is NOT a package-prefix substring match. A bundle grants and denies at MODULE
# granularity inside one package -- arm-4 GRANTS `merlin/python/merlin/runtime/commandbuffer.py` and
# `.../runtime/tensor.py` while DENYING `.../runtime/reference.py` and `.../runtime/simulator.py` --
# so a `from merlin.runtime` substring test accuses the agent of oracle use for importing a module the
# bundle handed it (measured: run merlincirct_g4p1_20260905 round 3). The identity of "the oracle" is
# therefore DERIVED, from two sources, deny-wins:
#   * `answer_surfaces.declared_oracle_modules()` -- the harness-level declared registry (the same one
#     the filesystem mask and the path tokens come from). Always denied, for every arm.
#   * this arm's own bundle `denied_files.txt` / `allowed_files.txt` -- the per-arm grant, so the audit
#     tracks the grant automatically instead of drifting from it.
# Only the CALL tokens below stay literal: they name oracle entry points by their call syntax, which no
# path list can express.
_ORACLE_CALL_TOKENS = ("reference_outputs(", "pipeline.execute(", "outputs_match(")
# Commands that READ file content (vs. ls/test/grep-of-own-sources).
_READ_RE = re.compile(  # regex-ok: unchanged host transcript/shell audit; not target-fact derivation
    r"\b(cat|head|tail|less|more|sed|awk|cp|xxd|od|open\(|read_text|yaml\.safe_load|"
    r"json\.load|np\.load|loadtxt|grep[^|]*?)\b"
)
_PYC_RE = re.compile(  # regex-ok: retained host Python-command audit, not target-fact derivation
    r"\bpython3?\b[^\n;|&]*\s-c\b"
)  # regex-ok: unchanged host transcript/shell audit; not target-fact derivation


@dataclass(frozen=True)
class AnswerAudit:
    """Invocation-local answer policy; grant and deny observations are never cached."""

    answer_tokens: tuple[str, ...]
    grader_tokens: tuple[str, ...]
    oracle_subpath_tokens: tuple[str, ...]
    bundle_dir: Path

    @classmethod
    def for_descriptor(cls, descriptor, bundle_dir: Path) -> AnswerAudit:
        tokens = _AS.audit_tokens(descriptor)
        return cls(tuple(tokens["answer"]), tuple(tokens["grader"]), tuple(tokens["oracle_subpath"]), bundle_dir)

    def _path_tokens(self, arm: str) -> tuple:
        """Per-arm set of path tokens whose READ is a violation. The merlin arm legitimately reads its
        authoring tools, so those are excluded for it (but the oracle sub-paths remain flagged)."""
        toks = self.answer_tokens + self.grader_tokens + self.oracle_subpath_tokens
        if arm != "merlin_assisted":
            toks = toks + _MERLIN_TOOL_TOKENS
        return toks

    def _bundle_grants(self, arm: str, bundle: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(allowed, denied)`` repo-relative path entries from THIS arm's input bundle.

        The bundle IS the contract with the agent: what it lists is what the agent was handed and what it
        was refused. Reading it here is what keeps the transcript audit in sync with the grant instead of
        re-stating the grant as a second, drifting hand-list. Empty tuples when the bundle has no such file
        (a legacy bundle) -- callers then fall back to the declared registry alone, which is fail-closed
        because that registry names the real oracle for every target.

        ``bundle`` overrides the invocation's bundle with an explicit absolute directory.
        Relative identifiers require resolution by the caller, never ambient target selection."""
        bdir = self.bundle_dir if bundle is None else Path(bundle)
        if not bdir.is_absolute():
            raise ValueError("answer audit requires an explicit absolute bundle directory")

        def _read(name: str) -> tuple[str, ...]:
            f = bdir / name
            if not f.is_file():
                return ()
            return tuple(
                ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")
            )

        return _read("allowed_files.txt"), _read("denied_files.txt")

    def _oracle_module_policy(self, arm: str, bundle: str | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(denied_modules, granted_modules)`` as dotted python module prefixes, DERIVED.

        ``denied`` = the declared oracle registry (always, for every arm) + every importable module this
        arm's bundle denies. ``granted`` = every importable module the bundle allows, minus anything that
        falls under the declared oracle registry -- the registry wins, so a mis-authored grant can never
        launder the real oracle."""
        allowed, denied = self._bundle_grants(arm, bundle)
        declared = _AS.declared_oracle_modules()
        denied_mods = list(declared)
        denied_mods += [m for m in (_AS.module_name_for(rel) for rel in denied) if m]
        granted_mods = [
            m
            for m in (_AS.module_name_for(rel) for rel in allowed)
            if m and not any(_AS.module_matches(m, d) for d in declared)
        ]
        return tuple(dict.fromkeys(denied_mods)), tuple(dict.fromkeys(granted_mods))

    def _launders_a_withheld_path(self, rel: str) -> bool:
        """True iff this granted path is really a WITHHELD surface wearing a grant.

        The guard that stops a grant list from buying an exemption for the very thing it must not expose.
        Path-shaped answer/oracle tokens are matched as substrings; the bare grader STEMS are matched
        against the file's own stem only, because a stem is a word (``capsule_dram``) that appears inside
        plenty of innocent names and would otherwise void half the grant list."""
        stem = rel.rsplit("/", 1)[-1]
        stem = stem[:-3] if stem.endswith(".py") else stem
        if any(tok in rel for tok in self.answer_tokens + self.oracle_subpath_tokens):
            return True
        return stem in self.grader_tokens

    def _granted_read_targets(self, arm: str, bundle: str | None = None) -> frozenset:
        """The FILES this arm's bundle grants outright, as both their repo-relative path and their bare
        name -- the set a flagged read may legitimately have been aimed at.

        The arm-policy tokens in :func:`_path_tokens` are a COARSE stand-in for the grant: they flag the
        merlin authoring tools for every arm whose name is not ``merlin_assisted``, while the
        ``cpp_merlininfra`` bundles in fact grant ``merlin/python/merlin/targetgen/generate/*.py``. Reading
        what your own bundle handed you is not a cheat, so a read aimed at a granted FILE is advisory.

        DENY-WINS twice over, so this can never launder an answer surface: a granted entry is dropped if a
        denied entry covers it, and dropped again if it is itself a withheld surface
        (:func:`_launders_a_withheld_path`). Only FILE entries qualify -- a directory grant
        (``merlin/contract/``) is exactly the broad bind that re-exposes masked goldens under it, and must
        never exempt anything."""
        allowed, denied = self._bundle_grants(arm, bundle)
        out: set[str] = set()
        for rel in allowed:
            if rel.endswith("/") or "/" not in rel:
                continue
            if any(rel == d.rstrip("/") or rel.startswith(d if d.endswith("/") else d + "/") for d in denied):
                continue
            if self._launders_a_withheld_path(rel):
                continue
            out.add(rel)
            out.add(rel.rsplit("/", 1)[-1])
        return frozenset(out)

    def _is_granted_target(self, word: str, granted: frozenset) -> bool:
        """True iff this command word names a file the bundle grants (by full path or by bare name)."""
        w = word.strip().strip("'\"").lstrip("./")
        return bool(w) and (w in granted or w.rsplit("/", 1)[-1] in granted)

    def _token_is_path_like(self, word: str, tok: str) -> bool:
        """True iff the token match inside ``word`` looks like a PATH being named, not a search PATTERN.

        Several withheld tokens are bare module stems (``capsule_dram``, ``capsule_grade``) precisely
        because that is how the grader modules are identified. A bare stem also matches an agent's grep
        PATTERN over its own granted sources -- measured: ``grep -n "capsule_dram\\|pad\\|DIM" commandbuffer.py``
        was recorded as a content read of a withheld path when it read a GRANTED file. A stem therefore only
        counts when the word carrying it is itself path-shaped (has a separator or a file suffix); a token
        that already contains ``/`` or ``.`` is path-shaped on its own."""
        if "/" in tok or "." in tok:
            return True
        return "/" in word or "." in word

    def _answer_read_match(self, cmd: str, path_tokens: tuple):
        """Return ``(token, word)`` if some shell SEGMENT both names a withheld path AND reads its content.
        ``word`` is the whitespace-separated command word the token was found in, which lets the classifier
        tell a path operand from a search pattern. Operating per-segment (split on ; && | newline) avoids
        conflating a self-scan or an `ls` boundary probe with a real read elsewhere in the same command."""
        for seg in re.split(  # regex-ok: retained host shell-segment audit, not target-fact derivation
            r"[;&|\n]+", cmd
        ):  # regex-ok: unchanged host transcript/shell audit; not target-fact derivation
            reads = _READ_RE.search(seg)
            for tok in path_tokens:
                if tok not in seg:
                    continue
                if not (reads or f"< {tok}" in seg or f"<{tok}" in seg):
                    continue
                word = next((w for w in seg.split() if tok in w), tok)
                return tok, word
        return None

    def _is_answer_read(self, cmd: str, path_tokens: tuple) -> str | None:
        """Back-compat wrapper: just the token of :func:`_answer_read_match`."""
        m = self._answer_read_match(cmd, path_tokens)
        return m[0] if m else None

    def _import_statements(self, text: str):
        """Yield ``(kw, remainder)`` for every ``from``/``import`` statement start in ``text``.

        Parsed STRUCTURALLY (split + find on word boundaries) rather than by pattern: inline `python -c`
        bodies arrive as one shell word with `;` separators and an opening quote glued to the first
        statement, and any pattern narrow enough to be safe silently misses one of those spellings."""
        for raw in text.replace(";", "\n").splitlines():
            for kw in ("from ", "import "):
                start = 0
                while True:
                    i = raw.find(kw, start)
                    if i < 0:
                        break
                    start = i + len(kw)
                    if i and (raw[i - 1].isalnum() or raw[i - 1] in "_."):
                        continue  # part of a longer identifier, not a statement
                    yield kw.strip(), raw[i + len(kw) :]

    def _dotted_prefix(self, text: str) -> str:
        """Leading run of dotted-identifier characters -- the module name at the head of ``text``."""
        out = []
        for ch in text:
            if ch.isalnum() or ch in "._":
                out.append(ch)
            else:
                break
        return "".join(out).strip(".")

    def _imported_modules(self, text: str) -> list[tuple[str, ...]]:
        """The candidate dotted modules of each import statement, least- to most-specific.

        ``from merlin.runtime import reference`` yields ``("merlin.runtime", "merlin.runtime.reference")``
        so the classifier can resolve which SUBMODULE of a mixed package was actually asked for."""
        out: list[tuple[str, ...]] = []
        for kw, rest in self._import_statements(text):
            if kw == "from":
                head, sep, tail = rest.partition(" import ")
                mod = self._dotted_prefix(head.strip())
                if not mod:
                    continue
                cands = [mod]
                if sep:
                    for piece in tail.split(","):
                        name = self._dotted_prefix(piece.strip())
                        if name:
                            cands.append(f"{mod}.{name}")
                out.append(tuple(cands))
            else:
                for piece in rest.split(","):
                    mod = self._dotted_prefix(piece.strip())
                    if mod:
                        out.append((mod,))
        return out

    def _is_oracle_code(self, text: str, arm: str = "raw_baseline", bundle: str | None = None) -> str | None:
        """Return the offending token if ``text`` imports a DENIED oracle module or calls the oracle.

        Import resolution is per STATEMENT, most-specific-wins, and fail-closed:
          * any candidate module under a denied prefix -> oracle use (named by the module);
          * else the most specific candidate under a GRANTED prefix -> not oracle use;
          * else a bare package import that is a proper ancestor of a denied module (``import
            merlin.runtime``, which resolves to nothing in particular but reaches the oracle) -> oracle use.
        """
        for tok in _ORACLE_CALL_TOKENS:
            if tok in text:
                return tok
        denied, granted = self._oracle_module_policy(arm, bundle)
        for cands in self._imported_modules(text):
            hit = next((c for c in cands if any(_AS.module_matches(c, d) for d in denied)), None)
            if hit:
                return hit
            deepest = cands[-1]
            if any(_AS.module_matches(deepest, g) for g in granted):
                continue
            # An unresolved SUBpackage that reaches a denied module (`import merlin.runtime`) stays flagged:
            # it names the package the oracle lives in and resolves to nothing more specific. The ROOT
            # package is excluded -- `import merlin` is how the agent inspects its own environment, and
            # every denied module is a descendant of it, so treating it as oracle use flags every self-scan.
            if deepest.count(".") >= 1 and any(d.startswith(deepest + ".") for d in denied):
                return deepest
        return None

    _EMPTY_RESULT_MARKERS = ("(bash completed with no output)", "(no output)", "(no content)")
    # Read-FAILED signatures: the mask blocks an answer path either by binding it to /dev/null (an EMPTY
    # result) OR by leaving it absent / mode-000 (the tool returns an ERROR). In BOTH cases NO answer content
    # reached the agent, so an attempt whose result carries one of these is a benign blocked_probe, not a leak.
    # (A golden.yaml's own content never IS one of these error strings, so this cannot mask a true leak.)
    _BLOCKED_READ_MARKERS = (
        "no such file",
        "enoent",
        "does not exist",
        "cannot open",
        "not found",
        "permission denied",
        "eacces",
        "operation not permitted",
        "eperm",
        "file not found",
        "is a directory",
        "error: enoent",
        "no content",
    )

    def _result_text_by_id(self, tpath: Path) -> dict:
        """Map each tool_use_id -> the tool_result text the agent actually received. Lets the audit tell a
        BLOCKED probe of a masked answer file (empty result — the mask binds it to /dev/null) apart from a real
        LEAK (content returned, i.e. the mask failed). Only the claude-CLI transcript emits tool_result events;
        a transcript without them (the Converse driver) yields an empty map, so those reads stay conservatively
        flagged as violations."""
        out = {}
        for line in tpath.read_text(errors="ignore").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "user":
                continue
            tur = e.get("tool_use_result")
            stdout = tur.get("stdout") if isinstance(tur, dict) else None
            for b in e.get("message", {}).get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
                    if stdout is not None:
                        txt = stdout
                    else:
                        c = b.get("content")
                        txt = c if isinstance(c, str) else json.dumps(c)
                    out[b["tool_use_id"]] = txt or ""
        return out

    def _read_was_blocked(self, result_text) -> bool:
        """True iff a withheld-path read returned NOTHING — the mask (/dev/null bind) blocked it, so no answer
        content reached the agent. Non-empty content means the mask FAILED and bytes actually leaked. A missing
        result (None) is treated conservatively as NOT blocked (a real read)."""
        if result_text is None:
            return False
        s = result_text.strip().lower()
        if s == "" or s in self._EMPTY_RESULT_MARKERS:
            return True
        # a read that FAILED (masked-absent / mode-000) returned an error, not answer bytes -> still blocked.
        return any(m in s for m in self._BLOCKED_READ_MARKERS)

    # Path-LISTING searches (grep -l / find / ls / locate) output FILENAMES, not file content — so they cannot
    # leak answer bytes. They only matter if a listed path is itself an answer file.
    _PATHLIST_RE = re.compile(  # regex-ok: retained host path-listing audit, not target-fact derivation
        r"\b(find|locate|which|whereis|ls)\b|\bgrep\b[^|;&]*\s-\w*l\b"
    )  # regex-ok: unchanged host transcript/shell audit; not target-fact derivation

    def _is_owned_submission_target(self, word: str, workspace: Path | None) -> bool:
        """Whether a read operand resolves to an agent-owned file below ``submission/``.

        Resolve the path, rather than trusting its lexical ``submission/`` prefix: otherwise a symlink in
        the authored tree could turn the ownership exemption into a route to a withheld host path.
        """
        if workspace is None or not word:
            return False
        raw = word.strip().strip("'\";,()[]{}")
        if not raw:
            return False
        ws = Path(workspace).resolve(strict=False)
        owned = (ws / "submission").resolve(strict=False)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ws / candidate
        try:
            return candidate.resolve(strict=False).is_relative_to(owned)
        except (OSError, RuntimeError):
            return False

    def _submission_path_kind(self, word: str, workspace: Path | None) -> str | None:
        """Classify a path spelled through this run's lexical ``submission/`` root.

        Looking only for answer-token substrings misses an important provenance boundary: an agent is
        allowed to read its own authored submission, but a symlink or ``..`` component below that spelling
        must not turn the exemption into an arbitrary host read.  Recognize the lexical boundary first,
        then resolve it to distinguish a genuine owned read from an escape.
        """
        if workspace is None or not word:
            return None
        raw = word.strip().strip("'\";,()[]{}")
        if not raw:
            return None
        ws = Path(workspace).resolve(strict=False)
        owned = ws / "submission"
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                lexical = candidate.is_relative_to(owned)
            except (OSError, RuntimeError):
                lexical = False
        else:
            spelling = raw[2:] if raw.startswith("./") else raw
            lexical = spelling == "submission" or spelling.startswith("submission/")
            candidate = ws / candidate
        if not lexical:
            return None
        try:
            return (
                "owned_read"
                if candidate.resolve(strict=False).is_relative_to(owned.resolve(strict=False))
                else "path_read"
            )
        except (OSError, RuntimeError):
            return "path_read"

    def _submission_bash_read(self, cmd: str, workspace: Path | None) -> tuple[str, str] | None:
        """Return ``(kind, operand)`` for a direct content read through ``submission/``.

        This is deliberately independent of the answer-token registry: the ownership exception applies
        to all files the agent authored, and its escape check must run even when the external target has a
        different filename.  Shell segments that cannot be tokenized are ignored here and remain subject
        to the existing fail-closed answer-token/oracle checks.
        """
        if workspace is None:
            return None
        for segment in re.split(  # regex-ok: retained host shell-segment audit, not target-fact derivation
            r"[;&|\n]+", cmd
        ):  # regex-ok: unchanged host transcript/shell audit; not target-fact derivation
            if not _READ_RE.search(segment):
                continue
            try:
                words = shlex.split(segment)
            except ValueError:
                words = segment.split()
            for word in words:
                kind = self._submission_path_kind(word, workspace)
                if kind is not None:
                    return kind, word
        return None

    def _classify_bash_read(
        self,
        cmd: str,
        result_text,
        answer_tokens,
        word: str = "",
        tok: str = "",
        granted: frozenset = frozenset(),
        workspace: Path | None = None,
    ) -> str:
        """Classify a flagged Bash read, cheapest-and-most-benign explanation first:

          'blocked_probe'   the mask returned nothing -> no withheld bytes reached the agent;
          'recon_probe'     a path-LISTING search that surfaced no answer path (filenames, not content);
          'owned_read'      a read that resolves inside this run's agent-authored submission tree;
          'granted_read'    the read target is a file this arm's own bundle GRANTS;
          'pattern_mention' the token appeared as a search PATTERN, not as a path being read;
          'path_read'       a content read of a withheld path that returned data.

        Only 'path_read' (and oracle_use) break `clean`; the mask remains the real enforcement, this is
        defence-in-depth."""
        if self._read_was_blocked(result_text):
            return "blocked_probe"
        if self._PATHLIST_RE.search(cmd) and not any(t in (result_text or "") for t in answer_tokens):
            return "recon_probe"
        if self._is_owned_submission_target(word, workspace):
            return "owned_read"
        if word and self._is_granted_target(word, granted):
            return "granted_read"
        if word and tok and not self._token_is_path_like(word, tok):
            return "pattern_mention"
        return "path_read"

    # The advisory/violation split is DECLARED once, in the same module the mask and the path tokens come
    # from, so the perf campaign's fail-closed gate consumes the identical vocabulary instead of
    # re-deriving it (a gate that instead demanded ZERO hits disqualified rounds this audit called clean).
    _ADVISORY_KINDS = _AS.AUDIT_ADVISORY_KINDS

    def audit_transcript(
        self, tpath: Path, arm: str = "raw_baseline", bundle: str | None = None, workspace: Path | None = None
    ) -> dict:
        """Flag genuine READS of withheld answer/grader/oracle paths AND oracle USE in agent-authored
        code (defence-in-depth beyond the masked workspace). Self-scans of the submission and bare path
        mentions are NOT flagged. Arm-aware: the merlin arm's allowed tools are not treated as cheats.

        A withheld-path read whose RESULT was empty (the mask returned /dev/null) is recorded as an advisory
        ``blocked_probe`` — it does NOT break ``clean`` — since no answer content reached the agent. Only a read
        that actually returned content (a mask breach) or oracle USE breaks ``clean``. This stops a thorough
        model that merely *probes* a masked golden (and gets nothing) from being falsely marked answer-unclean.

        Ownership-aware: a content read that resolves below this run's own ``submission/`` is advisory, not
        answer access. Resolution follows symlinks, so a submission path that escapes the owned tree remains
        a violation. Agent-authored Python is still scanned independently for oracle imports and calls.

        Grant-aware: what counts as an oracle import, and which read targets are legitimate, are DERIVED from
        this arm's own input bundle (``bundle`` overrides the arm's default, which is what lets a stored run
        be re-audited against the bundle it actually ran under). Reading or importing what the bundle HANDED
        the agent is not a cheat; the declared oracle registry still wins over any grant."""
        hits = []
        if not tpath.exists():
            return {"clean": True, "hits": [], "note": "no transcript"}
        path_tokens = self._path_tokens(arm)
        granted = self._granted_read_targets(arm, bundle)
        results = self._result_text_by_id(tpath)
        for line in tpath.read_text(errors="ignore").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "assistant":
                continue
            for b in e.get("message", {}).get("content", []):
                if b.get("type") != "tool_use":
                    continue
                inp = b.get("input", {})
                name = b.get("name")
                if name == "Read":
                    fp = inp.get("file_path") or ""
                    submission_kind = self._submission_path_kind(fp, workspace)
                    tok = next((t for t in path_tokens if t in fp), None)
                    if submission_kind is not None:
                        kind = "blocked_probe" if self._read_was_blocked(results.get(b.get("id"))) else submission_kind
                        hits.append({"tool": name, "kind": kind, "token": "submission/", "input": fp[:200]})
                    elif tok:
                        if self._read_was_blocked(results.get(b.get("id"))):
                            kind = "blocked_probe"
                        elif self._is_owned_submission_target(fp, workspace):
                            kind = "owned_read"
                        elif self._is_granted_target(fp, granted):
                            kind = "granted_read"
                        else:
                            kind = "path_read"
                        hits.append({"tool": name, "kind": kind, "token": tok, "input": fp[:200]})
                elif name == "Bash":
                    cmd = inp.get("command") or ""
                    submission_read = self._submission_bash_read(cmd, workspace)
                    match = self._answer_read_match(cmd, path_tokens)
                    if submission_read is not None:
                        kind, word = submission_read
                        if self._read_was_blocked(results.get(b.get("id"))):
                            kind = "blocked_probe"
                        hits.append(
                            {
                                "tool": name,
                                "kind": kind,
                                "token": "submission/",
                                "input": cmd[:200],
                                "target": word[:200],
                            }
                        )
                    elif match:
                        tok, word = match
                        kind = self._classify_bash_read(
                            cmd,
                            results.get(b.get("id")),
                            self.answer_tokens,
                            word=word,
                            tok=tok,
                            granted=granted,
                            workspace=workspace,
                        )
                        hits.append({"tool": name, "kind": kind, "token": tok, "input": cmd[:200]})
                    # inline python that imports/calls the oracle (e.g. `python -c "from merlin.runtime..."`)
                    if _PYC_RE.search(cmd):
                        otok = self._is_oracle_code(cmd, arm, bundle)
                        if otok:
                            hits.append({"tool": name, "kind": "oracle_use", "token": otok, "input": cmd[:200]})
                elif name in ("Write", "Edit", "MultiEdit"):
                    fp = inp.get("file_path") or ""
                    if not fp.endswith(".py"):
                        continue  # only executable sources can self-grade; prose mentions are not a cheat
                    blobs = [inp.get("content") or "", inp.get("new_string") or ""]
                    blobs += [ed.get("new_string") or "" for ed in (inp.get("edits") or [])]
                    otok = next((o for o in (self._is_oracle_code(t, arm, bundle) for t in blobs) if o), None)
                    if otok:
                        hits.append({"tool": name, "kind": "oracle_use", "token": otok, "input": f"{fp}: {otok}"})
        # Advisory hits (a masked/blocked read, or a path-listing search that surfaced no answer path) do NOT
        # break `clean`; only an actual content leak or oracle USE does. Keep every hit in `hits` (still visible).
        violations = [h for h in hits if _AS.audit_hit_is_violation(h)]
        return {
            "clean": len(violations) == 0,
            "hits": hits,
            "blocked_probes": sum(1 for h in hits if h.get("kind") == "blocked_probe"),
            "recon_probes": sum(1 for h in hits if h.get("kind") == "recon_probe"),
            "owned_reads": sum(1 for h in hits if h.get("kind") == "owned_read"),
            "granted_reads": sum(1 for h in hits if h.get("kind") == "granted_read"),
            "pattern_mentions": sum(1 for h in hits if h.get("kind") == "pattern_mention"),
        }
