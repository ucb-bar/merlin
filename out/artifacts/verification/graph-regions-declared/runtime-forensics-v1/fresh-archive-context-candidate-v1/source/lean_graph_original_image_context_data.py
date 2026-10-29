"""Data-only validation of a NEW reviewed original-image context selection.

This does not open SDK/runtime images, approve a context, or accept proof JSON.
Original paths are provenance strings only; only their exact private transports
may ever be read by the controller. The caller must independently bind/review
the three input documents and the immutable trace parser source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import PurePosixPath


class ContextRefusal(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def absolute(value):
    if type(value) is not str:
        raise ContextRefusal("path must be an exact string")
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts or "\x00" in value:
        raise ContextRefusal("noncanonical absolute path")
    return path


def module_name(value):
    if type(value) is not str or not value:
        raise ContextRefusal("module/root identifier required")
    for part in value.split("."):
        if not part or not part.isascii() or not (part[0].isalpha() or part[0] == "_"):
            raise ContextRefusal("invalid dotted module/root identifier")
        if any(not (c.isalnum() or c == "_") for c in part):
            raise ContextRefusal("invalid module/root identifier character")
    return value


def literal_imports(content):
    """Decode a CLOSED project header; trusted host code, not Lean reporting.

    Admit optional initial `prelude` and one literal dotted module per `import`
    line, with whitespace and nested/line comments. Other header forms refuse.
    Stop at the first body command; Lean's actual object report must independently
    equal this result before compilation. No source/object is opened here.
    """
    if type(content) is not bytes:
        raise ContextRefusal("exact project source bytes required")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ContextRefusal("invalid UTF-8 project source") from error
    if "\x00" in text or text.startswith("\ufeff"):
        raise ContextRefusal("unsupported source prefix/control")
    position = 0

    def skip():
        nonlocal position
        newline = False
        while position < len(text):
            if text[position] in " \t\r\n":
                newline |= text[position] in "\r\n"
                position += 1
            elif text.startswith("--", position):
                end = text.find("\n", position)
                position = len(text) if end < 0 else end
            elif text.startswith("/-", position):
                position += 2
                depth = 1
                while depth and position < len(text):
                    if text.startswith("/-", position):
                        depth += 1
                        position += 2
                    elif text.startswith("-/", position):
                        depth -= 1
                        position += 2
                    else:
                        newline |= text[position] in "\r\n"
                        position += 1
                if depth:
                    raise ContextRefusal("unterminated project header comment")
            else:
                break
        return newline

    def token():
        nonlocal position
        start = position
        while position < len(text) and (text[position].isalnum() or text[position] in "_."):
            position += 1
        return text[start:position]

    imports, first = ["Init"], True
    skip()
    while position < len(text):
        word = token()
        if word == "prelude":
            if not first:
                raise ContextRefusal("prelude must precede all explicit imports")
            imports = []
        elif word == "import":
            skip()
            name = module_name(token())
            if name in {"import", "prelude", "public", "private", "meta", "module"}:
                raise ContextRefusal("unsupported import modifier/name")
            imports.append(name)
        elif word in {"public", "private", "meta"}:
            while word in {"public", "private", "meta"}:
                skip()
                word = token()
            if word in {"import", "prelude"}:
                raise ContextRefusal("unsupported modified import")
            break  # A declaration modifier, not part of the import header.
        elif not word and text[position] in "#@":
            break  # Body command/attribute; never interpreted as an import.
        elif word == "module" or not word:
            raise ContextRefusal("unsupported project header form")
        else:
            break
        first = False
        newline = skip()
        if position < len(text) and not newline:
            raise ContextRefusal("one complete literal header directive per line required")
    return tuple(dict.fromkeys(imports))


@dataclass(frozen=True)
class FileRef:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self):
        absolute(self.path)
        if type(self.bytes) is not int or self.bytes < 0:
            raise ContextRefusal("nonnegative exact byte count required; bool refused")
        if type(self.sha256) is not str or len(self.sha256) != 64 or set(self.sha256) - set("0123456789abcdef"):
            raise ContextRefusal("lowercase SHA256 required")

    @classmethod
    def parse(cls, row):
        if type(row) is not dict or set(row) != {"path", "sha256", "bytes"}:
            raise ContextRefusal("closed file-reference fields required")
        return cls(**row)

    def record(self):
        return asdict(self)


def unique_refs(rows):
    result = {}
    for row in rows:
        ref = FileRef.parse(row)
        if ref.path in result:
            raise ContextRefusal("duplicate/conflicting selected path")
        result[ref.path] = ref
    return result


@dataclass(frozen=True)
class Selection:
    toolchain: str
    images: tuple[FileRef, ...]
    runtime: tuple[FileRef, ...]
    misses: tuple[tuple[str, str], ...]
    success_only: tuple[tuple[str, str], ...]
    executable: str
    interpreter: str
    forbidden_roots: tuple[str, ...]

    def record(self):
        return asdict(self)

    def allowed_misses(self, returncode):
        if type(returncode) is not int:
            raise ContextRefusal("exact process exit code required")
        return set(self.misses) if returncode == 0 else set(self.misses) - set(self.success_only)


def finite_pairs(rows, runtime):
    result = []
    if type(rows) is not list:
        raise ContextRefusal("explicit finite diagnostic list required")
    for row in rows:
        if type(row) is not list or len(row) != 2 or row[0] not in runtime:
            raise ContextRefusal("unbound finite symbol diagnostic")
        name = row[1]
        if type(name) is not str or not name or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in name):
            raise ContextRefusal("finite symbol must be an exact identifier")
        result.append(tuple(row))
    if len(result) != len(set(result)):
        raise ContextRefusal("duplicate finite diagnostic")
    return tuple(result)


def reviewed_selection(metadata, copied, probe, base_probe, review):
    if set(probe) != {
        "schema", "runtime", "finite_allowed_misses", "success_only_misses", "base_intent", "finite_review"
    }:
        raise ContextRefusal("closed reviewed runtime descriptor fields required")
    for key in ("base_intent", "finite_review"):
        FileRef.parse(probe[key])  # Content binding is performed by the IO controller.
    if base_probe is None or base_probe.get("schema") != "merlin.original-image.basic-kernel-probe.v1":
        raise ContextRefusal("original basic-intent provenance required")
    if review is None or review.get("schema") != "merlin.finite-stdlib-optional-diagnostics.review.v1":
        raise ContextRefusal("explicit finite stdlib review required")
    base = select(metadata, copied, base_probe)
    runtime = {ref.path: ref for ref in base.runtime}
    if unique_refs(probe["runtime"]) != runtime:
        raise ContextRefusal("reviewed descriptor changed original runtime references")
    owner = FileRef.parse(review["original_runtime_owner"])
    if runtime.get(owner.path) != owner:
        raise ContextRefusal("finite diagnostic owner differs from original runtime")
    baseline = finite_pairs(review["baseline_exact_pairs"], runtime)
    additional = finite_pairs(review["additional_exact_pairs"], runtime)
    union = finite_pairs(review["proposed_finite_union"], runtime)
    success = finite_pairs(review["proposed_success_only_pairs"], runtime)
    if (
        len(baseline) != 719 or set(baseline) != set(base.misses)
        or len(additional) != 180 or set(baseline) & set(additional)
        or any(path != owner.path for path, _ in additional)
        or len(union) != 899 or set(union) != set(baseline) | set(additional)
        or len(success) != 181 or set(success) != set(additional) | set(base.success_only)
    ):
        raise ContextRefusal("reviewed finite baseline/addition/success-only correspondence differs")
    actual = finite_pairs(probe["finite_allowed_misses"], runtime)
    only = finite_pairs(probe["success_only_misses"], runtime)
    if set(actual) != set(union) or set(only) != set(success):
        raise ContextRefusal("descriptor diagnostics differ from exact finite review")
    return replace(base, misses=actual, success_only=only)


def select(metadata, copied, probe, *, base_probe=None, finite_review=None):
    """Match all4527 ORIGINAL refs and the exact eleven-image reviewed runtime.

    Status strings are sanity checks, never evidence of successful checking.
    Every byte pin is compared across the reviewed documents; no observed file
    content is used to choose an expectation.
    """
    if probe["schema"] == "merlin.original-image.reviewed-runtime-intent.v2":
        return reviewed_selection(metadata, copied, probe, base_probe, finite_review)
    if base_probe is not None or finite_review is not None:
        raise ContextRefusal("legacy basic intent does not admit extra review inputs")
    if metadata["schema"] != "merlin.deadline.original-snapshot.metadata-review.v2":
        raise ContextRefusal("unknown original metadata schema")
    if copied["schema"] != "merlin.original-image-copy.v1":
        raise ContextRefusal("unknown original-copy schema")
    if probe["schema"] != "merlin.original-image.basic-kernel-probe.v1":
        raise ContextRefusal("unknown reviewed runtime-intent schema")
    originals = unique_refs(row["original"] for row in metadata["selected"])
    if len(originals) != 4527 or metadata["selected_count"] != 4527 or copied["file_count"] != 4527:
        raise ContextRefusal("complete ORIGINAL4527 membership required")
    if any(metadata[key] for key in ("conflicting_original_refs", "archive_missing", "archive_hash_differences")):
        raise ContextRefusal("original metadata reports unresolved selection conflict")
    destination, archived = absolute(copied["destination"]), absolute(copied["source_root"])
    original_exe = [absolute(path) for path in originals if path.endswith("/bin/lean")]
    if len(original_exe) != 1:
        raise ContextRefusal("one original Lean executable required")
    old = original_exe[0].parent.parent
    if destination == old or destination == archived or destination.is_relative_to(old):
        raise ContextRefusal("private transport must be separate from installed/archive roots")
    seen, images = set(), []
    for row in copied["files"]:
        original, selected = FileRef.parse(row["original"]), FileRef.parse(row["selected"])
        if original.path in seen or originals.get(original.path) != original:
            raise ContextRefusal("missing/duplicate/extra/wrong ORIGINAL copy association")
        path = absolute(original.path)
        if not path.is_relative_to(old):
            raise ContextRefusal("original image outside the reviewed SDK root")
        if selected.path != str(destination / path.relative_to(old)):
            raise ContextRefusal("incorrect original-to-private path transport")
        if (selected.sha256, selected.bytes) != (original.sha256, original.bytes):
            raise ContextRefusal("private expectation differs from original bytes")
        if row["archived"] != str(archived / path.relative_to(old)):
            raise ContextRefusal("archive provenance transport differs")
        seen.add(original.path)
        images.append(selected)
    if seen != set(originals):
        raise ContextRefusal("not every original image was selected")
    image_map = {ref.path: ref for ref in images}
    runtime = unique_refs(probe["runtime"])
    executable, interpreter = str(destination / "bin/lean"), "/usr/lib/ld-linux-x86-64.so.2"
    if len(runtime) != 11 or executable not in runtime or interpreter not in runtime:
        raise ContextRefusal("exact eleven-image runtime including executable/loader required")
    system = []
    for path, ref in runtime.items():
        if absolute(path).is_relative_to(destination):
            if image_map.get(path) != ref:
                raise ContextRefusal("runtime image not identical to selected private original")
        else:
            if not absolute(path).is_relative_to(PurePosixPath("/usr/lib")):
                raise ContextRefusal("unreviewed runtime location")
            system.append(ref)
    if len(system) != 6 or len({absolute(path).name for path in runtime}) != 11:
        raise ContextRefusal("six unique original system images required")
    misses = finite_pairs(probe["finite_allowed_misses"], runtime)
    return Selection(
        str(destination),
        tuple(images),
        tuple(runtime.values()),
        tuple(misses),
        tuple(pair for pair in misses if pair[1] == "l_Lean_Parser_Tactic_simpLemma___boxed"),
        executable,
        interpreter,
        (str(old), str(archived)),
    )


@dataclass(frozen=True)
class CheckedQuery:
    context_id: str
    module: str
    query: FileRef
    object: FileRef
    final_root: str
    support_roots: tuple[str, ...]
    axioms: tuple[tuple[str, tuple[str, ...]], ...]
    audit: FileRef
    types: FileRef
    imports: FileRef
    records: tuple[FileRef, ...]
    _key: object = field(repr=False, compare=False)


def diagnostic_failure(stdout, sources):
    """Closed one-line diagnostic protocol; unknown/mixed buffers stop context.

    This is intentionally narrower than Lean's full error grammar. In particular,
    a source location does NOT make interpreter/load/resource errors a logical
    refusal. Multi-line diagnostics remain infrastructure-unclassified until an
    exact grammar and retained examples receive separate review.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False
    for line in lines:
        filename, sep, tail = line.partition(":")
        row, sep2, tail = tail.partition(":")
        column, sep3, message = tail.partition(":")
        if not (
            sep
            and sep2
            and sep3
            and filename in sources
            and row.isascii()
            and row.isdigit()
            and column.isascii()
            and column.isdigit()
            and message in (" error: type mismatch", " error: unsolved goals")
        ):
            return False
    return True
