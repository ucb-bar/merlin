"""A MINIMAL C ISA header for a RoCC target, generated deterministically from RTL facts + contract.

A fresh compiler experiment must not find a vendor kernel library on its include path: an upstream
accelerator header can ship complete kernels next to its instruction macros, and a harness that
can ``#include`` it makes them callable by every candidate. So the
runner-owned build places only the bare-metal runtime (CRT, link script, printf/util basics) and this
generated header on the include path.

The header states hardware FACTS and nothing else:

* the RoCC custom opcode and every funct the elaborated decoder accepts, by the RTL source's own
  names (``interfaces.funct_decode_table``: header-only codes the decoder lacks are NOT emitted);
* array, memory and datapath geometry/dtypes from the fact bundle;
* the contract's reviewed encoding anchors (``encoding.addr_len``, ``encoding.rocc_custom_slot``) and
  the DMA payload bound (``memory_model.dma.max_transfer_bytes``);
* one ``.insn r`` macro per RoCC operand shape (standard RoCC xd/xs1/xs2 funct3 bits) and a fence.

No routine, loop, tiling or layout convention is generated. Output is a pure function of the
fact bytes and the contract's values, so its SHA-256 identifies it; :func:`materialize` writes it
under ``out/build/isa_headers/<target>/<sha16>/`` and never rewrites a different header in place.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- isa_header_gen ----------------------------------------------------------------------------

GENERATOR = "merlin.target.isa.headers.v1"


class IsaHeaderError(RuntimeError):
    """The facts or the contract cannot ground a header; nothing is guessed in its place."""


#: C fixed-width spelling of a fact-bundle dtype token.
_C_TYPES = {
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "f32": "float",
    "f64": "double",
}


def _ident(text: str) -> str:
    out = "".join(ch.upper() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    if not out or out[0].isdigit():
        raise IsaHeaderError(f"cannot form a C identifier from {text!r}")
    return out


def _decode_table(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in facts.get("interfaces") or ():
        if isinstance(row, Mapping) and row.get("name") == "funct_decode_table":
            return row
    raise IsaHeaderError("the RTL facts carry no interfaces.funct_decode_table")


def _int(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise IsaHeaderError(f"{what} must be a non-negative integer fact, got {value!r}")
    return value


def _register_layouts(prefix: str, body: Mapping[str, Any]) -> list[str]:
    """Each command-register field's bit offset and width, from the RTL's own Bundle declarations
    (``interfaces.register_bundle_layouts``). A field whose width is a parameter the facts did not
    resolve gets its packed slot width, which is the width the field occupies in the register."""
    row = next(
        (
            r
            for r in body.get("interfaces") or ()
            if isinstance(r, Mapping) and r.get("name") == "register_bundle_layouts"
        ),
        None,
    )
    if row is None:
        return []
    lines = ["", "/* command-register field layouts (RTL Bundle declarations): bit offset, bit width */"]
    for bundle, layout in sorted((row.get("bundles") or {}).items()):
        for field, spec in (layout.get("fields") or {}).items():
            if not isinstance(spec, Mapping) or spec.get("offset") is None:
                continue
            width = spec.get("width") if spec.get("width") is not None else spec.get("slot_width")
            name = f"{prefix}_{_ident(bundle)}_{_ident(field)}"
            lines.append(f"#define {name}_OFFSET {_int(spec['offset'], f'{bundle}.{field}.offset')}")
            if width is not None:
                lines.append(f"#define {name}_WIDTH {_int(width, f'{bundle}.{field}.width')}")
    return lines


def _contract_encodings(prefix: str, contract: Mapping[str, Any]) -> list[str]:
    """Reviewed encoding facts the RTL cannot ground: the CONFIG selector codes and the local-address
    flag bits (``rocc_operand_roles.local_address_flags``, counted down from the address MSB)."""
    encoding = contract.get("encoding") or {}
    lines: list[str] = []
    subtypes = encoding.get("config_subtype") or {}
    if isinstance(subtypes, Mapping) and subtypes:
        lines += ["", "/* CONFIG selector codes (contract encoding.config_subtype) */"]
        for code, name in sorted(subtypes.items(), key=lambda kv: int(kv[0])):
            lines.append(f"#define {prefix}_{_ident(name)} {_int(int(code), 'config_subtype code')}")
    flags = ((contract.get("rocc_operand_roles") or {}).get("local_address_flags")) or {}
    addr_len = encoding.get("addr_len")
    if isinstance(flags, Mapping) and flags and addr_len is not None:
        top = _int(addr_len, "encoding.addr_len") - 1
        lines += ["", "/* local-address flag bits (contract rocc_operand_roles.local_address_flags) */"]
        for name, spec in flags.items():
            offset = _int((spec or {}).get("msb_offset"), f"local_address_flags.{name}.msb_offset")
            lines.append(f"#define {prefix}_LOCAL_ADDR_{_ident(name)}_BIT {top - offset}")
    return lines


def render(target: str, *, facts: Mapping[str, Any], contract: Mapping[str, Any], facts_sha256: str) -> str:
    """The header text. Raises :class:`IsaHeaderError` when a required fact is absent."""
    body = facts.get("facts", facts)
    prefix = _ident(target)
    table = _decode_table(body)
    opcode = _int(table.get("custom_opcode"), "funct_decode_table.custom_opcode")
    legal = table.get("legal_funct")
    names = table.get("names") or {}
    if not isinstance(legal, list) or not legal or not isinstance(names, Mapping):
        raise IsaHeaderError("funct_decode_table needs legal_funct and names")
    encoding = contract.get("encoding") or {}
    lines = [
        f"/* GENERATED by {GENERATOR} for target {target!r}. DO NOT EDIT.",
        f" * RTL facts sha256 {facts_sha256}; contract encoding anchors and DMA bound.",
        " * Hardware facts only: no kernel routine, tiling or layout convention is provided here. */",
        f"#ifndef MERLIN_{prefix}_ISA_H",
        f"#define MERLIN_{prefix}_ISA_H",
        "",
        "#include <stdint.h>",
        "",
        "/* RoCC encoding (elaborated decoder; the contract's reviewed anchors) */",
        f"#define {prefix}_ROCC_OPCODE {opcode}",
    ]
    if encoding.get("rocc_custom_slot") is not None:
        lines.append(
            f"#define {prefix}_ROCC_CUSTOM_SLOT {_int(encoding['rocc_custom_slot'], 'encoding.rocc_custom_slot')}"
        )
    if encoding.get("addr_len") is not None:
        lines.append(f"#define {prefix}_ADDR_LEN {_int(encoding['addr_len'], 'encoding.addr_len')}")
    complete = table.get("complete_isa")
    lines += [
        "",
        f"/* funct values the RTL decoder accepts (complete_isa={str(complete).lower()}); a code the decoder",
        " * does not accept is not emitted even when a software header names it. */",
    ]
    seen: set[str] = set()
    for code in sorted(_int(value, "legal_funct entry") for value in legal):
        name = names.get(str(code))
        if not isinstance(name, str) or not name:
            lines.append(f"#define {prefix}_FUNCT_{code} {code}")
            continue
        ident = _ident(name)
        if ident in seen:
            raise IsaHeaderError(f"funct name {name!r} maps to a duplicate identifier")
        seen.add(ident)
        lines.append(f"#define {prefix}_FUNCT_{ident} {code}")
    lines += ["", "/* geometry and storage (RTL facts) */"]
    for array in body.get("arrays") or ():
        if not isinstance(array, Mapping) or not array.get("name"):
            continue
        name = _ident(array["name"])
        for key in ("rows", "cols"):
            if array.get(key) is not None:
                lines.append(f"#define {prefix}_{name}_{key.upper()} {_int(array[key], f'arrays.{key}')}")
    for memory in body.get("memories") or ():
        if not isinstance(memory, Mapping) or not memory.get("name"):
            continue
        name = _ident(memory["name"])
        for key in ("banks", "depth", "row_elems", "lanes", "elem_bits", "bytes"):
            if memory.get(key) is not None:
                lines.append(f"#define {prefix}_{name}_{key.upper()} {_int(memory[key], f'memories.{key}')}")
    dma = ((contract.get("memory_model") or {}).get("dma") or {}).get("max_transfer_bytes")
    if dma is not None:
        lines.append(f"#define {prefix}_DMA_MAX_TRANSFER_BYTES {_int(dma, 'memory_model.dma.max_transfer_bytes')}")
    lines += _register_layouts(prefix, body)
    lines += _contract_encodings(prefix, contract)
    lines += ["", "/* datapath element types (RTL facts) */"]
    for datapath in body.get("datapaths") or ():
        if not isinstance(datapath, Mapping) or not datapath.get("name"):
            continue
        ctype = _C_TYPES.get(str(datapath.get("dtype")))
        if ctype is None:
            raise IsaHeaderError(f"datapath {datapath.get('name')!r} has no C type for {datapath.get('dtype')!r}")
        lines.append(f"typedef {ctype} {prefix.lower()}_{_ident(datapath['name']).lower()}_t;")
    lines += [
        "",
        "/* one RoCC instruction: .insn r opcode, funct3={xd,xs1,xs2}, funct7, rd, rs1, rs2 */",
        f"#define {prefix}_ROCC(funct, rs1, rs2) \\",
        f'  __asm__ volatile(".insn r %0, 3, %1, x0, %2, %3" : : "i"({prefix}_ROCC_OPCODE), "i"(funct), '
        '"r"((uint64_t)(rs1)), "r"((uint64_t)(rs2)) : "memory")',
        f"#define {prefix}_ROCC_RD(rd, funct, rs1, rs2) \\",
        f'  __asm__ volatile(".insn r %1, 7, %2, %0, %3, %4" : "=r"(rd) : "i"({prefix}_ROCC_OPCODE), "i"(funct), '
        '"r"((uint64_t)(rs1)), "r"((uint64_t)(rs2)) : "memory")',
        f'#define {prefix}_FENCE() __asm__ volatile("fence" : : : "memory")',
        "",
        f"#endif /* MERLIN_{prefix}_ISA_H */",
        "",
    ]
    return "\n".join(lines)


def materialize(target: str, name: str, *, facts_path: Path | None = None, contract: Mapping | None = None) -> Path:
    """Write the header for ``target`` (once per content) and return its path."""
    from merlin.common.paths import build_dir
    from merlin.target.registry import identity as target_registry
    from merlin.target.rtl import facts as rtl_facts

    if not name or Path(name).name != name or not name.endswith(".h"):
        raise IsaHeaderError(f"generated header name must be a plain .h file name, got {name!r}")
    path = Path(facts_path) if facts_path is not None else rtl_facts.rtl_facts_path(target)
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise IsaHeaderError(f"{target}: RTL facts {path} cannot be read: {exc}") from exc
    declared = (document.get("facts") or {}).get("target") if isinstance(document, dict) else None
    if declared not in (None, target):
        raise IsaHeaderError(f"{path}: facts describe {declared!r}, not {target!r}")
    selected = dict(contract) if contract is not None else target_registry.resolve(target).load_contract()
    text = render(target, facts=document, contract=selected, facts_sha256=hashlib.sha256(raw).hexdigest())
    digest = hashlib.sha256(text.encode()).hexdigest()
    out = build_dir() / "isa_headers" / target / digest[:16] / name
    if out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == digest:
        return out
    if out.exists() or out.is_symlink():
        raise IsaHeaderError(f"{out} exists with different bytes; refusing to rewrite a content-addressed header")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(f".{digest[:8]}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(out)
    return out


# ---- isa_headers_spec --------------------------------------------------------------------------
# A target's bare-metal ISA-header environment, described as DATA and verified by content.
#
# A RoCC accelerator's software interface ships as C headers plus a small bare-metal runtime (CRT,
# syscalls, link script) in an upstream checkout. A support provider used to COPY that tree into
# itself, which made the provider a second, unpinned home for bytes whose identity matters: two
# elaborations of one generator differ in their generated parameter header, and a harness built
# against the wrong one returns wrong numbers rather than failing.
#
# So the provider carries only a spec (schema :data:`SCHEMA`) that names the upstream checkout by its
# registry pin and commit, locates it through an operator variable, and commits to the SHA-256 of
# every file the build reads. Resolution is fail-closed: a missing root, a missing file or different
# bytes raise :class:`IsaHeadersError`; an override variable may RELOCATE identical bytes, never
# authorize different ones. Nothing here names a target, a header or a flag; all of it is data::
#
#     schema: merlin.isa_headers.v1
#     target: <target>
#     source:
#       pin: <hardware_pins.yaml entry>
#       repository: <url>
#       commit: <40 hex>
#       root_env: <VARIABLE naming the container checkout>
#       root_ext: <MERLIN_EXT_<NAME> key, optional>
#       path: <checkout-relative directory of the header tree>
#     include_roots: [<tree-relative dir>, ...]          # compiler -I order
#     crt: {sources: [<file>, ...], link_script: <file>, stack_bytes_per_hart: <int>, evidence: <text>}
#     kernel_stack: {max_static_bytes: <int>, evidence: <text>}
#     cflags: [...]
#     ldflags: [...]
#     header_dependencies: [<file>, ...]                  # direct children of an include root
#     files: {<tree-relative file>: <sha256>, ...}         # every file the build reads
#     generated_header: {name: <file.h>}                  # merlin.target.isa.headers output
#     console_write: {protocol: htif_syscall, write_syscall: <int>, host_symbols: {request, response},
#                     symbol: <C name>}                   # a generated length-taking console write
#     excluded_from_include_path: [<relative name>, ...]  # must NOT resolve from any include root
#     evidence_files: {<tree-relative file>: <sha256>}     # reviewed bytes, never on the include path
#
# The exclusion list exists because an upstream accelerator header can ship a whole kernel library:
# a runner-owned build that can ``#include`` it makes those kernels callable by every candidate. Resolving
# any excluded name from any include root is a refusal, not a warning.

SCHEMA = "merlin.isa_headers.v1"


class IsaHeadersError(RuntimeError):
    """The declared header environment cannot be produced exactly as declared."""


#: Verified (path, size, mtime_ns) -> digest. Keyed on the stat fields so a rewritten file is re-hashed.
_VERIFIED: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    cached = _VERIFIED.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _VERIFIED[key] = value
    return value


def _relative(value: Any, *, what: str, spec: Path) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise IsaHeadersError(f"{spec}: {what} must be a non-empty tree-relative path, got {value!r}")
    return value


def _strings(value: Any, *, what: str, spec: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise IsaHeadersError(f"{spec}: {what} must be a list of non-empty strings")
    return tuple(value)


def _positive(value: Any, *, what: str, spec: Path) -> int:
    if type(value) is not int or value <= 0:
        raise IsaHeadersError(f"{spec}: {what} must be a positive integer")
    return value


@dataclass(frozen=True)
class IsaHeaders:
    """One resolved, content-pinned header environment."""

    target: str
    spec_path: Path
    spec_sha256: str
    root: Path
    root_source: str
    source: dict[str, Any]
    include_roots: tuple[Path, ...]
    crt_sources: tuple[Path, ...]
    link_script: Path
    stack_bytes_per_hart: int
    kernel_max_static_bytes: int
    cflags: tuple[str, ...]
    ldflags: tuple[str, ...]
    header_dependencies: tuple[Path, ...]
    link_first: tuple[Path, ...] = ()
    files: dict[str, str] = field(default_factory=dict)
    generated_header: str | None = None
    console_write: dict[str, Any] | None = None
    excluded: tuple[str, ...] = ()
    evidence_files: dict[str, str] = field(default_factory=dict)

    def path(self, relative: str) -> Path:
        """``relative`` under the header tree; only a DECLARED file may be asked for."""
        if relative not in self.files:
            raise IsaHeadersError(
                f"{self.target}: {relative!r} is not a declared file of {self.spec_path}; only files whose "
                "bytes the spec commits to may be read"
            )
        return self.root / relative

    def verify(self) -> None:
        """Every declared file exists and hashes to its declared digest, and no excluded header is
        reachable from an include root; or raise."""
        self.verify_exclusions(self.include_roots)
        for relative, declared in self.files.items():
            path = self.root / relative
            if path.is_symlink() or not path.is_file():
                raise IsaHeadersError(
                    f"{self.target}: declared ISA-header file {relative!r} is absent under {self.root} "
                    f"({self.root_source}); refusing to build against a different header tree"
                )
            actual = _sha256(path)
            if actual != declared:
                raise IsaHeadersError(
                    f"{self.target}: {path} has sha256 {actual}, but {self.spec_path} declares {declared} "
                    f"(pin {self.source.get('pin')!r} at {self.source.get('commit')}). These are DIFFERENT "
                    "bytes, so a harness built here is not the reviewed software interface"
                )

    def verify_exclusions(self, include_roots) -> None:
        """Refuse when any excluded name resolves from any of ``include_roots``."""
        for root in include_roots:
            for name in self.excluded:
                if (Path(root) / name).exists():
                    raise IsaHeadersError(
                        f"{self.target}: {name!r} is reachable from include root {root}; that header is excluded "
                        f"from the runner-owned build ({self.spec_path}), so the include path is refused"
                    )

    def record(self) -> dict[str, Any]:
        """A JSON-safe citation of what was resolved."""
        return {
            "schema": SCHEMA,
            "spec": str(self.spec_path),
            "spec_sha256": self.spec_sha256,
            "root": str(self.root),
            "root_source": self.root_source,
            "source": dict(self.source),
            "files": dict(self.files),
        }


def _root(target: str, source: dict[str, Any], *, override_env: str | None, spec: Path) -> tuple[Path, str]:
    from merlin.common.paths import ExternalPathUnset, env, ext_path

    if override_env:
        raw = env(override_env)
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                raise IsaHeadersError(f"{target}: {override_env}={raw!r} must be an absolute path")
            return path, f"override {override_env}"
    relative = _relative(source.get("path"), what="source.path", spec=spec)
    root_env = source.get("root_env")
    if isinstance(root_env, str) and root_env and env(root_env):
        return Path(str(env(root_env))) / relative, f"{root_env}/{relative}"
    root_ext = source.get("root_ext")
    if isinstance(root_ext, str) and root_ext:
        try:
            return ext_path(root_ext) / relative, f"MERLIN_EXT_{root_ext.upper()}/{relative}"
        except ExternalPathUnset as exc:
            raise IsaHeadersError(
                f"{target}: the ISA-header checkout is located by {root_env or '(no variable)'} or "
                f"MERLIN_EXT_{root_ext.upper()}, and neither is set"
            ) from exc
    raise IsaHeadersError(f"{target}: {spec} locates its header checkout by {root_env!r}, which is unset")


def _read(target: str, spec: Path) -> tuple[bytes, dict[str, Any]]:
    import yaml

    try:
        raw = spec.read_bytes()
        doc = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise IsaHeadersError(f"{spec}: cannot read ISA-header spec: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise IsaHeadersError(f"{spec}: expected schema {SCHEMA!r}")
    if doc.get("target") != target:
        raise IsaHeadersError(f"{spec}: declares target {doc.get('target')!r}, not {target!r}")
    return raw, doc


def _console_write(value: Any, *, spec: Path) -> dict[str, Any] | None:
    """A length-taking console write the runtime lacks, described as data (see ``console_write``)."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("protocol") != "htif_syscall":
        raise IsaHeadersError(f"{spec}: console_write.protocol must be htif_syscall")
    symbols = value.get("host_symbols")
    if (
        type(value.get("write_syscall")) is not int
        or not isinstance(symbols, dict)
        or set(symbols) != {"request", "response"}
        or not all(isinstance(v, str) and v.isidentifier() for v in symbols.values())
        or not isinstance(value.get("symbol"), str)
        or not str(value["symbol"]).isidentifier()
    ):
        raise IsaHeadersError(f"{spec}: console_write needs write_syscall, host_symbols request/response and symbol")
    return dict(value)


def _generated_name(value: Any, *, spec: Path) -> str | None:
    if value is None:
        return None
    name = value.get("name") if isinstance(value, dict) else None
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".h"):
        raise IsaHeadersError(f"{spec}: generated_header.name must be a plain .h file name")
    return name


def declared_files(target: str, spec_path: str | Path) -> dict[str, str]:
    """The spec's committed ``file -> sha256`` map, read WITHOUT locating the checkout.

    For cross-checking other data reviewed against these bytes (a readout contract) on a host that
    has no header checkout at all.
    """
    _, doc = _read(target, Path(spec_path))
    files = doc.get("files")
    if not isinstance(files, dict):
        raise IsaHeadersError(f"{spec_path}: files must map every file the build reads to its sha256")
    evidence = doc.get("evidence_files") or {}
    if not isinstance(evidence, dict):
        raise IsaHeadersError(f"{spec_path}: evidence_files must map a file to its sha256")
    return {**{str(k): str(v) for k, v in files.items()}, **{str(k): str(v) for k, v in evidence.items()}}


def load(target: str, spec_path: str | Path, *, override_env: str | None = None, verify: bool = True) -> IsaHeaders:
    """Read ``spec_path`` (a provider resource) and resolve it against this host, fail-closed."""
    spec = Path(spec_path)
    raw, doc = _read(target, spec)
    source = doc.get("source")
    if not isinstance(source, dict):
        raise IsaHeadersError(f"{spec}: source must be a mapping")
    commit = source.get("commit")
    if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
        raise IsaHeadersError(f"{spec}: source.commit must be a full lowercase 40-hex sha")
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        raise IsaHeadersError(f"{spec}: files must map every file the build reads to its sha256")
    for relative, digest in files.items():
        _relative(relative, what="files key", spec=spec)
        if not (isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
            raise IsaHeadersError(f"{spec}: files[{relative!r}] must be a lowercase sha256")
    crt = doc.get("crt")
    if not isinstance(crt, dict):
        raise IsaHeadersError(f"{spec}: crt must be a mapping")
    kernel_stack = doc.get("kernel_stack")
    if not isinstance(kernel_stack, dict):
        raise IsaHeadersError(f"{spec}: kernel_stack must be a mapping")
    root, root_source = _root(target, source, override_env=override_env, spec=spec)

    def declared_file(value: Any, what: str) -> Path:
        relative = _relative(value, what=what, spec=spec)
        if relative not in files:
            raise IsaHeadersError(f"{spec}: {what} {relative!r} is not committed to under files")
        return root / relative

    includes = tuple(
        root / _relative(item, what="include_roots entry", spec=spec) if item != "." else root
        for item in _strings(doc.get("include_roots"), what="include_roots", spec=spec)
    )
    stack = _positive(crt.get("stack_bytes_per_hart"), what="crt.stack_bytes_per_hart", spec=spec)
    kernel_max = _positive(kernel_stack.get("max_static_bytes"), what="kernel_stack.max_static_bytes", spec=spec)
    if kernel_max >= stack:
        raise IsaHeadersError(f"{spec}: kernel_stack.max_static_bytes must leave stack for the caller chain")
    resolved = IsaHeaders(
        target=target,
        spec_path=spec.resolve(),
        spec_sha256=hashlib.sha256(raw).hexdigest(),
        root=root,
        root_source=root_source,
        source=dict(source),
        include_roots=includes,
        crt_sources=tuple(
            declared_file(item, "crt.sources entry")
            for item in _strings(crt.get("sources"), what="crt.sources", spec=spec)
        ),
        link_script=declared_file(crt.get("link_script"), "crt.link_script"),
        link_first=tuple(
            declared_file(item, "crt.link_first entry")
            for item in _strings(crt.get("link_first") or [], what="crt.link_first", spec=spec)
        )
        if crt.get("link_first")
        else (),
        stack_bytes_per_hart=stack,
        kernel_max_static_bytes=kernel_max,
        cflags=_strings(doc.get("cflags"), what="cflags", spec=spec),
        ldflags=_strings(doc.get("ldflags") or [], what="ldflags", spec=spec) if doc.get("ldflags") else (),
        header_dependencies=tuple(
            declared_file(item, "header_dependencies entry")
            for item in _strings(doc.get("header_dependencies") or [], what="header_dependencies", spec=spec)
        )
        if doc.get("header_dependencies")
        else (),
        files={str(k): str(v) for k, v in files.items()},
        generated_header=_generated_name(doc.get("generated_header"), spec=spec),
        console_write=_console_write(doc.get("console_write"), spec=spec),
        excluded=_strings(doc.get("excluded_from_include_path") or [], what="excluded_from_include_path", spec=spec)
        if doc.get("excluded_from_include_path")
        else (),
        evidence_files={str(k): str(v) for k, v in (doc.get("evidence_files") or {}).items()},
    )
    if verify:
        resolved.verify()
    return resolved


def environment_root(target: str) -> str:
    """The per-target variable that may RELOCATE (never replace) the header tree."""
    from merlin.common.paths import target_env_name

    return target_env_name(target, "HARNESS_DIR")


__all__ = [
    "GENERATOR",
    "IsaHeaderError",
    "IsaHeaders",
    "IsaHeadersError",
    "SCHEMA",
    "load",
    "materialize",
    "render",
]
