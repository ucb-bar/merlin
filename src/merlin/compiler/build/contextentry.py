"""Original-caller admission and checked initialization for stateful C images.

This owns the software entry sequence, not a platform implementation or hardware
certificate. Opaque boot startup and platform effects remain caller obligations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from inspect import getsourcefile
from pathlib import Path, PurePosixPath

from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.backend.stateful import CContext, CRegionABI2, read_c_abi
from merlin.compiler.backend.workspace import CWorkspaceABI
from merlin.compiler.ir.semantics import validator_identity


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _member(value):
    if (type(value) is not str or not value or "\\" in value or ":" in value or
            PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts or
            PurePosixPath(value).as_posix() != value):
        raise ValueError("image entry members require portable relative paths")


def _identifier(value):
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    return (type(value) is str and bool(value) and value[0] in letters and
            all(c in letters + "0123456789" for c in value))


@dataclass(frozen=True)
class CImageContext:
    """An original platform contract and explicitly admitted initializer ABI.

    The caller validator must establish that this particular platform implements
    the selected initialization/loading/completion and error-lifetime contract.
    A matching digest establishes identity, not the truth of those claims.
    """
    context: CContext
    platform: Compilation
    contract: str
    initializer_abi: str
    workspace_initialization: str
    initializer_effects: str
    _record: str = field(init=False, repr=False)
    _manifest: Artifact = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.context) is not CContext or type(self.platform) is not Compilation:
            raise ValueError("image context requires original typed context and platform Compilation")
        _member(self.contract)
        if (self.initializer_abi != "int(context*,void*,size_t)" or
                self.workspace_initialization != "contents_not_retained" or
                self.initializer_effects != "host_state_only"):
            raise ValueError("image context has no admitted initializer signature")
        object.__setattr__(self, "_owners", (self.context, self.platform))
        object.__setattr__(self, "_record", self.record())
        object.__setattr__(self, "_manifest", Artifact.from_file(self.platform.manifest, kind="json", role="platform"))
        self.revalidate()

    def record(self):
        return _json({"context": self.context.record(), "platform": self.platform.to_dict(),
                      "contract": self.contract, "initializer_abi": self.initializer_abi,
                      "workspace_initialization": self.workspace_initialization,
                      "initializer_effects": self.initializer_effects})

    def revalidate(self):
        if (self.context is not self._owners[0] or self.platform is not self._owners[1] or
                self.record() != self._record):
            raise ValueError("original image context/platform selection changed")
        self._manifest.verify()
        self.platform.verify()
        if Compilation.load(self.platform.manifest).to_dict() != self.platform.to_dict():
            raise ValueError("image platform differs from its original saved receipt")
        product = next((a for a in self.platform.artifacts
                        if a.path.relative_to(self.platform.directory).as_posix() == self.contract), None)
        if product is None or product.kind != "json" or product.sha256 != self.context.contract_sha256:
            raise ValueError("image context is not bound to its selected platform contract artifact")


@dataclass(frozen=True)
class ReservedImageInterval:
    """Caller-approved address extent excluded from linked ELF LOAD segments."""
    address_space: str
    address: int
    bytes: int

    def __post_init__(self):
        if (self.address_space not in {"physical", "virtual"} or type(self.address) is not int or
                type(self.bytes) is not int or self.address < 0 or self.bytes <= 0 or
                self.address + self.bytes > 1 << 64):
            raise ValueError("image reservation requires an explicit address space and bounded extent")

    def record(self):
        self.__post_init__()
        return {"address_space": self.address_space, "address": self.address, "bytes": self.bytes}


class CImageEntry:
    """Opt-in generated initializer -> context table -> selected dispatcher.

    ``validate_platform(binding, region, image_selection, options)`` is the
    original caller's independent platform admission, not a backend hook. It
    raises on refusal and returns None. The caller owns the meaning of evidence
    and options; this mechanism never upgrades claims to execution evidence.
    ``linker_scripts`` and ``configuration`` name retained startup artifacts.
    """
    def __init__(self, *, contexts, symbol, validate_platform, options, reservations,
                 entry_state_bytes, entry_state_pre_reserved,
                 linker_scripts=(), configuration=(), evidence=()):
        if (type(contexts) is not tuple or not contexts or any(type(c) is not CImageContext for c in contexts) or
                len({c.context.identity for c in contexts}) != len(contexts) or not _identifier(symbol)):
            raise ValueError("stateful image entry needs distinct typed contexts and a C symbol")
        if (type(reservations) is not tuple or not reservations or
                any(type(r) is not ReservedImageInterval for r in reservations)):
            raise ValueError("stateful image entry needs explicit reserved address intervals")
        for members in (linker_scripts, configuration):
            if type(members) is not tuple or len(set(members)) != len(members):
                raise ValueError("image linker/configuration members must be distinct immutable tuples")
            for member in members:
                _member(member)
        if type(options) is not dict or type(evidence) is not tuple or any(type(a) is not Artifact for a in evidence):
            raise ValueError("image platform admission needs explicit options and immutable evidence artifacts")
        if (type(entry_state_bytes) is not int or entry_state_bytes <= 0 or
                entry_state_pre_reserved is not True):
            raise ValueError("image entry requires an explicit pre-reserved startup bookkeeping ceiling")
        self.entry_state_bytes = entry_state_bytes
        self.entry_state_pre_reserved = entry_state_pre_reserved
        self.contexts, self.symbol, self.options = contexts, symbol, options
        self.reservations, self.linker_scripts, self.configuration = reservations, linker_scripts, configuration
        self.validate_platform, self.evidence = validate_platform, evidence
        self._validator = validator_identity(validate_platform)
        source = getsourcefile(self._validator[1])
        if source is None:
            raise ValueError("image platform validator requires an observable direct source file")
        paths = (source, __file__, Path(__file__).with_name("image.py"), getsourcefile(CRegionABI2),
                 getsourcefile(CWorkspaceABI))
        self._sources = tuple(Artifact.from_file(Path(p), kind="source", role="image-admission") for p in paths)
        self._owners = (contexts, options, validate_platform, reservations, linker_scripts, configuration, evidence)
        self._identity = self.identity()
        self.revalidate()

    def identity(self):
        return _json({"contexts": [json.loads(c.record()) for c in self.contexts], "symbol": self.symbol,
                      "options": self.options, "reservations": [r.record() for r in self.reservations],
                      "linker_scripts": self.linker_scripts, "configuration": self.configuration,
                      "entry_state_bytes": self.entry_state_bytes,
                      "entry_state_pre_reserved": self.entry_state_pre_reserved,
                      "evidence": [(str(a.path), a.sha256, a.kind, a.role) for a in self.evidence]})

    def revalidate(self):
        current = (self.contexts, self.options, self.validate_platform, self.reservations,
                   self.linker_scripts, self.configuration, self.evidence)
        if any(a is not b for a, b in zip(current, self._owners, strict=True)) or self.identity() != self._identity:
            raise ValueError("original stateful image entry selection changed")
        if any(a is not b for a, b in zip(validator_identity(self.validate_platform), self._validator, strict=True)):
            raise ValueError("original image platform validator changed")
        for context in self.contexts:
            context.revalidate()
        for artifact in (*self.evidence, *self._sources):
            artifact.verify()

    def validate_flags(self, flags):
        # Scripts are added from retained members below. Linker response files
        # and path-bearing script flags cannot bypass that explicit selection.
        for flag in flags:
            parts = flag[4:].split(",") if flag.startswith("-Wl,") else (flag,)
            if any(p.startswith(("@", "-T", "--script", "--version-script", "--dynamic-list",
                                 "--retain-symbols-file", "--just-symbols", "-specs=")) for p in parts):
                raise ValueError("stateful image linker scripts require retained explicit members")

    def admit(self, selection, partition, products, description, *, directory):
        """Check actual ABI2 coverage and bracket the original caller validator."""
        self.revalidate()
        selection.revalidate()
        partition.revalidate()
        original = _json(description)
        directory = Path(directory)
        artifacts = {a.path.relative_to(selection.startup.directory).as_posix(): a for a in selection.startup.artifacts}
        for name in (*self.linker_scripts, *self.configuration):
            if name not in artifacts:
                raise ValueError("image linker/configuration inputs must be retained startup artifacts")
        configured = {b.context.identity: b for b in self.contexts}
        actual, payloads = {}, []
        for product in products:
            product.revalidate()
            region = product.compilation
            abi = read_c_abi(region)
            if not isinstance(abi, CRegionABI2):
                if (not isinstance(abi, CWorkspaceABI) and
                        region.metadata.get("c_call", {}).get("schema") != "merlin.c_call.v1" or abi.payloads):
                    raise ValueError("stateful image entry does not admit this artifact/workspace ABI")
                continue
            binding = configured.get(abi.context.identity)
            if binding is None or binding.context != abi.context:
                raise ValueError("stateful image entry differs from selected original contexts")
            if abi.context.identity not in actual:
                declaration = (region.directory / abi.context.declaration).relative_to(directory)
                header = next(a for a in region.artifacts if a.path == directory / declaration)
                actual[abi.context.identity] = (abi, declaration.as_posix(), header.sha256)
                payloads.extend((region.directory / p).relative_to(directory).as_posix()
                                for p in abi.payloads)
                result = self.validate_platform(binding, region, selection, self.options)
                if result is not None:
                    raise ValueError("image platform validator must raise on refusal and return None")
                self.revalidate()
                selection.revalidate()
                partition.revalidate()
                product.revalidate()
                if _json(description) != original:
                    raise ValueError("image platform validation changed its original control description")
        self.revalidate()
        selection.revalidate()
        partition.revalidate()
        for product in products:
            product.revalidate()
        if _json(description) != original:
            raise ValueError("image platform validation changed its original control description")
        if (set(actual) != set(configured) or sorted(payloads) != sorted(description["payloads"]) or
                set(actual) != set(description.get("context_indices", {}))):
            raise ValueError("stateful image entry must cover every selected context and payload exactly")
        return actual

    def sources(self, actual, description):
        """Generate exact typed initialization and persistent error retention."""
        entry = description["entry"]
        if not entry.endswith("_run") or not _identifier(entry[:-4]) or self.symbol == entry:
            raise ValueError("stateful image entry conflicts with selected dispatcher")
        name = entry[:-4]
        contexts = sorted(actual, key=lambda key: description["context_indices"][key])
        if [description["context_indices"][key] for key in contexts] != list(range(len(contexts))):
            raise ValueError("stateful image entry requires a complete ordered context table")
        guard = "MERLIN_IMAGE_ENTRY_" + self.symbol.upper() + "_H"
        header = f"""#ifndef {guard}
#define {guard}
#include "{name}.h"
int {self.symbol}({name}_state *, void *, size_t, const {name}_buffer *, size_t);
#endif
"""
        lines = ['#include "entry.h"', '#include <stdatomic.h>']
        declarations = {}
        for key in contexts:
            abi, declaration, sha = actual[key]
            declarations.setdefault((abi.context.ctype, sha), declaration)
        for declaration in declarations.values():
            lines.append('#include ' + json.dumps(declaration))
        domains = {}
        for key in contexts:
            c = actual[key][0].context
            size, alignment = domains.get(c.memory_domain, (0, 1))
            domains[c.memory_domain] = (max(size, c.workspace_bytes), max(alignment, c.workspace_alignment))
        domain_indices = {domain: i for i, domain in enumerate(domains)}
        for domain, (size, alignment) in domains.items():
            if size:
                lines.append(f'_Alignas({alignment}) static unsigned char workspace_{domain_indices[domain]}[{size}];')
        for index, key in enumerate(contexts):
            c = actual[key][0].context
            lines.extend((f'_Static_assert(sizeof({c.ctype})<={c.state_bytes}ULL, "image context state ceiling");',
                f'_Static_assert(_Alignof({c.ctype})<={c.state_alignment}ULL, "image context alignment");',
                f'_Static_assert(_Generic(&{c.initialize}, int (*)({c.ctype} *, void *, size_t):1, default:0), '
                '"admitted image initializer signature");',
                f'_Alignas({c.state_alignment}) static {c.ctype} context_{index};'))
        lines.extend(('typedef struct { atomic_flag lock; unsigned initialized,failed;',
            f' void *items[{len(contexts)}]; {name}_contexts contexts; }} entry_state;',
            f'_Static_assert(sizeof(entry_state)<={self.entry_state_bytes}ULL, "image startup state ceiling");',
            'static entry_state entry={.lock=ATOMIC_FLAG_INIT};'))
        if description.get("source_workspace"):
            lines.extend(self._mixed_preflight(contexts, domains, domain_indices, name, description))
        lines.extend((f'int {self.symbol}({name}_state *state, void *arena, size_t bytes,',
            f' const {name}_buffer *buffers, size_t count) {{',
            f' if(atomic_flag_test_and_set(&entry.lock)) return {name.upper()}_BUSY;',
            f' int result={name.upper()}_BUSY;', ' if(entry.failed) goto done;'))
        if description.get("source_workspace"):
            lines.append(f' if(!entry_preflight(state,arena,bytes,buffers,count)) '
                         f'{{result={name.upper()}_INVALID;goto done;}}')
        lines.append(' if(!entry.initialized) {')
        for index, key in enumerate(contexts):
            c = actual[key][0].context
            workspace = f'workspace_{domain_indices[c.memory_domain]}' if c.workspace_bytes else '0'
            lines.append(f'  if({c.initialize}(&context_{index},{workspace},{c.workspace_bytes}u)) '
                         f'{{entry.failed=1;result={name.upper()}_KERNEL_ERROR;goto done;}}')
        for index in range(len(contexts)):
            lines.append(f'  entry.items[{index}]=&context_{index};')
        lines.extend((f'  entry.contexts=({name}_contexts){{entry.items,{len(contexts)}u}};',
            '  entry.initialized=1;', ' }',
            f' result={entry}(state,arena,bytes,buffers,count,&entry.contexts);',
            ' if(result) entry.failed=1;', 'done:', ' atomic_flag_clear(&entry.lock); return result;', '}'))
        return header, '\n'.join(lines) + '\n'



    @staticmethod
    def _mixed_preflight(contexts, domains, domain_indices, name, description):
        """Disjoint storage before initialization; valid C pointers remain a caller premise."""
        upper = name.upper()
        lines = ["""static int entry_overlap(const void *a,size_t an,const void *b,size_t bn) {
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
 return x<y+bn && y<x+an;
}
static int entry_separate(const void *p,size_t n) {
 uintptr_t address=(uintptr_t)p;
 if(!p || !n || address>UINTPTR_MAX-n) return 0;
 if(entry_overlap(p,n,&entry,sizeof(entry))) return 0;"""]
        for index in range(len(contexts)):
            lines.append(f' if(entry_overlap(p,n,&context_{index},sizeof(context_{index}))) return 0;')
        for domain, (size, _) in domains.items():
            if size:
                i = domain_indices[domain]
                lines.append(f' if(entry_overlap(p,n,workspace_{i},sizeof(workspace_{i}))) return 0;')
        lines.extend((' return 1;', '}', f"""static int entry_preflight(
 {name}_state *state,void *arena,size_t bytes,const {name}_buffer *buffers,size_t count) {{
 if(count!={upper}_BUFFER_COUNT || !entry_separate(state,sizeof(*state)) ||
    bytes<{upper}_ARENA_BYTES || !entry_separate(arena,{upper}_ARENA_BYTES) ||
    (count && !entry_separate(buffers,sizeof(*buffers)*count))) return 0;
 if(entry_overlap(state,sizeof(*state),arena,{upper}_ARENA_BYTES) ||
    (count && (entry_overlap(buffers,sizeof(*buffers)*count,state,sizeof(*state)) ||
               entry_overlap(buffers,sizeof(*buffers)*count,arena,{upper}_ARENA_BYTES)))) return 0;"""))
        for index, row in enumerate(description["buffers"]):
            size = row["bytes"]
            data = f'buffers[{index}].data'
            lines.append(f' if(!entry_separate({data},{size}u) || '
                         f'entry_overlap({data},{size}u,buffers,sizeof(*buffers)*count) || '
                         f'entry_overlap({data},{size}u,state,sizeof(*state)) || '
                         f'entry_overlap({data},{size}u,arena,{upper}_ARENA_BYTES)) return 0;')
        lines.extend((' return 1;', '}'))
        return lines
