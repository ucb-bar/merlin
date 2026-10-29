"""Observe public native accessor semantics without a target encoding table.

The protected selector supplies a minimal type/member specification, never
instruction constants, shifts, masks, effects or schedules. A generated native
program calls those actual public C++ accessors over a complete single-bit basis
and independent mixed words. Only live issued objects authorize further native
decode invocations. This is model/header ABI evidence, not physical RTL proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.paths import module_source_path

from .command_intake import _git, _tracked_source
from .rtl_intake import RtlIntakePin, RtlIntakeRefusal, _json, _outside, _pin, _plain

SCHEMA = "merlin.independent_native_accessor.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_KEYS = {
    "header",
    "union_type",
    "word_member",
    "fields_member",
    "conversion_type",
    "length_member",
    "fields",
    "constants",
}
_UNKNOWN = (
    "physical_cpu_encoding_and_byte_order",
    "selected_rtl_and_bitstream_equivalence",
    "instruction_effects_and_legality",
    "prohibited_instruction_policy",
    "physical_memory_completion_ownership_and_timers",
    "host_compiler_and_shared_library_dependency_closure",
)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value or not (value[0].isalpha() or value[0] == "_"):
        raise RtlIntakeRefusal("native accessor identifiers must be simple public C++ names")
    if not value.isascii() or not all(c.isalnum() or c == "_" for c in value):
        raise RtlIntakeRefusal("native accessor specification cannot contain expressions or code")
    return value


def _spec(path: Path) -> dict:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, dict) or set(raw) != _KEYS:
        raise RtlIntakeRefusal("native accessor specification must contain exactly the reviewed type/member fields")
    header = raw["header"]
    if not isinstance(header, str) or not header or Path(header).is_absolute():
        raise RtlIntakeRefusal("native accessor header must be a relative public include")
    if any(part in {".", ".."} for part in header.split("/")) or not all(
        c.isascii() and (c.isalnum() or c in "_-/.") for c in header
    ):
        raise RtlIntakeRefusal("native accessor header cannot escape or inject code")
    for key in _KEYS - {"header", "fields", "constants"}:
        _identifier(raw[key])
    for key in ("fields", "constants"):
        values = raw[key]
        if not isinstance(values, list) or (key == "fields" and not values):
            raise RtlIntakeRefusal("native accessor fields/constants need explicit reviewed lists")
        if len(set(_identifier(v) for v in values)) != len(values):
            raise RtlIntakeRefusal("native accessor names must be unique")
    if set(raw["fields"]) & {"word", "length"}:
        raise RtlIntakeRefusal("native accessor fields collide with fixed observation keys")
    return raw


def _program(spec: dict) -> str:
    fields = "\n".join(f'  std::cout << " " << u.{spec["fields_member"]}.{name};' for name in spec["fields"])
    constants = "\n".join(f' std::cout << " {name}=" << (uint64_t){name};' for name in spec["constants"])
    return f"""#include <cstdint>
#include <climits>
#include <iostream>
#include <{spec["header"]}>
int main() {{
 std::cout << "META " << sizeof(uint64_t)*CHAR_BIT;
{constants}
 std::cout << "\\n";
 uint64_t word;
 while (std::cin >> word) {{
  {spec["union_type"]} u;
  u.{spec["word_member"]} = {spec["conversion_type"]}(word);
  std::cout << word << " " << u.{spec["word_member"]}.{spec["length_member"]}();
{fields}
  std::cout << "\\n";
 }}
 return std::cin.eof() ? 0 : 2;
}}
"""


def _rows(stdout: bytes, words: tuple[int, ...], names: list[str]) -> tuple[dict, list[dict]]:
    lines = stdout.decode("ascii").splitlines()
    if not lines or not lines[0].startswith("META ") or len(lines) != len(words) + 1:
        raise RtlIntakeRefusal("native accessor output is incomplete")
    meta = lines[0].split()
    constants = {}
    for token in meta[2:]:
        key, value = token.split("=", 1)
        if key in constants:
            raise RtlIntakeRefusal("native accessor constant output is duplicated")
        constants[key] = int(value)
    rows = []
    for line, word in zip(lines[1:], words, strict=True):
        values = [int(value) for value in line.split()]
        if len(values) != len(names) + 2 or values[0] != word or any(value < 0 for value in values):
            raise RtlIntakeRefusal("native accessor rows do not exactly match invocation inputs")
        rows.append(dict(zip(["word", "length", *names], values, strict=True)))
    return {"word_bits": int(meta[1]), "constants": constants}, rows


def _invoke(executable: Path, words: tuple[int, ...], names: list[str], destination: Path) -> tuple[dict, list[dict]]:
    destination.mkdir(parents=True, exist_ok=False)
    if any(type(word) is not int or word < 0 or word >= 1 << 64 for word in words):
        raise RtlIntakeRefusal("native accessor inputs must be unsigned 64-bit observation words")
    input_bytes = "".join(str(word) + "\n" for word in words).encode()
    (destination / "input.txt").write_bytes(input_bytes)
    completed = invocation_record.run(
        [str(executable)],
        directory=destination,
        stage="native_accessor_decode",
        inputs=(destination / "input.txt",),
        input=input_bytes,
        capture_output=True,
        timeout=30,
    )
    # The invocation record already retains both exact process streams. The
    # decode receipt below pins those files; another copy can be hundreds of MB.
    if completed.returncode:
        raise RtlIntakeRefusal("actual native accessor invocation failed")
    return _rows(completed.stdout, words, names)


def _observed_fields(rows: list[dict], names: list[str], bits: int) -> dict:
    if bits != 64:
        raise RtlIntakeRefusal("native generated observation word representation changed")
    fields = {}
    for name in names:
        if rows[0][name] != 0:
            raise RtlIntakeRefusal("native accessor field is not a pure zero-based bit projection")
        active = [(i, row[name]) for i, row in enumerate(rows[1 : bits + 1]) if row[name]]
        if not active:
            raise RtlIntakeRefusal("native accessor field has no observed bits")
        offset = active[0][0]
        if active != [(offset + i, 1 << i) for i in range(len(active))]:
            raise RtlIntakeRefusal("native accessor field is not an exact contiguous bit projection")
        width = len(active)
        if any(row[name] != (row["word"] >> offset) & ((1 << width) - 1) for row in rows):
            raise RtlIntakeRefusal("native accessor mixed-word observations disagree with the single-bit basis")
        fields[name] = {"offset": offset, "width": width}
    return fields


@dataclass(frozen=True, eq=False)
class IndependentAccessorIntake:
    """Live public accessor ABI observations, with explicit physical unknowns."""

    source_pins: tuple[RtlIntakePin, ...]
    checkout: Path
    commit: str
    executable: Path
    forbidden_roots: tuple[Path, ...]
    specification_json: bytes
    facts_json: bytes
    receipt_json: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self) -> str:
        return hashlib.sha256(
            _json(
                {
                    "pins": [p.record() for p in self.source_pins],
                    "receipt": self.sha256,
                    "checkout": str(self.checkout),
                    "commit": self.commit,
                    "executable": str(self.executable),
                    "forbidden_roots": [str(p) for p in self.forbidden_roots],
                    "spec_sha256": hashlib.sha256(self.specification_json).hexdigest(),
                    "facts_sha256": hashlib.sha256(self.facts_json).hexdigest(),
                }
            )
        ).hexdigest()

    def verify(self) -> None:
        if _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("native accessor evidence requires live independently issued authority")
        for pin in self.source_pins:
            pin.verify()
        if _git(self.checkout, "rev-parse", "HEAD") != self.commit or _git(
            self.checkout, "status", "--porcelain", "--untracked-files=no"
        ):
            raise RtlIntakeRefusal("public native accessor source checkout changed")
        if json.loads(self.receipt_json)["facts_sha256"] != hashlib.sha256(self.facts_json).hexdigest():
            raise RtlIntakeRefusal("native accessor fact projection changed")

    def public_facts(self) -> dict:
        self.verify()
        return json.loads(self.facts_json)

    def verify_public_facts(self, path: str | Path) -> None:
        self.verify()
        if _plain(path).read_bytes() != self.facts_json:
            raise RtlIntakeRefusal("public native accessor projection differs from actual issued observations")

    def decode_words(self, words, evidence_root: str | Path) -> dict:
        """Call actual native source accessors; persist this exact diagnostic run."""
        self.verify()
        destination = Path(evidence_root).absolute()
        _outside(destination, self.forbidden_roots)
        if destination.is_relative_to(self.checkout):
            raise RtlIntakeRefusal("native decode diagnostics cannot modify public source checkout")
        spec = json.loads(self.specification_json)
        meta, rows = _invoke(self.executable, tuple(words), spec["fields"], destination)
        facts = json.loads(self.facts_json)
        if meta != {"word_bits": facts["native_observation_word_bits"], "constants": facts["source_constants"]}:
            raise RtlIntakeRefusal("native accessor metadata changed on replay")
        pins = [_pin("native-invocation", p, ()) for p in sorted(destination.rglob("*")) if p.is_file()]
        receipt = _json({"schema": SCHEMA, "accessor_intake_sha256": self.sha256, "pins": [p.record() for p in pins]})
        receipt_path = destination / "receipt.json"
        receipt_path.write_bytes(receipt)
        self.verify()
        return {
            "rows": rows,
            "accessor_intake_sha256": self.sha256,
            "receipt_path": str(receipt_path),
            "receipt_sha256": hashlib.sha256(receipt).hexdigest(),
        }


def issue_independent_accessor_intake(
    *,
    public_checkout: str | Path,
    commit: str,
    include_root: str | Path,
    native_compiler: str | Path,
    reviewed_spec: str | Path,
    forbidden_roots: tuple[str | Path, ...],
    output_root: str | Path,
) -> IndependentAccessorIntake:
    """Compile a fixed observer and replay actual clean tracked public headers."""
    forbidden = tuple(Path(p).absolute() for p in forbidden_roots)
    sources = [public_checkout, include_root, native_compiler, reviewed_spec]
    for path in sources:
        _outside(Path(path).absolute(), forbidden)
    checkout, includes = _plain(public_checkout, directory=True), _plain(include_root, directory=True)
    compiler, spec_path = _plain(native_compiler), _plain(reviewed_spec)
    spec = _spec(spec_path)
    selected_header = _plain(includes / spec["header"])
    public_header = _plain(checkout / spec["header"])
    _outside(selected_header, forbidden)
    _outside(public_header, forbidden)
    _tracked_source(checkout, public_header, commit)
    if selected_header.read_bytes() != public_header.read_bytes():
        raise RtlIntakeRefusal("selected native header differs from the exact public source before compilation")
    destination = Path(output_root).absolute()
    _outside(destination, forbidden)
    if destination.is_relative_to(checkout) or destination.is_relative_to(includes):
        raise RtlIntakeRefusal("native observation output must be separate from selected public inputs")
    destination.mkdir(parents=True, exist_ok=False)
    program, binary, dependencies = (destination / name for name in ("observe.cc", "observe", "observe.d"))
    program.write_text(_program(spec))
    command = [
        str(compiler),
        "-std=c++20",
        "-O2",
        "-I" + str(includes),
        "-MMD",
        "-MF",
        str(dependencies),
        str(program),
        "-o",
        str(binary),
    ]
    completed = invocation_record.run(
        command,
        directory=destination,
        stage="native_accessor_build",
        inputs=(program, spec_path, selected_header),
        outputs=(binary, dependencies),
        capture_output=True,
        timeout=120,
    )
    (destination / "compile.stdout").write_bytes(completed.stdout)
    (destination / "compile.stderr").write_bytes(completed.stderr)
    if completed.returncode or not binary.is_file() or not dependencies.is_file():
        raise RtlIntakeRefusal("actual native public accessor compilation failed")
    dependency_paths = shlex.split(dependencies.read_text().split(":", 1)[1].replace("\\\n", " "))
    pins = [_pin("reviewed-accessor-spec", spec_path, forbidden), _pin("native-compiler", compiler, forbidden)]
    provenance = []
    for raw in sorted(set(dependency_paths)):
        unnormalized = Path(raw).absolute()
        if any(path.is_symlink() for path in (unnormalized, *unnormalized.parents)):
            raise RtlIntakeRefusal("native accessor dependency cannot traverse a symlink")
        selected = _plain(os.path.abspath(unnormalized))
        _outside(selected, forbidden)
        if selected == program:
            continue
        if not selected.is_relative_to(includes):
            raise RtlIntakeRefusal("native accessor non-system dependency escapes selected public headers")
        public = _plain(checkout / selected.relative_to(includes))
        _outside(public, forbidden)
        tracked = _tracked_source(checkout, public, commit)
        if selected.read_bytes() != public.read_bytes():
            raise RtlIntakeRefusal("native accessor installed header differs from selected tracked public source")
        provenance.append(tracked)
        pins.extend(
            (_pin("compiled-public-header", selected, forbidden), _pin("tracked-public-header", public, forbidden))
        )
    if not provenance:
        raise RtlIntakeRefusal("native accessor observer has no tracked public header dependency")
    words = [0, *(1 << i for i in range(64)), (1 << 64) - 1]
    mixed = 1
    for _ in range(64):
        mixed ^= (mixed << 13) & ((1 << 64) - 1)
        mixed ^= mixed >> 7
        mixed ^= (mixed << 17) & ((1 << 64) - 1)
        words.append(mixed)
    meta, rows = _invoke(binary, tuple(words), spec["fields"], destination / "basis")
    if set(meta["constants"]) != set(spec["constants"]):
        raise RtlIntakeRefusal("native observer did not return the exact selected source constants")
    fields = _observed_fields(rows, spec["fields"], meta["word_bits"])
    facts = {
        "schema": SCHEMA,
        "scope": "actual native public accessor ABI; physical effects unqualified",
        "native_observation_word_bits": meta["word_bits"],
        "fields": fields,
        "source_constants": meta["constants"],
        "observed_words": len(rows),
        "length_scope": "actual public source accessor calls, no transcribed length table",
        "unknowns": list(_UNKNOWN),
    }
    facts_json = _json(facts)
    facts_path = destination / "facts.json"
    facts_path.write_bytes(facts_json)
    for path in [
        program,
        binary,
        dependencies,
        facts_path,
        *(path for path in (destination / "basis").rglob("*") if path.is_file()),
        *(path for path in (destination / "invocations").rglob("*") if path.is_file()),
        destination / "compile.stdout",
        destination / "compile.stderr",
    ]:
        pins.append(_pin("native-observation-artifact", path, forbidden))
    pins.append(_pin("issuer-source", module_source_path(__name__), forbidden))
    pins.append(_pin("invocation-recorder-source", module_source_path(invocation_record.__name__), forbidden))
    receipt_json = _json(
        {
            "schema": SCHEMA,
            "sources": [p.record() for p in pins],
            "tracked_headers": provenance,
            "production": {"command": command, "returncode": completed.returncode},
            "facts_sha256": hashlib.sha256(facts_json).hexdigest(),
            "unknowns": list(_UNKNOWN),
        }
    )
    receipt_path = destination / "intake.json"
    receipt_path.write_bytes(receipt_json)
    authority = IndependentAccessorIntake(
        tuple([*pins, _pin("intake-receipt", receipt_path, forbidden)]),
        checkout,
        commit,
        binary,
        forbidden,
        _json(spec),
        facts_json,
        receipt_json,
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
