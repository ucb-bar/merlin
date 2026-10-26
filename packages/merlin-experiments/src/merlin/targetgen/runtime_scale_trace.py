"""Retired load-to-CONFIG_ST provenance for runtime f32 readout scales.

Replay token equality proves that the emitted plan names the source-bound
scale input. This module proves the executed program did it: in each kernel
call, a retired 32-bit load from that input's bound pointer (offset 0) supplied
bits equal to the frozen input, and only bit-preserving register steps carried
them into the scale slot of a retired accelerator command's rs2.

Register state comes from Spike's committed writes. Instruction forms come from
the retired words, which must equal the linked ELF's bytes at the same PC; the
disassembly text is never read. A register written by any step this module
does not model loses its provenance, so copies through memory, unknown
arithmetic or a value from an earlier call fail closed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "merlin.runtime_scale_loads.v1"
#: The host integer register width this module models (RV64 x registers), which
#: is also the width of a RoCC rs1/rs2 operand. The slot position comes from facts.
_XLEN = 64
_MASK = (1 << _XLEN) - 1
#: A scale is one binary32 value: a 32-bit load supplies it, whatever its slot offset.
_SCALE_BITS = 32


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sext(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _unknown(reason):
    return ValueError(f"UNKNOWN: runtime scale provenance: {reason}")


# ---------------------------------------------------------------------------
# Instruction forms (RV64GC subset) decoded from the retired word.
#
# Encodings follow the RISC-V Unprivileged ISA (document 20191213): the base
# opcode map of its "RV32/64G Instruction Set Listings" chapter, the RVC opcode
# map of its "C" chapter, and the ratified Zba extension (v1.0.0) for add.uw.
# Any other word is "other" and ends the provenance of the register it writes.
#   opcode  funct3/funct7         form                         defined in
#   0x03    f3 0-6                LB LH LW LD LBU LHU LWU      RV32I, RV64I
#   0x07    f3 2 / 3              FLW / FLD                    F, D
#   0x13    f3 0 1 5 6 7          ADDI SLLI SRLI ORI ANDI      RV32I, RV64I (6-bit shamt)
#   0x33    f7 0, f3 0 / 6        ADD / OR                     RV32I
#   0x3B    f7 0x04, f3 0, rs2 0  ADD.UW rd, rs, x0 (ZEXT.W)   Zba
#   0x53    f7 0x70, f3 0, rs2 0  FMV.X.W                      F
#   RVC Q0  f3 1 2 3              C.FLD C.LW C.LD              C (RV64)
#   RVC Q1  f3 0 / 4              C.ADDI / C.SRLI C.ANDI C.OR  C
#   RVC Q2  f3 0 / 1-3 / 4        C.SLLI / C.FLDSP C.LWSP C.LDSP / C.MV C.ADD  C (RV64)
# The accelerator opcode is not listed: it is the selected facts' custom_opcode.


def _op(word):
    """Return the modeled form of one retired instruction word."""
    if word & 3 != 3:
        return _compressed(word & 0xFFFF)
    opcode, rd, f3 = word & 0x7F, word >> 7 & 31, word >> 12 & 7
    rs1, rs2, f7 = word >> 15 & 31, word >> 20 & 31, word >> 25
    imm = _sext(word >> 20, 12)
    if opcode == 0x03:
        width, signed = {0: (1, 1), 1: (2, 1), 2: (4, 1), 3: (8, 1), 4: (1, 0), 5: (2, 0), 6: (4, 0)}.get(f3, (0, 0))
        return dict(op="load", rd=rd, width=width, signed=bool(signed), fp=False)
    if opcode == 0x07:
        return dict(op="load", rd=rd, width={2: 4, 3: 8}.get(f3, 0), signed=False, fp=True)
    if opcode == 0x13:
        if f3 == 0:
            return dict(op="addi", rd=rd, rs=rs1, imm=imm)
        if f3 == 1 and word >> 26 == 0:
            return dict(op="slli", rd=rd, rs=rs1, shamt=word >> 20 & 63)
        if f3 == 5 and word >> 26 == 0:
            return dict(op="srli", rd=rd, rs=rs1, shamt=word >> 20 & 63)
        if f3 == 6:
            return dict(op="ori", rd=rd, rs=rs1, imm=imm)
        if f3 == 7:
            return dict(op="andi", rd=rd, rs=rs1, imm=imm)
    if opcode == 0x33 and f7 == 0 and f3 in (0, 6):
        return dict(op="add" if f3 == 0 else "or", rd=rd, rs1=rs1, rs2=rs2)
    if opcode == 0x3B and f7 == 0x04 and f3 == 0 and rs2 == 0:
        return dict(op="zext_w", rd=rd, rs=rs1)  # add.uw rd, rs, zero
    if opcode == 0x53 and f7 == 0x70 and f3 == 0 and rs2 == 0:
        return dict(op="fmv_x_w", rd=rd, fs=rs1)
    return dict(op="other")


def _compressed(word):
    quadrant, f3 = word & 3, word >> 13 & 7
    big, small = word >> 7 & 31, 8 + (word >> 7 & 7)
    six = (word >> 12 & 1) << 5 | word >> 2 & 31
    if quadrant == 0 and f3 in (1, 2, 3):
        rd = 8 + (word >> 2 & 7)
        return dict(op="load", rd=rd, width=4 if f3 == 2 else 8, signed=f3 == 2, fp=f3 == 1)
    if quadrant == 2 and f3 in (1, 2, 3):
        return dict(op="load", rd=big, width=4 if f3 == 2 else 8, signed=f3 == 2, fp=f3 == 1)
    if quadrant == 1 and f3 == 0:
        return dict(op="addi", rd=big, rs=big, imm=_sext(six, 6))
    if quadrant == 1 and f3 == 4:
        kind = word >> 10 & 3
        if kind == 0:
            return dict(op="srli", rd=small, rs=small, shamt=six)
        if kind == 2:
            return dict(op="andi", rd=small, rs=small, imm=_sext(six, 6))
        if kind == 3 and not word >> 12 & 1 and word >> 5 & 3 == 2:
            return dict(op="or", rd=small, rs1=small, rs2=8 + (word >> 2 & 7))
    if quadrant == 2 and f3 == 0:
        return dict(op="slli", rd=big, rs=big, shamt=six)
    if quadrant == 2 and f3 == 4:
        rs2 = word >> 2 & 31
        if rs2 and not word >> 12 & 1:
            return dict(op="addi", rd=big, rs=rs2, imm=0)  # c.mv
        if rs2 and big:
            return dict(op="add", rd=big, rs1=big, rs2=rs2)
    return dict(op="other")


# ---------------------------------------------------------------------------
# Provenance state: value == ((ext(B) << shift) | low) mod 2**64.


def _normalize(state):
    if state is not None and state["shift"] + _SCALE_BITS >= _XLEN:
        state = dict(state, mode="zero")
    return state


def _value(state, bits):
    base = bits | (0xFFFFFFFF00000000 if state["mode"] == "sign" and bits >> 31 else 0)
    return (base << state["shift"] | state["low"]) & _MASK


def _with_constant(state, constant, combine):
    """Combine static bits on either side of the scale field, never into it."""
    field = (1 << _SCALE_BITS) - 1 << state["shift"]
    constant &= _MASK
    if state["mode"] != "zero" or constant & field:
        return None  # The constant would overlap the scale bits or sign copies.
    low = state["low"] + constant if combine == "add" else state["low"] | constant
    if low >> _XLEN or low & field:
        return None  # A carry reached the scale field or left the register.
    return dict(state, low=low)


def _transform(form, xstate, fstate, regs):
    op = form["op"]
    if op in ("addi", "ori", "andi", "slli", "srli", "zext_w"):
        source = xstate.get(form["rs"])
        if source is None:
            return None
        if op == "addi" and form["imm"] == 0:
            return source
        if op in ("addi", "ori"):
            return _with_constant(source, form["imm"], "add" if op == "addi" else "or")
        if op == "andi":
            mask, shift = form["imm"] & _MASK, source["shift"]
            field = (1 << _SCALE_BITS) - 1 << shift
            if mask & field != field:
                return None
            upper = _MASK & ~((1 << shift + _SCALE_BITS) - 1)
            mode = source["mode"]
            if mode == "sign" and mask & upper == 0:
                mode = "zero"
            elif mode == "sign" and mask & upper != upper:
                return None
            return dict(source, mode=mode, low=source["low"] & mask)
        if op == "slli":
            shift = source["shift"] + form["shamt"]
            if shift + _SCALE_BITS > _XLEN:
                return None
            return _normalize(dict(source, shift=shift, low=source["low"] << form["shamt"] & _MASK))
        if op == "srli":
            if source["mode"] != "zero" or form["shamt"] > source["shift"]:
                return None
            return dict(source, shift=source["shift"] - form["shamt"], low=source["low"] >> form["shamt"])
        if source["shift"] != 0:  # zext.w
            return None
        return dict(source, mode="zero")
    if op in ("add", "or"):
        left, right = xstate.get(form["rs1"]), xstate.get(form["rs2"])
        if (left is None) == (right is None):
            return None  # Untracked, or two scale values combined.
        tracked, other = (left, form["rs2"]) if left is not None else (right, form["rs1"])
        constant = regs[other]
        if constant is None:
            return None
        return _with_constant(tracked, constant, "add" if op == "add" else "or")
    if op == "fmv_x_w":
        source = fstate.get(form["fs"])
        return None if source is None else dict(source, mode="sign", shift=0, low=0)
    return None


# ---------------------------------------------------------------------------
# Trace and ELF readers.


def _record(line):
    from .native_instruction_audit import _trace_word

    record = _trace_word(line)
    if record is None:
        return None
    parts = line.split()
    return int(parts[1][:-1]), record[0], record[1], parts[5:]


def _effects(tokens, *, strict):
    xwrites, fwrites, memory = {}, {}, None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        try:
            if token == b"mem":
                address = int(tokens[i + 1], 16)
                i += 2
                if i < len(tokens) and tokens[i].startswith(b"0x"):
                    memory, i = ("store", address), i + 1
                else:
                    memory = ("load", address)
            elif token == b"x":
                xwrites[int(tokens[i + 1])], i = int(tokens[i + 2], 16), i + 3
            elif token[:1] == b"x" and token[1:].isdigit():
                xwrites[int(token[1:])], i = int(tokens[i + 1], 16), i + 2
            elif token[:1] == b"f" and token[1:].isdigit():
                fwrites[int(token[1:])], i = int(tokens[i + 1], 16), i + 2
            elif token[:1] == b"c" and b"_" in token:
                i += 2  # CSR write
            elif strict:
                raise _unknown(f"unmodeled committed effect {token!r} inside a scale kernel")
            else:
                i += 1
        except (IndexError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith("UNKNOWN"):
                raise
            raise _unknown("malformed committed effect record") from error
    return xwrites, fwrites, memory


class _ElfWords:
    def __init__(self, elf):
        from .elf_lanes import _endian, executable_sections

        self.blob = Path(elf).read_bytes()
        self.end = _endian(self.blob)
        self.sections = executable_sections(self.blob)

    def check(self, pc, word):
        size = 2 if word & 3 != 3 else 4
        for _name, offset, length, address in self.sections:
            if address <= pc and pc + size <= address + length:
                linked = int.from_bytes(
                    self.blob[offset + pc - address : offset + pc - address + size],
                    "little" if self.end == "<" else "big",
                )
                if linked != word:
                    raise _unknown(f"retired word at {pc:#x} differs from the linked ELF")
                return
        raise _unknown(f"retired PC {pc:#x} is outside the linked ELF text")


# ---------------------------------------------------------------------------


def _scale_bindings(runtime_abi, scale_inputs, input_bits):
    wanted = {}
    for row in scale_inputs:
        kernel, index = row.get("kernel"), row.get("arg_index")
        if (
            not isinstance(kernel, str)
            or type(index) is not int
            or row.get("offset") != 0
            or row.get("dtype") != "f32"
            or list(row.get("shape", ())) != [1]
        ):
            raise _unknown("scale input must be an f32[1] kernel argument at offset 0")
        arguments = [a for a in runtime_abi.get(kernel, ()) if a.get("arg_index") == index]
        if len(arguments) != 1 or arguments[0].get("byte_extent") != 4:
            raise _unknown("scale input has no unique four-byte pointer argument in the kernel ABI")
        bits = input_bits.get(index, input_bits.get(str(index)))
        if type(bits) is not int or not 0 <= bits < 1 << 32 or bits >> 23 & 0xFF == 0xFF:
            raise _unknown("scale input has no frozen finite f32 bit pattern")
        wanted.setdefault(kernel, {})[index] = dict(register=arguments[0]["register"], bits=bits)
    if not wanted:
        raise _unknown("no runtime scale inputs to audit")
    return wanted


def audit_runtime_scale_loads(
    *,
    elf: Path,
    trace: Path,
    runtime_abi: Mapping,
    scale_inputs: list[dict],
    input_bits: Mapping,
    slot: Mapping,
    custom_opcode: int,
) -> dict:
    """Return retired load/config provenance rows or raise ``ValueError``.

    ``runtime_abi`` is ``{symbol: [{arg_index, register, byte_extent}]}`` from
    the compiled pointer ABI; ``slot`` is the derived CONFIG_ST scale field
    (``offset``/``width``); ``custom_opcode`` the selected RoCC opcode. Command
    ordinals count every retired custom-opcode word, as the retirement decoder
    does, so composition audits can join a decoded CONFIG_ST row to its chain.
    """
    from merlin.kernels.decode.rocc import fields_of

    from .native_instruction_audit import _function_ranges

    offset, width = slot.get("offset"), slot.get("width")
    if type(width) is not int or width != _SCALE_BITS or type(offset) is not int or not 0 <= offset <= _XLEN - width:
        raise _unknown("selected CONFIG_ST scale slot is not one binary32 field of rs2")
    if type(custom_opcode) is not int:
        raise _unknown("selected accelerator opcode is unresolved")
    wanted = _scale_bindings(runtime_abi, scale_inputs, input_bits)
    ranges = _function_ranges(elf, set(wanted))
    if {row[2] for row in ranges} != set(wanted):
        raise _unknown("scale kernel has no linked function range")
    starts = [row[0] for row in ranges]
    words = _ElfWords(elf)
    digest = hashlib.sha256()
    registers, loads, uses = {}, [], []
    xstate, fstate, calls = {}, {}, {}
    active = None  # (kernel, call, {arg: (base, bits)})
    ordinal, only_hart = -1, None
    with Path(trace).open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = _record(line)
            if record is None:
                continue
            hart, pc, word, tokens = record
            if only_hart is None:
                only_hart = hart
            elif hart != only_hart:
                # Provenance and command ordinals are per hart; an interleaved trace is not modeled.
                raise _unknown(f"trace retires on hart {hart} and hart {only_hart}; one hart is modeled")
            regs = registers.setdefault(hart, [0, *([None] * 31)])
            index = bisect.bisect_right(starts, pc) - 1
            owner = ranges[index][2] if index >= 0 and pc < ranges[index][1] else None
            if owner is not None and pc == ranges[index][0]:
                calls[owner] = calls.get(owner, 0) + 1
                bases = {}
                for arg, row in wanted[owner].items():
                    base = regs[row["register"]]
                    if type(base) is not int or base <= 0:
                        raise _unknown("scale pointer register has no committed value at kernel entry")
                    bases[arg] = (base, row["bits"])
                if active is not None:
                    _close(active, loads, uses)
                active = (owner, calls[owner], bases)
                xstate, fstate = {}, {}
            if active is not None and owner != active[0]:
                # Leaving the kernel (a call or return) ends every register's
                # provenance; only a fresh load in this call can restore it.
                xstate, fstate = {}, {}
            fields = fields_of(word)
            command = fields["opcode"] == custom_opcode
            if command:
                ordinal += 1
            xwrites, fwrites, memory = _effects(tokens, strict=owner is not None)
            if active is not None and owner == active[0]:
                form = _op(word)
                if command:
                    _command(form, fields, word, pc, ordinal, regs, xstate, slot, active, uses, words)
                new_x, new_f = {}, {}
                if form["op"] == "load":
                    _load(form, pc, word, memory, xwrites, fwrites, active, loads, new_x, new_f, words)
                elif form.get("rd"):
                    state = _transform(form, xstate, fstate, regs)
                    if state is not None:
                        step = dict(pc=pc, word=word, op=form["op"])
                        new_x[form["rd"]] = dict(state, chain=(*state["chain"], step))
                for register, value in xwrites.items():
                    state = new_x.get(register)
                    if state is not None and _value(state, state["bits"]) != value:
                        raise _unknown(f"committed x{register} at {pc:#x} contradicts its scale provenance")
                    if state is not None:
                        words.check(pc, word)
                    xstate.pop(register, None)
                    if state is not None:
                        xstate[register] = state
                for register in fwrites:
                    fstate.pop(register, None)
                    if register in new_f:
                        fstate[register] = new_f[register]
            for register, value in xwrites.items():
                if register:
                    regs[register] = value
                elif value:
                    raise _unknown("committed zero register is nonzero")
    if active is not None:
        _close(active, loads, uses)
    for kernel, args in wanted.items():
        if not calls.get(kernel):
            raise _unknown("scale kernel was never entered")
    report = dict(
        schema=SCHEMA,
        status="pass",
        elf_sha256=hashlib.sha256(words.blob).hexdigest(),
        retirement_trace_sha256=digest.hexdigest(),
        slot=dict(offset=offset, width=width),
        loads=loads,
        config_uses=uses,
        evidence={},
    )
    for kernel, args in sorted(wanted.items()):
        for arg in sorted(args):
            report["evidence"][f"{kernel}:{arg}"] = scale_load_evidence_id(report, kernel, arg)
    report["sha256"] = _sha({k: v for k, v in report.items() if k != "sha256"})
    return report


def _load(form, pc, word, memory, xwrites, fwrites, active, loads, new_x, new_f, words):
    kernel, call, bases = active
    if memory is None or memory[0] != "load":
        raise _unknown(f"retired load at {pc:#x} has no observed address")
    address = memory[1]
    width = max(form["width"], 1)
    hits = [arg for arg, (base, _) in bases.items() if address < base + 4 and base < address + width]
    if not hits:
        return
    if len(hits) != 1:
        raise _unknown("one load reads several scale inputs")
    arg = hits[0]
    base, bits = bases[arg]
    if form["width"] != 4 or address != base:
        raise _unknown(f"scale input {arg} is read by a {form['width']}-byte load at offset {address - base}")
    written = fwrites.get(form["rd"]) if form["fp"] else xwrites.get(form["rd"])
    if written is None:
        raise _unknown(f"scale load at {pc:#x} has no committed destination value")
    loaded = written & 0xFFFFFFFF
    if form["fp"]:
        if written >> 32 not in (0, 0xFFFFFFFF):
            raise _unknown("f32 scale load is not NaN-boxed binary32")
    elif written != (loaded | (0xFFFFFFFF00000000 if form["signed"] and loaded >> 31 else 0)):
        raise _unknown("scale load destination differs from its 32-bit extension")
    if loaded != bits:
        raise ValueError(f"retired scale load of input {arg} read {loaded:#010x}, frozen input bits are {bits:#010x}")
    words.check(pc, word)
    row = dict(kernel=kernel, call=call, pc=pc, word=word, arg_index=arg, address=address, offset=0, loaded_bits=loaded)
    row["register"] = ("f" if form["fp"] else "x") + str(form["rd"])
    loads.append(row)
    state = dict(arg=arg, bits=bits, call=call, load_pc=pc, shift=0, low=0, chain=(dict(pc=pc, word=word, op="load"),))
    if form["fp"]:
        new_f[form["rd"]] = dict(state, mode="fp")
    elif form["rd"]:
        new_x[form["rd"]] = dict(state, mode="sign" if form["signed"] else "zero")


def _command(form, fields, word, pc, ordinal, regs, xstate, slot, active, uses, words):
    if word & 3 != 3:
        return
    if xstate.get(fields["rs1"]) is not None:
        raise _unknown(f"scale bits reach rs1 of the accelerator command at {pc:#x}")
    state = xstate.get(fields["rs2"])
    if state is None:
        return
    rs2 = regs[fields["rs2"]]
    if state["shift"] != slot["offset"] or state["mode"] != "zero" or rs2 != _value(state, state["bits"]):
        raise ValueError(f"scale bits reach rs2 of {pc:#x} outside the derived acc_scale slot")
    words.check(pc, word)
    kernel, call, _ = active
    if state["call"] != call:
        raise _unknown("scale provenance crosses kernel calls")
    chain = [dict(step) for step in state["chain"]]
    use = dict(
        kernel=kernel,
        call=call,
        command_ordinal=ordinal,
        pc=pc,
        word=word,
        rs2_register=f"x{fields['rs2']}",
        rs2=rs2,
        arg_index=state["arg"],
        scale_bits=state["bits"],
        static_bits=state["low"],
        load_pc=state["load_pc"],
        chain=chain,
    )
    use["chain_sha256"] = _sha(dict(kernel=kernel, call=call, arg_index=state["arg"], chain=chain, rs2=rs2))
    uses.append(use)


def _close(active, loads, uses):
    kernel, call, bases = active
    for arg in bases:
        if not any(r["kernel"] == kernel and r["call"] == call and r["arg_index"] == arg for r in loads):
            raise _unknown(f"call {call} of {kernel} has no retired load of scale input {arg}")
        if not any(r["kernel"] == kernel and r["call"] == call and r["arg_index"] == arg for r in uses):
            raise _unknown(f"call {call} of {kernel} never configures a store from scale input {arg}")


def scale_load_evidence_id(report, kernel, arg_index):
    """Canonical digest of one scale input's observed load-to-config chains."""
    loads = [r for r in report["loads"] if r["kernel"] == kernel and r["arg_index"] == arg_index]
    uses = [r for r in report["config_uses"] if r["kernel"] == kernel and r["arg_index"] == arg_index]
    if not loads or not uses:
        raise _unknown(f"scale input {arg_index} of {kernel} has no observed load-to-config chain")
    return _sha(
        dict(
            schema=SCHEMA,
            kernel=kernel,
            arg_index=arg_index,
            elf_sha256=report["elf_sha256"],
            retirement_trace_sha256=report["retirement_trace_sha256"],
            loads=loads,
            config_uses=uses,
        )
    )


def authenticate_report(report):
    """Reject a report whose rows, digests or status were altered after the audit."""
    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA or report.get("status") != "pass":
        raise _unknown("runtime scale load report is absent or did not pass")
    if report.get("sha256") != _sha({k: v for k, v in report.items() if k != "sha256"}):
        raise _unknown("runtime scale load report digest differs from its rows")
    for key, value in report["evidence"].items():
        kernel, _, arg = key.rpartition(":")
        if value != scale_load_evidence_id(report, kernel, int(arg)):
            raise _unknown("runtime scale evidence id differs from its observed chain")
    return report
