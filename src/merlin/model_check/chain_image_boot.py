"""Reopen actual ELF entry and selected host mailboxes after the chain link."""

from __future__ import annotations

from pathlib import Path

from merlin.common import invocation_record
from merlin.runtime.elf_audit import read_elf

from .chain_image_objects import pin


def verify(elf, selection, *, nm, directory, timeout):
    previous = set(directory.glob("invocations/*/invocation.json"))
    result = invocation_record.run(
        [str(nm), "--print-size", "--format=posix", str(elf)],
        directory=directory,
        stage="chain_image_boot_symbols",
        inputs=(elf,),
        dependencies=(Path(__file__),),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    symbols = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) in (3, 4) and parts[1] not in {"U", "u"}:
            symbols[parts[0]] = {"address": int(parts[2], 16), "bytes": int(parts[3], 16) if len(parts) == 4 else 0}
    entry, segments, _ = read_elf(elf, strict=True)
    if symbols[selection["entry_symbol"]]["address"] != entry:
        raise ValueError("chain ELF entry differs from the selected original boot symbol")
    mailboxes = []
    for name in selection["mailbox_symbols"]:
        symbol = symbols[name]
        if symbol["bytes"] <= 0 or not any(
            s.vaddr <= symbol["address"] and symbol["address"] + symbol["bytes"] <= s.end and "W" in s.flags
            for s in segments
        ):
            raise ValueError("selected host mailbox is absent from writable loaded memory")
        mailboxes.append({"symbol": name, **symbol})
    if not mailboxes or len({row["address"] for row in mailboxes}) != len(mailboxes):
        raise ValueError("host mailbox selection is empty or aliases storage")
    records = set(directory.glob("invocations/*/invocation.json")) - previous
    if len(records) != 1:
        raise ValueError("boot symbol query lost its original invocation")
    return {
        "entry": entry,
        "entry_symbol": selection["entry_symbol"],
        "mailboxes": mailboxes,
        "invocation": pin(records.pop()),
        "scope": "Actual selected ELF symbols and LOAD intervals; remote boot/AGFI remains separate.",
    }
