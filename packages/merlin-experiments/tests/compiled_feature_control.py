"""Owned diagnostic package commands; never selected as an author seed."""

import sys
from pathlib import Path

command, source, *rest = sys.argv[1:]
text = Path(source).read_text()
if command == "lower_interface_to_target":
    sys.stdout.write(text)
elif command == "emit_command_buffer":
    Path(rest[0]).write_bytes(Path("buffer.json").read_bytes())
elif command == "lower_target_to_llvm":
    sys.stdout.write(Path("emitted.mlir").read_text())
elif command != "parse":
    raise ValueError(command)
