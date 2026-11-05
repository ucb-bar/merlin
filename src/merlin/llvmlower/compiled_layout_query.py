"""Bounded layout queries for the ordinary LLVM object producer.

The selected LLVM driver emits its own pre-instruction-selection module.
That observation supplies no descriptor roles, storage or semantic theorem.
"""

from pathlib import Path

from merlin.common import invocation_record
from merlin.common.digest import sha256_file


def select_object(record, *, environment):
    """Reopen the actual object command; never discover a replacement tool."""
    document = invocation_record.verify(record)
    argv = document.get("argv", ())
    if document.get("kind") != "subprocess" or document.get("stage") != "object" or len(argv) < 5 or argv[-2] != "-o":
        raise ValueError("layout query requires the actual ordinary LLVM object command")
    original, obj = (Path(path).resolve(strict=True) for path in (argv[-3], argv[-1]))
    if original.suffix != ".ll" or document["inputs"] != [{"path": str(original), "sha256": sha256_file(original)}]:
        raise ValueError("layout query has no exact original LLVM input")
    if {"path": str(obj), "sha256": sha256_file(obj)} not in document["outputs"]:
        raise ValueError("layout query has no exact actual object")
    if environment is not None:
        invocation_record.require_environment(record, environment=environment)
    if len(argv) >= 5 and argv[-4] == "-c" and "-c" not in argv[1:-4] and "-o" not in argv[1:-4]:
        return document, original, obj, "clang", argv[1:-4]
    # This is the existing ordinary host LLC command. Other flags require a
    # separately supported query, rather than silently stripping selections.
    if argv[1:-3] != ["-O2", "-filetype=obj", "-relocation-model=pic"]:
        raise ValueError("layout query has unsupported ordinary object producer options")
    if environment is None:
        raise ValueError("LLVM object layout requires its exact selected process environment")
    return document, original, obj, "llvm", argv[1:-3]


def bounded_mir_module(path, *, max_bytes):
    """Read only the first complete embedded LLVM document, without YAML."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("LLVM object layout requires a positive explicit observation byte budget")
    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("LLVM object layout observation exceeds its byte budget")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("LLVM object layout has no UTF-8 embedded module") from error
    if not lines or lines[0] != "--- |":
        raise ValueError("LLVM object layout has no initial embedded module")
    module, closed = [], False
    for line in lines[1:]:
        if line == "...":
            closed = True
            break
        if not line.startswith("  "):
            raise ValueError("LLVM object layout has malformed embedded-module indentation")
        module.append(line[2:])
    if not closed:
        raise ValueError("LLVM object layout embedded module is incomplete")
    for prefix in ('target datalayout = "', 'target triple = "'):
        selected = [line.strip() for line in module if line.strip().startswith(prefix)]
        if len(selected) != 1 or not selected[0].endswith('"') or not selected[0][len(prefix) : -1]:
            raise ValueError("LLVM object layout has no unique complete layout and triple")
    return "\n".join(module) + "\n"


def query_object_layout(
    *,
    document,
    original,
    obj,
    record,
    driver,
    options,
    root,
    run,
    deadline,
    environment,
    max_observation_bytes,
):
    """Retain the actual driver output separately from the original artifact."""
    compiler = Path(document["executable"]["path"])
    dependencies = tuple(Path(row["path"]) for row in document["dependencies"])
    output = root / "selected.ll"
    common = dict(
        root=root,
        stage="layout_selected_object_compiler",
        deadline=deadline,
        inputs=(original,),
        dependencies=(record, obj, Path(__file__), *dependencies),
        env=environment,
    )
    if driver == "clang":
        run(
            [str(compiler), *options, "-S", "-emit-llvm", str(original), "-o", str(output)],
            outputs=(output,),
            **common,
        )
    else:
        if type(max_observation_bytes) is not int or max_observation_bytes <= 0:
            raise ValueError("LLVM object layout requires a positive explicit observation byte budget")
        mir = root / "selected.mir"
        common["cwd"] = document["cwd"]
        run(
            [
                str(compiler),
                *options,
                str(original),
                "-o",
                str(mir),
                "-stop-before=finalize-isel",
            ],
            outputs=(mir,),
            **common,
        )
        output.write_text(bounded_mir_module(mir, max_bytes=max_observation_bytes))
    invocation_record.verify(record)
    return output, compiler, dependencies
