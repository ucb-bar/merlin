"""Reproducible bounded native issue capture of selected generated C functions.

Source annotations supply tensor declarations, never mathematics. A selected
target observer must establish computation, layouts, and numerical boundaries
from the captured commands. Native issue capture supplies no target timing.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import math
import shutil
from pathlib import Path

from merlin.common import invocation_record
from merlin.kernels.reference_inventory import digest, inspect_source, read_pinned


def source_function(raw, symbol):
    """Read one actual parsed C definition and its adjacent Exo declarations."""
    inventory = inspect_source(raw, "c_source")
    matches = [row for row in inventory["functions"] if row["symbol"] == symbol]
    if len(matches) != 1 or not matches[0].get("syntax_complete"):
        raise ValueError("selected function requires one complete structural C definition")
    row = matches[0]
    lines = raw.decode().splitlines(keepends=True)
    annotations = []
    cursor = row["line"] - 2
    while cursor >= 0 and lines[cursor].lstrip().startswith("//"):
        annotations.append(lines[cursor].lstrip().removeprefix("//").strip())
        cursor -= 1
    declarations = {}
    for line in reversed(annotations):
        name, colon, value = line.partition(":")
        if not colon:
            continue
        name = name.strip()
        value = value.partition("@")[0].strip()
        dtype, opening, dimensions = value.partition("[")
        if not name.isidentifier() or dtype.strip() not in ("i8", "i16", "i32", "f32", "bool"):
            raise ValueError("unsupported generated source declaration")
        shape = ast.literal_eval("[" + dimensions) if opening else []
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 1 for dim in shape):
            raise ValueError("native capture needs complete static source tensor dimensions")
        declarations[name] = dict(dtype=dtype.strip(), shape=shape)
    if not declarations:
        raise ValueError("generated source tensor declarations are unavailable")
    return dict(row, source="".join(lines[row["line"] - 1:row["end_line"]]), declarations=declarations)


def _pin(path, kind="file"):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(kind=kind, uri=str(path), sha256=digest(data), size_bytes=len(data))


def _entrypoint(value):
    module, separator, name = value.partition(":")
    if not separator or not name.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("reference observer needs an explicit module:factory")
    return getattr(importlib.import_module(module), name)


def capture_native_function(raw, symbol, *, settings, workspace):
    """Execute unchanged scalar control with only the selected issue sinks replaced."""
    root = Path(settings["header_root"]).resolve()
    header = (root / settings["header"]).resolve()
    if not header.is_relative_to(root):
        raise ValueError("capture header escapes its selected root")
    sink, fence = settings["sink_macro"], settings["fence_macro"]
    if not sink.isidentifier() or not fence.isidentifier():
        raise ValueError("capture sink and fence need source macro identifiers")
    maximum = settings.get("max_commands", 100000)
    if type(maximum) is not int or not 1 <= maximum <= 1000000:
        raise ValueError("native issue capture needs a finite command bound")
    function = source_function(raw, symbol)
    declarations = function["declarations"]
    c_types = dict(i8="int8_t", i16="int16_t", i32="int32_t", f32="float", bool="bool")
    widths = dict(i8=1, i16=2, i32=4, f32=4, bool=1)
    allocations, arguments = [], []
    for name, expression in settings.get("probe_expressions", {}).items():
        if not name.isidentifier() or type(expression) is not str or not expression:
            raise ValueError("capture probes require named selected C expressions")
        allocations.append(f'printf("{{\\\"kind\\\":\\\"probe\\\",\\\"name\\\":\\\"{name}'
                           f'\\\",\\\"value\\\":%llu}}\\n",(unsigned long long)({expression}));')
    for parameter in function["parameters"]:
        name = parameter.rsplit(None, 1)[-1].lstrip("*")
        if name not in declarations:
            if len(arguments) == 0 and name == "ctxt":
                arguments.append("0")
                continue
            raise ValueError("function argument lacks a complete generated declaration")
        declaration = declarations[name]
        dtype, shape = declaration["dtype"], declaration["shape"]
        ctype = c_types[dtype]
        if shape:
            size = math.prod(shape) * widths[dtype]
            if size > 128 << 20:
                raise ValueError("source tensor exceeds native capture allocation bound")
            allocations.extend([f"{ctype} *{name}=calloc({size},1); if(!{name}) return 3;",
                                f'printf("{{\\\"kind\\\":\\\"buffer\\\",\\\"name\\\":\\\"{name}'
                                f'\\\",\\\"address\\\":%" PRIuPTR ",\\\"bytes\\\":{size}}}\\n",(uintptr_t){name});'])
            arguments.append(name)
        else:
            if name not in settings.get("scalar_values", {}):
                raise ValueError("scalar capture inputs require explicit runtime values")
            value = settings["scalar_values"][name]
            if dtype == "bool" and type(value) is bool:
                literal = "true" if value else "false"
            elif dtype == "f32" and type(value) in (float, int) and math.isfinite(value):
                literal = repr(float(value)) + "f"
            else:
                raise ValueError("unsupported native capture scalar input")
            allocations.append(f"{ctype} {name}={literal};")
            arguments.append(name if dtype == "bool" else "&" + name)
    header_text = f'#include {json.dumps(str(header))}\n#include <inttypes.h>\n'
    header_text += "\n".join(settings.get("preamble", ())) + "\n"
    header_text += f'''static unsigned long capture_count;
static void capture_issue(unsigned long slot,uint64_t x,uint64_t y,unsigned long funct) {{
 if (++capture_count>{maximum}) exit(2);
 printf("{{\\\"kind\\\":\\\"command\\\",\\\"custom_slot\\\":%lu,\\\"funct\\\":%lu,"
        "\\\"rs1\\\":%" PRIu64 ",\\\"rs2\\\":%" PRIu64 "}}\\n",slot,funct,x,y);
}}
#undef {sink}
#define {sink}(slot,x,y,funct) capture_issue((slot),(uint64_t)(x),(uint64_t)(y),(funct))
#undef {fence}
#define {fence}() puts("{{\\\"kind\\\":\\\"fence\\\"}}")
'''
    native_source = header_text + function["source"] + "\nint main(void){\n" + "\n".join(allocations)
    native_source += f"\n{symbol}({','.join(arguments)});\n"
    native_source += 'puts("{\\\"kind\\\":\\\"complete\\\"}"); return 0;}\n'
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    source, executable = ws / "capture.c", ws / "capture.native"
    source.write_text(native_source)
    compiler = shutil.which(settings.get("compiler", "cc"))
    if compiler is None:
        raise ValueError("selected native C compiler is unavailable")
    build = invocation_record.run([compiler, "-O2", "-I", str(root), str(source), "-lm", "-o", str(executable)],
                                  directory=ws / "compile", stage="compile_native_issue_capture",
                                  inputs=(source, header), outputs=(executable,), capture_output=True, timeout=60)
    if build.returncode:
        raise ValueError("native capture compilation failed: " + build.stderr.decode()[-2048:])
    trace = ws / "capture.jsonl"
    run = invocation_record.run([str(executable)], directory=ws / "execute", stage="native_issue_capture",
                                inputs=(executable,), outputs=(), capture_output=True, timeout=30)
    if run.returncode:
        raise ValueError("native issue capture failed or exceeded bound")
    trace.write_bytes(run.stdout)
    records = [json.loads(line) for line in run.stdout.splitlines()]
    if not records or records[-1] != {"kind": "complete"}:
        raise ValueError("native issue capture lacks its completion marker")
    # Dependency closure comes from the actual native compiler, including the
    # source header and selected parameter/header dependencies.
    dependency = invocation_record.run([compiler, "-M", "-I", str(root), str(source)],
                                       directory=ws / "dependencies", stage="native_capture_header_closure",
                                       inputs=(source, header), outputs=(), capture_output=True, timeout=30)
    dependency.check_returncode()
    paths = dependency.stdout.decode().replace("\\\n", " ").partition(":")[2].split()
    pins = [_pin(Path(path), "source_header") for path in dict.fromkeys(paths) if Path(path).is_file()]
    pins += [_pin(source, "capture_driver"), _pin(executable, "capture_executable"),
             _pin(trace, "capture"), _pin(compiler, "native_compiler")]
    return dict(function=function, records=records, artifacts=pins, scalar_values=settings.get("scalar_values", {}),
                native_timing=False, capture_path=str(trace))


def capture_reference(document, index, *, root, workspace):
    """Automatic selected-source capture and target-owned semantic observation."""
    settings = dict(document["capture_binding"]["settings"])
    selected_id = settings.get("reference_id")
    candidates = [row for row in document["references"] if row["format"] == "c_source"
                  and (selected_id is None or row["id"] == selected_id)]
    if len(candidates) != 1:
        raise ValueError("native capture requires one explicit source reference selection")
    selection = candidates[0]
    symbols = selection.get("symbols", ())
    if len(symbols) != 1 or "member" in selection["artifact"]:
        raise ValueError("capture requires one selected symbol and an already indexed ordinary source file")
    path = (Path(root) / selection["artifact"]["path"]).resolve()
    raw = read_pinned(path, selection["artifact"]["sha256"])
    capture = capture_native_function(raw, symbols[0], settings=settings, workspace=Path(workspace) / "native")
    capture["artifacts"].append(_pin(path, "source"))
    observer = _entrypoint(settings["observer"])
    observer_path = inspect.getsourcefile(observer)
    if observer_path is None:
        raise ValueError("selected reference observer must have pinned inspectable source")
    capture["artifacts"].extend((_pin(observer_path, "observer_source"), _pin(__file__, "capture_tool")))
    observed = observer(capture, settings=settings, workspace=Path(workspace) / "observation")
    return dict(observed, selected_reference=selection["id"], capture=capture["capture_path"])
