"""Pure whole-program caller source renderer, not an execution or admission service."""
from __future__ import annotations
from .format import CodegenError, container_for, container_words, ceil_dim, pad_rowmajor, buffer_extent
from .measurement import assemble_measurement_fragments

def render_whole_program(cb: dict, *, inputs: dict | None = None,
                         prepack_authorizations=None, legacy_helpers=None,
                         measurement_fragments=None, legacy_dim=None) -> str:
    """Call a submitted whole-program kernel through its explicitly declared pointer ABI.

    The harness allocates buffers, performs one unmeasured warm call, measures one call, and prints the
    declared result buffers. It does not interpret or reconstruct any operation between those calls;
    mesh and scalar-host work must both be present in the submitted ``gemmini_kernel`` artifact.
    """

    tensors = cb.get("tensors") or {}
    abi = cb.get("kernel_abi") or {}
    args = abi.get("args") or []
    outputs = abi.get("outputs") or []
    # Opt-in storage is a format contract, not permission to compute derived operands here.
    # This is a host allocation safety cap, not a discovered hardware memory capacity.
    from merlin.runtime.storage_binding import resolve_storage_bindings
    try:
        storage = resolve_storage_bindings(cb, inputs, max_storage_bytes=256 * 1024 * 1024,
                                           prepack_authorizations=prepack_authorizations)
    except ValueError as exc:
        error = CodegenError(str(exc))
        if hasattr(exc, "storage_obligations"):
            error.storage_obligations = exc.storage_obligations
        raise error from exc
    leaves = None
    if storage is None:
        if legacy_dim is not None:
            if legacy_helpers is not None or type(legacy_dim) is not int or legacy_dim <= 0:
                raise CodegenError("legacy exact inputs require one explicit positive host ABI dimension")
            if any(arg.get("access") not in {"read", "write"} for arg in args):
                raise CodegenError("legacy exact input renderer refuses readwrite warm inputs")
            reads = {arg["tensor"] for arg in args if arg["access"] == "read"}
            if not isinstance(inputs, dict) or set(inputs) != reads:
                raise CodegenError("legacy caller requires all and only exact read input tensors")
            for name in reads:
                leaf, spec = inputs[name], tensors[name]
                if tuple(getattr(leaf, "shape", ())) != tuple(spec["shape"]) or getattr(leaf, "dtype", None) != spec["dtype"]:
                    raise CodegenError("legacy exact input dtype/shape differs from caller tensor")
            _ceil_dim = lambda value: ceil_dim(value, legacy_dim)
            _pad_rowmajor, _buffer_extent, leaves = pad_rowmajor, buffer_extent, inputs
        elif legacy_helpers is None:
            raise CodegenError("legacy whole-program storage requires explicit host layout helpers")
        else:
            _ceil_dim, _pad_rowmajor, _buffer_extent, materialize_inputs = legacy_helpers
            leaves = materialize_inputs(cb, inputs)

    decls: list[str] = []
    layouts: dict[str, tuple[int, int, int]] = {}
    for arg in args:
        name, access = arg["tensor"], arg["access"]
        spec = tensors[name]
        if storage is not None:
            binding = storage[name]
            container = container_for(binding.encoding.dtype)
            initializer = None if binding.logical_values is None else ",".join(str(word) for word in
                binding.pack_words(container_words(binding.logical_values, binding.encoding.dtype)))
            decls.append(container.decl(f"T_{name}", binding.encoding.storage_elements,
                const=access == "read", initializer=initializer))
            continue
        rows, cols = _buffer_extent(spec, name=name)
        prows, pcols = _ceil_dim(rows), _ceil_dim(cols)
        layouts[name] = (rows, cols, pcols)
        container = container_for(str(spec.get("dtype") or ""))
        if access == "write":
            decls.append(container.decl(f"T_{name}", prows * pcols))
        else:
            padded = _pad_rowmajor(list(leaves[name].data), rows, cols, prows, pcols)
            decls.append(container.decl(
                f"T_{name}", prows * pcols, const=access == "read",
                initializer=",".join(str(word) for word in
                                     container_words(padded, str(spec.get("dtype") or "")))))

    call = ", ".join(f"(void*)T_{arg['tensor']}" for arg in args)
    prints: list[str] = []
    for name in outputs:
        if storage is not None:
            from math import prod
            binding = storage[name]
            shape = binding.encoding.logical_shape
            rows, cols = (prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            terms = [str(binding.encoding.offset_elements)]
            for divisor, extent, stride in binding.logical_offset_terms():
                terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
            element = f"T_{name}[{' + '.join(terms)}]"
            prints.extend([f'  printf("OUT {name} {rows} {cols}");',
                f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
                f" {container_for(binding.encoding.dtype).printf_element(element)}", '  printf("\\n");'])
            continue
        rows, cols, pcols = layouts[name]
        container = container_for(str(tensors[name].get("dtype") or ""))
        prints.extend([
            f'  printf("OUT {name} {rows} {cols}");',
            f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
            f" {container.printf_element(f'T_{name}[i * {pcols} + j]')}",
            '  printf("\\n");',
        ])

    measured_call = f"  gemmini_kernel({call});\n  gemmini_fence();"
    fragments = (measurement_fragments or assemble_measurement_fragments)(measured_call)
    return ("#include <stdint.h>\n#include <stdio.h>\n#include \"include/gemmini_testutils.h\"\n"
            + fragments["include"]
            + "extern void gemmini_kernel();\n" + "\n".join(decls) + "\nint main() {\n"
            + fragments["warmup"]
            + fragments["prologue"]
            + "  uint64_t c0 = read_cycles();\n"
            + measured_call + "\n"
            + "  uint64_t c1 = read_cycles();\n"
            + fragments["epilogue"]
            + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
            '  printf("METRIC cycle_window_gemmini_region 1\\n");\n'
            + "\n".join(prints) + "\n"
            '  printf("DONE\\n");\n  return 0;\n}\n')
