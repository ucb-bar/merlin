"""Render a checked no-op for a proven dense storage alias."""

def compile_view(region, name):
    """Emit a checked no-op for an already proved and allocated storage view."""
    from merlin.compiler.lower.c.dispatch import c_source_pair
    from .regions import CompiledCRegion

    nodes = [node for node in region.nodes if node.effect == "pure"]
    if len(nodes) != 1 or len(nodes[0].inputs) != 1 or region.outputs != (nodes[0].id,):
        raise ValueError("dense buffer view needs one source and one result")
    output = nodes[0]
    declaration = f"int {name}(void *,const void *const *,void *const *);"
    implementation = f"""int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  return !inputs || !outputs || !inputs[0] || inputs[0]!=outputs[0];
}}
"""
    files = c_source_pair(
        name, public_headers={}, private_headers={}, sources={"view.c": implementation}, declarations=declaration
    )
    return CompiledCRegion(
        name,
        name + ".h",
        output.inputs,
        region.outputs,
        {path: data.encode() for path, data in files.items()},
        (),
        (),
        {
            "kind": "dense_buffer_view",
            "launches_per_invocation": 0,
            "input": output.inputs[0],
            "output": output.id,
            "byte_offset": 0,
        },
    )
