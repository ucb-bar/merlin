# How-To Guides

Practical implementation guides for common Merlin extension tasks:

- add a new compiler dialect + plugin target
- add/modify a runtime HAL driver
- add a new sample application (including async-style samples)
- add a new compile target YAML for `./merlin compile`
- use `./merlin build` profiles/flags and find output artifacts

These guides are based on current in-tree implementations:

- Gemmini plugin + dialect stack
- NPU plugin + dialect stack
- Radiance runtime HAL driver
- SpacemiTX60 sample applications

Current caveat: these flows are under active development; successful build/test
here does not imply taped-out hardware validation.

## Guides

1. [Add A Compiler Dialect Plugin](add_compiler_dialect_plugin.md)
2. [Add Or Modify A HAL Driver](add_runtime_hal_driver.md)
3. [Add A Sample Application](add_sample_application.md)
4. [Add A Compile Target](add_compile_target.md)
5. [Use `./merlin build`](use_build_py.md) (includes packaging and release builds)
6. [Bring Up An External Backend With TargetGen](bring_up_external_backend_with_targetgen.md)
7. [Use The Merlin TargetGen MCP Server With Claude Code](use_merlin_mcp_with_claude_code.md)
8. [Inspect And Steer Dispatch Granularity](inspect_and_steer_dispatch_granularity.md)
9. [Embed A Custom Kernel Via Manifest](embed_custom_kernel_via_manifest.md)
10. [Inspect Kernel-Embedding MLIR Phases](inspect_kernel_embedding_phases.md)
11. [Extend Kernel Coverage To Any Model](extend_kernel_coverage_to_any_model.md)
12. [Kernel Embedding — Full Mechanism Walkthrough](kernel_embedding_walkthrough.md)
