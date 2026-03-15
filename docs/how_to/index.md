# How-To Guides

Practical implementation guides for common Merlin extension tasks:

- add a new compiler dialect + plugin target
- add/modify a runtime HAL driver
- add a new sample application (including async-style samples)
- add a new compile target YAML for `tools/compile.py`
- use `tools/build.py` profiles/flags and find output artifacts

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
5. [Use `tools/build.py`](use_build_py.md)
