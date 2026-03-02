# Tools

`tools/merlin.py` is the stable command entrypoint for maintainers and CI.

## Example

```bash
python3 tools/merlin.py --help
python3 tools/merlin.py targets list
python3 tools/merlin.py patches verify
python3 tools/merlin.py build host-release
python3 tools/merlin.py release-status
```

## Why

This keeps script sprawl manageable by exposing one interface while preserving
existing script internals in `scripts/`.
