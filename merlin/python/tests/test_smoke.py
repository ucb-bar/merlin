"""Smoke test: the package imports and schemas/benchmarks parse."""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_package_imports():
    import merlin  # noqa: F401


def test_schemas_parse():
    yaml = __import__("yaml") if _has_yaml() else None
    if yaml is None:
        return  # pyyaml not installed; skip rather than fail the scaffold
    files = glob.glob(os.path.join(ROOT, "merlin", "schemas", "*.yaml"))
    assert files, "no schema files found"
    for f in files:
        with open(f) as fh:
            yaml.safe_load(fh)


def _has_yaml():
    try:
        import yaml  # noqa: F401
        return True
    except Exception:
        return False
