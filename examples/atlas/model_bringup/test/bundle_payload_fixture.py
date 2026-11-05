"""Write supplied bundle bytes and their ordered checksum manifest."""
import hashlib


def write_bundle_payloads(root, files):
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (root / "SHA256SUMS").write_text("".join(
        hashlib.sha256(raw).hexdigest() + "  " + name + "\n" for name, raw in files.items()))
