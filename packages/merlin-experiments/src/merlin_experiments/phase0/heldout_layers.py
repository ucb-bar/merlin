"""Operator-side guard: no derived performance or holdout member may BE a layer of a held-out network.

Phase 2 tunes on performance-scale members and measures generalization on a disjoint holdout. Both are
meant to be generalization evidence about the held-out full models. A member whose contraction equals one
of those models' own layers would make a speedup on that model in-distribution. So the operator feeds
the held-out models' exact layer shapes (an operator-private file outside the repository, never granted
to an agent), and derivation and the form-holdout commit/reveal refuse any member that reproduces one.

Op families may overlap by design; only EXACT shapes are refused:
* a contraction ``(M, K, N)`` equal to a held-out layer's GEMM extents (rows, reduction depth, columns);
* a convolution whose window, stride and channel pair ``(kh, kw, sh, sw, Cin, Cout)`` equal a held-out
  convolution's, at any spatial size.

The file never reaches a repository or a public record: what is recorded is a count and the digest of
the private file.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "merlin.heldout_layer_shapes.v1"


class HeldoutLayerError(ValueError):
    """Operator error: a derived member reproduces a held-out network layer, or the private input is unfit."""


@dataclass(frozen=True)
class HeldoutLayers:
    sha256: str
    gemms: dict[tuple[int, int, int], tuple[str, str]] = field(default_factory=dict)
    convs: dict[tuple[int, int, int, int, int, int], tuple[str, str]] = field(default_factory=dict)

    @property
    def networks(self) -> list[str]:
        return sorted({network for network, _ in [*self.gemms.values(), *self.convs.values()]})

    def gemm(self, m: int, k: int, n: int) -> tuple[str, str] | None:
        return self.gemms.get((int(m), int(k), int(n)))

    def conv(self, kh: int, kw: int, sh: int, sw: int, cin: int, cout: int) -> tuple[str, str] | None:
        return self.convs.get((int(kh), int(kw), int(sh), int(sw), int(cin), int(cout)))

    def summary(self) -> dict[str, Any]:
        """What a public record may carry: counts and the private file's digest, never a shape."""
        return {
            "schema": SCHEMA,
            "heldout_layer_shapes_sha256": self.sha256,
            "networks": len(self.networks),
            "gemm_shapes": len(self.gemms),
            "conv_windows": len(self.convs),
        }


def _positive(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise HeldoutLayerError(f"held-out layer file: {where} must be a positive integer")
    return value


def load(path: str | Path, *, repository: str | Path | None = None) -> HeldoutLayers:
    """Read the operator-private held-out layer file.

    It must be an ordinary file readable by its owner only, and must not lie inside ``repository``
    (the checkout an agent reads): the held-out shapes are never public input."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise HeldoutLayerError(f"held-out layer file is absent, linked or not a file: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise HeldoutLayerError(f"held-out layer file must be owner-only (mode 0600): {path}")
    if repository is not None and path.resolve().is_relative_to(Path(repository).resolve()):
        raise HeldoutLayerError("held-out layer file must be operator-private, outside the repository")
    raw = path.read_bytes()
    document = json.loads(raw)
    networks = document.get("networks") if isinstance(document, Mapping) else None
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA or not isinstance(networks, Mapping):
        raise HeldoutLayerError(f"held-out layer file must be a {SCHEMA} document with networks")
    if not networks:
        raise HeldoutLayerError("held-out layer file declares no network")
    gemms: dict[tuple[int, int, int], tuple[str, str]] = {}
    convs: dict[tuple[int, int, int, int, int, int], tuple[str, str]] = {}
    for network, layers in sorted(networks.items()):
        if not isinstance(layers, Mapping) or set(layers) - {"contractions", "convs"}:
            raise HeldoutLayerError(f"held-out layer file: {network} declares contractions and convs only")
        for row in layers.get("contractions") or []:
            key = tuple(_positive(row.get(axis), f"{network}.{axis}") for axis in ("M", "K", "N"))
            gemms.setdefault(key, (str(network), str(row.get("layer") or "")))
        for row in layers.get("convs") or []:
            kernel, stride = row.get("kernel"), row.get("stride")
            if not (isinstance(kernel, list) and len(kernel) == 2 and isinstance(stride, list) and len(stride) == 2):
                raise HeldoutLayerError(f"held-out layer file: {network} conv needs kernel and stride pairs")
            key = (
                *(_positive(v, f"{network}.kernel") for v in kernel),
                *(_positive(v, f"{network}.stride") for v in stride),
                _positive(row.get("cin"), f"{network}.cin"),
                _positive(row.get("cout"), f"{network}.cout"),
            )
            convs.setdefault(key, (str(network), str(row.get("layer") or "")))
    if not gemms and not convs:
        raise HeldoutLayerError("held-out layer file declares no layer")
    return HeldoutLayers(hashlib.sha256(raw).hexdigest(), gemms, convs)


def from_inventories(inventories: Iterable[str | Path]) -> dict[str, Any]:
    """Build the private document from per-program operator inventories (``contractions`` rows with a
    ``gemm`` of M/K/N, and conv rows with kernel/stride/Cin/Cout); one network per inventory ``network``."""
    networks: dict[str, dict[str, list]] = {}
    for path in inventories:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        network = str(document.get("network") or Path(path).stem)
        layers = networks.setdefault(network, {"contractions": [], "convs": []})
        seen_gemm = {(r["M"], r["K"], r["N"]) for r in layers["contractions"]}
        seen_conv = {(tuple(r["kernel"]), tuple(r["stride"]), r["cin"], r["cout"]) for r in layers["convs"]}
        for row in document.get("contractions") or []:
            gemm = row.get("gemm") or {}
            extents = tuple(gemm.get(axis) for axis in ("M", "K", "N"))
            label = f"{document.get('entry', '')}:{row.get('module') or row.get('name')}"
            if all(isinstance(value, int) and value > 0 for value in extents) and extents not in seen_gemm:
                seen_gemm.add(extents)
                layers["contractions"].append(dict(zip(("M", "K", "N"), extents, strict=True), layer=label))
            if row.get("family") == "conv2d":
                key = (tuple(row["kernel"]), tuple(row["stride"]), row["Cin"], row["Cout"])
                if key not in seen_conv:
                    seen_conv.add(key)
                    layers["convs"].append(
                        {
                            "kernel": list(row["kernel"]),
                            "stride": list(row["stride"]),
                            "cin": row["Cin"],
                            "cout": row["Cout"],
                            "layer": label,
                        }
                    )
    return {"schema": SCHEMA, "networks": networks}


def write_private(document: Mapping[str, Any], output: str | Path) -> Path:
    """Write the document as a fresh owner-only file."""
    output = Path(output)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, sort_keys=True, indent=1) + "\n")
    output.chmod(0o600)
    return output


def form_scope_collisions(forms: Mapping[str, Any], layers: HeldoutLayers) -> list[dict[str, Any]]:
    """Every device contraction member of a derived form scope whose extents equal a held-out layer, or
    a convolution member whose window, stride and channel pair equal a held-out convolution's."""
    from .form_perf import gemm_extents

    found = []
    for row in forms.get("classes") or []:
        for member in row.get("members") or []:
            entry = member.get("entry") or {}
            if str(entry.get("op")) not in {"matmul", "conv2d", "batch_matmul"}:
                continue
            try:
                extents = gemm_extents(entry)
            except (KeyError, TypeError, ValueError):
                continue
            hit = layers.gemm(*extents)
            if hit is None and str(entry.get("op")) == "conv2d":
                stride = list(entry.get("stride") or [1, 1])
                try:
                    window = (entry["kh"], entry["kw"], stride[0], stride[1], entry["ci"], entry["N"])
                    hit = layers.conv(*window)
                except (KeyError, TypeError, ValueError, IndexError):
                    hit = None
            if hit is not None:
                found.append(
                    {
                        "application": member.get("application"),
                        "class": row.get("label"),
                        "extents": list(extents),
                        "network": hit[0],
                        "layer": hit[1],
                    }
                )
    return found


def workload_collision(shape: Mapping[str, Any], layers: HeldoutLayers) -> tuple[str, str] | None:
    """A derived capsule workload shape (``m``/``k``/``n``) equal to a held-out layer, or ``None``."""
    try:
        return layers.gemm(int(shape["m"]), int(shape["k"]), int(shape["n"]))
    except (KeyError, TypeError, ValueError):
        return None


def refuse(collisions: list[dict[str, Any]], *, what: str) -> None:
    """Raise the operator error naming each colliding member (operator-side; never an agent record)."""
    if not collisions:
        return
    lines = [
        f"{row.get('application') or row.get('member')} {row.get('class') or ''} "
        f"{'x'.join(str(v) for v in row['extents'])} = {row['network']} {row['layer']}"
        for row in collisions
    ]
    raise HeldoutLayerError(
        f"OPERATOR: {len(collisions)} {what} member(s) reproduce a held-out network layer exactly; "
        "re-pick the generator widths so no member is an evaluation layer:\n  " + "\n  ".join(lines)
    )


__all__ = [
    "SCHEMA",
    "HeldoutLayerError",
    "HeldoutLayers",
    "form_scope_collisions",
    "from_inventories",
    "load",
    "refuse",
    "workload_collision",
    "write_private",
]
