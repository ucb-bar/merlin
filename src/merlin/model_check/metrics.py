"""The numerics policy's metrics and the ratio rule (contract §4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

RATIO = 1.1
FLOOR_INT8_WORST = 0.5  # LSB
FLOOR_REL = 2.0**-20
FLOOR_KL = 1e-6
#: Metrics where larger is better (agreement); every other float metric is an error, smaller is better.
AGREEMENT = frozenset({"top1_agree", "top5_agree", "cosine"})


def rel_rmse(y: np.ndarray, y_fp: np.ndarray) -> float:
    y, y_fp = np.asarray(y, np.float64), np.asarray(y_fp, np.float64)
    denom = float(np.linalg.norm(y_fp))
    return float(np.linalg.norm(y - y_fp)) / denom if denom else float(np.linalg.norm(y - y_fp))


def worst(y: np.ndarray, y_fp: np.ndarray, scale: float | None = None) -> float:
    """``max|y - y_fp|`` over the output scale (int8 tensors) or ``max|y_fp|`` (float tensors)."""
    y, y_fp = np.asarray(y, np.float64), np.asarray(y_fp, np.float64)
    s = float(scale) if scale else float(np.max(np.abs(y_fp))) or 1.0
    return float(np.max(np.abs(y - y_fp))) / s


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl(fp_logits: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """``KL(p_fp || p)`` per distribution (last axis is the vocabulary)."""
    lp, lq = log_softmax(fp_logits), log_softmax(logits)
    return np.sum(np.exp(lp) * (lp - lq), axis=-1)


def topk_agree(fp_logits: np.ndarray, logits: np.ndarray, k: int) -> float:
    """Fraction of rows whose FP top-1 is in the other's top-k."""
    fp, q = (
        np.asarray(fp_logits).reshape(-1, np.shape(fp_logits)[-1]),
        np.asarray(logits).reshape(-1, np.shape(logits)[-1]),
    )
    top1 = fp.argmax(axis=-1)
    topk = np.argsort(-q, axis=-1)[:, :k]
    return float(np.mean([t in row for t, row in zip(top1, topk, strict=True)]))


def model_metrics(kind: str, judged: Sequence[np.ndarray], fp: Sequence[np.ndarray], spec: dict[str, Any]) -> dict:
    """One input's metrics of ``judged`` against the FP reference, per the model's judged kind."""
    if kind == "logits":
        y, y_fp = (
            np.concatenate([np.asarray(a).reshape(1, -1) for a in judged]),
            np.concatenate([np.asarray(a).reshape(1, -1) for a in fp]),
        )
        out = {"rel_rmse": rel_rmse(y, y_fp), "worst": worst(y, y_fp)}
        for k in spec.get("topk", [1]):
            out[f"top{k}_agree"] = topk_agree(y_fp, y, int(k))
        cos = float(np.sum(y * y_fp) / (np.linalg.norm(y) * np.linalg.norm(y_fp)))
        return {**out, "cosine": cos}
    if kind == "next_token_distribution":
        y = np.concatenate([np.asarray(a).reshape(-1, np.shape(a)[-1]) for a in judged])
        y_fp = np.asarray(fp[0]).reshape(-1, np.shape(fp[0])[-1])
        if y.shape != y_fp.shape:
            raise ValueError(f"{y.shape[0]} judged distributions, {y_fp.shape[0]} FP ones")
        divergence = kl(y_fp, y)
        return {
            "kl_worst": float(divergence.max()),
            "kl_mean": float(divergence.mean()),
            "top1_agree": topk_agree(y_fp, y, 1),
            "distributions": int(y.shape[0]),
        }
    if kind == "per_channel":
        y, y_fp = np.asarray(judged[-1], np.float64), np.asarray(fp[0], np.float64)
        first, last = int(spec["channels"]["first"]), int(spec["channels"]["last"])
        rows = {}
        for channel in range(first, last + 1):
            a, b = y[..., channel], y_fp[..., channel]
            rows[str(channel)] = {"rel_rmse": rel_rmse(a, b), "worst": worst(a, b)}
        return {
            "per_channel": rows,
            "rel_rmse_worst": max(r["rel_rmse"] for r in rows.values()),
            "worst_worst": max(r["worst"] for r in rows.values()),
        }
    raise ValueError(f"unknown judged kind {kind!r}")


def ratio_rule(value: float, reference: float, floor: float, ratio: float = RATIO) -> dict[str, Any]:
    """``value <= max(ratio * reference, floor)`` with the threshold it was held to."""
    threshold = max(ratio * reference, floor)
    return {"value": value, "reference": reference, "threshold": threshold, "pass": bool(value <= threshold)}


def worst_of(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Each float metric's worst value over ``rows``: the minimum of an agreement, the maximum of an error."""
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, float)})
    return {
        key: (min if key in AGREEMENT else max)(row[key] for row in rows if isinstance(row.get(key), float))
        for key in keys
    }
