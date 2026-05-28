"""`./merlin verify-output` — cross-hash check between backends + golden CPU reference."""

from __future__ import annotations

from pathlib import Path


def setup_parser(parser):
    parser.add_argument("model", type=Path, help="Quantized .q.int8.onnx model")
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        help="Input shape comma-separated (repeat per input)",
    )
    parser.add_argument(
        "--observed",
        action="append",
        default=[],
        help="Backend hash to verify: <hex_hash>:<label> (e.g. 0x498...:gemmini)",
    )
    parser.add_argument(
        "--uartlog",
        action="append",
        type=Path,
        default=[],
        help="FireSim uartlog file to extract hashes from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xCAFE,
        help="RNG seed for x86 reference input when --random-input (default 0xCAFE)",
    )
    parser.add_argument(
        "--random-input",
        action="store_true",
        help=(
            "Use random input instead of all-zero (the runner uses zeros "
            "via ZERO_FILL buffer alloc). Use this only for sanity checks."
        ),
    )
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="Skip the onnxruntime baseline (just cross-check observed hashes)",
    )


def main(args) -> int:
    # Lazy-import the heavy onnxruntime dep.
    from verify_int8_output import (  # noqa: PLC2701
        _golden_hash,
        _parse_observed,
        _parse_shape,
        _parse_uartlog,
    )

    shapes = [_parse_shape(s) for s in args.shape]

    rows = []
    for spec in args.observed:
        h, label = _parse_observed(spec)
        rows.append({"source": "--observed", "label": label, "hash": h})
    for uart in args.uartlog:
        if not uart.exists():
            print(f"uartlog not found: {uart}")
            return 1
        for r in _parse_uartlog(uart):
            rows.append(
                {
                    "source": uart.name,
                    "label": f"job={r['job']} hart={r['hart']}",
                    "hash": r["hash"],
                    "rc": r["rc"],
                }
            )

    golden = None
    if not args.skip_golden:
        print(f"==> computing CPU x86 golden hash from {args.model}")
        golden = _golden_hash(
            args.model,
            shapes,
            zero_input=not args.random_input,
            seed=args.seed,
        )
        print(f"    golden hash = 0x{golden:016x} (zero_input={not args.random_input})")

    if not rows:
        print("no observed hashes to compare. Pass --observed HASH:LABEL or --uartlog FILE.")
        return 0 if args.skip_golden else 0

    print()
    print(f"{'source':<46} {'label':<32} {'hash':<20} {'verdict'}")
    print("-" * 110)
    any_mismatch = False
    for r in rows:
        h = r["hash"]
        verdict = "OK"
        if golden is not None and h != golden:
            verdict = "DIFF-FROM-GOLDEN"
            any_mismatch = True
        elif r.get("rc", 0) != 0:
            verdict = f"rc={r['rc']}"
            any_mismatch = True
        print(f"{r['source']:<46} {r['label']:<32} 0x{h:016x}   {verdict}")

    distinct_hashes = sorted({r["hash"] for r in rows})
    if len(distinct_hashes) > 1:
        print()
        print(f"WARNING: {len(distinct_hashes)} DISTINCT hashes across observed runs:")
        for h in distinct_hashes:
            labels = [r["label"] for r in rows if r["hash"] == h]
            print(f"  0x{h:016x}  →  {', '.join(labels)}")
        any_mismatch = True

    return 1 if any_mismatch else 0
