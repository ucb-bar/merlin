"""The existing recipe JSON transformation command and its exact option surface."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from .dynamic import (
    with_dynamic_contractions as with_dynamic_contractions,
)
from .dynamic import (
    with_site_agreement as with_site_agreement,
)
from .recipe import (
    ACTIVATION_CONTRACTION as ACTIVATION_CONTRACTION,
)
from .recipe import (
    NON_REDUCTION as NON_REDUCTION,
)
from .recipe import (
    SOFTMAX_OUTPUT as SOFTMAX_OUTPUT,
)
from .recipe import (
    STATIC_HEADS as STATIC_HEADS,
)
from .static import (
    with_activation_contractions as with_activation_contractions,
)
from .static import (
    with_activation_observer as with_activation_observer,
)
from .static import (
    with_gptq_weights as with_gptq_weights,
)
from .static import (
    with_native_output_quantization as with_native_output_quantization,
)
from .static import (
    with_shared_grid_operand_sums as with_shared_grid_operand_sums,
)
from .static import (
    with_static_planes as with_static_planes,
)


def main(argv: Sequence[str] | None = None, *, api=None) -> int:
    """``python -m merlin.compiler.quantization.capabilities {activation-contractions|dynamic-contractions|static-planes|
    shared-grid-sums|native-outputs|site-agreement|activation-observer|gptq-weights} IN OUT``."""
    if api is None:
        api = sys.modules[__name__]
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m merlin.compiler.quantization.capabilities")
    sub = parser.add_subparsers(dest="command", required=True)
    variant = sub.add_parser("activation-contractions", help=api.with_activation_contractions.__doc__.splitlines()[0])
    variant.add_argument("recipe", type=Path)
    variant.add_argument("out", type=Path)
    variant.add_argument(
        "--minimum-reduction",
        type=int,
        default=None,
        help="explicit inclusive activation reduction minimum; default retains the profitability screen",
    )
    residual = sub.add_parser("shared-grid-sums", help=api.with_shared_grid_operand_sums.__doc__.splitlines()[0])
    residual.add_argument("recipe", type=Path)
    residual.add_argument("out", type=Path)
    residual.add_argument(
        "--output-boundary",
        default="quantized_readers",
        choices=("quantized_readers", "dequantize"),
        help="dequantize explicitly permits floating-point consumers after the new INT8 residual output",
    )
    residual.add_argument("--residual-selection", default=None, choices=("distinct_contraction_frontiers",))
    outputs = sub.add_parser("native-outputs", help=api.with_native_output_quantization.__doc__.splitlines()[0])
    outputs.add_argument("recipe", type=Path)
    outputs.add_argument("out", type=Path)
    outputs.add_argument("--operators", nargs="+", default=["linear"], choices=("linear", "conv2d"))
    outputs.add_argument("--reader-policy", default="non_add_compute", choices=("compute", "non_add_compute"))
    dynamic = sub.add_parser("dynamic-contractions", help=api.with_dynamic_contractions.__doc__.splitlines()[0])
    dynamic.add_argument("recipe", type=Path)
    dynamic.add_argument("out", type=Path)
    dynamic.add_argument("--families", default=ACTIVATION_CONTRACTION, help="comma-separated contraction families")
    dynamic.add_argument("--planes", default="", help=f"selector=k pairs, e.g. {SOFTMAX_OUTPUT}=2")
    dynamic.add_argument("--smooth", action="store_true", help="subtract the mean over a softmax's axis")
    dynamic.add_argument("--static-families", default=None, help="families kept static (default: the recipe's)")
    dynamic.add_argument("--scales", default=NON_REDUCTION, choices=(NON_REDUCTION, STATIC_HEADS))
    dynamic.add_argument("--context-output", choices=("int8_value_grid",))
    wide = sub.add_parser("static-planes", help=api.with_static_planes.__doc__.splitlines()[0])
    wide.add_argument("recipe", type=Path)
    wide.add_argument("out", type=Path)
    wide.add_argument("--planes", type=int, default=2)
    wide.add_argument("--observer", default="minmax")
    wide.add_argument("--owners", default=None, help="comma-separated module-path prefixes (default: all)")
    observe = sub.add_parser("activation-observer", help=api.with_activation_observer.__doc__.splitlines()[0])
    observe.add_argument("recipe", type=Path)
    observe.add_argument("out", type=Path)
    observe.add_argument("observer")
    bar = sub.add_parser("site-agreement", help=api.with_site_agreement.__doc__.splitlines()[0])
    bar.add_argument("recipe", type=Path)
    bar.add_argument("out", type=Path)
    bar.add_argument("--cosine-min", type=float, required=True, dest="cosine_min")
    bar.add_argument("--rel-rmse-max", type=float, required=True, dest="rel_rmse_max")
    bar.add_argument("--basis", required=True, help="where the bar came from (recorded in the recipe's why)")
    gptq = sub.add_parser("gptq-weights", help=api.with_gptq_weights.__doc__.splitlines()[0])
    gptq.add_argument("recipe", type=Path)
    gptq.add_argument("out", type=Path)
    gptq.add_argument("--calibration-samples", type=int, dest="calibration_samples")
    gptq.add_argument("--act-order", action=argparse.BooleanOptionalAction, dest="act_order")
    gptq.add_argument("--damp", type=float)
    gptq.add_argument("--block", type=int)
    args = parser.parse_args(argv)
    source = json.loads(args.recipe.read_text(encoding="utf-8"))
    if args.command == "shared-grid-sums":
        body = api.with_shared_grid_operand_sums(
            source, output_boundary=args.output_boundary, residual_selection=args.residual_selection
        )
    elif args.command == "native-outputs":
        body = api.with_native_output_quantization(source, operators=args.operators, reader_policy=args.reader_policy)
    elif args.command == "activation-contractions":
        body = api.with_activation_contractions(source, minimum_reduction=args.minimum_reduction)
    elif args.command == "static-planes":
        owners = None if args.owners is None else [o for o in args.owners.split(",") if o]
        body = api.with_static_planes(source, planes=args.planes, observer=args.observer, owners=owners)
    elif args.command == "gptq-weights":
        chosen = {name: getattr(args, name) for name in ("calibration_samples", "act_order", "damp", "block")}
        body = api.with_gptq_weights(source, **{name: value for name, value in chosen.items() if value is not None})
    elif args.command == "site-agreement":
        body = api.with_site_agreement(
            source, cosine_min=args.cosine_min, rel_rmse_max=args.rel_rmse_max, basis=args.basis
        )
    elif args.command == "activation-observer":
        body = api.with_activation_observer(source, args.observer)
    else:
        planes = {k: int(v) for k, _, v in (p.partition("=") for p in args.planes.split(",") if p)}
        static = None if args.static_families is None else [f for f in args.static_families.split(",") if f]
        body = api.with_dynamic_contractions(
            source,
            families=[f for f in args.families.split(",") if f],
            planes=planes,
            smooth=args.smooth,
            static_families=static,
            scales=args.scales,
            context_output=args.context_output,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out}: recipe_sha256 {body['recipe_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
