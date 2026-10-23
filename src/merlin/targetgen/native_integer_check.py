"""Reference CPU exception domains and complete signed-byte quotient checks."""


def generate():
    import hashlib
    import json
    import subprocess
    import sys

    import torch

    # Check trapping domains in isolated processes. A CPU SIGFPE is evidence
    # for retaining the entire value-producing program on host, never a value.
    exceptions = []
    for width in (8, 16, 32, 64):
        for mode in ("trunc", "floor", None):
            for divisor in (0, -1):
                code = f"""import torch,json,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
torch.set_num_threads(1)
x=torch.tensor([-(1<<{width - 1}),0,1],dtype=torch.int{width})
try:
 y=torch.div(x,{divisor},rounding_mode={mode!r})
 print(json.dumps(dict(status="value",dtype=str(y.dtype),bits=y.view(torch.uint8).tolist())))
except Exception as e:print(json.dumps(dict(status="error",type=type(e).__name__,message=str(e))))
"""
                child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
                observation = (
                    json.loads(child.stdout)
                    if child.returncode == 0
                    else dict(status="signal_or_exit", code=child.returncode)
                )
                exceptions.append(dict(width=width, mode=mode, divisor=divisor, observation=observation))
    numerators, divisors = [], []
    for a in range(-128, 128):
        for b in range(-128, 128):
            if b:
                numerators.append(a)
                divisors.append(b)
    rows = []
    for mode in ("trunc", "floor"):
        overflow = next(
            row["observation"]
            for row in exceptions
            if row["width"] == 8 and row["mode"] == mode and row["divisor"] == -1
        )
        selected = [
            (a, b)
            for a, b in zip(numerators, divisors)
            if not (a == -128 and b == -1 and overflow["status"] != "value")
        ]
        x, y = (torch.tensor([v[i] for v in selected], dtype=torch.int8) for i in (0, 1))
        actual = torch.div(x, y, rounding_mode=mode).tolist()
        expected = []
        for a, b in selected:
            q = a // b if mode == "floor" else (-1 if (a < 0) != (b < 0) else 1) * (abs(a) // abs(b))
            expected.append((q + 128) % 256 - 128)
        if actual != expected:
            raise ValueError("full signed-byte CPU quotient theorem failed")
        rows.append(
            dict(
                width=8,
                rounding=mode,
                count=len(actual),
                overflow_domain=overflow["status"],
                output_sha256=hashlib.sha256(bytes(v % 256 for v in actual)).hexdigest(),
                status="proven",
            )
        )
    return dict(
        quotients=rows,
        exceptions=exceptions,
        wider_value_theorem="reciprocal_universal_interval_and_exact_correction",
        integer_true_division="retain_integer_RNE_promotion_before_typed_floating_division",
    )
