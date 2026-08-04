#!/usr/bin/env python3
"""Reconcile experiment cost against the REAL AWS Bedrock bill from the terminal.

Our per-run ``estimated_cost_usd`` is a list-price ESTIMATE (see targetgen/experiment_tokens.py). This
pulls the authoritative numbers straight from AWS:

  * ``aws ce get-cost-and-usage``  → actual Bedrock $ (month-to-date or daily), optionally grouped by a
    cost-allocation tag (use a per-experiment Bedrock *application inference profile* tag to attribute
    spend per experiment — https://docs.aws.amazon.com/bedrock/latest/userguide/cost-management.html).
  * ``aws cloudwatch get-metric-data`` → real InputTokenCount / OutputTokenCount by ModelId (AWS/Bedrock),
    so you can see burn BEFORE the bill finalizes.

Auth note: the Bedrock *bearer* token (AWS_BEARER_TOKEN_BEDROCK) is INFERENCE-ONLY; the cost/metric APIs
need standard IAM creds (access key / SSO) with ce:GetCostAndUsage + cloudwatch:GetMetricData. Without
them this prints a clear, actionable NO-CREDS message instead of a wrong number.

Examples:
  bedrock_cost.py cost --month
  bedrock_cost.py cost --daily --days 14 --group-tag Experiment
  bedrock_cost.py tokens --days 7
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys

_SERVICE = "Amazon Bedrock"


def _aws(args: list[str]) -> tuple[bool, dict | str]:
    """Run an aws CLI subcommand returning JSON. (ok, parsed|error-string). Date.now is not used in a
    workflow context here — this is a CLI, so real clock access is fine."""
    try:
        p = subprocess.run(["aws", *args, "--output", "json"], capture_output=True, text=True, timeout=90)
    except FileNotFoundError:
        return False, "aws CLI not found (install awscli v2)"
    except Exception as e:  # noqa: BLE001
        return False, f"aws invocation failed: {e}"
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip()
        if "NoCredentials" in err or "Unable to locate credentials" in err:
            return False, ("no AWS cost-API credentials. The Bedrock BEARER token is inference-only; "
                           "configure IAM creds with ce:GetCostAndUsage + cloudwatch:GetMetricData "
                           "(aws configure / aws sso login) and retry.")
        if "AccessDenied" in err:
            return False, f"access denied (the principal lacks ce/cloudwatch read permission): {err[:200]}"
        return False, err[:400]
    try:
        return True, json.loads(p.stdout or "{}")
    except Exception as e:  # noqa: BLE001
        return False, f"could not parse aws output: {e}"


def cmd_cost(a) -> int:
    today = _dt.date.today()
    if a.month:
        start, end, gran = today.replace(day=1).isoformat(), (today + _dt.timedelta(days=1)).isoformat(), "MONTHLY"
    else:
        start = (today - _dt.timedelta(days=a.days)).isoformat()
        end = (today + _dt.timedelta(days=1)).isoformat()
        gran = "DAILY"
    args = ["ce", "get-cost-and-usage", "--time-period", f"Start={start},End={end}",
            "--granularity", gran, "--metrics", "UnblendedCost",
            "--filter", json.dumps({"Dimensions": {"Key": "SERVICE", "Values": [_SERVICE]}})]
    if a.group_tag:
        args += ["--group-by", json.dumps([{"Type": "TAG", "Key": a.group_tag}])]
    ok, res = _aws(args)
    if not ok:
        print(f"NO-COST: {res}", file=sys.stderr)
        return 2
    total = 0.0
    for period in res.get("ResultsByTime", []):
        ts = period["TimePeriod"]["Start"]
        if period.get("Groups"):
            for g in period["Groups"]:
                amt = float(g["Metrics"]["UnblendedCost"]["Amount"])
                total += amt
                print(f"{ts}  {g['Keys'][0]:40}  ${amt:.2f}")
        else:
            amt = float(period["Total"]["UnblendedCost"]["Amount"])
            total += amt
            print(f"{ts}  {_SERVICE:40}  ${amt:.2f}")
    print(f"\nBedrock {'month-to-date' if a.month else f'last {a.days}d'} total: ${total:.2f}  (AUTHORITATIVE)")
    return 0


def cmd_tokens(a) -> int:
    end = _dt.datetime.now(_dt.timezone.utc)
    start = end - _dt.timedelta(days=a.days)
    queries = [{"Id": mid, "MetricStat": {
        "Metric": {"Namespace": "AWS/Bedrock", "MetricName": name},
        "Period": 86400, "Stat": "Sum"}}
        for mid, name in (("input", "InputTokenCount"), ("output", "OutputTokenCount"))]
    ok, res = _aws(["cloudwatch", "get-metric-data",
                    "--start-time", start.isoformat(), "--end-time", end.isoformat(),
                    "--metric-data-queries", json.dumps(queries)])
    if not ok:
        print(f"NO-TOKENS: {res}", file=sys.stderr)
        return 2
    for r in res.get("MetricDataResults", []):
        print(f"{r['Id']:8} Input/OutputTokenCount sum over {a.days}d: {sum(r.get('Values') or []):,.0f}")
    print("(per-ModelId dimensions available with --group by ModelId; this is the account rollup)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Query the REAL AWS Bedrock cost / token metrics.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost", help="actual Bedrock $ from Cost Explorer")
    c.add_argument("--month", action="store_true", help="month-to-date total (else daily)")
    c.add_argument("--days", type=int, default=30)
    c.add_argument("--group-tag", help="cost-allocation tag to group by (e.g. per-experiment)")
    c.set_defaults(fn=cmd_cost)
    t = sub.add_parser("tokens", help="real InputTokenCount/OutputTokenCount from CloudWatch")
    t.add_argument("--days", type=int, default=7)
    t.set_defaults(fn=cmd_tokens)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
