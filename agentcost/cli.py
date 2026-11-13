"""
    agentcost claude                 # every Claude Code session on this machine
    agentcost claude --last 7d       # sessions touched in the last 7 days
    agentcost claude --project <dir> # one project folder under ~/.claude/projects
    agentcost claude --json          # machine-readable
    agentcost session <file.jsonl>   # one session, full cost tree
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import claude_code


def _since(spec: str | None) -> float | None:
    if not spec:
        return None
    m = re.fullmatch(r"(\d+)([hdw])", spec)
    if not m:
        raise SystemExit(f"--last wants e.g. 24h, 7d, 2w; got {spec!r}")
    n, unit = int(m.group(1)), m.group(2)
    return time.time() - n * {"h": 3600, "d": 86400, "w": 604800}[unit]


def _compare(pa: Path, pb: Path) -> int:
    """Side-by-side totals. The number people act on is the delta column."""
    ta = json.loads(pa.read_text())["totals"]
    tb = json.loads(pb.read_text())["totals"]
    rows = [
        ("calls", ta["calls"], tb["calls"], "{:,}"),
        ("tokens", ta["tokens_total"], tb["tokens_total"], "{:,}"),
        ("cost $", ta["cost_usd"], tb["cost_usd"], "{:.4f}"),
        ("waste % tok", ta["waste_share"] * 100, tb["waste_share"] * 100, "{:.1f}"),
    ]
    if ta.get("waste_share_cost") is not None and tb.get("waste_share_cost") is not None:
        rows.append(("waste % cost", ta["waste_share_cost"] * 100, tb["waste_share_cost"] * 100, "{:.1f}"))
    for k in ("served", "retried", "discarded", "late", "error"):
        va = ta["by_outcome"].get(k, {}).get("tokens", 0)
        vb = tb["by_outcome"].get(k, {}).get("tokens", 0)
        if va or vb:
            rows.append((f"{k} tok", va, vb, "{:,}"))
    print(f"{'':<14}{ta['run'][:18]:>20}{tb['run'][:18]:>20}{'delta':>14}")
    for name, va, vb, fmt in rows:
        delta = vb - va
        pct = f" ({delta / va * 100:+.0f}%)" if va else ""
        print(f"{name:<14}{fmt.format(va):>20}{fmt.format(vb):>20}{fmt.format(delta) + pct:>14}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentcost", description="What did your agent run cost, and how much was waste?")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("claude", help="summarize Claude Code sessions")
    c.add_argument("--root", type=Path, default=claude_code.DEFAULT_ROOT)
    c.add_argument("--project", default=None)
    c.add_argument("--last", default=None, help="24h, 7d, 2w")
    c.add_argument("--json", action="store_true")
    c.add_argument("--top", type=int, default=5, help="show the N most expensive sessions")

    s = sub.add_parser("session", help="full cost tree for one session file")
    s.add_argument("path", type=Path)
    s.add_argument("--slo-ms", type=float, default=None)
    s.add_argument("--json", action="store_true")
    s.add_argument("--strict", action="store_true")

    d = sub.add_parser("compare", help="diff two run JSONs (from `session --json` or Tracer.to_json())")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)

    a = ap.parse_args(argv)

    if a.cmd == "compare":
        return _compare(a.a, a.b)

    if a.cmd == "session":
        run = claude_code.parse_session(a.path, strict=a.strict)
        run.slo_ms = a.slo_ms
        print(run.to_json() if a.json else run.render())
        return 0

    since = _since(a.last)
    paths = [p for p in claude_code.find_sessions(a.root, a.project) if since is None or p.stat().st_mtime >= since]
    if not paths:
        print(f"no sessions under {a.root}", file=sys.stderr)
        return 1
    runs = [claude_code.parse_session(p) for p in paths]
    agg = claude_code.aggregate(runs)
    if a.json:
        print(json.dumps({"aggregate": agg, "sessions": [r.totals() for r in runs]}, indent=2))
        return 0

    empty = f" ({agg['sessions_empty']} empty skipped)" if agg.get("sessions_empty") else ""
    print(f"{agg['sessions']} sessions{empty}, {agg['calls_total']:,} calls, {agg['tokens_total']:,} tokens")
    print(f"cost  total ${agg['cost_usd_total']:.2f}   p50 ${agg['cost_usd_p50']:.2f}   p90 ${agg['cost_usd_p90']:.2f}   max ${agg['cost_usd_max']:.2f}")
    print(f"waste p50 {agg['waste_share_p50']*100:.1f}%   p90 {agg['waste_share_p90']*100:.1f}%   (retried tool calls / all tokens)")
    if agg["unpriced_models"]:
        print(f"unpriced models: {', '.join(agg['unpriced_models'])}; costs above exclude them")
    print()
    top = sorted((r for r in runs if r.totals()["calls"] > 0), key=lambda r: r.totals()["cost_usd"], reverse=True)[: a.top]
    print(f"top {len(top)} sessions by cost:")
    for r in top:
        t = r.totals()
        print(f"  ${t['cost_usd']:>8.3f}  {t['tokens_total']:>10,} tok  waste {t['waste_share']*100:5.1f}%  {t['run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
