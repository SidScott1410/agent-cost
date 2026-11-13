"""
Claude Code session logs -> Run.

Claude Code writes one JSONL per session under ~/.claude/projects/<project>/.
Each line is a message envelope; assistant messages carry `message.usage`
with input/output/cache token counts and `message.model`. Tool calls are
`tool_use` content blocks on assistant turns; their results come back as
`tool_result` blocks on the next user turn, with `is_error` when they failed.

Mapping to the cost model:
  step      one assistant turn; named after its first tool_use, else "respond"
  retried   an assistant turn whose tool call came back is_error (the agent
            will spend more tokens to recover, so this turn's tokens are
            waste by the served-token rule)
  latency   wall-clock from the preceding user turn to this assistant turn

One API response can be written as several assistant records (text block,
then tool_use block) that share `message.id` and repeat the same usage.
Records are merged by id so tokens are counted once.

The format is undocumented and changes. The parser is tolerant: unknown
lines are skipped and counted, and `--strict` turns that into an error.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .core import Run, Step

DEFAULT_ROOT = Path.home() / ".claude" / "projects"


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _blocks(msg: dict) -> list[dict]:
    c = msg.get("content")
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    return []


def parse_session(path: Path, strict: bool = False) -> Run:
    run = Run(name=path.stem, retry_by_name=False)
    skipped = 0
    last_user_ts: float | None = None
    pending: list[Step] = []  # assistant steps awaiting their tool_result
    by_msg_id: dict[str, Step] = {}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            typ = rec.get("type")
            msg = rec.get("message") or {}
            ts = _ts(rec.get("timestamp"))

            if typ == "user":
                # tool results decide the fate of the previous assistant step
                for b in _blocks(msg):
                    if b.get("type") == "tool_result" and b.get("is_error") and pending:
                        pending[-1].outcome = "retried"
                        pending[-1].meta["error"] = True
                pending = []
                last_user_ts = ts
            elif typ == "assistant":
                usage = msg.get("usage") or {}
                tools = [b.get("name") for b in _blocks(msg) if b.get("type") == "tool_use"]
                mid = msg.get("id")
                if mid and mid in by_msg_id:
                    # continuation of a response already counted: add its tools only
                    prev = by_msg_id[mid]
                    prev.meta["tools"].extend(tools)
                    if tools and not prev.tool:
                        prev.tool = tools[0]
                        prev.name = tools[0]
                    continue
                s = Step(
                    name=tools[0] if tools else "respond",
                    model=msg.get("model"),
                    input_tokens=int(usage.get("input_tokens") or 0),
                    cache_write_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                    cached_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
                    tool=tools[0] if tools else None,
                    latency_ms=((ts - last_user_ts) * 1e3) if (ts and last_user_ts) else 0.0,
                    meta={"tools": tools, "uuid": rec.get("uuid")},
                )
                if s.total_tokens == 0 and not tools:
                    skipped += 1
                    continue
                run.root.children.append(s)
                pending.append(s)
                if mid:
                    by_msg_id[mid] = s
            else:
                skipped += 1
    run.root.meta["skipped_lines"] = skipped
    if strict and skipped:
        raise ValueError(f"{path}: {skipped} lines not understood")
    return run


def find_sessions(root: Path = DEFAULT_ROOT, project: str | None = None) -> list[Path]:
    if not root.exists():
        return []
    paths: Iterable[Path] = root.glob("*/*.jsonl") if project is None else (root / project).glob("*.jsonl")
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def aggregate(runs: list[Run]) -> dict:
    """Totals across sessions plus the distribution a post needs."""
    per = [r.totals() for r in runs]
    costs = sorted(t["cost_usd"] for t in per)
    waste = sorted(t["waste_share"] for t in per)

    def pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        i = min(int(q * (len(xs) - 1)), len(xs) - 1)
        return xs[i]

    return {
        "sessions": len(per),
        "cost_usd_total": round(sum(costs), 4),
        "cost_usd_p50": pct(costs, 0.5),
        "cost_usd_p90": pct(costs, 0.9),
        "cost_usd_max": costs[-1] if costs else 0.0,
        "waste_share_p50": pct(waste, 0.5),
        "waste_share_p90": pct(waste, 0.9),
        "tokens_total": sum(t["tokens_total"] for t in per),
        "calls_total": sum(t["calls"] for t in per),
        "unpriced_models": sorted({m for t in per for m in t["unpriced_models"]}),
    }
