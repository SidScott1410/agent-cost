"""
The cost model.

A run is a tree of steps. Each step is one model call (or one tool call)
with its token counts, latency, and outcome. The report classifies every
token as one of:

  served    the call succeeded, its output was kept, and it landed inside
            the run's latency bound (if one was declared)
  retried   the call failed or was rolled back and the same step ran again;
            every token spent before the final attempt is retried
  discarded the call succeeded but its output was thrown away (a branch not
            taken, a draft replaced, a tool result never read)
  late      the call succeeded but landed outside the latency bound

Waste share = (retried + discarded + late) / total. This is the number the
report leads with, because it is the one nobody measures.

The classification is conservative: a token is served unless the trace says
otherwise. Integrations mark retries and discards; a bare trace reports 0%
waste and says so.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Literal

from .prices import lookup, normalize

Outcome = Literal["served", "retried", "discarded", "late", "error"]


@dataclass
class Step:
    name: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0   # Anthropic cache_creation; billed above input rate
    latency_ms: float = 0.0
    outcome: Outcome = "served"
    tool: str | None = None
    attempt: int = 1
    children: list[Step] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- derived -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_input_tokens + self.cache_write_input_tokens

    def cost_usd(self) -> float | None:
        if not self.model:
            return 0.0
        p = lookup(self.model)
        if p is None:
            return None
        cached_rate = p.cached_input_per_m if p.cached_input_per_m is not None else p.input_per_m
        write_rate = p.cache_write_per_m if p.cache_write_per_m is not None else 1.25 * p.input_per_m
        return (
            self.input_tokens * p.input_per_m
            + self.cached_input_tokens * cached_rate
            + self.cache_write_input_tokens * write_rate
            + self.output_tokens * p.output_per_m
        ) / 1e6

    def walk(self) -> Iterator[Step]:
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class Run:
    name: str = "run"
    slo_ms: float | None = None          # per-step latency bound, if any
    # Same-name sibling steps are attempts of one step (all but the last are
    # retried). True for the Tracer, where names are chosen by the caller.
    # False for log parsers that name steps after a tool ("Bash" x 40 is
    # forty steps, not forty attempts).
    retry_by_name: bool = True
    root: Step = field(default_factory=lambda: Step("root"))
    started: float = field(default_factory=time.time)

    def steps(self) -> list[Step]:
        return [s for s in self.root.walk() if s is not self.root]

    # ---- classification ------------------------------------------------
    def classify(self) -> None:
        """Apply the SLO and retry rules in place."""
        for s in self.steps():
            if s.outcome == "error":
                continue
            if self.slo_ms is not None and s.latency_ms > self.slo_ms and s.outcome == "served":
                s.outcome = "late"
        if not self.retry_by_name:
            return
        # Retry rule: siblings with the same name, all but the last attempt
        # are retried, whatever their own outcome said.
        for parent in self.root.walk():
            by_name: dict[str, list[Step]] = {}
            for c in parent.children:
                by_name.setdefault(c.name, []).append(c)
            for group in by_name.values():
                if len(group) > 1:
                    for s in group[:-1]:
                        s.outcome = "retried"

    # ---- totals --------------------------------------------------------
    def totals(self) -> dict:
        self.classify()
        by: dict[str, dict] = {}
        unpriced: set[str] = set()
        cost_total = 0.0
        for s in self.steps():
            b = by.setdefault(s.outcome, {"tokens": 0, "cost": 0.0, "calls": 0})
            b["tokens"] += s.total_tokens
            b["calls"] += 1
            c = s.cost_usd()
            if c is None:
                unpriced.add(normalize(s.model or ""))
            else:
                b["cost"] += c
                cost_total += c
        tokens_total = sum(b["tokens"] for b in by.values())
        waste_keys = ("retried", "discarded", "late", "error")
        wasted = sum(by.get(k, {"tokens": 0})["tokens"] for k in waste_keys)
        wasted_cost = sum(by.get(k, {"cost": 0.0})["cost"] for k in waste_keys)
        return {
            "run": self.name,
            "slo_ms": self.slo_ms,
            "calls": len(self.steps()),
            "tokens_total": tokens_total,
            "cost_usd": round(cost_total, 6),
            "waste_share": round(wasted / tokens_total, 4) if tokens_total else 0.0,
            # cost-weighted: cached tokens are cheap, so this is the number the bill follows
            "waste_share_cost": round(wasted_cost / cost_total, 4) if cost_total else None,
            "by_outcome": by,
            "unpriced_models": sorted(m for m in unpriced if m),
            "models": sorted({normalize(s.model) for s in self.steps() if s.model}),
        }

    # ---- rendering -----------------------------------------------------
    def render(self) -> str:
        t = self.totals()
        lines = []
        cost = f"${t['cost_usd']:.4f}" if not t["unpriced_models"] else f"${t['cost_usd']:.4f} (+ unpriced)"
        lines.append(f"{t['run']}: {t['calls']} calls, {t['tokens_total']:,} tokens, {cost}, waste {t['waste_share']*100:.1f}% of tokens"
                     + (f", {t['waste_share_cost']*100:.1f}% of cost" if t['waste_share_cost'] is not None else ""))
        if t["slo_ms"] is not None:
            lines.append(f"  SLO {t['slo_ms']:.0f} ms per call")
        for k in ("served", "retried", "discarded", "late", "error"):
            b = t["by_outcome"].get(k)
            if b:
                lines.append(f"  {k:<9} {b['calls']:>5} calls {b['tokens']:>12,} tok  ${b['cost']:.4f}")
        if t["unpriced_models"]:
            lines.append(f"  unpriced: {', '.join(t['unpriced_models'])} (add to agentcost/prices.py or install tokenprice)")
        lines.append("")
        lines.extend(self._tree(self.root, ""))
        return "\n".join(lines)

    def _tree(self, node: Step, indent: str) -> list[str]:
        out = []
        for i, c in enumerate(node.children):
            last = i == len(node.children) - 1
            branch = "└─ " if last else "├─ "
            cost = c.cost_usd()
            cost_s = f"${cost:.4f}" if cost is not None else "unpriced"
            tag = "" if c.outcome == "served" else f" [{c.outcome}]"
            model = f" {normalize(c.model)}" if c.model else (f" tool:{c.tool}" if c.tool else "")
            out.append(f"{indent}{branch}{c.name}{model}  {c.total_tokens:,} tok  {c.latency_ms:.0f} ms  {cost_s}{tag}")
            out.extend(self._tree(c, indent + ("   " if last else "│  ")))
        return out

    def to_json(self) -> str:
        return json.dumps({"totals": self.totals(), "root": asdict(self.root)}, indent=2)
