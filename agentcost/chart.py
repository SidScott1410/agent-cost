"""
One SVG: per session, tokens by outcome next to dollars by outcome.

The two bars rarely match. Retries happen late in a run, when most of the
context is cache reads at a tenth of the input rate, so a retry's share of
tokens overstates its share of the bill. That gap is the point of the chart.

No dependencies; fixed colors on a white card so it reads on any GitHub theme.
"""
from __future__ import annotations

from agentcost.core import Run

OUTCOMES = ["served", "retried", "discarded", "late", "error"]
COLOR = {"served": "#2f6fdd", "retried": "#d6453b", "discarded": "#e0a106", "late": "#7c3aed", "error": "#6b7280"}
W, PAD_L, PAD_R, ROW, GAP = 920, 150, 120, 22, 10


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def waste_chart(runs: list[Run], max_sessions: int = 12) -> str:
    tots = [(r, r.totals()) for r in runs]
    tots = [(r, t) for r, t in tots if t["calls"] > 0]
    tots.sort(key=lambda rt: -rt[1]["cost_usd"])
    tots = tots[:max_sessions]
    H = 70 + len(tots) * (2 * ROW + GAP) + 40
    sub = "Top bar = share of tokens. Bottom bar = share of dollars. Late retries are mostly cache reads, so the bars differ."
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"'
    o = [
        head + ' font-family="system-ui,sans-serif" font-size="12" fill="#222">',
        f'<rect width="{W}" height="{H}" fill="#fff" rx="6"/>',
        f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600">Where the spend went: tokens vs dollars, by outcome</text>',
        f'<text x="{PAD_L}" y="42" opacity=".65">{sub}</text>',
    ]
    x = PAD_L
    for k in OUTCOMES:
        o.append(f'<rect x="{x}" y="50" width="10" height="10" fill="{COLOR[k]}"/>')
        o.append(f'<text x="{x + 14}" y="59">{k}</text>')
        x += 24 + 7 * len(k)
    if not tots:
        o.append(f'<text x="{PAD_L}" y="100">no sessions with API calls</text></svg>')
        return "\n".join(o)
    bw = W - PAD_L - PAD_R
    y = 80
    for r, t in tots:
        by = t["by_outcome"]
        tok_total = sum(b["tokens"] for b in by.values()) or 1
        cost_total = sum(b["cost"] for b in by.values()) or 1
        name = (t["run"] or "run")[:18]
        o.append(f'<text x="{PAD_L - 10}" y="{y + ROW - 2}" text-anchor="end" font-weight="600">{_esc(name)}</text>')
        o.append(f'<text x="{PAD_L - 10}" y="{y + 2 * ROW - 4}" text-anchor="end" opacity=".6">${t["cost_usd"]:.3f}</text>')
        for row, (key, total) in enumerate((("tokens", tok_total), ("cost", cost_total))):
            x = PAD_L
            yy = y + row * ROW
            for k in OUTCOMES:
                v = by.get(k, {}).get(key, 0)
                if not v:
                    continue
                w = v / total * bw
                o.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="{ROW - 4}" fill="{COLOR[k]}" fill-opacity="{0.9 if row else 0.55}"/>')
                x += w
        ws, wc = t["waste_share"] * 100, (t.get("waste_share_cost") or 0) * 100
        o.append(f'<text x="{PAD_L + bw + 8}" y="{y + ROW - 6}" opacity=".7">waste {ws:.0f}% tok</text>')
        o.append(f'<text x="{PAD_L + bw + 8}" y="{y + 2 * ROW - 8}" opacity=".7">waste {wc:.0f}% $</text>')
        y += 2 * ROW + GAP
    o.append("</svg>")
    return "\n".join(o)
