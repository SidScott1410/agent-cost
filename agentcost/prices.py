"""
Price table: USD per million tokens, by model.

Provenance matters here. Every entry carries the date it was read and where.
The table is a fallback; if the `tokenprice` package is installed, its live
feed is used first (see `lookup`). Unknown models cost zero and are reported
as unpriced, never silently estimated.

Prices are list prices for the base tier. Cached-input and batch discounts
are applied only where the provider exposes them in usage data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_m: float        # USD per 1M input tokens
    output_per_m: float       # USD per 1M output tokens
    cached_input_per_m: float | None = None
    source: str = ""
    as_of: str = ""
    cache_write_per_m: float | None = None   # None -> 1.25 x input (Anthropic 5-minute cache convention)


# Keys are normalized model ids (lowercase, no provider prefix, no date suffix
# unless the vendor prices dated snapshots differently). `normalize` maps
# vendor ids onto these.
#
# Fill this table from the vendor price pages and record the date. The
# entries below are deliberately few; an unpriced model is reported as such.
TABLE: dict[str, Price] = {
    # Anthropic, https://www.anthropic.com/pricing (read the page, record the date)
    # "claude-sonnet-4-5": Price(3.00, 15.00, 0.30, "anthropic.com/pricing", "2026-11-10"),
    # OpenAI, https://openai.com/api/pricing
    # "gpt-5": Price(1.25, 10.00, 0.125, "openai.com/api/pricing", "2026-11-10"),
}


_DATE_SUFFIX = re.compile(r"[-_]?\d{8}$|[-_]?\d{4}-\d{2}-\d{2}$")


def normalize(model: str) -> str:
    """'anthropic/claude-sonnet-4-5-20250929' -> 'claude-sonnet-4-5'."""
    m = model.strip().lower()
    if "/" in m:
        m = m.rsplit("/", 1)[1]
    m = _DATE_SUFFIX.sub("", m)
    return m


def lookup(model: str) -> Price | None:
    """Live feed first (tokenprice), then the static table, else None."""
    key = normalize(model)
    try:  # optional dependency
        import tokenprice  # type: ignore
    except ImportError:
        tokenprice = None
    if tokenprice is not None:
        p = tokenprice.get(key)
        if p is not None:
            return Price(p.input_per_m, p.output_per_m, p.cached_input_per_m, p.source, p.as_of)
    return TABLE.get(key)
