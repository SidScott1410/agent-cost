"""
Tracer: the in-process API, and the OpenAI / Anthropic client wrappers.

    from agentcost import Tracer

    t = Tracer("plan-and-write", slo_ms=4000)
    with t.step("plan") as s:
        r = client.chat.completions.create(...)
        s.record(r)                       # reads usage + model off the response
    with t.step("write", discard=True):   # branch not taken
        ...
    print(t.report())

Or patch a client once and every call is recorded under the current step:

    t.wrap_openai(client)   # also: t.wrap_anthropic(client)

A step that raises is marked "error". Running the same step name twice under
the same parent marks the earlier attempts "retried" at report time.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any

from .core import Run, Step


def _usage_from_response(r: Any) -> tuple[str | None, int, int, int, int]:
    """Best-effort extraction across OpenAI and Anthropic response shapes."""
    model = getattr(r, "model", None)
    u = getattr(r, "usage", None)
    if u is None and isinstance(r, dict):
        model = r.get("model")
        u = r.get("usage")
    if u is None:
        return model, 0, 0, 0, 0
    g = (lambda k: (getattr(u, k, None) if not isinstance(u, dict) else u.get(k)) or 0)
    # OpenAI chat: prompt_tokens / completion_tokens (+ prompt_tokens_details.cached_tokens)
    # OpenAI responses: input_tokens / output_tokens (+ input_tokens_details.cached_tokens)
    # Anthropic: input_tokens / output_tokens / cache_read_input_tokens
    inp = g("prompt_tokens") or g("input_tokens")
    out = g("completion_tokens") or g("output_tokens")
    cached = g("cache_read_input_tokens")
    cache_write = g("cache_creation_input_tokens")
    for det in ("prompt_tokens_details", "input_tokens_details"):
        d = getattr(u, det, None) if not isinstance(u, dict) else u.get(det)
        if d is not None:
            cached = cached or ((getattr(d, "cached_tokens", None) if not isinstance(d, dict) else d.get("cached_tokens")) or 0)
    # Anthropic reports cache reads separately from input; OpenAI includes cached inside prompt_tokens.
    if g("cache_read_input_tokens"):
        pass
    elif cached:
        inp = max(inp - cached, 0)
    return model, int(inp), int(out), int(cached), int(cache_write)


class _StepCtx:
    def __init__(self, tracer: Tracer, step: Step):
        self.tracer, self.step = tracer, step
        self._t0 = 0.0

    def record(self, response: Any, tool: str | None = None) -> None:
        model, i, o, c, w = _usage_from_response(response)
        self.step.model = self.step.model or model
        self.step.input_tokens += i
        self.step.output_tokens += o
        self.step.cached_input_tokens += c
        self.step.cache_write_input_tokens += w
        if tool:
            self.step.tool = tool

    def tokens(self, input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0, model: str | None = None) -> None:
        self.step.input_tokens += input_tokens
        self.step.output_tokens += output_tokens
        self.step.cached_input_tokens += cached_input_tokens
        if model:
            self.step.model = model

    def discard(self) -> None:
        self.step.outcome = "discarded"


class Tracer:
    def __init__(self, name: str = "run", slo_ms: float | None = None):
        self.run = Run(name=name, slo_ms=slo_ms)
        self._stack: list[Step] = [self.run.root]

    @contextlib.contextmanager
    def step(self, name: str, model: str | None = None, tool: str | None = None, discard: bool = False):
        s = Step(name=name, model=model, tool=tool)
        self._stack[-1].children.append(s)
        self._stack.append(s)
        ctx = _StepCtx(self, s)
        t0 = time.perf_counter()
        try:
            yield ctx
            if discard:
                s.outcome = "discarded"
        except Exception:
            s.outcome = "error"
            raise
        finally:
            s.latency_ms = (time.perf_counter() - t0) * 1e3
            self._stack.pop()

    # ---- client wrappers ----------------------------------------------
    def _current(self) -> Step:
        cur = self._stack[-1]
        if cur is self.run.root:  # calls outside any step get their own
            s = Step(name="call")
            cur.children.append(s)
            return s
        return cur

    def wrap_openai(self, client: Any) -> Any:
        """Patch chat.completions.create and responses.create on an OpenAI client."""
        tracer = self

        def patch(obj: Any, attr: str) -> None:
            orig = getattr(obj, attr, None)
            if orig is None:
                return

            def wrapped(*a: Any, **kw: Any) -> Any:
                step = tracer._current()
                t0 = time.perf_counter()
                try:
                    r = orig(*a, **kw)
                except Exception:
                    step.outcome = "error"
                    step.latency_ms += (time.perf_counter() - t0) * 1e3
                    raise
                step.latency_ms += (time.perf_counter() - t0) * 1e3
                _StepCtx(tracer, step).record(r)
                return r

            setattr(obj, attr, wrapped)

        chat = getattr(getattr(client, "chat", None), "completions", None)
        if chat is not None:
            patch(chat, "create")
        resp = getattr(client, "responses", None)
        if resp is not None:
            patch(resp, "create")
        return client

    def wrap_anthropic(self, client: Any) -> Any:
        tracer = self
        msgs = getattr(client, "messages", None)
        if msgs is None:
            return client
        orig = msgs.create

        def wrapped(*a: Any, **kw: Any) -> Any:
            step = tracer._current()
            t0 = time.perf_counter()
            try:
                r = orig(*a, **kw)
            except Exception:
                step.outcome = "error"
                step.latency_ms += (time.perf_counter() - t0) * 1e3
                raise
            step.latency_ms += (time.perf_counter() - t0) * 1e3
            _StepCtx(tracer, step).record(r)
            return r

        msgs.create = wrapped
        return client

    # ---- output --------------------------------------------------------
    def report(self) -> str:
        return self.run.render()

    def totals(self) -> dict:
        return self.run.totals()

    def to_json(self) -> str:
        return self.run.to_json()
