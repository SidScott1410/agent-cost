import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentcost import TABLE, Price, Tracer, claude_code
from agentcost.cli import main


@pytest.fixture(autouse=True)
def priced(monkeypatch):
    monkeypatch.setitem(TABLE, "test-model", Price(1.0, 2.0, 0.1, "test", "2026-01-01"))
    monkeypatch.setitem(TABLE, "claude-test", Price(3.0, 15.0, 0.3, "test", "2026-01-01"))


def fake_resp(model="test-model", inp=1000, out=500, cached=0):
    return SimpleNamespace(model=model, usage=SimpleNamespace(prompt_tokens=inp + cached, completion_tokens=out,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached)))


def test_tracer_costs_and_waste():
    t = Tracer("demo", slo_ms=1000)
    with t.step("plan") as s:
        s.record(fake_resp())
    with t.step("write") as s:          # first attempt
        s.record(fake_resp(out=100))
    with t.step("write") as s:          # retry, same name -> first is retried
        s.record(fake_resp(out=100))
    with t.step("draft", discard=True) as s:
        s.record(fake_resp(inp=200, out=50))
    tot = t.totals()
    assert tot["calls"] == 4
    assert tot["by_outcome"]["retried"]["calls"] == 1
    assert tot["by_outcome"]["discarded"]["calls"] == 1
    # cost: plan 1000*1 + 500*2 = 2000 -> $0.002 ; writes 2 * (1000+200)=2400 -> 0.0024 ; draft 200+100 -> 0.0003
    assert abs(tot["cost_usd"] - 0.0047) < 1e-9
    wasted = 1100 + 250
    assert tot["waste_share"] == round(wasted / (1500 + 1100 + 1100 + 250), 4)
    text = t.report()
    assert "waste" in text and "[retried]" in text and "[discarded]" in text


def test_cached_tokens_priced_at_cached_rate():
    t = Tracer()
    with t.step("x") as s:
        s.record(fake_resp(inp=0, out=0, cached=1_000_000))
    assert abs(t.totals()["cost_usd"] - 0.1) < 1e-9


def test_unpriced_model_reported_not_estimated():
    t = Tracer()
    with t.step("x") as s:
        s.record(fake_resp(model="mystery-9000"))
    tot = t.totals()
    assert tot["unpriced_models"] == ["mystery-9000"]
    assert tot["cost_usd"] == 0.0
    assert "unpriced" in t.report()


def test_wrap_openai_records_under_current_step():
    class Completions:
        def create(self, **kw):
            return fake_resp(out=10)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()), responses=None)
    t = Tracer()
    t.wrap_openai(client)
    with t.step("ask"):
        client.chat.completions.create(model="test-model", messages=[])
        client.chat.completions.create(model="test-model", messages=[])
    tot = t.totals()
    assert tot["calls"] == 1 and tot["tokens_total"] == 2 * 1010


def test_error_marks_step():
    t = Tracer()
    with pytest.raises(RuntimeError), t.step("boom") as s:
        s.tokens(100, 0, model="test-model")
        raise RuntimeError("x")
    assert t.totals()["by_outcome"]["error"]["calls"] == 1


def write_session(path: Path):
    lines = [
        {"type": "user", "timestamp": "2026-11-10T10:00:00Z", "message": {"role": "user", "content": "fix the bug"}},
        {"type": "assistant", "timestamp": "2026-11-10T10:00:03Z", "uuid": "a1", "message": {
            "model": "claude-test", "content": [{"type": "tool_use", "name": "Bash", "input": {}}],
            "usage": {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 5000, "cache_creation_input_tokens": 0}}},
        {"type": "user", "timestamp": "2026-11-10T10:00:04Z", "message": {"role": "user", "content": [
            {"type": "tool_result", "is_error": True, "content": "command failed"}]}},
        {"type": "assistant", "timestamp": "2026-11-10T10:00:07Z", "uuid": "a2", "message": {
            "model": "claude-test", "content": [{"type": "tool_use", "name": "Edit", "input": {}}],
            "usage": {"input_tokens": 1200, "output_tokens": 300, "cache_read_input_tokens": 5000}}},
        {"type": "user", "timestamp": "2026-11-10T10:00:08Z", "message": {"role": "user", "content": [
            {"type": "tool_result", "is_error": False, "content": "ok"}]}},
        {"type": "assistant", "timestamp": "2026-11-10T10:00:10Z", "uuid": "a3", "message": {
            "model": "claude-test", "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 100, "output_tokens": 50}}},
        {"type": "summary", "summary": "fixed"},
        "not json",
    ]
    with path.open("w") as f:
        for l in lines:
            f.write((json.dumps(l) if not isinstance(l, str) else l) + "\n")


def test_claude_code_parser(tmp_path):
    p = tmp_path / "proj" / "sess.jsonl"
    p.parent.mkdir()
    write_session(p)
    run = claude_code.parse_session(p)
    steps = run.steps()
    assert [s.name for s in steps] == ["Bash", "Edit", "respond"]
    tot = run.totals()
    assert steps[0].outcome == "retried"       # its tool_result was an error
    assert steps[1].outcome == "served"
    assert steps[0].latency_ms == 3000.0
    assert tot["by_outcome"]["retried"]["tokens"] == 6200
    assert run.root.meta["skipped_lines"] == 2  # summary line + bad json
    with pytest.raises(ValueError):
        claude_code.parse_session(p, strict=True)

    agg = claude_code.aggregate([run, run])
    assert agg["sessions"] == 2 and agg["cost_usd_total"] > 0

    assert main(["claude", "--root", str(tmp_path)]) == 0
    assert main(["session", str(p)]) == 0
    assert main(["claude", "--root", str(tmp_path / "empty")]) == 1


def test_claude_code_same_tool_is_not_a_retry(tmp_path):
    import json

    from agentcost.claude_code import parse_session
    lines = []
    for i in range(5):
        lines.append({"type": "user", "timestamp": f"2026-11-10T00:00:{i*2:02d}Z",
                      "message": {"content": [{"type": "tool_result", "is_error": False}]}})
        lines.append({"type": "assistant", "timestamp": f"2026-11-10T00:00:{i*2+1:02d}Z",
                      "message": {"id": f"m{i}", "model": "claude-sonnet-4-5",
                                  "usage": {"input_tokens": 100, "output_tokens": 50},
                                  "content": [{"type": "tool_use", "name": "Bash"}]}})
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    t = parse_session(p).totals()
    assert t["waste_share"] == 0.0 and t["calls"] == 5


def test_claude_code_split_records_counted_once(tmp_path):
    import json

    from agentcost.claude_code import parse_session
    lines = [
        {"type": "user", "timestamp": "2026-11-10T00:00:00Z", "message": {"content": "hi"}},
        {"type": "assistant", "timestamp": "2026-11-10T00:00:01Z",
         "message": {"id": "A", "model": "m", "usage": {"input_tokens": 1000, "output_tokens": 10,
                     "cache_creation_input_tokens": 500, "cache_read_input_tokens": 200},
                     "content": [{"type": "text", "text": "ok"}]}},
        {"type": "assistant", "timestamp": "2026-11-10T00:00:01Z",
         "message": {"id": "A", "model": "m", "usage": {"input_tokens": 1000, "output_tokens": 10},
                     "content": [{"type": "tool_use", "name": "Read"}]}},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    run = parse_session(p)
    steps = run.steps()
    assert len(steps) == 1 and steps[0].name == "Read"
    assert run.totals()["tokens_total"] == 1710
    assert steps[0].cache_write_input_tokens == 500


def test_cache_write_priced_above_input():
    from agentcost.core import Step
    from agentcost.prices import TABLE, Price
    TABLE["unit-test-model"] = Price(1.0, 2.0, 0.1)
    try:
        s = Step("x", model="unit-test-model", input_tokens=1_000_000, cache_write_input_tokens=1_000_000,
                 cached_input_tokens=1_000_000)
        assert abs(s.cost_usd() - (1.0 + 1.25 + 0.1)) < 1e-9
    finally:
        del TABLE["unit-test-model"]


def test_compare_cli(tmp_path, capsys):
    from agentcost.cli import main
    from agentcost.tracer import Tracer
    for name, n in (("a", 2), ("b", 3)):
        t = Tracer(name)
        for _ in range(n):
            with t.step("plan") as s:
                s.tokens(input_tokens=100, output_tokens=10, model="m")
        (tmp_path / f"{name}.json").write_text(t.to_json())
    assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    out = capsys.readouterr().out
    assert "delta" in out and "+50%" in out


def test_lookup_uses_tokenprice_record_shape(monkeypatch):
    """tokenprice.Record has read_at and provider, not as_of; lookup must map them (found on a real machine)."""
    import sys
    import types

    from agentcost import prices

    class Rec:
        input_per_m = 0.05
        output_per_m = 0.08
        cached_input_per_m = None
        source = "https://openrouter.ai/api/v1/models/x/endpoints"
        read_at = "2026-11-13T00:00:00Z"
        provider = "deepinfra"

    fake = types.ModuleType("tokenprice")
    fake.get = lambda key: Rec() if key == "llama-3-1-8b-instruct" else None
    monkeypatch.setitem(sys.modules, "tokenprice", fake)
    p = prices.lookup("meta-llama/llama-3.1-8b-instruct")
    assert p is not None and p.input_per_m == 0.05 and p.as_of == "2026-11-13T00:00:00Z"
    assert p.source.startswith("tokenprice:deepinfra:")
    assert prices.lookup("nope") is None

    fake.get = lambda key: (_ for _ in ()).throw(RuntimeError("feed down"))
    assert prices.lookup("llama-3-1-8b-instruct") is None  # feed failure degrades to unpriced, not a crash
