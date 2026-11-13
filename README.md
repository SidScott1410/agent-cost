# agentcost

**What did your agent run cost, and how much of it was waste.**

```
$ agentcost claude --last 7d
41 sessions, 1,872 calls, 6,214,330 tokens
cost  total $31.18   p50 $0.41   p90 $2.07   max $6.90
waste p50 18.4%   p90 41.0%   (retried tool calls / all tokens)
```

No server. No account. No telemetry. Reads the logs that are already on your
machine, or wraps the client you already use.

```
pip install agentcost
```

## Three ways in

**1. Claude Code sessions you already ran.** Claude Code writes one JSONL per
session under `~/.claude/projects/`. This reads them.

```
agentcost claude                # everything on this machine
agentcost claude --last 24h     # recent
agentcost session ~/.claude/projects/<project>/<session>.jsonl   # one run, full tree
```

**2. Wrap a client.** Every call is recorded under the current step.

```python
from agentcost import Tracer
from openai import OpenAI

t = Tracer("plan-and-write", slo_ms=4000)
client = t.wrap_openai(OpenAI())        # or t.wrap_anthropic(Anthropic())

with t.step("plan"):
    client.chat.completions.create(model="gpt-5", messages=[...])
with t.step("write"):
    client.chat.completions.create(model="gpt-5", messages=[...])

print(t.report())
```

**3. Record by hand.** For frameworks without a hook.

```python
with t.step("search", tool="web") as s:
    r = my_call()
    s.record(r)                      # reads .model and .usage off the response
    # or: s.tokens(input_tokens=1200, output_tokens=300, model="gpt-5")
```

## Compare two runs

```
agentcost session before.jsonl --json > a.json
agentcost session after.jsonl  --json > b.json
agentcost compare a.json b.json
```

Prints calls, tokens, cost and waste side by side with the delta. This is the screenshot for a "we cut retries by X%" post.

## What "waste" means

Every token is classified as one of:

| outcome   | meaning |
| --- | --- |
| served    | the call succeeded, its output was kept, and it landed inside the latency bound |
| retried   | the same step ran again; everything before the last attempt |
| discarded | the output was thrown away (a branch not taken, a draft replaced) |
| late      | succeeded, but outside the `slo_ms` you declared |
| error     | the call raised |

Waste share = (retried + discarded + late + error) / total. This is the
served-token rule from [Reckon Research](https://reckonresearch.com): a token
that did not land inside its deadline, or was not used, is not a token you
bought anything with.

The classification is conservative. A bare trace with no retries, no discards
and no SLO reports 0% waste and says so. For Claude Code logs, a tool call
whose result came back `is_error` marks that turn retried, because the agent
spends more tokens to recover.

## Prices

Costs come from `agentcost/prices.py`, a table with a source and a date on
every entry, or from the [`tokenprice`](https://github.com/reckonresearch/tokenprice)
feed if it is installed (`pip install agentcost[live]`). An unknown model is
reported as **unpriced**, never estimated. Pull requests adding a price need
the vendor page URL and the date you read it.

## Example tree

```
plan-and-write: 4 calls, 3,950 tokens, $0.0047, waste 34.2%
  SLO 1000 ms per call
  served        2 calls        2,600 tok  $0.0032
  retried       1 calls        1,100 tok  $0.0012
  discarded     1 calls          250 tok  $0.0003

├─ plan test-model  1,500 tok  812 ms  $0.0020
├─ write test-model  1,100 tok  640 ms  $0.0012 [retried]
├─ write test-model  1,100 tok  655 ms  $0.0012
└─ draft test-model  250 tok  210 ms  $0.0003 [discarded]
```

## Status

v0.1. The Claude Code log format is undocumented and changes; the parser
skips lines it does not understand and counts them (`--strict` makes that an
error). Wrappers cover the OpenAI chat and responses APIs and the Anthropic
messages API. LangGraph and OpenAI Agents SDK callbacks are next.

Apache-2.0. Built by [Sidney Scott](https://www.sidney.fyi) at Reckon Research.
