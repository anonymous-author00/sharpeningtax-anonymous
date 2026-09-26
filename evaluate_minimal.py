#!/usr/bin/env python3
"""Base vs. post-trained agent evaluation (Sec. 2-3).

The base model acts through a text harness over /v1/completions: tools shown as Python stubs,
calls written as a fenced JSON ``tool_call`` block, and a plain-text transcript. The
post-trained model uses its chat template and vLLM's tool-call parser. Benchmarks plug in via
``Environment``; a small tool-use task is included.

    python evaluate_minimal.py --base-model Qwen/Qwen2.5-7B --base-url http://127.0.0.1:8000/v1 \\
        --rl-model Qwen/Qwen2.5-7B-Instruct --rl-url http://127.0.0.1:8001/v1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import sharpening_tax_minimal as st

# Text harness for base models

MARKERS = {"user": "### User", "assistant": "### Assistant", "tool_output": "### Tool Output"}
CALL_FENCE = "tool_call"


def stop_sequences(fence_stop: str = "default") -> list[str]:
    """Base models have no end-of-turn token. "closing" is used for Qwen3.5 base on BFCL,
    which writes prose before the fence (the default stop would match the opening fence)."""
    fence = {"default": "\n```", "closing": "\n```\n"}[fence_stop]
    return [MARKERS["tool_output"], MARKERS["user"], fence]


# Tool catalog: OpenAI JSON schema -> Python stubs

_SCALAR = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "null": "None"}


def _lit(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)


def _py_type(schema: Any) -> str:
    """JSON-Schema fragment -> Python type hint."""
    if not isinstance(schema, dict):
        return "Any"
    if schema.get("enum"):
        return "Literal[" + ", ".join(_lit(v) for v in schema["enum"]) + "]"
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            subs = list(dict.fromkeys(_py_type(s) for s in schema[key]))
            if not subs:
                return "Any"
            return subs[0] if len(subs) == 1 else "Union[" + ", ".join(subs) + "]"
    t = schema.get("type")
    if isinstance(t, list):
        subs = list(dict.fromkeys(_SCALAR.get(x, "Any") for x in t if x != "null")) or ["Any"]
        base = subs[0] if len(subs) == 1 else "Union[" + ", ".join(subs) + "]"
        return f"Optional[{base}]" if "null" in t else base
    if t == "array":
        items = schema.get("items")
        return f"list[{_py_type(items)}]" if isinstance(items, dict) else "list"
    if t == "object":
        return "dict"
    return _SCALAR.get(t, "Any")


def _unwrap(tool: dict) -> dict:
    return tool["function"] if isinstance(tool.get("function"), dict) else tool


def render_tool_stub(tool: dict) -> str:
    """One tool as a Python ``def`` stub with a Google-style docstring."""
    fn = _unwrap(tool)
    name = fn.get("name", "tool")
    desc = (fn.get("description") or "").strip()
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    ordered = [k for k in props if k in required] + [k for k in props if k not in required]

    sig = ", ".join(
        f"{k}: {_py_type(props[k])}" if k in required
        else f"{k}: {_py_type(props[k])} = {_lit(props[k].get('default', None))}"
        for k in ordered)
    doc = [desc or f"Call the {name} tool."]
    if ordered:
        doc += ["", "Args:"]
        for k in ordered:
            p = props[k]
            line = f"    {k} ({_py_type(p)}, {'required' if k in required else 'optional'}):"
            if (p.get("description") or "").strip():
                line += " " + p["description"].strip()
            doc.append(line)
            if p.get("enum"):
                doc.append(f"        choices: {p['enum']}")

    ind = "    "
    out = [f"def {name}({sig}) -> dict:"]
    if len(doc) == 1:
        out.append(f'{ind}"""{doc[0]}"""')
    else:
        out.append(f'{ind}"""{doc[0]}')
        out += [ind + l if l else "" for l in doc[1:]]
        out.append(f'{ind}"""')
    out.append(f"{ind}...")
    return "\n".join(out)


CALL_CONTRACT = (
    "# How to call tools\n"
    f"To call one or more tools, emit a fenced ```{CALL_FENCE}``` block containing "
    "a JSON array of calls. Each call is an object "
    '{"name": <tool_name>, "arguments": {<param>: <value>, ...}} whose '
    "argument keys match the function parameters above. Format:\n"
    f"```{CALL_FENCE}\n"
    '[{"name": "<tool_name>", "arguments": {"<param>": "<value>"}}]\n'
    "```\n"
    f'After the block, stop — the result will be provided under "{MARKERS["tool_output"]}". '
    "You may place several calls in the array to call tools in parallel. "
    "When the task is complete, reply in plain text with no "
    f"```{CALL_FENCE}``` block."
)


# Transcript + prompt

def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join((p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p)
                       for p in content)
    return str(content)


def render_call_block(tool_calls: list[dict]) -> str:
    """OpenAI tool_calls -> the fenced JSON array the model is asked to emit."""
    arr = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        arr.append({"name": fn.get("name"), "arguments": args or {}})
    return f"```{CALL_FENCE}\n{json.dumps(arr, ensure_ascii=False)}\n```"


def serialize(messages: list[dict]) -> str:
    """Plain-text transcript. Assistant turns keep the raw completion plus the re-rendered call
    block, as in the reported runs."""
    id2name = {tc.get("id"): (tc.get("function") or {}).get("name")
               for m in messages for tc in (m.get("tool_calls") or []) if tc.get("id")}
    blocks = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            blocks.append(f"{MARKERS['user']}\n{_text(m.get('content'))}")
        elif role == "assistant":
            parts = [p for p in (_text(m.get("content")),) if p]
            if m.get("tool_calls"):
                parts.append(render_call_block(m["tool_calls"]))
            blocks.append(f"{MARKERS['assistant']}\n" + "\n".join(parts))
        elif role == "tool":
            name = id2name.get(m.get("tool_call_id")) or m.get("name") or "tool"
            blocks.append(f"{MARKERS['tool_output']}\n[{name}] -> {_text(m.get('content'))}")
        else:
            blocks.append(f"### {role}\n{_text(m.get('content'))}")
    return "\n\n".join(blocks)


def build_prompt(messages: list[dict], tools: list[dict] | None = None,
                 system_prompt: str | None = None) -> str:
    """Raw completion prompt for one generation step."""
    if system_prompt is None:
        system_prompt = next((_text(m.get("content")) for m in messages if m.get("role") == "system"), None)
    blocks = []
    if system_prompt and system_prompt.strip():
        blocks.append(system_prompt.strip())
    if tools:
        blocks.append("# Available tools\n```python\n"
                      + "\n\n".join(render_tool_stub(t) for t in tools) + "\n```")
        blocks.append(CALL_CONTRACT)
    transcript = serialize(messages)
    if transcript:
        blocks.append(transcript)
    return "\n\n".join(blocks) + f"\n\n{MARKERS['assistant']}\n"


# Completion -> OpenAI tool calls

_FENCE_RE = re.compile(r"```(?:tool_call|json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def _repair(s: str) -> str:
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _loads_lenient(s: str):
    if not s or not s.strip():
        return None
    s = s.strip()
    attempts = [s, _repair(s)]
    if '"' not in s and "'" in s:
        attempts.append(_repair(s).replace("'", '"'))
    for a in attempts:
        try:
            return json.loads(a)
        except ValueError:
            continue
    return None


def _first_json_blob(text: str) -> str | None:
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        depth, in_str, esc, quote = 0, False, False, ""
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    in_str = False
            elif c in "\"'":
                in_str, quote = True, c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
    return None


def _as_calls(obj) -> list[dict]:
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        if "name" in obj:
            return [obj]
        for key in ("tool_calls", "calls", "tools"):
            if isinstance(obj.get(key), list):
                return [o for o in obj[key] if isinstance(o, dict)]
    return []


def _coerce(v, prop):
    """Cast a string argument to its declared JSON-Schema scalar type."""
    if not isinstance(v, str) or not isinstance(prop, dict):
        return v
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    s = v.strip()
    try:
        if t == "integer":
            return int(s)
        if t == "number":
            return float(s)
    except ValueError:
        return v
    if t == "boolean" and s.lower() in ("true", "false"):
        return s.lower() == "true"
    return v


def make_tool_call(name: str, args) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}}


def parse_tool_calls(text: str, tools: list[dict] | None = None) -> list[dict]:
    """Completion -> OpenAI tool calls ([] means a final answer). Unknown tools are dropped."""
    if not text:
        return []
    props = {}
    for t in tools or []:
        fn = _unwrap(t) if isinstance(t, dict) else {}
        if fn.get("name"):
            props[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    known = set(props) if tools else None

    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    if not candidates:
        blob = _first_json_blob(text)
        candidates = [blob] if blob else []
    for block in candidates:
        obj = _loads_lenient(block)
        if obj is None:
            continue
        calls = []
        for call in _as_calls(obj):
            name = call.get("name")
            if not isinstance(name, str) or (known is not None and name not in known):
                continue
            args = call.get("arguments", call.get("parameters", {}))
            if isinstance(args, str):
                parsed = _loads_lenient(args)
                args = parsed if isinstance(parsed, dict) else {}
            if not isinstance(args, dict):
                args = {}
            schema = props.get(name, {})
            calls.append(make_tool_call(name, {k: _coerce(v, schema.get(k)) for k, v in args.items()}))
        if calls:
            return calls
    return []


# Policies

@dataclass
class Reply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)   # OpenAI shape; [] = final answer


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024


class HarnessPolicy:
    """Base model: plain-text harness prompt -> /v1/completions -> parsed tool calls."""

    def __init__(self, model: str, base_url: str, sampling: Sampling):
        from openai import OpenAI
        self.model, self.sampling = model, sampling
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def __call__(self, messages, tools, seed=None) -> Reply:
        resp = self.client.completions.create(
            model=self.model, prompt=build_prompt(messages, tools),
            stop=stop_sequences(), seed=seed, **vars(self.sampling))
        text = resp.choices[0].text or ""
        return Reply(text, parse_tool_calls(text, tools))


class ChatPolicy:
    """Post-trained model: native chat template + vLLM tool-call parser."""

    def __init__(self, model: str, base_url: str, sampling: Sampling, chat_template_kwargs=None):
        self.model, self.sampling = model, sampling
        from openai import OpenAI
        self.extra = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else None
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def __call__(self, messages, tools, seed=None) -> Reply:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, seed=seed,
            extra_body=self.extra, **vars(self.sampling))
        msg = resp.choices[0].message
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                 for tc in (msg.tool_calls or [])]
        return Reply(msg.content or "", calls)


# Environments and rollouts

class Environment(Protocol):
    def reset(self, task: dict) -> tuple[list[dict], list[dict]]:
        """Start an episode: (initial messages, OpenAI tool schemas)."""

    def step(self, tool_calls: list[dict]) -> list[str]:
        """Execute the calls, one result string per call."""

    def success(self) -> bool:
        """Process-checked verdict on the state the episode reached."""


def rollout(policy, env: Environment, task: dict, max_steps: int = 20, seed=None) -> bool:
    """One trajectory: the policy calls tools until it answers without a tool call."""
    messages, tools = env.reset(task)
    for _ in range(max_steps):
        reply = policy(messages, tools, seed)
        if not reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.text})
            break
        messages.append({"role": "assistant", "content": reply.text or None, "tool_calls": reply.tool_calls})
        for tc, result in zip(reply.tool_calls, env.step(reply.tool_calls)):
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return env.success()


def collect(policy, make_env, tasks: list[dict], n: int, seed: int = 0, workers: int = 32) -> dict:
    """task_id -> list of n pass flags."""
    def one(job):
        task, i = job
        try:
            return task["id"], rollout(policy, make_env(), task, seed=seed + i)
        except Exception:      # a failed request counts as a failed rollout
            return task["id"], False
    flags = {t["id"]: [] for t in tasks}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, ok in ex.map(one, [(t, i) for t in tasks for i in range(n)]):
            flags[tid].append(ok)
    return flags


# Toy environment: multi-step tool use over hidden state

def _tool(name, desc, params):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": {p: {"type": t, "description": d}
                                                        for p, (t, d) in params.items()},
                       "required": list(params)}}}


INVENTORY_TOOLS = [
    _tool("list_warehouses", "List the names of all warehouses.", {}),
    _tool("get_stock", "Number of units of an item stored in one warehouse.",
          {"warehouse": ("string", "warehouse name"), "item": ("string", "item name")}),
    _tool("submit", "Submit the final answer.", {"total": ("integer", "total number of units")}),
]


def inventory_tasks(n_tasks: int, seed: int = 0) -> list[dict]:
    """Stock levels are only visible through the tools: list warehouses, query each, submit."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n_tasks):
        item = rng.choice(["apples", "bolts", "cables", "drills", "filters", "gloves", "hammers"])
        stock = {w: rng.randint(0, 250) for w in rng.sample(["north", "south", "east", "west", "central"],
                                                            rng.randint(2, 5))}
        tasks.append({"id": f"inventory_{i}", "item": item, "stock": stock, "answer": sum(stock.values()),
                      "question": f"How many {item} do we have in stock in total, across all warehouses? "
                                  "Submit the total."})
    return tasks


class InventoryEnv:
    """Success iff exactly one submitted answer, equal to the true total."""

    def reset(self, task):
        self.task, self.submitted = task, []
        return [{"role": "system", "content": "You are a helpful assistant that can call tools."},
                {"role": "user", "content": task["question"]}], INVENTORY_TOOLS

    def step(self, tool_calls):
        out = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                if name == "list_warehouses":
                    out.append(json.dumps(sorted(self.task["stock"])))
                elif name == "get_stock":
                    if args["item"] != self.task["item"] or args["warehouse"] not in self.task["stock"]:
                        raise KeyError(f"no record for {args['item']!r} in {args['warehouse']!r}")
                    out.append(json.dumps({"units": self.task["stock"][args["warehouse"]]}))
                elif name == "submit":
                    self.submitted.append(int(args["total"]))
                    out.append(json.dumps({"status": "submitted"}))
                else:
                    raise KeyError(f"unknown tool {name!r}")
            except Exception as e:
                out.append(json.dumps({"error": str(e)}))
        return out

    def success(self):
        return len(self.submitted) == 1 and self.submitted[0] == self.task["answer"]



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--base-url", required=True, help="vLLM /v1 endpoint serving the base model")
    ap.add_argument("--rl-model", required=True)
    ap.add_argument("--rl-url", required=True, help="vLLM /v1 endpoint serving the post-trained model")
    ap.add_argument("--n", type=int, default=16, help="rollouts per task")
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="outcomes", help="prefix for the outcome JSONL files")
    args = ap.parse_args()

    sampling = Sampling(args.temperature, args.top_p, args.max_tokens)
    tasks = inventory_tasks(args.tasks)
    policies = {"base": HarnessPolicy(args.base_model, args.base_url, sampling),
                "rl": ChatPolicy(args.rl_model, args.rl_url, sampling)}
    paths = {}
    for arm, policy in policies.items():
        flags = collect(policy, InventoryEnv, tasks, args.n)
        paths[arm] = f"{args.out}_{arm}.jsonl"
        with open(paths[arm], "w") as f:
            for tid, fl in flags.items():
                f.write(json.dumps({"task_id": tid, "pass_flags": fl}) + "\n")
        print(f"{arm}: mean success {sum(map(sum, flags.values())) / (len(tasks) * args.n):.3f} -> {paths[arm]}")
    print()
    st.report(st.load_counts(paths["base"]), st.load_counts(paths["rl"]),
                          K=args.n, n_boot=1000)


if __name__ == "__main__":
    main()
