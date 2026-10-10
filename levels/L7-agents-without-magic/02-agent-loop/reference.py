"""
Title: Agent Loop
XP: 35

## The job
Write `run_agent(model, registry, user_message, max_steps, system)`. Start the transcript
with an optional system message and the user message. Each step, call `model(messages)`:
a `final` reply ends the run; a `tool_call` reply is recorded, run through
`registry.call`, and its result appended as a `tool` message; anything else gets a system
message explaining the valid reply shapes. Stop after `max_steps` model calls. Return
`{"answer", "steps", "stopped", "messages", "tool_calls"}`.

## Why it matters
This loop is what every agent framework wraps. Getting the transcript right is the whole
game: the model only knows what you put back into the messages, so a tool error that never
makes it into the transcript is a mistake the model can never fix. The step limit is the
seatbelt that keeps a confused model from looping until the bill arrives.

## Hints
- Message shapes: user {"role": "user", "content": text}; tool request {"role": "assistant", "tool_call": {"name", "args"}}; tool result {"role": "tool", "name", "content": the dict from registry.call}; final {"role": "assistant", "content": text}.
- `tool_calls` in the result is a list of (name, ok) pairs, one per tool call, in order.
- `stopped` is "final" when the model answered and "max_steps" when you cut it off (then answer is None).
- An invalid reply still counts as a step. Append {"role": "system", "content": ...} that names the two valid reply types.
- `kit.tools.ToolRegistry` is the registry; `kit.fakemodel.ScriptedModel` is what the tests use as a model.
"""
from __future__ import annotations


def run_agent(model, registry, user_message, max_steps=6, system=None):
    """Run the model/tool loop. See the docstring above for the exact message shapes."""
    # >>> fullsend: build the transcript, loop up to max_steps, branch on the reply type
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    tool_calls = []
    for step in range(1, max_steps + 1):
        reply = model(messages)
        kind = reply.get("type") if isinstance(reply, dict) else None
        if kind == "final":
            text = reply.get("text", "")
            messages.append({"role": "assistant", "content": text})
            return {"answer": text, "steps": step, "stopped": "final",
                    "messages": messages, "tool_calls": tool_calls}
        if kind == "tool_call":
            name, args = reply.get("name"), reply.get("args", {})
            messages.append({"role": "assistant", "tool_call": {"name": name, "args": args}})
            result = registry.call(name, args)
            messages.append({"role": "tool", "name": name, "content": result})
            tool_calls.append((name, result["ok"]))
            continue
        messages.append({
            "role": "system",
            "content": f"Invalid reply {reply!r}. Reply with type 'final' or type 'tool_call'.",
        })
    return {"answer": None, "steps": max_steps, "stopped": "max_steps",
            "messages": messages, "tool_calls": tool_calls}
    # <<< fullsend
