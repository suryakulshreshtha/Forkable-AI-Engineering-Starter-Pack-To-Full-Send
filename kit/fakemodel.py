"""A scripted stand-in for a language model.

Real models are slow, cost money and never say the same thing twice. For
learning how an agent loop works, none of that helps. ScriptedModel replays a
list of replies, one per call, and records every conversation it was shown so
tests can check what the agent actually sent.

A reply is either a dict or a function that takes the message list and
returns a dict. Reply shapes the agent understands:

    {"type": "final", "text": "..."}
    {"type": "tool_call", "name": "add", "args": {"a": 1, "b": 2}}
"""
from __future__ import annotations

import copy


class ScriptedModel:
    def __init__(self, replies, repeat_last=False):
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.seen = []

    def __call__(self, messages):
        self.seen.append(copy.deepcopy(messages))
        index = len(self.seen) - 1
        if index >= len(self.replies):
            if not (self.repeat_last and self.replies):
                raise RuntimeError("ScriptedModel ran out of replies")
            index = len(self.replies) - 1
        reply = self.replies[index]
        return reply(messages) if callable(reply) else copy.deepcopy(reply)

    @property
    def calls(self):
        return len(self.seen)
