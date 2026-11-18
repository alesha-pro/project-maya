"""serve/test_hy4_frontend.py - Tencent Hy4 flash (maya-hy4f) in the frontend: its output (reasoning, answer, GLM-form
tool calls with the template's ':opensource' suffix) parsed from any split of the stream, and - where the model's own
chat template is at hand - its thinking levels.

    python -m unittest serve.test_hy4_frontend      (MAYA_HY4_TEMPLATE=<the model's chat_template.jinja>; default: the
                                                     rig's /mnt/ssd/models/hy4-flash/maya-tokenizer copy, else skipped)
"""
from __future__ import annotations

import os
import random
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, OutputParser

TEMPLATE = Path(os.environ.get("MAYA_HY4_TEMPLATE", "/mnt/ssd/models/hy4-flash/maya-tokenizer/chat_template.jinja"))
S = ":opensource"
TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}}}}]
OUT = (f"Let me think about <b>it</b>.</think{S}>Sure.<tool_calls{S}><tool_call{S}>get_weather<arg_key{S}>city"
       f"</arg_key{S}><arg_value{S}>Paris</arg_value{S}><arg_key{S}>days</arg_key{S}><arg_value{S}>3</arg_value{S}>"
       f"</tool_call{S}></tool_calls{S}>")


def parse(pieces, thinking=True):
    p = OutputParser(thinking=thinking, tools=TOOLS, stream_tools=True, hy4=S)
    evs = []
    for x in pieces:
        evs += p.feed(x)
    evs += p.finish()
    text = lambda kind: "".join(e.text for e in evs if e.kind == kind)   # noqa: E731
    return text("reasoning"), text("content"), [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class Hy4Output(unittest.TestCase):
    def test_whole_and_every_split(self):
        want = ("Let me think about <b>it</b>.", "Sure.", [("get_weather", {"city": "Paris", "days": 3})])
        self.assertEqual(parse([OUT]), want)
        rng = random.Random(7)
        for _ in range(200):
            cuts = sorted(rng.sample(range(1, len(OUT)), 12))
            self.assertEqual(parse([OUT[a:b] for a, b in zip([0] + cuts, cuts + [len(OUT)])]), want)

    def test_plain_answer_and_thinking_off(self):
        self.assertEqual(parse([f"hmm</think{S}>", "\n\nHello <there>."]), ("hmm", "Hello <there>.", []))
        self.assertEqual(parse(["Just text, no tags."], thinking=False), ("", "Just text, no tags.", []))


@unittest.skipUnless(TEMPLATE.exists(), f"the Hy4 chat template is not at {TEMPLATE}")
class Hy4Template(unittest.TestCase):
    def test_levels(self):
        t = ChatTemplate(TEMPLATE)
        self.assertEqual((t.hy4, t.think_open), (S, f"<think{S}>"))
        msgs = [{"role": "user", "content": "Hi"}]
        mode = lambda **kw: t.render(msgs, **kw).split("reasoning_effort:")[1].split("<")[0]   # noqa: E731
        self.assertEqual([mode(), mode(reasoning_effort="low"), mode(reasoning_effort="high"),
                          mode(reasoning_effort="max"), mode(enable_thinking=False)],
                         ["high", "high", "high", "max", "no_think"])
        self.assertTrue(t.render(msgs).endswith(f"assistant<｜hy_middle{S}｜><think{S}>"))
        self.assertTrue(t.render(msgs, enable_thinking=False).endswith(f"<think{S}></think{S}>"))


if __name__ == "__main__":
    unittest.main()
