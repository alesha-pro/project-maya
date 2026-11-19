"""Exercise HY4's pre-decode yield through the single-place Service FIFO."""
import os
import threading
import time
import unittest
from unittest.mock import patch

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, FairLock, Service
from serve.test_fair import ROOT, ResumingEngine


class CheckpointEngine(ResumingEngine):
    can_prefill_yield = True

    def __init__(self, tok):
        super().__init__(tok, {"long request": "long result", "short request": "short result"}, delay_s=0)
        self.reading, self.queued, self.short, self.finish_short = (threading.Event() for _ in range(4))
        self.last, self.progress, self.signal, self.attempts = {}, None, False, []

    def yield_prefill(self):
        self.signal = True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        long = len(ids) > 8192
        order = sampling.get("_resume_prefill")
        self.attempts.append((long, order, tuple(ids), dict(sampling)))
        self.progress = None
        if long and not order:
            self.reading.set()
            assert self.queued.wait(3)
            self.progress = (2048, len(ids), 0, 40)
            yield None
            if self.signal:
                self.last = dict(finish="prefill_yield", prompt_tokens=len(ids), prompt_ms=60, decode_ms=0,
                                 generated=0, reused=0, prefill_order=37, prefill_read=4096)
                return
        if not long:
            self.short.set()
            assert self.finish_short.wait(3)
        self.progress = (len(ids), len(ids), 4096 if order else 0, 80)
        yield None
        count = 0
        try:
            for token in super().generate(ids, max_new, sampling, cancel):
                count += 1
                yield token
        finally:
            self.last = dict(finish="stop", prompt_tokens=len(ids), prompt_ms=80, decode_ms=15, generated=count,
                             reused=4096 if order else 0)


class PrefillYield(unittest.TestCase):
    def test_only_one_waiter_leapfrogs_and_a_cancelled_claim_preserves_the_reader(self):
        for cancelled in (False, True):
            fifo = FairLock()
            self.assertTrue(fifo.acquire())
            short = fifo.enqueue(1, short_prefill=True)
            later = fifo.enqueue(1, short_prefill=True)
            self.assertIs(fifo.claim_short_prefill(), short)
            self.assertIsNone(fifo.claim_short_prefill())
            if cancelled:
                fifo.leave(short)
            reader = fifo.yield_place(1, after_waiter=short)
            if not cancelled:
                self.assertTrue(fifo.wait_turn(short, 0))
                fifo.release()
            self.assertTrue(fifo.wait_turn(reader, 0))
            fifo.release()
            self.assertTrue(fifo.wait_turn(later, 0))
            fifo.release()
        with fifo:
            older = fifo.enqueue()
            later = fifo.enqueue(1, short_prefill=True)
            self.assertIsNone(fifo.claim_short_prefill())
            fifo.leave(older)
            fifo.leave(later)

    def exercise(self, abort=False, enabled=True, short_limit=32):
        tok = ByteTokenizer()
        engine = CheckpointEngine(tok)
        engine.can_prefill_yield = enabled
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        output, errors = {}, []
        cancel = threading.Event()
        def turn(tag, message, limit, stop):
            try:
                ids, think, limit = svc.prepare([{"role": "user", "content": message}], None,
                                                {"enable_thinking": False}, limit)
                events = list(svc.run(ids, think, None, limit, {"seed": 19}, stop))
                output[tag] = "".join(e.text for kind, e in events if kind == "event" and e.kind == "content")
                if tag == "a" and enabled and not abort and short_limit <= 256:
                    progress = [pp for kind, pp in events if kind == "ping" and pp]
                    assert all(pp["cache"] == 0 for pp in progress)
                    assert progress[-1]["processed"] == len(ids) and progress[-1]["time_ms"] == 140
            except BaseException as exc:
                errors.append(exc)
        a = threading.Thread(target=turn, args=("a", "long request " + "x" * 16000, 48, cancel))
        b = threading.Thread(target=turn, args=("b", "short request", short_limit, threading.Event()))
        with patch.dict(os.environ, {"MAYA_FAIR_SLICE_S": "0"}):
            a.start()
            self.assertTrue(engine.reading.wait(3))
            b.start()
            until = time.monotonic() + 3
            while not svc.fifo.waiting() and time.monotonic() < until:
                time.sleep(.005)
            self.assertEqual(svc.fifo.waiting(), 1)
            engine.queued.set()
            self.assertTrue(engine.short.wait(3))
            if abort:
                cancel.set()
                a.join(2)
                self.assertFalse(a.is_alive())
            engine.finish_short.set()
            for thread in (a, b):
                thread.join(3)
                self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(output["b"], "short result")
        self.assertEqual(svc.fifo.waiting(), 0)
        self.assertFalse(svc.fifo.locked())
        if abort:
            partial = next(row for row in svc.history if row["prompt_tokens"] > 8192)
            self.assertEqual((partial["prompt_ms"], partial["engine_generated"], partial["reused"]), (60, 0, 0))
            self.assertEqual(partial["prefill_tok_s"], round(4096 / .06, 1))
        return engine, output

    def test_resume_preserves_prompt_and_seed(self):
        engine, output = self.exercise()
        self.assertEqual(output["a"], "long result")
        self.assertEqual([(a, b) for a, b, _, _ in engine.attempts], [(True, None), (False, None), (True, 37)])
        self.assertEqual(engine.attempts[0][2], engine.attempts[2][2])
        self.assertEqual(engine.attempts[2][3], {**engine.attempts[0][3], "_resume_prefill": 37})
        self.assertEqual((engine.last["prompt_ms"], engine.last["reused"]), (140, 0))

    def test_abort_while_queued_does_not_resume(self):
        eng, _ = self.exercise(abort=True)
        self.assertEqual([(a, b) for a, b, _, _ in eng.attempts], [(True, None), (False, None)])

    def test_no_capability_or_long_waiter(self):
        for enabled, short_limit in ((False, 32), (True, 1024)):
            eng, _ = self.exercise(enabled=enabled, short_limit=short_limit)
            self.assertFalse(eng.signal)
            self.assertEqual(len(eng.attempts), 2)
