"""tools/hy4f_tokenizer.py - Hy4 flash's tokenizer in the form Maya's server reads (strata_tokenizer.Tokenizer.from_dir).

   python tools/hy4f_tokenizer.py --src <HF model dir> --out <dir> [--gguf <shard-00001.gguf>] [--check]

The model's Hugging Face tokenizer.json (byte-level BPE, 120,000 tokens + 832 added) is written as vocab.json (added tokens
included, so every id has its string), merges.txt, token_type.json (1 normal, 3 an added token marked special, 4 one that
is not: <think:opensource>, <tool_call:opensource>, <arg_key:opensource>, ...) and tokenizer.json {"pre": "deepseek3",
"special_ids": {bos, eos}}, beside the model's chat_template.jinja.  Its pre-tokenizer is DeepSeek-V3's three Isolated
splits verbatim (digits {1,3}, CJK / kana runs, the word pattern) with no normalizer - checked here, not assumed.

The GGUF spells the added tokens with another suffix (<think:6124c78e> where Hugging Face has <think:opensource>); the ids
are the same, and the server renders with Hugging Face's template, so the strings come from tokenizer.json.  --gguf
checks that the ordinary tokens and the merges agree with the GGUF's.  --check compares the ids with the `tokenizers`
library's over a corpus (chat markup included).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strata_tokenizer import DEEPSEEK3_PATTERNS, Tokenizer  # noqa: E402


def _spelled(p: str) -> str:
    """A pattern with its escapes written out (\\uXXXX, \\r, \\n, \\") - tokenizer.json holds the characters themselves."""
    p = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), p)
    return p.replace(r"\r", "\r").replace(r"\n", "\n").replace('\\"', '"')


def export(src: pathlib.Path, out: pathlib.Path) -> dict:
    t = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    m = t["model"]
    if m.get("type") != "BPE" or m.get("ignore_merges") or m.get("byte_fallback"):
        raise SystemExit("expected a plain byte-level BPE model")
    if (t.get("normalizer") or {}).get("normalizers"):
        raise SystemExit(f"unexpected normalizer: {t['normalizer']}")
    pts = (t.get("pre_tokenizer") or {}).get("pretokenizers") or []
    splits = [p for p in pts if p.get("type") == "Split"]
    bl = [p for p in pts if p.get("type") == "ByteLevel"]
    if (len(splits) != 3 or len(bl) != 1 or bl[0].get("use_regex") or bl[0].get("add_prefix_space")
            or any(p.get("behavior") != "Isolated" or p.get("invert") for p in splits)
            or tuple(_spelled(p["pattern"]["Regex"]) for p in splits) != tuple(map(_spelled, DEEPSEEK3_PATTERNS))):
        raise SystemExit("the pre-tokenizer is not deepseek3's (strata_tokenizer.DEEPSEEK3_PATTERNS)")
    vocab = dict(m["vocab"])
    types = {i: 1 for i in vocab.values()}
    for a in t.get("added_tokens") or []:
        if a["content"] in vocab and vocab[a["content"]] != a["id"]:
            raise SystemExit(f"added token {a['content']!r} clashes with the vocabulary")
        vocab[a["content"]] = a["id"]
        types[a["id"]] = 3 if a.get("special") else 4
    n = len(vocab)
    if sorted(vocab.values()) != list(range(n)):
        raise SystemExit("the ids are not 0 .. n-1")
    merges = [x if isinstance(x, str) else " ".join(x) for x in m["merges"]]
    if any(len(x.split(" ")) != 2 for x in merges):
        raise SystemExit("a merge with a space inside a token")
    cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    special = {"bos": vocab[cfg["bos_token"]], "eos": vocab[cfg["eos_token"]]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps([types[i] for i in range(n)]), encoding="utf-8")
    meta = {"model": "gpt2", "pre": "deepseek3", "vocab_size": n, "n_merges": len(merges), "special_ids": special,
            "add_bos_token": False, "source": str(src / "tokenizer.json")}
    (out / "tokenizer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copyfile(src / "chat_template.jinja", out / "chat_template.jinja")
    return meta


def check_gguf(out: pathlib.Path, gguf: str) -> None:
    from gguf_reader import GGUFFile
    md = GGUFFile(pathlib.Path(gguf)).metadata
    toks, merges = list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"])
    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    mine = [None] * len(vocab)
    for s, i in vocab.items():
        mine[i] = s
    types = json.loads((out / "token_type.json").read_text())
    normal = [i for i in range(len(mine)) if types[i] == 1]
    bad = [i for i in normal if i >= len(toks) or toks[i] != mine[i]]
    mm = (out / "merges.txt").read_text(encoding="utf-8").split("\n")
    print(f"gguf: {len(toks)} tokens ({len(mine)} here), ordinary tokens differing: {len(bad)}, merges equal: "
          f"{mm == merges}, eos {md.get('tokenizer.ggml.eos_token_id')} bos {md.get('tokenizer.ggml.bos_token_id')}")
    if bad or mm != merges or len(toks) != len(mine):
        raise SystemExit("the GGUF's tokenizer differs")


def check_hf(src: pathlib.Path, out: pathlib.Path) -> int:
    from tokenizers import Tokenizer as HF
    hf = HF.from_file(str(src / "tokenizer.json"))
    mine = Tokenizer.from_dir(out)
    corpus = [l for l in (HERE / "tokenizer_corpus.txt").read_text(encoding="utf-8").split("\n") if l] \
        if (HERE / "tokenizer_corpus.txt").exists() else []
    corpus += ["Hello, world!", "Привет, мир! Как дела? 12345 тысяч", "def f(x):\n\treturn x  # comment\n",
               "你好，世界こんにちは", "é combining", "  spaces   and\n\n\nnewlines",
               "<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>Hi<｜hy_end:opensource｜><｜hy_start:opensource｜>"
               "assistant<｜hy_middle:opensource｜><think:opensource>hmm</think:opensource>ok<tool_calls:opensource>"
               "<tool_call:opensource>f<arg_key:opensource>a</arg_key:opensource><arg_value:opensource>1"
               "</arg_value:opensource></tool_call:opensource></tool_calls:opensource><｜hy_end:opensource｜>",
               "<｜reasoning_mode:opensource｜>reasoning_effort:high", "x" * 3000, "1234567890 3.14159 1e-9",
               "Ünïcödé — “quotes” ‘and’ «ёлочки» ½ ² ™ \U0001f600\U0001f680"]
    bad = 0
    for s in corpus:
        a, b = hf.encode(s, add_special_tokens=False).ids, mine.encode(s, parse_special=True)
        if a != b:
            bad += 1
            if bad <= 5:
                print(f"  DIFF {s[:60]!r}: hf {a[:12]} mine {b[:12]}")
        if mine.decode(b) != s:
            bad += 1
            print(f"  ROUND TRIP {s[:60]!r}")
    print(f"tokenizers check: {len(corpus)} strings, {bad} differing")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the Hugging Face folder (tokenizer.json, chat_template.jinja)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", help="the first GGUF shard: its tokens / merges checked against tokenizer.json's")
    ap.add_argument("--check", action="store_true", help="compare with the tokenizers library")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    meta = export(src, out)
    print(f"{out}: vocab {meta['vocab_size']}, merges {meta['n_merges']}, pre {meta['pre']}, {meta['special_ids']}")
    if a.gguf:
        check_gguf(out, a.gguf)
    return 1 if a.check and check_hf(src, out) else 0


if __name__ == "__main__":
    raise SystemExit(main())
