# Hy4-flash Uncensored on RTX 3090s

**Tencent's 220B MoE with the full 256K context, vision and MTP. On 4 cards, on 2, or on ONE RTX 3090 in a 64 GB box.**

[Weights on Hugging Face](https://huggingface.co/alesha-pro/Hy4-flash-Uncensored-GGUF) · [Quick start](#quick-start) · [Uncensored](#uncensored) · [Numbers](#numbers)

| | 4x RTX 3090 | 2x RTX 3090 | 1x RTX 3090 |
|---|---:|---:|---:|
| Decode, tok/s | **176 to 251** | **70 to 104** (IQ2: 95 to 123) | **37 to 54** (IQ2: 54 to 68) |
| Prefill, tok/s | **2947** | **1225** (IQ2: 1999) | **1510** |
| Context | 262144 | 262144 | 262144 |
| RAM it needs | about 60 GiB | 137 GiB | 137 GiB, down to **45 GiB** with IQ2 + NVMe |

Tested on 4x RTX 3090 at 300 W, PCIe 3.0, EPYC 7642, 8 channel DDR4.

## Why this exists

[Hy4-flash](https://huggingface.co/tencent/Hy4-flash) is a 220B parameter MoE: 512 experts per layer, 16 active per token, a 63 GB Engram table, image input and a 1M context. Stock llama.cpp doesn't load it, and the official GGUF wants a lot more memory than a home rig has.

`maya-hy4f` is an engine for exactly this model, built on [Project Maya](https://github.com/mw00/project-maya)'s kernels:

* **All of the model, no pruning.** On 1 and 2 cards the hot experts stay in VRAM and the CPU computes the rest straight from RAM. Same weights, same answers: 2 cards agree with 4 cards as well as two 4 card runs agree with each other.
* **Uncensored.** One 4 MB file projects the refusal directions out of the model as it runs. Refusals with thinking off drop from 139 to 32 out of 146. The weights stay stock, and deleting one line in the config brings the stock model back.
* **Fits a 64 GB machine.** IQ2 experts plus the Engram table read from NVMe: 45 GiB of RAM, 54 to 68 tok/s on one card.
* **Built for agents.** OpenAI and Anthropic APIs, tool calls, images, KV snapshots in RAM so going back to an earlier conversation reads only the new tail (4 s instead of 36 s at 44K tokens).

## Quick start

**1. Build** (Linux, CUDA 12.8, RTX 30 series)

```bash
git clone -b hy4-flash https://github.com/alesha-pro/project-maya && cd project-maya
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-hy4f
```

**2. Download** one trunk plus everything else

```bash
# Q3_K trunk (150 GB): the full quality model
hf download alesha-pro/Hy4-flash-Uncensored-GGUF --exclude "Hy4-flash-IQ2-*" --local-dir /models/Hy4-flash-Uncensored-GGUF
# or the IQ2 trunk (115 GB): smaller and faster, further from the base
hf download alesha-pro/Hy4-flash-Uncensored-GGUF --exclude "Hy4-flash-Q3_K-00001-*" --local-dir /models/Hy4-flash-Uncensored-GGUF
```

**3. Pick a config and run**

| Your machine | Config |
|---|---|
| 4x 24 GB | `hy4f-4gpu.json` |
| 2x 24 GB, 160 GB RAM | `hy4f-2gpu.json` |
| 1x 24 GB, 160 GB RAM | `hy4f-1gpu.json` |
| 1x 24 GB, 64 GB RAM and an NVMe drive | `hy4f-1gpu-iq2-nvme.json` |

The configs are in [`docs/hy4-flash/`](hy4-flash/) and in the weights repo. Set `exe` and `cwd` to your build, check the model paths, then:

```bash
python -m serve.server --engine maya --config docs/hy4-flash/hy4f-2gpu.json --host 127.0.0.1 --port 18000
```

The server speaks the OpenAI chat API on `http://127.0.0.1:18000/v1` (model `hy4-flash`) and has a chat page on `/`.

## Uncensored

The configs ship with the vector **on**. This is the line that does it, inside `env`:

```json
"HY4F_ABLATE": "/models/Hy4-flash-Uncensored-GGUF/Hy4-flash-refusal-directions.bin",
```

* **Uncensored:** keep the line.
* **Stock Hy4-flash:** delete the line and restart the server.
* **In between:** add `"HY4F_ABLATE_SCALE": "0.5"` (1.0 is full strength).

Running the engine directly works the same way:

```bash
HY4F_ABLATE=/models/Hy4-flash-Uncensored-GGUF/Hy4-flash-refusal-directions.bin \
build/maya-hy4f --gguf Hy4-flash-Q3_K-00001-of-00002.gguf --gguf Hy4-flash-Q3_K-00002-of-00002.gguf \
  --gpus 0,1,2,3 --max-context 262144 --tokens-i32 prompt.i32 --gen 400 --spec 2
```

Refusals on held-out prompt sets, greedy, judged by the stock model:

| | Stock | Uncensored |
|---|---:|---:|
| AdvBench 64 + JailbreakBench 82, thinking off | 139 | **32** |
| AdvBench 24, thinking on | 22 | **3** |
| Harmless requests 64 | 1 | **0** |

Most refusals are gone, not all. On ordinary long text the top token matches the stock model in 85.7% of positions (two stock runs agree on 88.3%).

<details>
<summary>How the vector was made</summary>

The residual streams at the last prompt token were recorded for 256 harmful and 256 harmless requests in several framings: thinking on and off, with and without a coding agent system prompt. Hy4 has 4 hyper-connection streams, and they carry the refusal signal differently (their directions overlap at a cosine of 0.1 to 0.6). So the vector holds its own directions for every layer and every stream: the harmful minus harmless mean, with the harmless means projected out. The engine removes them from the stream entering each layer.

Removing global directions from the weights was tried first. A few directions did nothing, 40 to 66 directions removed refusals but pushed the model far from the base (68% same top token). The scripts are in `tools/abliteration/` of the weights repo.

</details>

## Numbers

Context 262144, Q8 KV cache, MTP with 2 or 3 drafts. Tokens per second.

| | 4 GPUs | 2 GPUs | 1 GPU |
|---|---:|---:|---:|
| Decode, short coding prompt | 251 | 70 | 44 |
| Decode after a 49K to 57K prompt | 176 to 221 | 81 to 104 | 37 to 54 |
| Decode at 255K to 262K | 159 to 197 | 61 | 33 |
| Prefill, 57K prompt | 2947 | 1225 | 1510 |
| Filling 255K to 262K | 2880 | 1239 (206 s) | 1423 (184 s) |
| Coding agent session, average decode | 223 | 74 | n/a |

| IQ2 trunk | 2 GPUs | 1 GPU, Engram in RAM | 1 GPU, Engram on NVMe |
|---|---:|---:|---:|
| Decode | 95 to 123 | 58 | 54 to 68 |
| Prefill, 49K to 57K prompt | 1999 | 1534 | 1184 to 1199 |

| RAM pinned by the server | |
|---|---:|
| 1 or 2 GPUs, Q3_K | 137 GiB |
| 1 GPU, Q3_K, Engram on NVMe | 78 GiB |
| 1 GPU, IQ2, Engram on NVMe | 45 GiB (runs under a 56 GiB limit) |

Quality against the 4 GPU base, 512 teacher-forced positions of long text:

| Run | Same top token | Median KL |
|---|---:|---:|
| Two stock runs, different expert layout (the noise floor) | 88.3% | 0.005 |
| 2 GPUs | 88.3% | 0.005 |
| 1 GPU (2560 positions, 4 texts) | 88.5% | 0.007 |
| Uncensored vector | 85.7% | 0.014 |
| IQ2 experts | 74 to 76% | 0.070 |

On 1 and 2 cards decode speed depends on how many picked experts are already in VRAM. The cache follows the answer, so long sessions run at the high end.

<details>
<summary>How 1 and 2 card offload works</summary>

* Experts of each layer are renumbered hot to cold by a routing profile. Each card keeps its hottest experts in VRAM; every expert also has a record in pinned RAM.
* In decode the experts picked from RAM go through a host mailbox to CPU threads (AVX2 over the GGUF blocks) while the card computes its own. The results come back in the same layer. The CPU lane is bound by memory bandwidth, not cores.
* The cache follows the answer: every 8 steps the coldest VRAM experts trade places with the hottest RAM ones. Prompt routing doesn't predict answer routing (48% against 96% coverage), so only generated tokens count.
* Prefill streams RAM experts over PCIe and fetches each one once per chunk. The 1 GPU path reads up to 57600 tokens per expert load, then reuses its 7 GiB prefill workspace for 1934 extra experts while it generates.
* Engram on NVMe: `O_DIRECT` reads of only the rows a step needs (659 MiB for a 49K prompt), byte-identical to the table in RAM.

</details>

<details>
<summary>All settings</summary>

| Variable | Meaning |
|---|---|
| `HY4F_EXPERT_VRAM_MB` | VRAM for experts per card in MiB, or `auto` |
| `HY4F_VRAM_RESERVE_MB` | what `auto` leaves free: 2048 on 2 GPUs, 8050 on the 1 GPU path |
| `HY4F_CPU_LANE` | `1`: CPU threads compute RAM experts in decode (3 to 6 times faster than PCIe reads) |
| `HY4F_CPU_THREADS` | CPU lane threads, default 40 |
| `HY4F_EXPERT_ADAPT` | `1`: the VRAM set follows the routing while generating |
| `HY4F_EXPERT_PROFILE`, `HY4F_ROUTE_STATS` | routing profile to start from, and where to save it at exit |
| `HY4F_ABLATE`, `HY4F_ABLATE_SCALE` | refusal vector file, strength (1.0) |
| `HY4F_ENGRAM_IO` | `direct`: Engram rows from disk (1 GPU, 512 byte sectors) |
| `HY4F_EXPERT_GGUF`, `HY4F_HOT_VRAM_MB` | experts from another GGUF of the model, optionally the hottest in the base quant |
| `HY4F_SINGLE_*`, `HY4F_LARGE_HOST_PREFIX` | the fast 1 GPU path (`hy4f-1gpu.json` sets them) |
| `HY4F_RESID_DUMP` | record the residual streams per layer (used to make the vector) |

</details>

## Limits

* Tested on RTX 3090s only, with 1, 2 or 4 cards.
* 2 cards don't have the big prefill chunk of the 1 card path yet, so their prefill is slower.
* On 1 card a prompt extension of 130 to a few thousand tokens costs 2.4 to 6 s; turns up to 128 tokens skip that. Only about 0.5 GiB of VRAM stays free.
* The vector turns off a few fused kernels, so decode with it may run a few percent slower.

## Credits

Hy4-flash by [Tencent](https://huggingface.co/tencent/Hy4-flash). Q3_K GGUF and the llama.cpp format patch by [AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF). IQ2 experts by [AtomicChat](https://huggingface.co/AtomicChat/Hy4-flash-GGUF). [Project Maya](https://github.com/mw00/project-maya) by mw00 and Peasant Smith, grown out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. GGUF formats and kernels from [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp). Model license: [Tencent Hy License 1.0](https://huggingface.co/alesha-pro/Hy4-flash-Uncensored-GGUF/blob/main/LICENSE).
