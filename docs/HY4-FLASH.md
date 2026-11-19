# Hy4-flash on RTX 3090s: 4, 2 or 1 GPU, uncensored option

This branch adds `maya-hy4f`, an engine for [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (220B, 512 routed experts per layer, 16 active per token) built on Project Maya's kernels: full 262144 token context, image input, MTP speculative decoding.

* 4 cards: every expert in VRAM.
* 2 or 1 card: the hottest experts in VRAM, the CPU computes the rest from RAM. Same weights, same quality.
* Uncensored: a refusal vector the engine projects out of the residual stream at every layer (`HY4F_ABLATE`). The weights stay stock.
* Compact: a trunk file with IQ2 experts, and on 1 card the Engram table can be read from NVMe.

Weights, vector, tokenizer and configs: [alesha-pro/Hy4-flash-Uncensored-GGUF](https://huggingface.co/alesha-pro/Hy4-flash-Uncensored-GGUF).

## Speed

4x RTX 3090 at 300 W, PCIe 3.0 x16, EPYC 7642, 512 GB DDR4 in 8 channels. Context 262144, Q8 KV cache, MTP. Tokens per second.

| | 4 GPUs | 2 GPUs | 1 GPU |
|---|---:|---:|---:|
| Decode, short coding prompt | 251 | 70 | 44 |
| Decode after a 49K to 57K prompt | 176 to 221 | 81 to 104 | 37 to 54 |
| Decode at 255K to 262K | 159 to 197 | 61 | 33 |
| Prefill, 57K prompt | 2947 | 1225 | 1510 |
| Coding agent session, average decode | 223 | 74 | not measured |

| IQ2 trunk | 2 GPUs | 1 GPU, Engram in RAM | 1 GPU, Engram on NVMe |
|---|---:|---:|---:|
| Decode | 95 to 123 | 58 | 54 to 68 |
| Prefill, 49K to 57K prompt | 1999 | 1534 | 1184 to 1199 |

On 2 and 1 card decode speed depends on how many picked experts are already in VRAM. The cache follows the answer, so a long session runs at the high end of the range.

## Memory

| Setup | RAM pinned by the API server |
|---|---:|
| 1 or 2 GPUs, Q3_K, Engram in RAM | 137 GiB |
| 1 GPU, Q3_K, Engram on NVMe | 78 GiB |
| 1 GPU, IQ2, Engram on NVMe | 45 GiB (works under a 56 GiB limit) |

## Quality

| Run, against the 4 GPU base | Same top token | Median KL |
|---|---:|---:|
| Noise of the test (the base with another expert layout) | 88.3% | 0.005 |
| 2 GPUs, CPU lane | 88.3% | 0.005 |
| 1 GPU, fast path (2560 positions, 4 texts) | 88.5% | 0.007 |
| Uncensored vector, 4 GPUs | 85.7% | 0.014 |
| IQ2 experts | 74 to 76% | 0.070 |

512 teacher-forced positions of one long text unless noted. Engram read from NVMe gives byte-identical logits.

## Uncensored

`HY4F_ABLATE=Hy4-flash-refusal-directions.bin` in the config's `env`. Delete the line for stock behavior. Works on 1, 2 and 4 cards, with the Q3_K and the IQ2 trunk.

Refusals on held-out sets, greedy, judged by the stock model:

| | Stock | Uncensored |
|---|---:|---:|
| AdvBench (64) + JailbreakBench (82), thinking off | 139 | 32 |
| AdvBench, thinking on (24) | 22 | 3 |
| Harmless requests (64) | 1 | 0 |

How it was made: the residual streams at the last prompt token were recorded for 256 harmful and 256 harmless requests in five prompt framings (thinking on and off, with and without a coding agent system text) and at the end of the model's own reasoning. Hy4's four hyper-connection streams carry this differently, so the vector has its own directions for every layer and every stream: harmful minus harmless means, with the harmless means projected out. One rank-k removal of global directions from the weights cut refusals too, but moved the model far from the base (68% same top token), so the projection stays in the engine.

With the vector the model takes on more tasks with thinking on, and those answers are longer. Decode speed with the vector was not measured separately: it turns off a few fused kernels.

## Build

Linux, CUDA 12.8, compute capability 8.6 (RTX 30 series). Other GPUs were not tested.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-hy4f
```

## Run

Put your paths into a config from [`docs/hy4-flash/`](hy4-flash/), then

```bash
python -m serve.server --engine maya --config docs/hy4-flash/hy4f-2gpu.json --host 127.0.0.1 --port 18000
```

| Config | Cards | What it sets |
|---|---|---|
| `hy4f-4gpu.json` | 4 | all experts in VRAM, prefill chunk 2048 |
| `hy4f-2gpu.json` | 2 | expert offload, CPU lane, adaptive cache, chunk 4096 |
| `hy4f-1gpu.json` | 1 | the fast 1 GPU path, chunk 57600 |
| `hy4f-1gpu-iq2-nvme.json` | 1 | IQ2 trunk, Engram table read from NVMe |

The configs in the weights repo have the vector on. To use the IQ2 trunk anywhere, replace the first `--gguf` path. For the NVMe mode put `Hy4-flash-Q3_K-00002-of-00002.gguf` on an NVMe drive and set `HY4F_ENGRAM_IO=direct` (1 GPU).

## How offload works

* Experts of each layer are renumbered hot to cold by a routing profile; each card keeps its hottest experts in VRAM and every expert has a record in pinned RAM.
* In decode, experts picked from RAM go through a host mailbox to CPU threads (AVX2 over the GGUF blocks) while the card computes its own; results come back in the same layer.
* The cache follows the answer: every 8 steps the coldest VRAM experts trade places with the hottest RAM ones. Prompt routing does not predict answer routing (48% against 96% coverage), so only generated tokens count.
* Prefill copies RAM experts to staging buffers on a copy stream. Each chunk fetches each expert once, so bigger chunks move fewer bytes per token. The 1 GPU path reads up to 57600 tokens per expert load and reuses its 7 GiB prefill workspace for extra experts while generating.
* The server keeps KV snapshots in RAM at chunk boundaries, so returning to an earlier conversation reads only the tail (2 GPUs: 4.1 s instead of 35.8 s at 44.6K).

## Settings

| Variable | Meaning |
|---|---|
| `HY4F_EXPERT_VRAM_MB` | VRAM for experts per card in MiB, or `auto` (free VRAM less the reserve) |
| `HY4F_VRAM_RESERVE_MB` | what `auto` leaves free: 2048 on 2 GPUs, 8050 for the 1 GPU path |
| `HY4F_CPU_LANE` | `1`: CPU threads compute the RAM experts in decode (3 to 6 times faster than reading them over PCIe) |
| `HY4F_CPU_THREADS` | CPU lane threads, default 40 |
| `HY4F_EXPERT_ADAPT` | `1`: the VRAM set follows the routing while generating |
| `HY4F_EXPERT_PROFILE`, `HY4F_ROUTE_STATS` | routing profile to start from, and where to write it at exit |
| `HY4F_ABLATE`, `HY4F_ABLATE_SCALE` | refusal directions file, projection strength (1.0) |
| `HY4F_ENGRAM_IO` | `direct`: read Engram rows from disk (1 GPU, 512 byte sectors) |
| `HY4F_EXPERT_GGUF`, `HY4F_HOT_VRAM_MB` | experts from another GGUF of the model, optionally the hottest kept in the base quant |
| `HY4F_SINGLE_*`, `HY4F_LARGE_HOST_PREFIX` | the fast 1 GPU path; `hy4f-1gpu.json` sets them all |
| `HY4F_RESID_DUMP` | record the residual streams per layer (how the vector was measured) |

## Limits

* Tested on RTX 3090s only, 1, 2 or 4 cards.
* 2 cards do not have the large prefill chunk of the 1 card path yet, so 2 card prefill is slower.
* On 1 card a prompt extension of 130 to a few thousand tokens costs 2.4 to 6 s; turns up to 128 tokens avoid it. The 1 card path leaves about 0.5 GiB of VRAM free.
* The vector removes most refusals, not all: about a fifth stay with thinking off.

## Credits

| Part | Authors |
|---|---|
| Project Maya | [mw00 and Peasant Smith](https://github.com/mw00/project-maya) |
| Strata, which Maya grew out of | [Niko1221](https://github.com/Niko1221/Strata) |
| Hy4-flash | [Tencent](https://huggingface.co/tencent/Hy4-flash) |
| Q3_K GGUF and the llama.cpp patch for the format | [AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) |
| IQ2 experts | [AtomicChat](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) |
| GGUF formats, CPU and MMQ kernels | [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) |
