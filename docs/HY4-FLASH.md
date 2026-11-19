# Hy4-flash on RTX 3090s: 4, 2 or 1 GPU

This branch adds `maya-hy4f`, an engine for [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (220B parameters, 512 routed experts per layer, 16 active per token) built on Project Maya's kernels. It runs the Q3_K GGUF of Tencent's AngelSlim team with the full 262144 token context, image input and MTP speculative decoding.

With 4 cards every expert is in VRAM. With 2 cards or 1 card the hottest experts stay in VRAM and the CPU computes the rest from RAM. The weights and the quality are the same in all 3 setups.

Two more options cut the memory a machine needs:

* a trunk file with IQ2 experts (51 GB instead of 86 GB), faster on 1 and 2 cards and further from the base model;
* on 1 card the Engram table (59 GiB) can be read from an NVMe drive instead of RAM.

Together they run the model on 1 RTX 3090 in a machine with 64 GB of RAM: the API service pins 45 GiB.

Weights, tokenizer and starting routing profiles: [alesha-pro/Hy4-flash-Q3_K-MTP-GGUF](https://huggingface.co/alesha-pro/Hy4-flash-Q3_K-MTP-GGUF).

## Numbers

Test machine: 4x RTX 3090 at 300 W each on PCIe 3.0 x16, EPYC 7642 (47 cores), 512 GB DDR4 in 8 channels, WD Black SN850X for the NVMe runs. Context 262144, Q8 KV cache, MTP with 2 or 3 drafts per step, greedy unless noted. Speeds are tokens per second.

Base weights (Q3_K), Engram in RAM:

| | 4 GPUs | 2 GPUs | 1 GPU |
|---|---:|---:|---:|
| Routed experts in VRAM | all | 43% | 7%, and 16% while it generates |
| Decode, short coding prompt, 600 tokens | 251 | 70 | 44 |
| Decode after a 49K to 57K prompt, 600 tokens | 176 to 221 | 81 to 104 | 37 to 54 |
| Decode with 255K to 262K tokens in context | 159 to 197 | 61 | 33 |
| Coding agent session, average over about 50K generated tokens | 223 | 74 | not measured |
| Prefill, 57K prompt | 2947 | 1225 | 1510 |
| Filling the context, 255K to 262K tokens | 2880 | 1239 (206 s) | 1423 (184 s) |
| Image, 1920x1080 (2074 tokens), encode time | 0.8 s | 0.7 s | works, not timed |

The agent session is one task in a coding harness with tools and high reasoning effort: a landing page in one HTML file, 47 to 50 model calls, context growing to about 100K tokens.

On 2 and 1 GPU the decode speed follows how many of the picked experts are already in VRAM. The cache adapts to the answer while it is generated, so a short answer on a new topic runs at the low end of the range and a long session reaches the high end. On 1 GPU the first 100 tokens after a long new prompt ran at 31 to 37 tokens per second at every depth from 57K to 262K.

1 GPU prefill is faster than 2 GPU prefill because the 1 GPU path reads the prompt in 57600 token chunks and the 2 GPU path still uses 4096 token chunks (see Limits).

One card, by expert quantization and by where the Engram table lives:

| 1 GPU | Q3_K, Engram in RAM | Q3_K, Engram on NVMe | IQ2, Engram in RAM | IQ2, Engram on NVMe |
|---|---:|---:|---:|---:|
| Prefill, 49K code prompt | 1516 to 1528 | 1173 to 1188 | 1534 | 1184 |
| Decode after that prompt, 600 tokens | 37 to 38 | 34 to 39 | 58 | 54 |
| Prefill, 57K prompt | 1510 | 1123 | not measured | 1199 |
| Decode after that prompt, 600 tokens | 54 | 52 | not measured | 68 |
| Decode, short coding prompt, 600 tokens | 44 | 44 | not measured | 58 |
| Decode after 262K tokens of context, 192 tokens | 33 | 21 | not measured | not measured |
| Routed experts in VRAM while it generates | 16% | 16% | 30% | 30% |

Two cards with the IQ2 trunk file keep 76% of the experts in VRAM:

| 2 GPUs | Q3_K | IQ2 |
|---|---:|---:|
| Decode, short coding prompt, 600 tokens | 70 to 72 | 123 |
| Decode after a 49K code prompt, 600 tokens | not measured | 95 |
| Prefill, 49K code prompt | not measured | 1999 |
| Prefill, 57K prompt | 1225 | not measured |

## Memory

| Setup | Pinned by the API service | Memory limit it was tested under |
|---|---:|---:|
| 4 GPUs, Q3_K | not measured: the 59 GiB Engram table and the session snapshots | none |
| 1 GPU, Q3_K, Engram in RAM (2 GPUs pin about the same) | 137 GiB | 148 GiB |
| 1 GPU, Q3_K, Engram on NVMe | 78 GiB | 88 GiB |
| 1 GPU, IQ2, Engram on NVMe | 45 GiB | 56 GiB |

The limits are cgroup v2 `memory.max` values with swap off, covering the Python server and the engine, a cold start and real requests. The smallest working limit was not searched for. Under a tight limit the cold start is slow because the page cache has no room: 389 s at 88 GiB for Q3_K. The IQ2 test started with the trunk file already in the page cache of the machine, so its start time (22 s) is not a cold one. Through the API it read a fresh 44.8K token prompt at 1136 tokens per second and generated at 34 to 60.

The routed experts stay in RAM in every offload setup: 73 GiB for Q3_K, 40 GiB for IQ2. The CPU lane wants memory bandwidth more than cores: 40 threads on 8 DDR4 channels saturate it here.

## Quality

The model, the quantization and the arithmetic of the VRAM experts are the same as on 4 GPUs. Experts computed by the CPU use llama.cpp's CPU arithmetic for the same GGUF blocks.

Agreement with the 4 GPU run on 512 teacher-forced positions of one long text (top token the same, median KL divergence):

| Run | Same top token | Median KL |
|---|---:|---:|
| 4 GPUs again, experts dealt to the cards in another order (the noise of this test) | 88.3% | 0.0054 |
| 2 GPUs, experts in RAM read by the GPU kernels | 89.1% | 0.0053 |
| 2 GPUs, CPU lane | 88.3% | 0.0047 |
| 2 GPUs, CPU lane, prefill chunk 4096 | 87.9% | 0.0051 |

The test text has a high entropy, and the partial sums that the cards exchange are rounded to BF16, so two runs of the same weights with another expert layout already differ this much. The 2 GPU runs are inside that noise.

The fast 1 GPU path against the plain 1 GPU offload on 4 texts (prose, Russian, code, the end of a 262K context), full distributions over the vocabulary:

| Text | Positions | NLL change | 95% interval | Same top token | Median KL |
|---|---:|---:|---|---:|---:|
| Long prose | 512 | -0.087 | -0.234 to +0.078 | 90.2% | 0.0057 |
| Russian, 12K | 1024 | +0.013 | -0.033 to +0.058 | 88.7% | 0.0110 |
| Code, 48K | 768 | -0.003 | -0.122 to +0.110 | 85.4% | 0.0023 |
| End of the 262K context | 256 | -0.007 | -0.090 to +0.073 | 93.4% | 0.0071 |
| All | 2560 | -0.014 | -0.065 to +0.038 | 88.5% | 0.0070 |

No significant change. With the new flags off, the logits of the 1 GPU run are byte-identical to the previous build, and the 4 GPU run produces the same tokens.

Engram on NVMe reads the same bytes as the table in RAM. With the same expert placement all 258 compared logit vectors were byte-identical, and 98544 rows read through the disk path matched the GGUF.

IQ2 experts change the model. Against the Q3_K experts on the same build and the same text:

| Run | Same top token | Median KL |
|---|---:|---:|
| 2 GPUs, IQ2 experts, 512 positions | 75.6% | 0.070 |
| 1 GPU, IQ2 experts, 128 positions | 74.2% | 0.070 |

AtomicChat, who made the IQ2 quantization, report 76.4% top token agreement with BF16 for their file and 89.3% for Tencent's Q4_K_M. I did not run task benchmarks on either.

## Build

Linux, CUDA 12.8, CMake 3.24 or newer, an NVIDIA GPU with compute capability 8.6 (RTX 30 series). Other architectures were not tested.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-hy4f
```

## Run

The engine runs behind Maya's OpenAI compatible server. Example configs are in [`docs/hy4-flash/`](hy4-flash/): put your paths into `exe`, `cwd`, `args`, `tokenizer` and `log`, then

```bash
python -m serve.server --engine maya --config docs/hy4-flash/hy4f-2gpu.json --host 127.0.0.1 --port 18000
```

| Config | Cards | Weights | What it sets |
|---|---|---|---|
| `hy4f-4gpu.json` | 4x 24 GB | Q3_K | all experts in VRAM, prefill chunk 2048 |
| `hy4f-2gpu.json` | 2x 24 GB | Q3_K | expert offload with the CPU lane, automatic VRAM budget, adaptive cache, prefill chunk 4096 |
| `hy4f-1gpu.json` | 1x 24 GB | Q3_K | the fast 1 GPU path: 57600 token prefill chunk, prefill workspace reused for experts while it generates |
| `hy4f-1gpu-iq2-nvme.json` | 1x 24 GB | IQ2 | the same with the IQ2 trunk file and the Engram table read from NVMe |

Every config takes two `--gguf` files: the trunk (`Hy4-flash-Q3_K-00001-of-00002.gguf` or `Hy4-flash-IQ2-00001-of-00002.gguf`) and the Engram table (`Hy4-flash-Q3_K-00002-of-00002.gguf`, the same file for both trunks). To use the IQ2 trunk in another config, replace the first path. To read the Engram table from disk on 1 GPU, put the second file on an NVMe drive and set `HY4F_ENGRAM_IO=direct`.

A direct run without the server, for a quick check:

```bash
HY4F_EXPERT_VRAM_MB=auto HY4F_CPU_LANE=1 HY4F_EXPERT_ADAPT=1 \
build/maya-hy4f --gguf Hy4-flash-Q3_K-00001-of-00002.gguf --gguf Hy4-flash-Q3_K-00002-of-00002.gguf \
  --gpus 0,1 --max-context 262144 --prefill 4096 --tokens-i32 prompt.i32 --gen 400 --spec 2 -v
```

## How the 2 GPU and 1 GPU modes work

* Per MoE layer the experts are renumbered from hot to cold by a routing profile, and the router rows move with them. The cards take the experts in turn, so each card owns half of every layer on 2 GPUs.
* Each card keeps its hottest experts in VRAM, as many as fit. Every expert also has a `[gate | up | down]` record in pinned host memory.
* In decode a small kernel splits the experts a step picked into the ones in VRAM and the ones in RAM. The first group runs on the card as before. The second group goes to a mailbox in host mapped memory together with the tokens' latent inputs, host threads compute it with AVX2 kernels over the GGUF blocks, and the card picks the results up in the same layer. The card works on its own experts in the meantime.
* The cache follows the answer. Every 8 steps the coldest experts in VRAM trade places with the hottest ones in RAM, by picks counted over the last 1024 tokens or so. A swap is one copy of about 3.7 MB and a swap of two router rows.
* In prefill the experts from RAM are copied into two staging buffers on a copy stream while the previous batch computes. A chunk fetches each expert once, so a larger chunk means fewer bytes per token: 846 tokens per second at chunk 2048, 1225 at 4096 and 1588 at 8192 on 2 GPUs.
* The routing of a prompt does not predict the routing of the answer. On one 57K prompt the 220 hottest experts per layer by the prompt covered 48% of the answer's picks, and the 220 hottest by the answer itself covered 96%. The cache therefore counts generated tokens only.
* The server keeps snapshots of the KV state in RAM at chunk boundaries. A return to an earlier conversation restores the nearest snapshot and reads only the tail. On 2 GPUs a return to a 44.6K token prompt took 4.1 s instead of 35.8 s.

## The fast 1 GPU path

One card holds 15% of the Q3_K experts at best, so the prefill is limited by how many times each expert crosses PCIe, and the decode by how much VRAM is left for experts.

* A layer reads up to 57600 prompt tokens with one load of its experts. Attention and the dense parts still go through that block in tiles of 2048 tokens, which keeps the work buffers small. The PCIe traffic per token falls about 8 times against a 4096 token chunk: one profiled run moved 67.7 GB for 32768 tokens where the old path moved 66.0 GB for 4096.
* The prefill workspace is 7 GiB. While the model generates, that memory holds 1934 more Q3_K experts (3549 with IQ2) on top of the permanent set. Before the next long prefill they are dropped, and afterwards loaded again: 6.8 GiB over PCIe.
* A text turn of up to 128 new tokens does not go through the prefill at all. It runs through the decode graphs and the expert cache stays in place.
* Single card attention kernels for both attention types: 2 groups of 32 heads for the sliding window layers, explicit score, softmax and value tiles for the sparse layers. A decode step went from 68 ms to 50 ms.
* Session snapshots use a grid that fits the large chunk: 4096, 32768, then every 57344 tokens.

## Engram on NVMe

The Engram table is 660 million rows of 96 bytes, 59 GiB. The default loader pins all of it. With `HY4F_ENGRAM_IO=direct` the engine opens the second GGUF with `O_DIRECT` and reads only the rows a step asks for through Linux AIO at queue depth 64, with a cache of 1 million rows (104 MiB).

A 49K token prompt read 659 MiB from the drive, a 262K one 3.3 GiB. The prefill is 13% to 23% slower. The decode is the same on short and medium contexts and slower near the end of the context (21 against 33 tokens per second after 262K). The mode is for machines without the RAM for the table. With enough RAM keep the table in memory.

## Settings

Environment variables of the offload. The 4 GPU path does not read them, and without `HY4F_EXPERT_VRAM_MB` nothing changes.

| Variable | Meaning |
|---|---|
| `HY4F_EXPERT_VRAM_MB` | VRAM for routed experts per card in MiB, or `auto`: the experts load last and take the free VRAM less the reserve |
| `HY4F_VRAM_RESERVE_MB` | what `auto` leaves free for the prefill buffers, staging and decode buffers. Default 2048 fits a 4096 token chunk on 2 GPUs. The fast 1 GPU path with its 57600 token chunk uses 8050 |
| `HY4F_CPU_LANE` | `1`: host threads compute the experts held in RAM during decode. Without it the GPU kernels read them over PCIe, which is 3 to 6 times slower |
| `HY4F_CPU_THREADS` | threads of the CPU lane, split between the cards. Default 40 |
| `HY4F_EXPERT_ADAPT` | `1`: the VRAM set follows the routing while the model generates |
| `HY4F_EXPERT_PROFILE` | routing statistics to start from (which experts are hot) |
| `HY4F_ROUTE_STATS` | file the engine writes its routing statistics to at exit. Give it the same file as `HY4F_EXPERT_PROFILE` and the next start begins with a warm cache |
| `HY4F_ADAPT_HORIZON`, `HY4F_ADAPT_EVERY`, `HY4F_ADAPT_MARGIN` | cache policy: memory in tokens (1024), steps between rebalances (8), picks a RAM expert must lead by to enter VRAM (3) |
| `HY4F_OFFLOAD_STAGE_MB` | staging buffers of the prefill, both together. Default 256 |
| `HY4F_EXPERT_GGUF` | take the routed experts of the trunk from another GGUF of the same model instead of the first `--gguf` |
| `HY4F_HOT_VRAM_MB` | with `HY4F_EXPERT_GGUF`: keep the hottest experts in the base quantization within this many MiB per card |
| `HY4F_ENGRAM_IO` | `direct`: read Engram rows from the second GGUF on disk (1 GPU). `ram` or unset: pin the table |

The fast 1 GPU path, all read only with 1 card:

| Variable | Meaning |
|---|---|
| `HY4F_SINGLE_PF` | `mqa`: tiled single card prefill attention. Turns the path on |
| `HY4F_SINGLE_LAYER_TILE` | tokens per attention tile inside a chunk, 2048 |
| `HY4F_SINGLE_SPLIT_PF`, `HY4F_SINGLE_DSA_HALF`, `HY4F_SINGLE_COMBINE` | `1`: score, softmax and value kernels for the prefill attention, half precision KV rows for the sparse layers, fused expert combine and residual normalization |
| `HY4F_SINGLE_DEC_ATTN` | `1`: the single card decode attention kernels |
| `HY4F_SINGLE_WORKSPACE_CACHE` | `1`: the prefill workspace holds experts while the model generates |
| `HY4F_SINGLE_ADAPT_WORKSPACE` | `1`: the adaptive cache manages the permanent and the workspace experts as one set |
| `HY4F_SINGLE_SHORT_VERIFY` | text turns of up to this many new tokens run through the decode graphs, 128 |
| `HY4F_SINGLE_PAR_QUANT` | `1`: the CPU lane quantizes the intermediate activations on all its threads |
| `HY4F_LARGE_HOST_PREFIX` | `1`: session snapshots on the grid of the large chunk |

Use `--chunk` in the server config (`--prefill` in the direct run) for the prefill chunk: 2048 on 4 cards, 4096 on 2, 57600 on 1.

## The IQ2 trunk file

`Hy4-flash-IQ2-00001-of-00002.gguf` is the Q3_K trunk file with the routed experts of layers 1 to 41 taken byte for byte from [AtomicChat/Hy4-flash-GGUF](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) (gate and up at IQ2_XXS, down at IQ2_S). The attention, the output head and the MTP layer are the ones of the Q3_K file, and the Engram table is the same second file. An expert takes 1.97 MB instead of 3.65 MB, so twice as many fit into VRAM and the CPU reads half the bytes for the rest. `tools/merge_experts_gguf.py` in the weights repo builds the file and verifies it.

`HY4F_EXPERT_GGUF` does the same swap at load time from any GGUF of the model, and with `HY4F_HOT_VRAM_MB` it makes a hybrid: the hottest experts in the base quantization, the rest in IQ2.

| 2 GPUs, short coding prompt | Decode | Same top token as the base | Median KL |
|---|---:|---:|---:|
| Base Q3_K and Q4_K experts, CPU lane | 70 | 88.3% | 0.0047 |
| Hybrid, about a third of the picks in the base quantization | 90 | 77.0% | 0.041 |
| IQ2 experts | 126 | 75.6% | 0.070 |

The base experts keep the quality of the 4 GPU run. IQ2 is for machines where the base experts do not fit, or where speed matters more.

## Limits

* Tested on RTX 3090s only. The kernels assume 24 GB cards and the geometry of Hy4-flash.
* 1, 2 or 4 cards. The 2 card mode has not received the large prefill chunk or the workspace reuse of the 1 card path yet, so its prefill is slower than on 1 card. By my estimate a 2 card decode step with every expert in VRAM would take about 26 ms, against 11 ms measured on 4 cards.
* On 1 card a prompt extension of 130 to a few thousand tokens costs 2.4 to 6 s whatever its size, because even a small chunk touches almost every expert and most of them come from RAM. A 215 token tool result took 4.25 s. Turns of up to 128 tokens avoid this.
* A 57600 token chunk takes about 38 s and cannot be interrupted in the middle. A return to an earlier session can re-read up to 57K tokens when the prompt ends just before a snapshot boundary.
* The fast 1 GPU path leaves about 0.5 GiB of VRAM free. Nothing else can run on that card.
* `HY4F_ENGRAM_IO=direct` works with 1 card and assumes a drive with 512 byte logical sectors.
* A swap of two experts takes 0.25 to 0.55 ms and the cache did up to 2 swaps per generated token in a long session, about 6% of the decode time on 2 cards.
* Expert quantizations other than Q3_K, Q4_K and Q8_0 need the offload path even on 4 cards.

## Credits

| Part | Authors |
|---|---|
| Project Maya, the engine family and kernels this is built on | [mw00 and Peasant Smith](https://github.com/mw00/project-maya) |
| Strata, the engine Maya grew out of | [Niko1221](https://github.com/Niko1221/Strata) |
| Hy4-flash | [Tencent](https://huggingface.co/tencent/Hy4-flash) |
| The Q3_K GGUF and the llama.cpp patch that defines the format | [AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) |
| IQ2 experts of the compact trunk file | [AtomicChat](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) |
| GGUF block formats and the CPU and MMQ kernels | [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) |
