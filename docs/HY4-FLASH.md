# Hy4-flash on RTX 3090s: 4, 2 or 1 GPU

This branch adds `maya-hy4f`, an engine for [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (220B parameters, 512 routed experts per layer, 16 active per token) built on Project Maya's kernels. It runs the Q3_K GGUF of Tencent's AngelSlim team with the full 262144 token context, image input and MTP speculative decoding.

With 4 cards every expert is in VRAM. With 2 cards or 1 card the hottest experts stay in VRAM and the CPU computes the rest from RAM. The weights and the quality are the same in all 3 setups.

Weights, tokenizer and a starting routing profile: [alesha-pro/Hy4-flash-Q3_K-MTP-GGUF](https://huggingface.co/alesha-pro/Hy4-flash-Q3_K-MTP-GGUF).

## Numbers

Test machine: 4x RTX 3090 at 300 W each on PCIe 3.0 x16, EPYC 7642 (47 cores), 512 GB DDR4 in 8 channels. Context 262144, Q8 KV cache, MTP with 2 or 3 drafts per step, greedy unless noted. Speeds are tokens per second.

| | 4 GPUs | 2 GPUs | 1 GPU |
|---|---:|---:|---:|
| Routed experts in VRAM | all | 43% | 15% |
| Decode, short coding prompt, 600 tokens | 251 | 70 | 40 |
| Decode after a 57K prompt | 176 to 221 | 81 to 104 | 40 to 47 |
| Decode with 255K to 261K tokens in context | 159 to 197 | 61 | not measured |
| Coding agent session, average over about 50K generated tokens | 223 | 74 | not measured |
| Prefill, 57K prompt | 2947 | 1225 | 415 |
| Filling 255K tokens of context | 2880 | 1239 (206 s) | not measured |
| Image, 1920x1080 (2074 tokens), encode time | 0.8 s | 0.7 s | 1.2 s |

The 2 GPU and 1 GPU columns use a 4096 token prefill chunk. The agent session is one task in a coding harness with tools and high reasoning effort: a landing page in one HTML file, 47 to 50 model calls, context growing to about 100K tokens.

On 2 and 1 GPU the decode speed follows how many of the picked experts are already in VRAM. The cache adapts to the answer while it is generated, so a short answer on a new topic runs at the low end of the range and a long session reaches the high end. The table rows give the measured range, not one number.

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

| Config | Cards | What it sets |
|---|---|---|
| `hy4f-4gpu.json` | 4x 24 GB | all experts in VRAM, prefill chunk 2048 |
| `hy4f-2gpu.json` | 2x 24 GB | expert offload with the CPU lane, automatic VRAM budget, adaptive cache, prefill chunk 4096 |
| `hy4f-1gpu.json` | 1x 24 GB | the same with a larger VRAM reserve for the prefill buffers |

The 2 GPU and 1 GPU configs need RAM for all routed experts (73 GiB pinned) and the Engram table (59 GiB, memory mapped), and the CPU lane wants memory bandwidth more than cores: 40 threads on 8 DDR4 channels saturate it here.

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

## Settings

Environment variables of the offload. The 4 GPU path does not read them, and without `HY4F_EXPERT_VRAM_MB` nothing changes.

| Variable | Meaning |
|---|---|
| `HY4F_EXPERT_VRAM_MB` | VRAM for routed experts per card in MiB, or `auto`: the experts load last and take the free VRAM less the reserve |
| `HY4F_VRAM_RESERVE_MB` | what `auto` leaves free for the prefill buffers, staging and decode buffers. Default 2048 fits a 4096 token chunk on 2 GPUs. 1 GPU needs 2500 at chunk 2048 and 3600 at chunk 4096 |
| `HY4F_CPU_LANE` | `1`: host threads compute the experts held in RAM during decode. Without it the GPU kernels read them over PCIe, which is 3 to 6 times slower |
| `HY4F_CPU_THREADS` | threads of the CPU lane, split between the cards. Default 40 |
| `HY4F_EXPERT_ADAPT` | `1`: the VRAM set follows the routing while the model generates |
| `HY4F_EXPERT_PROFILE` | routing statistics to start from (which experts are hot) |
| `HY4F_ROUTE_STATS` | file the engine writes its routing statistics to at exit. Give it the same file as `HY4F_EXPERT_PROFILE` and the next start begins with a warm cache |
| `HY4F_ADAPT_HORIZON`, `HY4F_ADAPT_EVERY`, `HY4F_ADAPT_MARGIN` | cache policy: memory in tokens (1024), steps between rebalances (8), picks a RAM expert must lead by to enter VRAM (3) |
| `HY4F_OFFLOAD_STAGE_MB` | staging buffers of the prefill, both together. Default 256 |
| `HY4F_EXPERT_GGUF` | take the routed experts of the trunk from another GGUF of the same model (see below) |
| `HY4F_HOT_VRAM_MB` | with `HY4F_EXPERT_GGUF`: keep the hottest experts in the base quantization within this many MiB per card |

Use `--chunk 4096` in the server config (`--prefill 4096` in the direct run) for the larger prefill chunk.

## Smaller experts as an option

`HY4F_EXPERT_GGUF` swaps the routed experts of layers 1 to 41 for the same tensors of another quantization and keeps the attention, the output head, the Engram table and the MTP layer of the base files. With the IQ2 experts of [AtomicChat/Hy4-flash-GGUF](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) (1.97 MB per expert instead of 3.65 MB) 78% of the experts fit into 2 cards.

| 2 GPUs, short coding prompt | Decode | Same top token as the base | Median KL |
|---|---:|---:|---:|
| Base Q3_K and Q4_K experts, CPU lane | 70 | 88.3% | 0.0047 |
| Hybrid, about a third of the picks in the base quantization | 90 | 77.0% | 0.041 |
| IQ2 experts | 126 | 75.6% | 0.070 |

IQ2 experts are faster and further from the base model. With the CPU lane the base experts are fast enough for me, so this stays an option.

## Limits

* Tested on RTX 3090s only. The kernels assume 24 GB cards and the geometry of Hy4-flash.
* 1, 2 or 4 cards. With 2 cards the engine has not been tuned as far as with 4: by my estimate a step with every expert in VRAM would take about 26 ms, against 11 ms measured on 4 cards.
* The CPU lane serves decode. Prefill streams experts over PCIe, which is why 1 GPU prefill is slow.
* A swap of two experts takes 0.25 to 0.55 ms and the cache did up to 2 swaps per generated token in a long session, about 6% of the decode time.
* Expert quantizations other than Q3_K, Q4_K and Q8_0 need the offload path even on 4 cards.

## Credits

| Part | Authors |
|---|---|
| Project Maya, the engine family and kernels this is built on | [mw00 and Peasant Smith](https://github.com/mw00/project-maya) |
| Strata, the engine Maya grew out of | [Niko1221](https://github.com/Niko1221/Strata) |
| Hy4-flash | [Tencent](https://huggingface.co/tencent/Hy4-flash) |
| The Q3_K GGUF and the llama.cpp patch that defines the format | [AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) |
| IQ2 experts used for the optional mode | [AtomicChat](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) |
| GGUF block formats and the CPU and MMQ kernels | [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) |
