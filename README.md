<h1 align="center">Hy4-flash Uncensored on your RTX 3090s</h1>

<p align="center"><b>Tencent's 220B MoE with 256K context, vision and MTP. 251 tok/s on 4 cards, 70 to 104 on 2, and it runs on ONE 3090 in a 64 GB box.</b></p>

<p align="center">
<a href="docs/HY4-FLASH.md"><b>Read the guide</b></a> ·
<a href="https://huggingface.co/alesha-pro/Hy4-flash-Uncensored-GGUF"><b>Weights on Hugging Face</b></a> ·
<a href="docs/HY4-FLASH.md#quick-start"><b>Quick start</b></a>
</p>

> **Start with [docs/HY4-FLASH.md](docs/HY4-FLASH.md).** Speeds, memory per setup, how to switch the uncensored vector on and off, and every config.
>
> Boards without P2P (for example a Xeon E5 v4 / C612 with two RTX 3090s) work too: the engine falls back to host-mapped exchange buffers. See [Two GPUs without P2P](docs/HY4-FLASH.md#two-gpus-without-p2p).

| | 4x RTX 3090 | 2x RTX 3090 | 1x RTX 3090 |
|---|---:|---:|---:|
| Decode, tok/s | 176 to 251 | 70 to 104 (IQ2: up to 123) | 37 to 54 (IQ2: up to 68) |
| Prefill, tok/s | 2947 | 1225 (IQ2: 1999) | 1510 |
| RAM | about 60 GiB | 137 GiB | 137 GiB, or 45 GiB with IQ2 + NVMe |

## What you get

* **`maya-hy4f`**, an engine for [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash): the full 262144 token context, image input, MTP speculative decoding, OpenAI and Anthropic APIs, tool calls.
* **1, 2 or 4 cards from the same weights.** What doesn't fit in VRAM stays in RAM and the CPU computes it. 2 cards agree with 4 as closely as two 4 card runs agree with each other.
* **Uncensored with one file.** A 4 MB refusal vector the engine projects out of the model as it runs: refusals with thinking off drop from 139 to 32 out of 146, with thinking on from 22 to 3 out of 24. Delete one config line and you have the stock model again.
* **Small machines too.** An IQ2 trunk and an Engram table read from NVMe bring one card down to 45 GiB of RAM.

## Quick start

```bash
git clone -b hy4-flash https://github.com/alesha-pro/project-maya && cd project-maya
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-hy4f
hf download alesha-pro/Hy4-flash-Uncensored-GGUF --exclude "Hy4-flash-IQ2_XXS-*" --local-dir /models/Hy4-flash-Uncensored-GGUF
# set exe and cwd in the config to this checkout, then
python -m serve.server --engine maya --config docs/hy4-flash/hy4f-2gpu.json --host 127.0.0.1 --port 18000
```

Configs for 4, 2 and 1 card are in [`docs/hy4-flash/`](docs/hy4-flash/). The full walkthrough, with the IQ2 and NVMe options, is in the [guide](docs/HY4-FLASH.md#quick-start).

## About this repo

This is a fork of [Project Maya](https://github.com/mw00/project-maya) (v1.1.66), the engine family by Peasant Smith ([@mw00](https://github.com/mw00)) and the Maya contributors, which grew out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. The `hy4-flash` branch adds the Hy4-flash engine (`src/hy4f/`), its offload modes and the refusal vector support. The rest of Maya (GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash and the app around them) is still in the tree but isn't maintained here: for those models use [upstream Maya](https://github.com/mw00/project-maya) and its README.

Weights: Tencent Hy4-flash, quantized by [AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) (Q3_K) and [AtomicChat](https://huggingface.co/AtomicChat/Hy4-flash-GGUF) (IQ2 experts), packaged with a requantized MTP layer in [alesha-pro/Hy4-flash-Uncensored-GGUF](https://huggingface.co/alesha-pro/Hy4-flash-Uncensored-GGUF).

## License

The code is MIT, as upstream ([LICENSE](LICENSE)). The model weights are under the [Tencent Hy License 1.0](https://huggingface.co/alesha-pro/Hy4-flash-Uncensored-GGUF/blob/main/LICENSE).
