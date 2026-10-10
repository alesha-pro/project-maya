# Refusal projection for GLM-5.3-Flash in Maya

This branch is Project Maya v1.0.27 plus one small patch. The patch removes chosen directions from the model's residual streams while it runs. With my vector file it removes refusals, and the GGUF stays as published. Without the settings below the engine behaves exactly like upstream.

Maya and its quants are the work of mw00 and PeasantSmith: [mw00/project-maya](https://github.com/mw00/project-maya), [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF). Maya grew out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. My part is the 3 files listed below and the vector.

## The patch

Strata has control vectors for Qwen. Maya's GLM path had none. I added `src/core/glm_ablate.cuh`, which holds a CUDA kernel that projects K orthonormal directions out of every row it is given and the loader for the vector file. The kernel is called from the decode path in `src/core/glm_fast_path.cu` and from the batched prompt path in `src/core/glm_prefill.cu`.

Per-layer mode, the one I use: before layer `l`'s FFN output is written into the hyper-connection streams, both operands of that write (the FFN output and the 4 streams) lose their components along layer `l`'s directions:

```text
x = x - scale * (x . v) v      for each of the layer's directions v
```

After the write every stream is orthogonal to the directions of that layer. It is the same operation my [exllamav3 fork](https://github.com/alesha-pro/exllamav3) does after each block. The MTP draft block is left alone.

There is also an "outputs" mode: one set of directions for all layers, removed from the token embedding and from every attention and FFN output.

## Use

Build Maya as upstream says. Then get the vector from [alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF) and turn it on in the config the installer wrote:

```bash
tools/glm_ablate_config.py maya-maya-m.json /path/to/glm53-refusal-per-block-3dir.layers45.f32
./maya.sh            # restart Maya
tools/glm_ablate_config.py maya-maya-m.json --off     # back to the stock model
```

The script adds 2 entries to the `env` section of the config:

| Variable | Meaning |
|---|---|
| `STRATA_GLM_ABLATE` | Path to a raw float32 file |
| `STRATA_GLM_ABLATE_LAYERS` | `45`: the file is 45 x K x 4096 values, each layer has its own K directions. A layer of zeros is not touched. Unset: the file is K x 4096 values for the outputs mode |
| `STRATA_GLM_ABLATE_SCALE` | Strength, default 1 |

My file has 3 directions per layer for layers 1 to 44 (2,211,840 bytes, SHA256 `b0debc326b06c956039fd75dc1cb759dd46c4aa5e868bbfef1a5b0607c3552b7`).

## Measurements

Rig: 4x RTX 3090 at 300 W, PCIe 3.0, EPYC 7642, 2 memory channels of DDR4-2400. Maya-M, 256K context, text only, after `./maya.sh --calibrate`.

Refusals, 178 answers (64 held-out AdvBench and 82 JailbreakBench requests without thinking, 32 with thinking), judged by the stock GLM-5.3-Flash:

| | Refused | Disputes the premise | Cut off while thinking |
|---|---:|---:|---:|
| Maya-M with the vector | 2 | 4 | 0 |

The 2 refusals are requests about suicide and self-harm that the model turns into text about prevention. I left them as they are. The directions were found on the EXL3 2.05 bpw quant of the same model, where the full run (146 requests in 3 modes) gives 0 refused of 438 answers. How they were found: [abliteration write-up](https://github.com/alesha-pro/exllamav3/blob/glm53-4x3090/glm53/abliteration/README.md).

Speed through the chat API (answers run to their natural end, thinking off, temperature 0):

| | Without the vector | With the vector |
|---|---|---|
| Decode, 1 request | 58.4 tokens/s | 56.1 |
| Decode at 32K depth | 57.2 | 55.9 |
| Prefill 8K | 1078 to 1113 tokens/s | 1119 to 1153 |
| Prefill 32K | 1860 to 1896 | 1938 to 1953 |

## Limits

- Tested on Maya-M only, on 4 NVIDIA cards, text only. In my build the vision encoder did not link (CMake picked the system CUDA 11), so I did not run images with the patch.
- One rig. I did not test the AMD path, Windows or other GPU counts.
- The judge is the same model without the vector. There are no human labels.
- The patch is not sent upstream.
