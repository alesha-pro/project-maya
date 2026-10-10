#!/usr/bin/env python3
"""Turn the refusal projection on or off in a Maya config written by the installer (maya-<model>.json).

    tools/glm_ablate_config.py maya-maya-m.json /path/to/glm53-refusal-per-block-3dir.layers45.f32
    tools/glm_ablate_config.py maya-maya-m.json --off

The engine reads the settings from the "env" section of the config when the server starts it:
STRATA_GLM_ABLATE (the vector file) and STRATA_GLM_ABLATE_LAYERS (45: the file holds its own directions for
every layer). Restart Maya after changing the config.
"""
import json, os, sys

if len(sys.argv) != 3:
    sys.exit(__doc__)
path, vec = sys.argv[1], sys.argv[2]
cfg = json.load(open(path))
env = dict(cfg.get("env") or {})
for k in ("STRATA_GLM_ABLATE", "STRATA_GLM_ABLATE_LAYERS", "STRATA_GLM_ABLATE_SCALE"):
    env.pop(k, None)
if vec != "--off":
    vec = os.path.abspath(vec)
    n = os.path.getsize(vec) // 4
    assert n % 4096 == 0, "the file is not a float32 array of 4096-wide vectors"
    env["STRATA_GLM_ABLATE"] = vec
    if ".layers" in os.path.basename(vec):   # per-layer file: <name>.layers<N>.f32
        env["STRATA_GLM_ABLATE_LAYERS"] = os.path.basename(vec).split(".layers")[1].split(".")[0]
cfg["env"] = env
json.dump(cfg, open(path, "w"), indent=1)
print("projection", "off" if vec == "--off" else f"on: {vec}", "| env:", env)
