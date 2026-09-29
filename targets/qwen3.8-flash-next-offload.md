# Target: Qwen3.8-Flash-Next (EXL3 3.05 bpw) on one consumer GPU, experts in RAM, n-gram table on NVMe

- Weights: `turboderp/Qwen3.8-Flash-Next-exl3`, branch `3.05bpw_h5_ng5`, revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`
  (85.14 GB: 7 shards 52.3 GB read once at load + `ngram_embedding.safetensors` 32.64 GB read randomly while serving).
- Model: 180B MoE, 6B active (48 layers, 512 routed experts, top-10), 5-bit trellis n-gram table.
  Artificial Analysis Intelligence Index v4.3: 40 (reasoning), 2026-09-29.
- Engine: SGLang 0.5.20 + the trellis-serve EXL3 plugin (github.com/0xSero/trellis-serve).
- How it fits: 45.8 GB of routed experts live in pinned host RAM behind a GPU expert cache (sized automatically from
  free VRAM); the 32.6 GB n-gram table is read from NVMe through an 8 GB RAM row cache; KV fp8.

## Needs (measured on the reference machines — your probe must meet these)
| | RTX 3090 24 GB | Arc Pro B70 32 GB |
|---|---|---|
| VRAM | the whole card (21,562 MiB at ready, 24,026 MiB peak with a 4096² image) | the whole card (31.85 GiB peak) |
| Free system RAM | 75 GB (MemAvailable drop at ready; 55.0 GB of it page-locked) | 68 GB (peak host total 67.26 GB) |
| Disk | 86 GB free on a local NVMe SSD (random reads while serving; no network FS / HDD) | same |
| PCIe | 4.0 x16 measured 26.2 GB/s host→device; a rented 3090 at 14-16 GB/s decoded 15 % slower | 4.0 x16, 28.2 GB/s |

No NVMe but ≥ 100 GB free RAM: set `SGLANG_EXL3_NGRAM_TIER=pinned` (table in RAM, +24.6 GB RAM) — a different config,
measure it as such.

## Known-good configs
### RTX 3090 (sm_86) — `rtx-3090-24gb`
- Image: `ghcr.io/0xsero/sglang-exl3-flashnext@sha256:95b1a85d53812ea73e58638d545a9b8d2a5c46d2494bf6cd4baf97f3b6440dda`
  (text; a vision build is being published — check `local-ai-registry` recipe `rtx-3090-24gb/qwen3.8-flash-next.sglang.200k`
  for the current digest).
- Launch (bridge network, `--shm-size 64g`, `-v <models dir>:/models:ro`, `--entrypoint /opt/entrypoint.sh`):
  `python3 -m sglang.launch_server --model-path /models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 --quantization exl3
  --trust-remote-code --host 0.0.0.0 --port 30100 --served-model-name flashnext --disable-shared-experts-fusion
  --kv-cache-dtype fp8_e4m3 --context-length 204800 --mem-fraction-static 0.88 --chunked-prefill-size 8192
  --max-running-requests 4 --cuda-graph-max-bs-decode 8 --cuda-graph-backend-prefill disabled --max-mamba-cache-size 16
  --max-total-tokens 210000`
- Env: `SGLANG_EXL3_MODEL_PATH=/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 SGLANG_EXL3_MOE_OFFLOAD=gpu_cache
  SGLANG_EXL3_EXPERT_CACHE_GB=auto SGLANG_EXL3_EMBED_HOST=1 SGLANG_EXL3_OFFLOAD_STAGING_PARTS=4 SGLANG_EXL3_OFFLOAD_FUSED=1
  SGLANG_EXL3_OFFLOAD_COMPACT_PARTS=1 SGLANG_EXL3_MOE_PREFILL_FP16_ACC=1 SGLANG_EXL3_NGRAM_TIER=nvme HF_HUB_OFFLINE=1`
- Reference result (EPYC 7443P, PCIe 4.0, 990 Pro; protocol sweep): prefill 8k 2,627 / 16k 2,785 / 32k 2,846 /
  64k 2,824 tok/s; decode 1 user 62.7, 2 users 77.4, 3 users 84.8 (max), 4 users 79.3 tok/s aggregate; KV pool
  209,984 tokens shared, one session up to 204,800.

### Arc Pro B70 (Battlemage) — `intel-arc-pro-b70-32gb`
- Image: `ghcr.io/0xsero/sglang-exl3-xpu-flashnext@sha256:26f5b8e8b372c4401f25ef93306c064f5c9b8d64e4ec99775b375dcafdf9f3de`
- Launch: see `local-ai-registry` recipe `intel-arc-pro-b70-32gb/qwen3.8-flash-next.sglang.256k` (EXL3_NGRAM_TIER=nvme,
  9,000 expert slots, KV 270,336 fp8, XPU decode graphs bs 1-2).
- Reference result: prefill 8k 1,023 / 16k 946 / 32k 800 / 64k 604 tok/s; decode 1 user 32.8 tok/s; serves one
  request at a time; one session up to 262,144 tokens.

## Quality band (reference panel, `tools/score_ref_panel.py`)
top-1 agreement ≥ 0.987 and mean KL ≤ 0.0013 vs the exllamav3 reference (reference runs: 0.988-0.993 / 0.00090-0.00124).

## Safe knobs to tune
`SGLANG_EXL3_EXPERT_CACHE_GB` (auto or a number; bigger = faster decode, but must leave room for prefill activations),
`--chunked-prefill-size` (8192 default; 16384 needs ~2 GB more VRAM), `--max-mamba-cache-size` / `--max-running-requests`
(more concurrent users cost cache), `SGLANG_EXL3_NGRAM_TIER` (nvme | pinned), `SGLANG_EXL3_NGRAM_RAM_GB` (8 default).
Anything else: describe it in the PR.
