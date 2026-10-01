# Target: GLM-5.3-Flash (EXL3 3.05 bpw) on one 24 GB GPU, all routed experts in RAM, AVX2 CPU tier

- Weights: `turboderp/GLM-5.3-Flash-exl3`, branch `3.05bpw`, revision `332ab457b709b7ba30dd9a448be5de03b80a7ac9`
  (125.3 GB / 117 GiB: 15 shards + `mtp.safetensors` + `kpool_aux.safetensors`; read once at load).
- Model: GLM-5.3-Flash, 45 layers (42 MoE x 288 routed experts, 9.44 MB each at 3.05 bpw), KDA linear attention +
  DSA sparse attention, mHC hyper-connections.
- Engine: stock exllamav3 1.5.1 + [glm53-flash-offload](https://github.com/0xSero/glm53-flash-offload) (monkeypatches
  and small kernels; OpenAI-compatible server with SGLang-shaped `/generate` + `/tokenize` for the kit tools).
- How it fits: every routed expert (114.2 GB) lives in pinned host RAM and is readable by the GPU over PCIe
  (zero-copy); an elastic CLOCK cache keeps hot experts in all free VRAM (~1,376 slots, 13 GB); in decode an AVX2
  kernel computes the coldest cache misses on the host cores from a second, block-contiguous RAM copy (114.2 GB);
  prefill stages each layer's non-cached experts into VRAM one layer ahead. KV fp16, 131,072 tokens.
- Registry recipe (authoritative launch): local-ai-registry
  [`rtx-3090-24gb/glm-5.3-flash.exllamav3.128k`](https://github.com/0xSero/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3.128k.json)
  (status: candidate, `reported` proof, until the CPU-tier decode-KL fix below lands).

## Needs (measured on the reference machine; your probe must meet these)
| | RTX 3090 24 GB, `GLM53_MODE=fast` (default) | `GLM53_MODE=exact` |
|---|---|---|
| VRAM | the whole card (the expert cache takes all free VRAM; ~22.7 GiB at ready) | same |
| Free system RAM | **~218 GiB / 234 GB** (MemAvailable drop at ready 213.2-218.3 GiB; RSS 217 GiB: two 114.2 GB expert copies). A single-copy fast mode (~170 GB) is in progress | **~111 GiB / 119 GB** (one copy; the low-RAM option) |
| CPU | x86-64 with AVX2 + FMA + F16C, ~24 physical cores (reference: EPYC 7443P, 22 tier threads) | any |
| Memory bandwidth | 8-channel DDR4 or better recommended (CPU tier ~80 GB/s + PCIe ~25 GB/s from the same DRAM) | - |
| Disk | 126 GB free, local NVMe recommended (load ~2 min from page cache) | same |
| PCIe | 4.0 x16 (cache misses are read over the link, ~25 GB/s) | same |

## Known-good config
### RTX 3090 (sm_86) — `rtx-3090-24gb`
- Image: `ghcr.io/0xsero/glm53-flash-offload@sha256:1159044a73d91804c25ffa0851d1668518220c631e2efd4e904b7cc0578a7e39` (glm53-flash-offload [`c4b9160`](https://github.com/0xSero/glm53-flash-offload/tree/c4b9160bdbd3bdbda9b2d50b763332c8580280fb), exllamav3 v1.5.1 for
  sm_86; built and attested by github.com/0xSero/local-ai-images `release-image`).
- Launch (bridge network; the image entrypoint holds the defaults below):
  ```
  docker run -d --name glm53 --gpus '"device=0"' --ulimit memlock=-1 --shm-size 16g -p 30000:30000 \
    -v <models dir>/GLM-5.3-Flash-exl3-3.05bpw:/models ghcr.io/0xsero/glm53-flash-offload@sha256:1159044a73d91804c25ffa0851d1668518220c631e2efd4e904b7cc0578a7e39
  ```
  Server argv (from the entrypoint): `python3 /opt/glm53/glm53/serve.py -m /models -cs 131072 --max-batch-size 8
  -chunk_size 8192 -ambs 4 --host 0.0.0.0 --port 30000 --served-name glm-5.3-flash`
- Env (fast-mode defaults = campaign run G067): `GLM53_ZC_VRAM=0 GLM53_EC=1 GLM53_EC_RESERVE_GB=1.5
  GLM53_EC_STAGE_GB=2.6 GLM53_EC_ELASTIC_GB=10 GLM53_CPU_TIER=1 GLM53_CT_CPUS=auto GLM53_K_HCFUSE=1 GLM53_K_FTSPLIT=1
  GLM53_K_OVL=1 GLM53_TRITON_PIN=/opt/glm53/data/triton_pin_exact.json` (+ routing stats and tune cache from the image).
- Reference result (EPYC 7443P, 8ch DDR4, PCIe 4.0 x16, 990 Pro; protocol sweep `--template glm --prefill 8192 32768
  --conc 1 2 4`): prefill 8k 710 / 32k 951 tok/s; decode 1 user 28.15, 2 users 31.57, 4 users 33.98 tok/s aggregate,
  1 user at 32k context 26.89. Exact mode (G066a): prefill 697 / 945, decode ~12.6 at 1-4 users.

## Quality band (reference panel `reference/glm-5.3-flash-exl3-ref-panel.json`, `tools/score_ref_panel.py`)
- Teacher-forced panel (8 prompts, 2,154 positions; prefill path) in both modes: **top-1 1.0000, mean KL 0**
  (exact). Anything below 1.0000 means a different Triton autotune pick set or a broken cache: check that
  `GLM53_TRITON_PIN` is set (a different pick set measured 0.989 / 0.0034).
- Decode in `fast` mode is **not exact** and the panel does not see it: a paired decode check (tier off vs on, same
  tokens, 1,288 positions) measured mean KL 0.0053 (p99 0.086), top-1 0.977 vs the exact GPU-only path. The CPU
  kernel alone is ~1e-3 relative error per expert (like exllamav3's own GPU MoE kernel); a decode-path bug is
  suspected and a fix will ship as a new image digest. Need exllamav3-identical output: `GLM53_MODE=exact`.

## Safe knobs to tune
- `GLM53_CT_CPUS` / `GLM53_CT_THREADS` (`auto` = one logical CPU per physical core minus the first 2; give the tier one
  thread per physical core, SMT does not help) and `GLM53_CT_B` (ms per CPU expert at your thread count, about
  `0.104 x 22 / threads`; the cost model splits misses between CPU and PCIe with it).
- `GLM53_MODE=exact` (or `GLM53_CPU_TIER=0`): no CPU tier, ~111 GiB RAM, bit-exact, ~12.6 tok/s decode.
- `-ambs N` (appended after the image name): recurrent-cache slots = concurrent decode streams; `-ambs 1` serialises
  users and frees a little VRAM for the expert cache.
- `GLM53_EC_STAGE_GB` (2.6; 2.0 loses ~12 % prefill, more than 2.6 gives nothing), `GLM53_EC_RESERVE_GB` (1.5; lower =
  more cache slots, risk of OOM on long prefill).
- `-cs` (KV tokens, 131072; 65536 gives ~100 more expert-cache slots, +2 % decode).
Anything else: describe it in the PR.

## Benchmark notes
`tools/sweep.py --card rtx-3090-24gb/glm-5.3-flash-offload --template glm --prefill 8192 32768 --conc 1 2 4` (the GLM
target's best-known key; GLM prompts need `--template glm`). Prefill sizes 8192 and 32768 are required for this
target.
