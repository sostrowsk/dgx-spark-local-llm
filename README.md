# DGX Spark — Local LLM Setup and Benchmarks

Measured inference performance of three MoE models on an **NVIDIA DGX Spark (GB10, sm_121a)**,
comparing **vLLM against SGLang** and **NVFP4 against INT4**.

Everything here was actually run on the hardware — 14 server starts, of which 6 failed, all
documented including the failures. No estimates, no extrapolations except where explicitly marked.

> The detailed write-ups in `handouts/` are in German. This README, the code and all data are
> language-neutral or English.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1 (`sm_121a`) |
| Memory | 121 GB unified, shared between CPU and GPU, ~273 GB/s |
| CPU | 20 cores, aarch64 |
| OS | Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia |
| Driver / CUDA | 580.173.02 / 13.0 |
| Engines | vLLM 0.25.1, SGLang 0.5.16, both on torch 2.11.0+cu130 |

## Headline results

Total throughput in tok/s across all concurrent requests, median of 3 runs:

| Configuration | active params | decode c=1 | decode c=4 | best TTFT | KV cache |
|---|---|---|---|---|---|
| Laguna-S-2.1 NVFP4 | 8 B | 17.3 | 42.7 | 0.352 s | 94,135 |
| Laguna-S-2.1 INT4 (vLLM) | 8 B | 20.6 | 52.8 | 0.260 s | 1,001,532 |
| Laguna-S-2.1 INT4 (SGLang) | 8 B | 18.2 | 45.3 | 0.193 s | 185,897 |
| **Qwen3.6-35B-A3B NVFP4** | 3 B | **60.7** | **201.2** | 0.094 s | 3,366,051 |
| **Gemma-4-26B-A4B NVFP4** | 4 B | 53.0 | 170.6 | **0.036 s** | **3,912,140** |
| Gemma-4-26B-A4B INT4 (QAT) | 4 B | 43.9 | 177.1 | 0.038 s | 3,875,400 |

## What the data says

**Active parameters dominate everything else.** 8 B active → 20.6 tok/s, 4 B → 53.0, 3 B → 60.7.
Notably, Gemma-4 reaches 2.6× Laguna's speed with half the active parameters *and without any
speculative decoding*, while Laguna uses DFlash.

**Prefix caching depends on the attention architecture, not on the engine.**

| Model | Attention | cold (16k) | warm | speed-up |
|---|---|---|---|---|
| Laguna-S-2.1 | sliding window 512 | 9.32 s | 0.30 s | ~30× |
| Gemma-4-26B | sliding window 1024 | 3.13 s | 0.093 s | ~34× |
| Qwen3.6-35B | linear attention | 3.09 s | 2.81 s | **1.0×** |

Qwen3.6 shows **no prefix cache effect at all** — warm and cold are identical. Linear attention
carries a sequentially updated state that cannot be reused piecewise like a KV cache. For an agent
carrying the same 16k context across 20 turns, that is 56 s versus 5 s in total.

**NVFP4 vs INT4 is load-dependent, and memory pressure confounds it.** On Laguna, INT4 won by 19–246 %
— but only because NVFP4 weighed 95.63 GiB of 121, leaving 6.97 GiB of KV cache and 1.44 concurrent
full-context requests. On Gemma-4, where both variants are within 0.75 GiB of each other, the picture
inverts with load:

| concurrency | NVFP4 | INT4 | |
|---|---|---|---|
| 1 | **53.0** | 43.9 | NVFP4 +17 % |
| 2 | **101.0** | 94.9 | NVFP4 +6 % |
| 4 | 170.6 | **177.1** | INT4 +4 % |

vLLM logs the kernel path it picks: `FLASHINFER_CUTLASS` for NVFP4, `MARLIN` for INT4. Marlin
amortises its unpacking cost over the batch and holds 44–47 tok/s across all load levels, while
FlashInfer-CUTLASS is already near optimum at a single stream and falls from 53.5 to 43.2.

**vLLM and SGLang split the disciplines** (measured on Laguna INT4, both with DFlash):

| | vLLM | SGLang |
|---|---|---|
| Throughput c=4 | **52.8** | 45.3 |
| TTFT c=4 | 0.513 s | **0.193 s** |
| Repeated prefix c=4 | 39.5 | **70.6** |
| KV cache | **950,420** | 185,897 |
| Cold prefill 16k | **2243** | 1686 |

vLLM's throughput lead *grows* with concurrency (13 % → 17 %), contrary to the common expectation
that SGLang scales better. SGLang wins decisively where RadixAttention applies: repeated prefixes.

## Pitfalls worth knowing

These cost the most time and are the reason this repo exists.

| Pitfall | Symptom | Fix |
|---|---|---|
| `MAX_JOBS` too high | OOM killer terminates `cicc` (7.5 GB each), **no error in the application log** | `MAX_JOBS=1` on first start; kernel cache is warm afterwards |
| `gpu-memory-utilization` > 0.89 | abort before loading | the OS holds ~11 GiB that must stay free |
| `mem-fraction-static` misread | a smaller model gives you **no** extra system headroom | it reserves a fraction of total memory regardless of model size |
| First request treated as a measurement | 3.8 vs 75 tok/s — off by 20× | run an explicit warm-up |
| Output in the `reasoning` field | TTFT never set, decode rate computed over total time | check `content`, `reasoning` *and* `reasoning_content` |
| tqdm ETA on non-uniform work | showed 1:48:53, actually 3:33 | watch the rate over several steps |
| HF API `size` fields missing | 28 GB under-counted | verify the response is complete |
| Prefix cache silently ineffective | "warm" values identical to cold, no error | verify against a true cold measurement with a unique prefix |
| Wrong CUDA wheel | missing sm_121a kernels on cu12x builds | on the Spark use `-cu130` on aarch64 only |
| Missing Python headers | Triton JIT fails without `python3.12-dev` | `uv python install 3.12` ships them, no sudo needed |

One claim circulating about this hardware is **wrong**: that `sgl_kernel`'s prebuilt wheels lack
sm_121a and require a source rebuild. Verified with `cuobjdump` on the installed binaries:

```
common_ops.abi3.so   80M   sm_90 sm_90a sm_100a sm_103a sm_110a sm_120a sm_121a
flash_ops.abi3.so   309M   sm_90a                    # FA3, Hopper-only — irrelevant on GB10
```

SGLang ran without any rebuild.

## Reproducing

```bash
# Toolchain — no sudo needed, uv ships Python with dev headers
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv venv-vllm -p 3.12
uv pip install -p venv-vllm vllm==0.25.1 --torch-backend=cu130

# SGLang, if you want the engine comparison
uv venv venv-sglang -p 3.12
uv pip install -p venv-sglang sgl-kernel --prerelease=allow \
  --index-url https://sgl-project.github.io/whl/cu130/
uv pip install -p venv-sglang sglang --prerelease=allow
# sglang 0.5.16 pins torch 2.11.0 — do NOT force-reinstall a newer one

# Start a server (Gemma-4 is the fastest to get running)
./start-vllm-gemma4.sh nvfp4

# Measure
python3 bench.py    http://localhost:8000 <model-id> <label> results.json
python3 prefill.py  http://localhost:8000 <model-id> <label> prefill.json
```

Both scripts use only the Python standard library and speak the OpenAI-compatible API that vLLM and
SGLang both serve, so every configuration receives identical requests.

## Repository layout

```
bench.py               throughput and TTFT: 4 scenarios × 3 concurrency levels × 3 repeats
prefill.py             cold prefill across length stages, unique UUID prefix per request
start-vllm*.sh         vLLM launch scripts per model and quantization
start-sglang*.sh       SGLang launch scripts
ergebnisse_*.json      raw results, 12 measurement points each
prefill_*.json         raw prefill results
*.log                  server logs of all 14 starts, including the failed ones
handouts/              four detailed write-ups (German, self-contained HTML)
```

The `handouts/` directory holds the full story: one document per model, plus a synthesis across all
six configurations. They are standalone HTML with no external dependencies — open them directly in a
browser.

## Caveats

- Response **quality was not measured**. The QAT advantage of the Gemma-4 INT4 checkpoint appears in
  none of these numbers, nor does any quality difference between models.
- Laguna and Qwen3.6 emit reasoning tokens that the parser did not separate, so their code-vs-prose
  comparison is meaningless. Only Gemma-4 produced plain answers.
- Concurrency was only measured up to 4. Gemma-4 INT4 still retains 103 % of its single-stream rate
  there, so the ceiling is untested.
- One measurement series per configuration, three repeats per point, no averaging across sessions.
  A spot check showed 1.4 % deviation between two runs of the same configuration.
- `--mem-fraction-static 0.85` (SGLang) and `--gpu-memory-utilization 0.89` (vLLM) are not exactly
  equivalent; part of the KV cache gap between the engines follows from that.

## License

MIT for the code in this repository. The models themselves carry their own licenses.
