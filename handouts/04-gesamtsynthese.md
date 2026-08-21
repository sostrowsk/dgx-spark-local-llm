# LLM inference on DGX Spark GB10

*Test report · Inference benchmark · Part 4 — Synthesis*

> Six server configurations, three models, two engines, two quantization formats — on the same hardware, with the same harness and the same prompts. What that teaches us about the machine.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB unified memory · 273 GB/s **Configurations** 6 **Measurement points** 72 plus 22 prefill stages **Server starts** 14 · 6 of them failed

## Contents

1. [At a glance](#at-a-glance)
2. [The five key findings](#the-five-key-findings)
3. [The six configurations](#the-six-configurations)
4. [Rankings](#rankings)
5. [Active parameters](#active-parameters)
6. [vLLM against SGLang](#vllm-against-sglang)
7. [INT4 against NVFP4](#int4-against-nvfp4)
8. [The prefix cache](#the-prefix-cache)
9. [Prefill scaling](#prefill-scaling)
10. [Memory as the main variable](#memory-as-the-main-variable)
11. [Complete measurement data](#complete-measurement-data)
12. [Transferable pitfalls](#transferable-pitfalls)
13. [Refuted assumptions](#refuted-assumptions)
14. [Recommendation matrix](#recommendation-matrix)
15. [Methodology](#methodology)
16. [Open questions](#open-questions)
17. [Glossary](#glossary)
18. [Individual reports](#individual-reports)

## At a glance

The test series began with two assumptions, both of which were wrong: that NVFP4 was the right quantization for this hardware, and that SGLang would fail on it for lack of kernels. What counts in the end is a third thing — the number of active parameters, and whether the context can be reused.

### Active parameters decide

8 B → 20.6 tok/s, 4 B → 53.0, 3 B → 60.7. The number after the “A” in the model name says more about speed than format, engine and total size combined.

### The prefix cache depends on the architecture

Sliding window: a factor of 30 to 34. Linear attention: a factor of 1.0 — no effect at all. For agents with a growing history that is the single biggest lever.

### INT4 against NVFP4 is load-dependent

Under memory pressure the smaller format wins. Without memory pressure NVFP4 wins by 17 % at low load, and INT4 draws level from four streams on.

### vLLM and SGLang split the disciplines

vLLM: +17 % throughput, five times the KV cache. SGLang: three times better latency, +79 % on a recurring prefix.

- **201.2** — tok/s best value Qwen3.6, code, c=4

- **0.036** — seconds TTFT Gemma-4

- **3.9 M** — tokens KV cache Gemma-4

- **9.8×** — spread between best and worst configuration

## The five key findings

### 1. At 273 GB/s, what counts is how many bytes are moved per token

Everything else follows from that. Active parameters determine decode speed, the weight size determines how much memory is left for the KV cache, and the KV cache determines how many users can be served concurrently. Compute formats and kernel paths act only in second order.

### 2. The prefix cache is the second most important property

An agent carrying the same 16k context across twenty turns pays full price every time on a model with linear attention. On sliding-window models the cost drops by a factor of 30 to 34 after the first time. No change of engine and no quantization format compensates for that.

### 3. Memory pressure distorts every format comparison

The Laguna test appeared to show that INT4 beats NVFP4 — what it actually showed was that a model occupying 95.63 of 121 GiB leaves hardly any KV cache. Only the Gemma-4 test, with variants of practically equal size, measures the formats themselves.

### 4. Choosing an engine is not a performance question but a profile question

vLLM and SGLang are 13 to 17 % apart on throughput — noticeable, but small next to the factor of 3 between the models. What is more interesting is that they win different disciplines: throughput and memory against latency and prefix reuse.

### 5. Most pitfalls are measurement errors, not hardware problems

Of the documented stumbling blocks, the most consequential concerned not the machine but the observation: an extrapolated remaining time taken for a measurement, the first request taken as representative, an output field overlooked, incomplete API data counted as complete.

## The six configurations

| # | Model | Format | Engine | active | Weights | Speculation |
|---|---|---|---|---|---|---|
| 1 | Laguna-S-2.1 | NVFP4 | vLLM | 8 B | 95.63 GiB | DFlash |
| 2 | Laguna-S-2.1 | INT4 | vLLM | 8 B | 69.34 GiB | DFlash |
| 3 | Laguna-S-2.1 | INT4 | SGLang | 8 B | 67.56 GB | DFlash |
| 4 | Qwen3.6-35B-A3B | NVFP4 | vLLM | 3 B | 23.45 GiB | MTP |
| 5 | Gemma-4-26B-A4B | NVFP4 | vLLM | 4 B | 15.88 GiB | none |
| 6 | Gemma-4-26B-A4B | INT4 (QAT) | vLLM | 4 B | 16.63 GiB | none |

All three models are mixture-of-experts architectures with hybrid attention and multimodal capabilities that went unused in these tests. All support a 262,144-token context.

> [!IMPORTANT]
> **On comparability:** Laguna and Qwen3.6 use speculative decoding (DFlash and MTP respectively), Gemma-4 does not — its checkpoints ship no speculation head. Gemma-4 therefore reaches its values without that multiplier, which makes its position in the rankings all the more notable.

## Rankings

*Best measured value per discipline*

| Discipline | Best value | Configuration | Worst value | Spread |
|---|---|---|---|---|
| Throughput c=4 | 201.2 tok/s | Qwen3.6 NVFP4 | 42.4 | 4.7× |
| Throughput c=1 | 63.5 tok/s | Qwen3.6 NVFP4 | 15.0 | 4.2× |
| TTFT | 0.036 s | Gemma-4 NVFP4 | 0.352 s | 9.8× |
| KV cache | 3,912,140 | Gemma-4 NVFP4 | 94,135 | 41.6× |
| Cold prefill 16k | 5301 tok/s | Qwen3.6 NVFP4 | 1686 | 3.1× |
| Prefix cache factor | ≈ 34 | Gemma-4 | 1.0 | 34× |
| Repeated prefix c=4 | 141.1 tok/s | Gemma-4 NVFP4 | 17.6 | 8.0× |

The spread between the best and worst configuration is nearly tenfold on latency and more than fortyfold on KV cache. **Laguna takes first place in no discipline** — which says nothing about model quality, something this series does not measure.

## Active parameters

*Decode on a single stream, best variant of each model*

| Model | active | Decode c=1 | c=4 total | Speculation |
|---|---|---|---|---|
| Laguna-S-2.1 | 8 B | 20.6 | 52.8 | DFlash |
| Gemma-4-26B | 4 B | 53.0 | 177.1 | none |
| Qwen3.6-35B | 3 B | 60.7 | 201.2 | MTP |

The relationship is clear but not exactly proportional. What stands out is the middle row: **Gemma-4 reaches 2.6× the speed of Laguna with half as many active parameters — and does so without any speculative decoding**, while Laguna works with DFlash. Factor out the speculation advantage and the link between active parameters and speed is tighter still than the raw numbers show.

### Scaling over concurrency

*Share of the single-stream rate retained at four concurrent streams*

| Configuration | c=1 | c=4 | retained |
|---|---|---|---|
| Laguna INT4 | 21.2 | 13.7 | 65 % |
| Qwen3.6 NVFP4 | 62.2 | 47.8 | 77 % |
| Gemma-4 NVFP4 | 53.5 | 43.2 | 81 % |
| Gemma-4 INT4 | 44.1 | 45.5 | 103 % |

The fewer the active parameters, the further the machine sits from the bandwidth ceiling and the better it scales over parallelism. At four streams Gemma-4 INT4 even gets faster per stream than at one — a batching effect of the Marlin kernels.

## vLLM against SGLang

Tested on Laguna-S-2.1-INT4, both with DFlash speculation, both on torch 2.11.0+cu130.

| Discipline | vLLM 0.25.1 | SGLang 0.5.16 | Lead |
|---|---|---|---|
| Prose c=1 | 20.6 | 18.2 | vLLM +13 % |
| Prose c=2 | 31.7 | 27.8 | vLLM +14 % |
| Prose c=4 | 52.8 | 45.3 | vLLM +17 % |
| Code c=4 | 64.3 | 56.2 | vLLM +14 % |
| TTFT c=2 | 0.635 s | 0.201 s | SGLang 3.2× |
| TTFT c=4 | 0.513 s | 0.193 s | SGLang 2.7× |
| Repeated prefix c=4 | 39.5 | 70.6 | SGLang +79 % |
| KV cache | 950,420 | 185,897 | vLLM 5.1× |
| Cold prefill 16k | 2243 | 1686 | vLLM +33 % |

### What the numbers mean

**vLLM wins on throughput, and the lead grows with concurrency** — the opposite of the common expectation that SGLang scales better over parallelism.

**SGLang wins on latency, clearly and stably.** TTFT sits between 0.19 and 0.29 s almost independently of load and prompt length, while vLLM scatters up to 0.78 s.

**On a recurring prefix the picture inverts.** SGLang reaches 70.6 against vLLM's 39.5 tok/s — in the scenario before it, on first contact with that same prefix, it was the other way round (38.0 against 64.6). RadixAttention therefore delivers in exactly the predicted case.

<details>
<summary>The refuted claim about sm_121a</summary>

The starting point was the claim that SGLang fails on GB10 because the prebuilt `sgl_kernel` wheels contain no sm_121a kernels and only a rebuild from source helps. Checked in two stages:

**Source** — in `sgl-kernel/CMakeLists.txt` the gencode is emitted when CUDA ≥ 13.0 *and* aarch64 apply. Precisely the cu130 build on this machine.

**Binary** — `cuobjdump` over the installed libraries:

```bash
common_ops.abi3.so   80M   sm_90 sm_90a sm_100a sm_103a sm_110a sm_120a sm_121a
spatial_ops.abi3.so  196K  sm_90 sm_90a sm_100a sm_103a sm_110a sm_120a sm_121a
flashmla_ops.abi3.so 12M   sm_90a sm_100a sm_103a          — no sm_121
flash_ops.abi3.so    309M  sm_90a                          — FA3, Hopper-exclusive
```

The main kernel library contains sm_121a. That FA3 and FlashMLA are missing is correct and irrelevant — neither is usable on GB10 anyway. SGLang ran without any source build.

</details>

## INT4 against NVFP4

Two tests, two opposite results — and the difference lies not in the format but in the boundary conditions.

| Test case | Weights NVFP4 / INT4 | KV cache NVFP4 / INT4 | Result |
|---|---|---|---|
| Laguna-S-2.1 | 95.63 / 69.34 GiB | 94k / 950k | **INT4**, +19 to +246 % |
| Gemma-4-26B | 15.88 / 16.63 GiB | 3.91 / 3.88 M | **load-dependent** |

### The Laguna case: memory decides

NVFP4 occupied 95.63 of 121 GiB, leaving only 6.97 GiB for the KV cache — arithmetically **1.44 concurrent requests** at full context. The checkpoint also consumed twice as much KV memory per token (77.7 against 38.8 KiB), which points to a build fault. INT4 won there not because of the compute path but because it was 26 GiB smaller.

### The Gemma-4 case: the kernel path becomes visible

The two variants differ by 0.75 GiB of weights and 0.9 % of KV cache. What remains is logged by vLLM at startup:

```bash
NVFP4: Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
INT4:  Using 'MARLIN' WNA16 MoE backend
```

*Gemma-4 · total throughput and decode rate per stream, prose*

| c | NVFP4 total | INT4 total | NVFP4 per stream | INT4 per stream |
|---|---|---|---|---|
| 1 | 53.0 | 43.9 | 53.5 | 44.1 |
| 2 | 101.0 | 94.9 | 50.9 | 46.7 |
| 4 | 170.6 | 177.1 | 43.2 | 45.5 |

**Marlin amortises its unpacking cost over the batch** and holds 44–47 tok/s across all load levels. **FlashInfer-CUTLASS already works near its optimum at a single stream** and loses accordingly as load rises.

On cold prefill — the most compute-intensive case — NVFP4 stays ahead throughout, with a shrinking margin: 28 % at 4k tokens, 22 % at 16k, 12 % at 65k, 6 % at 132k.

> [!NOTE]
> **Rule of thumb**
>
> If memory is tight, file size decides — regardless of format. If it is not, NVFP4 wins by around 17 % at low load and draws level from four concurrent streams on. The blanket claim “NVFP4 is native on Blackwell and therefore faster” holds only in the partial-load range.

## The prefix cache

The most consequential finding of the series, and it depends neither on the engine nor on the format but on the model's attention architecture.

*Time to first token at a ~16k-token prompt*

| Model | Attention | cold | warm | Factor |
|---|---|---|---|---|
| Laguna-S-2.1 | sliding window 512 | 9.32 s | 0.30 s | ≈ 30 |
| Gemma-4-26B | sliding window 1024 | 3.13 s | 0.093 s | ≈ 34 |
| Qwen3.6-35B | linear attention | 3.09 s | 2.81 s | 1.0 |

With Qwen3.6 the warm and cold runs are **identical**. The evidence is twofold: the measured cold rate of 5301 tok/s works out to 3.09 s for 16,384 tokens — and 2.81 s was measured in the main run. There never was a cache hit. On top of that, TTFT there rose linearly with concurrency (2.8 / 4.4 / 7.2 s), while for the other two it stayed nearly constant.

**Attempted explanation:** linear attention carries a sequentially updated state that cannot be addressed and reused piecewise the way a KV cache can. The hypothesis arose in the Qwen test and was supported by the counter-check in the Gemma-4 test — it is not directly demonstrated.

> [!IMPORTANT]
> **Practical scope**
>
> An agent carrying the same 16k context across twenty turns pays 2.8 s every time with Qwen3.6. With Gemma-4 it is 0.093 s after the first time. Over twenty turns that adds up to 56 against 5 seconds — a difference no throughput advantage makes up for.

## Prefill scaling

All three models successfully dampen the quadratic attention term — Laguna and Gemma-4 via sliding windows, Qwen3.6 via linear attention.

*Cold prefill in tok/s, unique prefix per request*

| Prompt tokens | Laguna INT4 | Qwen3.6 | Gemma-4 NVFP4 |
|---|---|---|---|
| ≈ 4,000 | 2255 | 5064 | 7161 |
| ≈ 16,000 | 2301 | 5301 | 5238 |
| ≈ 65,000 | 1850 | 4033 | 2396 |
| ≈ 132,000 | 1484 | 3060 | 1416 |
| ≈ 200,000 | 1224 | 2429 | — |

*Complexity, derived from the ratio of length growth to time growth*

| Model | Complexity | Rate at 132k | Architecture |
|---|---|---|---|
| Laguna | O(n1.25) | 64 % | window 512, 12 of 48 full |
| Qwen3.6 | O(n1.29) | 58 % | linear attention, 3 of 4 |
| Gemma-4 | steeper | 27 % | window 1024, 5 of 30 full |

With pure full attention a collapse to below 10 % would have been expected. Gemma-4 falls off most steeply — plausibly because of its twice-as-large windows and higher share of full attention. That makes it the fastest model of the series on short prompts and the slowest on very long ones.

## Memory as the main variable

*121 GB unified memory, shared between operating system, weights, KV cache and compiler*

| Configuration | Weights | KV cache | KV per token | Concurrency |
|---|---|---|---|---|
| Laguna NVFP4 | 95.63 GiB | 94,135 | 77.7 KiB | 1.44× at 65k |
| Laguna INT4 (65k) | 69.34 GiB | 950,420 | 38.8 KiB | 14.50× at 65k |
| Laguna INT4 (262k) | 69.34 GiB | 1,001,532 | 38.8 KiB | 3.82× at 262k |
| Laguna INT4 SGLang | 67.56 GB | 185,897 | — | 48 requests |
| Qwen3.6 NVFP4 | 23.45 GiB | 3,366,051 | ≈ 25 KiB | 12.84× at 262k |
| Gemma-4 NVFP4 | 15.88 GiB | 3,912,140 | — | 14.92× at 262k |
| Gemma-4 INT4 | 16.63 GiB | 3,875,400 | — | 14.78× at 262k |

The spread in KV cache is a **factor of 41.6**. At 1.44 concurrent full-context requests, Laguna NVFP4 is practically unusable for multi-user operation; Gemma-4 at 14.92 is comfortable.

> [!NOTE]
> **An incidental finding: the context window is free**
>
> Raising `--max-model-len` for Laguna INT4 from 65,536 to 262,144 cost no KV cache — it even yielded 5 % more (1,001,532 instead of 950,420 tokens), presumably through a more favourable block layout. There is no reason to stay below the checkpoint's maximum.

## Complete measurement data

<details>
<summary>Total throughput of all six configurations (tok/s)</summary>

| Scenario | c | Lag NVFP4 | Lag INT4 | Lag SGLang | Qwen3.6 | Gem NVFP4 | Gem INT4 |
|---|---|---|---|---|---|---|---|
| Prose | 1 | 17.3 | 20.6 | 18.2 | 60.7 | 53.0 | 43.9 |
| Prose | 2 | 25.5 | 31.7 | 27.8 | 101.3 | 101.0 | 94.9 |
| Prose | 4 | 42.7 | 52.8 | 45.3 | 180.7 | 170.6 | 177.1 |
| Code | 1 | 15.0 | 23.8 | 21.9 | 63.5 | 53.1 | 44.1 |
| Code | 2 | 27.4 | 39.0 | 33.2 | 121.2 | 101.5 | 95.0 |
| Code | 4 | 42.4 | 64.3 | 56.2 | 201.2 | 173.6 | 172.5 |
| Prefill 16k | 1 | 5.8 | 13.1 | 6.3 | 16.0 | 27.6 | 22.8 |
| Prefill 16k | 2 | 11.6 | 38.9 | 25.1 | 18.8 | 82.3 | 75.9 |
| Prefill 16k | 4 | 18.7 | 64.6 | 38.0 | 20.3 | 142.5 | 140.0 |
| Prefill rep. | 1 | 7.5 | 24.0 | 25.4 | 16.5 | 44.8 | 38.3 |
| Prefill rep. | 2 | 11.8 | 31.1 | 41.1 | 19.2 | 85.5 | 77.8 |
| Prefill rep. | 4 | 17.6 | 39.5 | 70.6 | 20.3 | 141.1 | 136.2 |

</details>

<details>
<summary>Time to first token (seconds, median)</summary>

| Scenario | c | Lag NVFP4 | Lag INT4 | Lag SGLang | Qwen3.6 | Gem NVFP4 | Gem INT4 |
|---|---|---|---|---|---|---|---|
| Prose | 1 | 0.352 | 0.276 | 0.314 | 0.094 | 0.036 | 0.038 |
| Prose | 2 | 0.443 | 0.635 | 0.201 | 0.259 | 0.036 | 0.037 |
| Prose | 4 | 0.427 | 0.513 | 0.193 | 0.184 | 0.069 | 0.064 |
| Code | 4 | 0.374 | 0.480 | 0.194 | 0.130 | 0.078 | 0.076 |
| Prefill 16k | 1 | 0.307 | 0.309 | 0.200 | 2.806 | 0.093 | 0.084 |
| Prefill 16k | 4 | 0.686 | 0.551 | 0.269 | 7.226 | 0.153 | 0.180 |
| Prefill rep. | 4 | 0.781 | 0.630 | 0.285 | 7.265 | 0.142 | 0.128 |

</details>

## Transferable pitfalls

From fourteen server starts, six of which failed.

### Memory and compiler

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| `MAX_JOBS` too high | OOM kill of `cicc` (7.5 GB per process), **no error in the application log** | `MAX_JOBS=1` on the first start; the kernel cache is warm afterwards |
| `gpu-memory-utilization` > 0.89 | abort before loading | The OS holds ~11 GiB that must stay free |
| `mem-fraction-static` misunderstood | a smaller model creates *no* system headroom | The parameter reserves independently of model size; the saving goes into the KV cache |
| Delayed memory release | restart fails immediately after stopping | Wait for the release, here 5–15 s |

### Measurement and observation

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| First request as a measurement | 3.8 instead of 75 tok/s — off by a factor of 20 | Explicit warm-up run before measuring |
| Output in the `reasoning` field | TTFT stays unset, decode rate computed over the total time | Check `content`, `reasoning` and `reasoning_content` |
| Extrapolated remaining time as fact | tqdm showed 1:48:53, it was actually 3:33 | On non-uniform work, watch the rate over several steps |
| Stationary progress bar | read as “hung”, when compilation was in fact running | Check secondary indicators: CPU load, growing kernel cache |
| Incomplete API data | missing `size` fields counted as 0, 28 GB too low | Verify the response is complete |
| Prefix cache silently ineffective | “warm” values identical to cold ones, with no error message | Check against a true cold measurement with a unique prefix |
| Result file written only at the end | an abort discards every stage already measured | Write incrementally or reconstruct from the log |

### Tools and environment

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Exit code swallowed by a pipeline | `docker pull … \| tail -5` reports success despite `permission denied` | Check the exit code of the first stage |
| `pkill -f` hits your own shell | exit 143/144, the target keeps running | Determine PIDs via `ps`, kill them specifically |
| Server dies with the session | log breaks off after a few lines, no error | `setsid nohup … < /dev/null &` |
| Torch ABI break | `undefined symbol: _ZNK2at10TensorBase…` | Use the torch version pinned by the library |
| Wrong CUDA wheel | missing sm_121a kernels on cu12x | On the Spark use `-cu130` on aarch64 exclusively |
| Missing Python headers | Triton JIT fails without `python3.12-dev` | `uv python install 3.12` ships them, no sudo needed |
| Model revision not pinned | upstream moves `main`; 93 GB re-downloaded in silence, then the start aborts on the KV cache | Pin `--revision` for model and draft; a checkpoint can stay “INT4” in name while quantizing less |

## Refuted assumptions

| Assumption | Finding | Evidence |
|---|---|---|
| `sgl_kernel` lacks sm_121a, runs only after a rebuild | false | cuobjdump shows sm_121a; SGLang ran without a source build |
| NVFP4 is generally the choice for the Spark | false | depends on memory pressure; with Laguna, NVFP4 lost clearly |
| INT4 beats NVFP4 on this hardware | only under memory pressure | with Gemma-4, NVFP4 wins in the partial-load range |
| SGLang scales better over concurrency | false | vLLM's lead grows from 13 to 17 % |
| Prefill 600–800 tok/s | far too low | 1224 to 7161 tok/s depending on model and length |
| `MAX_JOBS=4` is enough | not strict enough | led to the OOM kill; what was needed was `MAX_JOBS=1` |
| SGLang's graph capturing hangs on sm_121 | false | 3 min 33 s, not the extrapolated 1 h 50 min |

## Recommendation matrix

| Use case | Recommendation | Rationale |
|---|---|---|
| Maximum throughput | **Qwen3.6-35B-A3B NVFP4** on vLLM | 201 tok/s at c=4 |
| Minimum response latency | **Gemma-4-26B-A4B NVFP4** on vLLM | 0.036 s TTFT |
| Agent with a growing history | **Gemma-4**, alternatively **Laguna on SGLang** | cache factor 34, or +79 % on a repeated prefix |
| Many concurrent users | **Gemma-4** (either format) | 14.9 full-context requests at once |
| Very long prompts | **Qwen3.6** | 3060 tok/s even at 132k tokens |
| Largest model that still runs | **Laguna-S-2.1 INT4** | but 3× slower than the alternatives |
| do not use | Laguna NVFP4, the FP8 and BF16 variants | 1.44× concurrency, or they do not fit in 121 GB |

> [!NOTE]
> **If only one configuration could be chosen**
>
> **Gemma-4-26B-A4B NVFP4 on vLLM.** It wins on latency, KV cache and memory footprint, sits only just behind Qwen3.6 on throughput — and is the only one of the three models with both a working prefix cache *and* high decode speed, without depending on speculative decoding for it.

#### Launch configuration that proved itself across all runs

```bash
export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=1          # on the first start; cache is warm afterwards

vllm serve <model> \
  --max-num-seqs 32 \
  --max-model-len 262144 \  # costs no KV cache
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

Do not set `--attention-backend`: on sm_121 the auto-selection picks `flashinfer`, because `trtllm_mha` does not support SM120 and `fa3` is Hopper-exclusive.

## Methodology

**Hardware** — NVIDIA GB10, sm_121a, 121 GB unified memory, ~273 GB/s, aarch64

**System** — Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia, driver 580.173.02, CUDA 13.0

**Engines** — vLLM 0.25.1 and SGLang 0.5.16, both torch 2.11.0+cu130, native venvs

### Two purpose-built measurement scripts, standard library only

- `bench.py` — 4 scenarios × 3 concurrency levels (1, 2, 4) × 3 repeats. Streaming for the TTFT, `ignore_eos` for exact token counts, `temperature 0`, an explicit warm-up run, medians as the result.

- `prefill.py` — cold prefill across up to 5 length stages. Every request gets its own UUID in the prompt, so that neither Automatic Prefix Caching nor RadixAttention can take effect.

Both speak the OpenAI-compatible API that vLLM and SGLang alike provide — so every configuration receives exactly the same requests.

> [!IMPORTANT]
> **Two flaws that only surfaced along the way**
>
> **The warm-cold comparison in `bench.py` failed.** Both prefill scenarios measure the warm case, because the prefix is already in the cache after the first of the three repeat runs. Fixed after the fact by `prefill.py`.

> [!IMPORTANT]
> **The size labels were off by a factor of two.** The prompts labelled “8k” actually contained 16,000 to 18,800 tokens. In `prefill.py` the stages were later named after measured token counts.

## Open questions

- **Concurrency beyond four streams.** Gemma-4 INT4 still holds 103 % of its single-stream rate there, and the KV cache would allow 14.8 full-context requests. The scaling potential is exhausted for none of the models.

- **SGLang for Qwen3.6 and Gemma-4.** The engine comparison rests on a single model; whether the profile carries over is open.

- **Answer quality.** Entirely unmeasured. The QAT advantage of the Gemma-4 INT4 checkpoint appears in none of these numbers, nor does the quality difference between the models.

- **Reasoning cleanly separated.** With Laguna and Qwen3.6 the output consisted predominantly of reasoning, which devalues the code-versus-prose comparison. Only Gemma-4 produced genuine answers.

- **The 200k prefill stage** is missing for Gemma-4 in both variants.

- **Multimodality.** All three models ship vision encoders that occupy memory and were exercised in no test.

- **Reproducibility across sessions.** Three repeats per measurement point, but no averaging across sessions. A spot check showed 1.4 % deviation between two runs of the same configuration.

## Glossary

**Active parameters (A3B, A4B)**
In mixture-of-experts models, the number of parameters actually evaluated per token. Determines decode speed; the total parameter count determines the memory footprint.

**KV cache**
Cache of the key and value tensors of processed tokens. Its size limits how many requests at what context length run concurrently.

**Prefix cache**
Reuse of the KV tensors of an already-processed prompt prefix. Automatic Prefix Caching in vLLM, RadixAttention in SGLang.

**Sliding-window attention**
Attention over a bounded window of preceding tokens. Dampens the quadratic cost while remaining cacheable.

**Linear attention**
Attention with a sequentially updated state of constant size. Saves memory, but prevents piecewise reuse of a prefix.

**NVFP4**
NVIDIA's microscaling format: FP4 values in E2M1 layout with FP8 block scales. Blackwell tensor cores process it natively via `FLASHINFER_CUTLASS`.

**MARLIN**
CUDA kernel for weight-quantized formats with unquantized activations. Unpacks 4-bit weights and multiplies at higher precision; amortises over larger batches.

**Speculative decoding**
A cheap method proposes several tokens, which the main model verifies in parallel. The variants here: DFlash (a dedicated draft model), MTP (a bundled head).

**TTFT**
Time to first token. Determines perceived responsiveness; dominated by prefill for long prompts and by scheduling for short ones.

**Unified memory**
On GB10, CPU and GPU share one physical memory pool. Weights, KV cache, operating system and compiler processes compete for the same 121 GB.

**sm_121a**
Compute capability 12.1 of the GB10 GPU. Kernels must be compiled for it; the gencode is emitted only for CUDA ≥ 13.0 on aarch64.

**cicc**
Compiler frontend of the CUDA toolchain. Occupied up to 7.5 GB per process and triggered the OOM kill on the first Laguna start.

## Individual reports

This synthesis summarises three detailed test reports. They contain the full timelines, log excerpts and detailed findings.

| Part | Content | File |
|---|---|---|
| 1 | Laguna-S-2.1 — vLLM against SGLang, 10 start attempts, 2 OOM kills, 4 misdiagnoses | `01-laguna-s-2.1.html` |
| 2 | Qwen3.6-35B-A3B — three times the decode speed, missing prefix cache | `02-qwen3.6-35b-a3b.html` |
| 3 | Gemma-4-26B-A4B — NVFP4 against INT4, documented kernel paths | `03-gemma-4-26b-a4b.html` |
| 4 | This synthesis | `04-gesamtsynthese.html` |

#### Tools and raw data

**bench.py** — Throughput and TTFT across 4 scenarios × 3 concurrency levels

**prefill.py** — Cold prefill with a unique UUID prefix per request

**start-vllm-*.sh** — Launch scripts per model and variant

**start-sglang-*.sh** — SGLang launch scripts

**ergebnisse_*.json** — 6 measurement series of 12 points each

**prefill_*.json** — 6 prefill series

***.log** — startup logs of all 14 attempts, including the failed ones

Everything in `~/bench/`.

---

*Overall synthesis · 6 server configurations, 3 models, 2 engines, 2 quantization formats · 72 throughput and latency measurement points, 22 prefill stages, 14 server starts · all values measured, extrapolations expressly marked.*
