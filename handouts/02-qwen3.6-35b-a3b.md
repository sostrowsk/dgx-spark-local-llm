# Qwen3.6-35B-A3B on DGX Spark GB10

*Test report · Inference benchmark · Part 2*

> NVFP4 with multi-token prediction, measured with the same harness as Laguna-S-2.1 — three times the decode speed at a third of the memory footprint, but without effective prefix reuse.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB unified memory **Model** RedHatAI/Qwen3.6-35B-A3B-NVFP4 · 35B MoE, 3B active **Engine** vLLM 0.25.1 **Server starts** 1 · succeeded on the first attempt

## Contents

1. [At a glance](#at-a-glance)
2. [Key findings](#key-findings)
3. [Model and variants](#model-and-variants)
4. [Test setup](#test-setup)
5. [Startup](#startup)
6. [Decode throughput](#decode-throughput)
7. [Response latency](#response-latency)
8. [The missing prefix cache](#the-missing-prefix-cache)
9. [Prefill scaling](#prefill-scaling)
10. [Memory](#memory)
11. [Pitfalls](#pitfalls)
12. [Comparison with Laguna](#comparison-with-laguna-s-21)
13. [Recommendation](#recommendation)
14. [Limitations](#limitations)
15. [Glossary](#glossary)
16. [Artefacts](#artefacts)

## At a glance

One measurement series, a single server start, no failed attempts — in contrast to the ten runs of the Laguna tests. The result is unambiguous on two axes and surprising on a third.

### Three times the decode speed

60.7 to 201.2 tok/s against Laguna's 20.6 to 64.3 — a factor of 2.7 to 3.4 across all decode scenarios. Matches the ratio of active parameters, 3 to 8 B.

### A third of the memory

23.45 GiB of weights instead of 69.34. Hence 3,366,051 tokens of KV cache and 12.84× concurrency at the full 262k context.

### No prefix cache effect

Warm and cold runs identical: 2.822 against 2.806 s. Laguna resolved the same prefix in 0.30 s — a factor of 30 from caching that is absent here entirely.

### Twice the prefill speed

5301 tok/s against 2301 at 16k tokens, and 2429 against 1224 at just under 200k. Raw throughput is a good two times higher throughout.

- **201.2** — tok/s peak throughput code, c=4

- **0.094** — seconds TTFT best result

- **3,366,051** — tokens KV cache at 262k context

- **1.00×** — cache gain on a repeated prefix

## Key findings

### 1. Active parameters determine the speed

Qwen3.6 activates 3 B parameters per token, Laguna 8 B. The measured throughput ratio of 2.7 to 3.4 matches that almost exactly. On a machine with 273 GB/s of memory bandwidth this is the dominant quantity — total parameter count and quantization format come second.

### 2. The machine is not at its limit with this model

Going from one to four concurrent streams, the single stream drops only to 77 % (Laguna: 61 %), while total throughput nearly triples. Concurrency beyond the four levels measured would probably keep scaling.

### 3. Linear attention prevents prefix reuse

Qwen3.6 uses linear attention with a sequentially updated state in roughly three out of four layers. Such a state cannot be reused piecewise the way a KV cache can. The measurement confirms the behaviour unambiguously; the architectural explanation is derived from the configuration, not directly demonstrated.

### 4. The model is multimodal — which the name does not reveal

Architecture `Qwen3_5MoeForConditionalGeneration` with its own `model_visual.safetensors` (0.89 GB) and `vision_config`. The vision encoder is excluded from quantization and occupies memory even when only text is processed.

### 5. The harness had to be corrected

Qwen3.6 returns its output in the streaming field `reasoning` instead of `content`. Without an adjustment, time to first token would never have been set and the decode rate would have been computed over the total time instead of the decode phase.

## Model and variants

> [!IMPORTANT]
> **On the name:** A model called *Qwen3.6-36B-A3B* does not exist. The MoE variant of the Qwen3.6 line is called **Qwen3.6-35B-A3B**; alongside it there is the dense **Qwen3.6-27B**.

### Architecture of the checkpoint measured

**Repo** — `RedHatAI/Qwen3.6-35B-A3B-NVFP4`

**Architectures** — `Qwen3_5MoeForConditionalGeneration` · speculation head `Qwen3_5MoeMTP`

**model_type** — `qwen3_5_moe`

**Parameters** — 35 B total, 3 B active per token

**Layers** — 40 — linear attention in roughly three of four positions, full attention in every fourth

**Context** — `max_position_embeddings` 262,144

**Quantization** — `compressed-tensors`, format `nvfp4-pack-quantized`, group size 16, FP8-E4M3 scales

**Excluded** — vision encoder, `lm_head`, MTP weights (`re:^mtp.*`), gate and linear-attention projections

#### Files in the repo

| File | Size | Function |
|---|---|---|
| model.safetensors | 22.46 GB | language model, NVFP4-quantized |
| model_mtp.safetensors | 1.69 GB | multi-token prediction, unquantized |
| model_visual.safetensors | 0.89 GB | vision encoder, unquantized |

### Available 4-bit variants

A pure `INT4` in the compressed-tensors sense does not exist for Qwen3.6 — the integer paths are called AWQ, GPTQ and AutoRound. All sizes measured via the HuggingFace API.

*Sorted by format, then by downloads*

| Repo | Size | Format | MTP |
|---|---|---|---|
| nvidia/Qwen3.6-35B-A3B-NVFP4 | 23.5 GB | modelopt | no |
| RedHatAI/Qwen3.6-35B-A3B-NVFP4 | 25.1 GB | compressed-tensors | **yes** |
| unsloth/Qwen3.6-35B-A3B-NVFP4-Fast | 23.7 GB | compressed-tensors | no |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | 26.5 GB | compressed-tensors | no |
| Intel/…-int4-mixed-AutoRound | 21.5 GB | auto-round | no |
| palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 | 24.5 GB | gptq | **yes** |
| cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | 25.0 GB | awq | no |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | 25.5 GB | awq | no |
| Qwen/Qwen3.6-35B-A3B-FP8 | 37.5 GB | fp8 | **yes** |
| Qwen/Qwen3.6-35B-A3B | 71.9 GB | BF16 | no |

`RedHatAI/…-NVFP4` was chosen as the only NVFP4 variant with MTP weights. `compressed-tensors` is also the format that vLLM and SGLang process directly. The MLX variants target Apple Silicon and are useless on GB10.

> [!NOTE]
> **Why NVFP4 and not AWQ or GPTQ**
>
> The Laguna test established that NVFP4 runs natively on sm_121 via `FLASHINFER_CUTLASS`, with no Marlin fallback. AWQ and GPTQ go through Marlin kernels on this architecture — the slower path. The finding there, *INT4 beats NVFP4*, does not carry over, because it was driven by memory pressure at 99.7 GB of weights. At 23–26 GB, memory is simply not an issue.

## Test setup

Hardware, engine and measurement methodology identical to the Laguna test, so that the values stay comparable.

**GPU** — NVIDIA GB10, compute capability 12.1 (sm_121a)

**Memory** — 121 GB unified memory

**Operating system** — Ubuntu 24.04.4 LTS, aarch64

**Driver** — NVIDIA 580.173.02, CUDA 13.0

**Engine** — vLLM 0.25.1, torch 2.11.0+cu130, FlashInfer 0.6.13

#### Launch configuration

```bash
export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=1

vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

`MAX_JOBS=1` was carried over from the Laguna test, where four parallel `cicc` processes triggered the OOM killer. At 23 GB of weights there would have been considerably more headroom here, but the rule costs nothing.

### Measurement methodology

- `bench.py` — 4 scenarios × 3 concurrency levels (1, 2, 4) × 3 repeats, streaming for the TTFT, `ignore_eos` for exact token counts, `temperature 0`, warm-up run beforehand, medians as the result

- `prefill.py` — cold prefill across 5 length stages, a unique UUID in the prompt per request so that no cache can take effect

## Startup

A single attempt, no corrections needed — in contrast to the ten runs of the Laguna tests. The settings worked out there carried over unchanged.

*Timeline of the start*

| Time | Event | Duration |
|---|---|---|
| 09:08–09:19 | Download, 24 GB | ≈ 10 min |
| 09:19:34 | Architecture detected: `Qwen3_5MoeForConditionalGeneration` | — |
| 09:19:39 | Speculation head detected: `Qwen3_5MoeMTP` | — |
| 09:22:17 | Weights loaded, 23.45 GiB | 125.7 s |
| 09:33:53 | KV cache allocated, 3,366,051 tokens | — |
| ≈ 09:34 | Server ready | ≈ 14 min |

The method `mtp` was correct — vLLM 0.25.1 also knows `qwen3_next_mtp`, which was not needed here. No OOM, no kernel error, no restart.

> [!IMPORTANT]
> **The first request is not a measurement**
>
> The smoke test returned 120 tokens in 31.4 s, that is **3.8 tok/s**. The second request returned 60 tokens in 0.797 s, that is **75 tok/s** — a difference of a factor of 20. The first request after startup carries warm-up costs that must not enter any measurement. `bench.py` therefore runs an explicit warm-up before measuring.

## Decode throughput

*Total throughput in tok/s across all concurrent requests · median of 3 runs*

| Scenario | c | per stream | total | Laguna INT4 | Factor |
|---|---|---|---|---|---|
| Prose | 1 | 62.21 | 60.7 | 20.6 | 2.94× |
| Prose | 2 | 56.15 | 101.3 | 31.7 | 3.19× |
| Prose | 4 | 47.81 | 180.7 | 52.8 | 3.42× |
| Code | 1 | 65.99 | 63.5 | 23.8 | 2.67× |
| Code | 2 | 67.01 | 121.2 | 39.0 | 3.11× |
| Code | 4 | 52.84 | 201.2 | 64.3 | 3.13× |

### Scaling behaviour

*Prose scenario · loss per stream against total gain*

| c | per stream | retained | total | Scaling |
|---|---|---|---|---|
| 1 | 62.2 | 100 % | 60.7 | — |
| 2 | 56.2 | 90 % | 101.3 | 1.67× |
| 4 | 47.8 | 77 % | 180.7 | 2.98× |

For comparison, Laguna retained only 61 % of its single-stream rate at c=4. With 3 B active parameters Qwen3.6 sits further from the bandwidth ceiling and can exploit parallelism better — the four levels measured probably do not exhaust the potential.

## Response latency

*Time to first token in seconds · median*

| Scenario | c | Qwen3.6 | Laguna INT4 |
|---|---|---|---|
| Prose | 1 | 0.094 | 0.276 |
| Prose | 2 | 0.259 | 0.635 |
| Prose | 4 | 0.184 | 0.513 |
| Code | 1 | 0.128 | 0.260 |
| Code | 2 | 0.121 | 0.266 |
| Code | 4 | 0.130 | 0.480 |
| Prefill 16k | 1 | 2.806 | 0.309 |
| Prefill 16k | 2 | 4.360 | 0.458 |
| Prefill 16k | 4 | 7.226 | 0.551 |
| Prefill 16k repeated | 1 | 2.822 | 0.328 |
| Prefill 16k repeated | 2 | 4.362 | 0.484 |
| Prefill 16k repeated | 4 | 7.265 | 0.630 |

On short prompts Qwen3.6 reaches the first token two to three times faster throughout. On long prompts the picture inverts — see the following section.

## The missing prefix cache

The most striking finding of this measurement series. The scenario `prefill_16k_wiederholt` uses the same long prefix as the preceding scenario and should therefore be served from the cache.

*TTFT on first contact against a repeat of the same prefix*

| c | first contact | repeated | Difference | Laguna for comparison |
|---|---|---|---|---|
| 1 | 2.806 s | 2.822 s | +0.6 % | 0.309 → 0.328 s |
| 2 | 4.360 s | 4.362 s | +0.0 % | 0.458 → 0.484 s |
| 4 | 7.226 s | 7.265 s | +0.5 % | 0.551 → 0.630 s |

### The proof via the cold measurement

16,384 tokens at a measured cold rate of 5301 tok/s work out to **3.09 s**. What was measured in the main run was **2.806 s**. The values labelled “warm” were therefore recomputations at full price — there never was a cache hit.

A second indication comes from the scaling: TTFT rises almost linearly with concurrency (2.8 / 4.4 / 7.2 s). With a working prefix cache it would stay largely constant, as with Laguna (0.31 / 0.46 / 0.55 s).

### Attempted explanation

The checkpoint's quantization ignore list shows `linear_attn` projections in roughly three of four layers — Qwen3.6 mixes linear with full attention. Linear attention carries a sequentially updated state that cannot be addressed and reused piecewise the way a KV cache can.

> [!IMPORTANT]
> The behaviour is unambiguously measured. The architectural justification is derived from the configuration and was not demonstrated directly — for instance by comparison with a full-attention-only model of the same size.

### Practical significance

For an agent carrying the same 16k-token context across twenty turns, Qwen3.6 pays about 2.8 s every time. Laguna pays 9.3 s the first time and 0.3 s thereafter. From the fourth turn on Laguna is ahead, and the gap widens with every further turn.

## Prefill scaling

Measured with a unique prefix per request. Here Qwen3.6 is a good two times faster throughout.

*Cold prefill · median of 3 runs, each with its own UUID in the prompt*

| Prompt tokens | TTFT | Prefill | relative | Laguna INT4 |
|---|---|---|---|---|
| 4,053 | 0.80 s | 5064 tok/s | — | 2255 tok/s |
| 16,384 | 3.09 s | 5301 tok/s | 100 % | 2301 tok/s |
| 65,362 | 16.21 s | 4033 tok/s | 76 % | 1850 tok/s |
| 129,280 | 42.24 s | 3060 tok/s | 58 % | 1484 tok/s |
| 198,163 | 81.58 s | 2429 tok/s | 46 % | 1224 tok/s |

From 16,384 to 198,163 tokens the length grows 12.1-fold while the time grows 26.4-fold. That corresponds to a complexity of about **O(n1.29)** — practically identical to Laguna's O(n1.25), only at twice the level. Both models successfully dampen the quadratic attention term, Qwen3.6 via linear attention, Laguna via sliding windows.

> [!NOTE]
> **Mind the tokenization**
>
> The same prompt text yields **16,384** tokens with Qwen3.6 and **18,699** with Laguna. Qwen3.6 tokenizes roughly 12 % more densely. This has to be kept in mind when comparing absolute TTFT values; the tok/s rates are unaffected.

## Memory

| Metric | Qwen3.6 NVFP4 | Laguna INT4 |
|---|---|---|
| Weights | 23.45 GiB | 69.34 GiB |
| Load time | 125.7 s | 115.1 s |
| KV cache | 3,366,051 | 1,001,532 |
| KV per token | ≈ 25 KiB | 38.8 KiB |
| Concurrency at 262k | 12.84× | 3.82× |
| Total startup time | ≈ 14 min | ≈ 3 min |

The threefold KV cache draws on two sources: 46 GiB fewer weights and a third lower consumption per token. The latter follows from linear attention — only about a quarter of the 40 layers scales with sequence length.

That the load time is no shorter despite a third of the data volume is down to the cold page cache: Laguna had already been loaded several times before its comparison run.

## Pitfalls

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Output in the `reasoning` field | Streaming deltas contain no `content`; TTFT stays unset, decode rate is computed over the total time | Have the harness check `content`, `reasoning` and `reasoning_content` |
| First request taken as a measurement | 3.8 instead of 75 tok/s — off by a factor of 20 | Explicit warm-up run before measuring |
| Prefix cache silently ineffective | “warm” values identical to cold ones, with no error message | Check against a true cold measurement with a unique prefix |
| Multimodality unexpected | `model_visual.safetensors` occupies memory even in text-only operation | Check `config.json` for `vision_config` before loading |
| Wrong model name | Searching for *36B-A3B* returns zero hits | The MoE variant is called **35B**-A3B |
| Guessing the speculation method | vLLM knows both `mtp` and `qwen3_next_mtp` | For Qwen3.6, `mtp` is correct; the architecture detected in the log confirms it |
| Different tokenization | the same text yields 12 % fewer tokens than with Laguna | Compare absolute TTFT only at equal token counts, otherwise normalise via tok/s |

<details>
<summary>The harness correction in detail</summary>

Qwen3.6's streaming delta looks like this:

```bash
data: {"choices":[{"delta":{"reasoning":"Here"},...}]}
data: {"choices":[{"delta":{"reasoning"":"'s a thinking process"},...}]}
```

The original evaluation checked `delta["content"]` exclusively. Consequence: `ttft` stayed `None`, the chunk counter stayed at zero, and decode time was computed as `total - 0` — that is, including prefill.

The correction is additive and does not change the behaviour for answers in the `content` field:

```bash
# before
if delta.get("content"):

# after
if any(delta.get(f) for f in ("content", "reasoning", "reasoning_content")):
```

The Laguna measurements therefore remain valid — there the reasoning parser failed to initialise, so the entire output arrived as `content` anyway.

</details>

## Comparison with Laguna-S-2.1

*Both on vLLM 0.25.1, identical harness, identical prompts*

| Scenario | c | Laguna INT4 | Qwen3.6 NVFP4 | Factor |
|---|---|---|---|---|
| Prose | 1 | 20.6 | 60.7 | 2.94× |
| Prose | 2 | 31.7 | 101.3 | 3.19× |
| Prose | 4 | 52.8 | 180.7 | 3.42× |
| Code | 1 | 23.8 | 63.5 | 2.67× |
| Code | 2 | 39.0 | 121.2 | 3.11× |
| Code | 4 | 64.3 | 201.2 | 3.13× |
| Prefill 16k | 1 | 13.1 | 16.0 | 1.23× |
| Prefill 16k | 2 | 38.9 | 18.8 | 0.48× |
| Prefill 16k | 4 | 64.6 | 20.3 | 0.31× |
| Prefill 16k rep. | 1 | 24.0 | 16.5 | 0.69× |
| Prefill 16k rep. | 2 | 31.1 | 19.2 | 0.62× |
| Prefill 16k rep. | 4 | 39.5 | 20.3 | 0.51× |

The profile splits cleanly: **decode goes to Qwen3.6 threefold, long recurring context goes to Laguna.** The prefill rows do not measure compute performance — Qwen3.6 is twice as fast there — but the cache effect that only Laguna has.

## Recommendation

### Qwen3.6-35B-A3B-NVFP4 for throughput and interactivity

Three times the decode speed, three times the KV cache, a third of the memory footprint, TTFT under 0.2 s. For chat, code completion and anything where each request carries a manageable context, it is the clearly better choice on this box.

### Laguna-S-2.1-INT4 for long recurring contexts

From roughly the fourth turn with the same 16k context Laguna is ahead, because its prefix cache takes effect. For agents carrying a growing history, that can outweigh Qwen3.6's threefold compute advantage.

### Open optimisation

The concurrency measurement stops at four streams, even though Qwen3.6 still holds 77 % of its single-stream rate there and the KV cache would allow 12.84 concurrent full-context requests. Higher levels were not measured and should keep scaling.

## Limitations

- **Reasoning dominates the output.** Qwen3.6 thinks visibly; at a 256-token limit the output consists almost entirely of reasoning. What was measured is the token generation rate, not the time to a finished answer. For hardware and engine comparisons that is the right quantity, for wait times from a user's perspective it is not.

- **One engine only.** vLLM 0.25.1 exclusively; SGLang was not tested for Qwen3.6.

- **One quantization only.** Of the eight available 4-bit variants only `RedHatAI/…-NVFP4` was measured. Whether AWQ, GPTQ or AutoRound differ appreciably on sm_121 is open.

- **Explanation of the missing cache effect unproven.** The behaviour is measured, the attribution to linear attention is derived from the configuration.

- **Concurrency only up to 4.** The scaling potential was not exhausted.

- **Multimodality unused.** All measurements with text-only requests; the vision encoder occupied memory but was never exercised.

- **One measurement series each.** Three repeats per point, no averaging across sessions.

## Glossary

**A3B**
Naming convention for “3 B active parameters”. In MoE models this number governs decode speed, the total parameter count governs the memory footprint.

**MTP**
Multi-token prediction. A bundled head predicts several tokens at once, which the main model verifies in parallel. Acts like speculative decoding without a separate draft model.

**Linear attention**
Attention variant with a sequentially updated state of constant size instead of a KV cache that grows with sequence length. Saves memory, but prevents piecewise reuse of a prefix.

**NVFP4**
NVIDIA's microscaling format: FP4 values in E2M1 layout with FP8-E4M3 scales per 16 values. Blackwell tensor cores process it natively.

**compressed-tensors**
Quantization container that vLLM and SGLang read directly. An alternative to NVIDIA's `modelopt` format; both can hold NVFP4.

**Prefix cache**
Reuse of the KV tensors of an already-processed prompt prefix. Called Automatic Prefix Caching in vLLM, RadixAttention in SGLang.

**TTFT**
Time to first token. Determines perceived responsiveness and is dominated by prefill for long prompts.

**sm_121a**
Compute capability 12.1 of the GB10 GPU. CUDA kernels must be compiled for it; on this architecture NVFP4 runs natively via FlashInfer-CUTLASS.

## Artefacts

**start-vllm-qwen36.sh** — Launch script, NVFP4 with MTP, 262k context

**bench.py** — Throughput and TTFT, with the correction for the `reasoning` field

**prefill.py** — Cold prefill across 5 length stages up to 198,163 tokens

**ergebnisse_qwen36.json** — all 12 measurement points

**prefill_qwen36.json** — 5 prefill stages

**qwen36.log** — startup log

> [!NOTE]
> All files in `~/bench/`. The model remains in the HuggingFace cache at 24 GB, and the FlashInfer kernel cache is warm — a restart takes around four minutes.

---

*Test report Qwen3.6-35B-A3B-NVFP4 on DGX Spark GB10 · 1 server start, 12 measurement points, 5 prefill stages · companion document to the Laguna-S-2.1 test report · all values measured.*
