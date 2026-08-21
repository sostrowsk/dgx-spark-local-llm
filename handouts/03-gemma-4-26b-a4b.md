# Gemma-4-26B-A4B on DGX Spark GB10

*Test report · Inference benchmark · Part 3*

> NVFP4 against INT4 under conditions that, for the first time, measure almost nothing but the kernel path — same model, same engine, practically the same memory. The outcome depends on the load.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB unified memory **Model** Gemma-4-26B-A4B · 26B MoE, 4B active **Engine** vLLM 0.25.1 **Runs** 2 configurations · both succeeded on the first attempt

## Contents

1. [At a glance](#at-a-glance)
2. [Key findings](#key-findings)
3. [Model and variants](#model-and-variants)
4. [Test setup](#test-setup)
5. [Startup](#startup)
6. [The two kernel paths](#the-two-kernel-paths)
7. [Decode throughput](#decode-throughput)
8. [Response latency](#response-latency)
9. [Prefill](#prefill)
10. [Memory](#memory)
11. [Pitfalls](#pitfalls)
12. [Placing it in the series](#placing-it-in-the-series)
13. [Recommendation](#recommendation)
14. [Limitations](#limitations)
15. [Glossary](#glossary)
16. [Artefacts](#artefacts)

## At a glance

Two variants of the same model whose weights differ by 0.75 GiB and whose KV cache differs by 1 %. What remains is the difference between two CUDA kernel paths on sm_121 — and it inverts as load rises.

### NVFP4 leads at low load

17 % faster on a single stream, and 28 % faster on cold prefill of short prompts. That is where the native FP4 path plays its advantage.

### INT4 catches up at four streams

Marlin delivers a steady 44–47 tok/s per stream across all load levels, while NVFP4 falls from 53.5 to 43.2. At c=4, INT4 leads by 4 % in the prose scenario.

### Best latency of the whole series

0.036 s to the first token — an eighth of Laguna's, less than half of Qwen3.6's.

### Prefix cache works

16k tokens in 0.093 s warm against 3.13 s cold, a factor of 34. The counter-check to Qwen3.6, where the effect was absent entirely.

- **177.1** — tok/s peak INT4, prose, c=4

- **0.036** — seconds TTFT best of the series

- **3,912,140** — tokens KV cache best of the series

- **7161** — tok/s cold prefill NVFP4 at 4k

## Key findings

### 1. The format advantage is load-dependent, not absolute

NVFP4 wins by 17 % at one stream, 6 % at two, and at four streams INT4 leads the prose scenario by 4 %. The common claim that “NVFP4 is native on Blackwell and therefore faster” thus holds only in the partial-load range.

### 2. Marlin scales over the batch, FlashInfer-CUTLASS does not

Per-stream decode rate shows two distinct characteristics: INT4 stays at 44–47 tok/s across all load levels, NVFP4 starts at 53.5 and falls to 43.2. The FlashInfer path is already near its optimum at a single stream and has correspondingly less headroom.

### 3. NVFP4 stays clearly ahead on prefill — with a shrinking margin

28 % ahead at 4k tokens, 22 % at 16k, 12 % at 65k, 6 % at 132k. The more compute-bound and the shorter the prompt, the more the native FP4 path is worth.

### 4. Sliding-window attention can be cached, linear attention cannot

Gemma-4 shows the prefix cache effect at a factor of 34 that Qwen3.6 lacked in the previous test. Both models dampen the quadratic attention term, but only the sliding-window approach permits reuse — which supports the hypothesis raised in the Qwen test.

### 5. A meaningful code-versus-prose comparison, for the first time

Gemma-4 works without a reasoning mode and returns actual answers. Result: 53.1 against 53.0 tok/s — practically identical. Without speculative decoding, decode is purely bandwidth-driven and independent of content. With Laguna and Qwen3.6 this comparison was devalued by the reasoning share.

## Model and variants

### Architecture

**Architecture** — `Gemma4ForConditionalGeneration` — multimodal, with `vision_config`

**Parameters** — 26 B total, 4 B active per token

**Experts** — 128

**Layers** — 30 — pattern 5 × `sliding_attention` to 1 × `full_attention`

**Sliding window** — 1024

**Context** — `max_position_embeddings` 262,144

**MTP** — not included in any of the checkpoints examined

### Variants measured

| Role | Repo | Size | Format | Kernel path |
|---|---|---|---|---|
| NVFP4 | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 16.5 GB | compressed-tensors | `FLASHINFER_CUTLASS` |
| INT4 | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` | 17.2 GB | compressed-tensors, QAT | `MARLIN` |

Both in the same container format, so that vLLM treats them identically and only the number format differs. The INT4 is a **QAT** checkpoint, that is, quantization-aware retrained — a quality advantage this speed measurement does not surface.

### Other available variants

*Sizes measured via the HuggingFace API, GGUF and MLX omitted*

| Repo | Size | Format |
|---|---|---|
| Intel/…-int4-AutoRound | 15.4 GB | auto-round |
| Intel/…-int4-mixed-AutoRound | 16.2 GB | auto-round |
| RedHatAI/…-NVFP4 | 16.5 GB | compressed-tensors |
| unsloth/…-NVFP4 | 16.9 GB | compressed-tensors |
| cyankiwi/…-AWQ-4bit | 17.2 GB | compressed-tensors |
| cyankiwi/…-qat-AWQ-INT4 | 17.2 GB | compressed-tensors, QAT |
| nvidia/Gemma-4-26B-A4B-NVFP4 | 18.8 GB | modelopt |
| RedHatAI/…-FP8-dynamic | 28.7 GB | compressed-tensors |
| google/gemma-4-26B-A4B-it | 51.6 GB | BF16 |

All repos are openly accessible, no gating — not a given with Google models.

## Test setup

Hardware, engine and measurement methodology identical to the Laguna and Qwen3.6 tests. Both variants via the same launch script with a variant argument, which rules out diverging parameters.

```bash
#!/usr/bin/env bash
VARIANTE="${1:-nvfp4}"
case "$VARIANTE" in
  nvfp4) MODELL="RedHatAI/gemma-4-26B-A4B-it-NVFP4" ;;
  int4)  MODELL="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4" ;;
esac

export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=1

exec vllm serve "$MODELL" \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

No `--speculative-config`: the checkpoints ship no MTP head. Gemma-4 therefore measures pure decode performance without the speculative multiplier that both Laguna (DFlash) and Qwen3.6 (MTP) had.

#### Measurement tools

- `bench.py` — 4 scenarios × 3 concurrency levels × 3 repeats, streaming for the TTFT, `ignore_eos`, `temperature 0`, warm-up run beforehand, medians

- `prefill.py` — cold prefill with a unique UUID prefix per request

## Startup

Both runs succeeded on the first attempt. The settings worked out during the Laguna tests carried over unchanged.

| Metric | NVFP4 | INT4 |
|---|---|---|
| Weights | 15.88 GiB | 16.63 GiB |
| Load time | 101.9 s | 39.2 s |
| KV cache | 3,912,140 | 3,875,400 |
| Concurrency at 262k | 14.92× | 14.78× |
| MoE backend | FLASHINFER_CUTLASS | MARLIN |
| Result | succeeded | succeeded |

> [!IMPORTANT]
> The load times are **not comparable**: the INT4 run followed immediately after the NVFP4 run, so the operating system's page cache was warm. The 39.2 s therefore measure cache warmth, not model properties.

## The two kernel paths

vLLM logs at startup which MoE backend it selects. That makes the difference underlying this comparison directly evidenced rather than cited from the literature.

```bash
NVFP4: INFO [nvfp4.py:285] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
INT4:  INFO [int_wna16.py:197] Using 'MARLIN' WNA16 MoE backend
```

### What separates the two paths

**FLASHINFER_CUTLASS** addresses the FP4 tensor cores of the Blackwell generation directly. The weights are held in E2M1 layout with FP8 block scales and are computed without an intermediate step.

**MARLIN** is a kernel for weight-quantized formats with unquantized activations (WNA16). The 4-bit weights are unpacked and multiplied at higher precision. The overhead amortises over larger batches — which is exactly what the measurement shows.

### The effect in numbers

*Decode rate per individual stream, prose scenario*

| Concurrency | NVFP4 | INT4 | Characteristic |
|---|---|---|---|
| 1 | 53.5 | 44.1 | NVFP4 near optimum |
| 2 | 50.9 | 46.7 | INT4 gains through batching |
| 4 | 43.2 | 45.5 | INT4 overtakes |

INT4 stays within a band of 3 tok/s across the entire load range, NVFP4 loses 19 %. The Marlin kernels win back their initial deficit over the batch.

## Decode throughput

*Total throughput in tok/s across all concurrent requests · median of 3 runs*

| Scenario | c | NVFP4 | INT4 | Delta |
|---|---|---|---|---|
| Prose | 1 | 53.0 | 43.9 | −17 % |
| Prose | 2 | 101.0 | 94.9 | −6 % |
| Prose | 4 | 170.6 | 177.1 | +4 % |
| Code | 1 | 53.1 | 44.1 | −17 % |
| Code | 2 | 101.5 | 95.0 | −6 % |
| Code | 4 | 173.6 | 172.5 | −1 % |
| Prefill 16k | 1 | 27.6 | 22.8 | −18 % |
| Prefill 16k | 2 | 82.3 | 75.9 | −8 % |
| Prefill 16k | 4 | 142.5 | 140.0 | −2 % |
| Prefill 16k rep. | 1 | 44.8 | 38.3 | −15 % |
| Prefill 16k rep. | 2 | 85.5 | 77.8 | −9 % |
| Prefill 16k rep. | 4 | 141.1 | 136.2 | −3 % |

The same pattern across all four scenarios: at c=1 NVFP4 leads by 15–18 %, at c=2 still by 6–9 %, at c=4 the gap has shrunk to −1 to +4 %.

### Scaling behaviour

*Prose scenario · how much of the single-stream rate is retained at higher load*

| c | NVFP4 retained | INT4 retained | NVFP4 total | INT4 total |
|---|---|---|---|---|
| 1 | 100 % | 100 % | 53.0 | 43.9 |
| 2 | 95 % | 106 % | 101.0 | 94.9 |
| 4 | 81 % | 103 % | 170.6 | 177.1 |

At two and four streams INT4 gets *faster* per stream than at one — a clear batching signal from the Marlin kernels.

## Response latency

*Time to first token in seconds · median*

| Scenario | c | NVFP4 | INT4 |
|---|---|---|---|
| Prose | 1 | 0.036 | 0.038 |
| Prose | 2 | 0.036 | 0.037 |
| Prose | 4 | 0.069 | 0.064 |
| Code | 1 | 0.042 | 0.042 |
| Code | 2 | 0.045 | 0.044 |
| Code | 4 | 0.078 | 0.076 |
| Prefill 16k | 1 | 0.093 | 0.084 |
| Prefill 16k | 2 | 0.112 | 0.114 |
| Prefill 16k | 4 | 0.153 | 0.180 |
| Prefill 16k rep. | 1 | 0.077 | 0.078 |
| Prefill 16k rep. | 2 | 0.097 | 0.098 |
| Prefill 16k rep. | 4 | 0.142 | 0.128 |

TTFT barely differs between the formats — the kernel path affects throughput, not start-up time. What stands out is the absolute level: **0.036 s** is the best value of the entire test series, against 0.094 s for Qwen3.6 and 0.276 s for Laguna.

## Prefill

### The prefix cache works

Unlike with Qwen3.6, reuse takes effect: 16,386 tokens need **3.13 s** cold and **0.093 s** in the main run with a warm cache — a **factor of 34**. This supports the hypothesis raised in the Qwen test, that sliding-window attention is cacheable and linear attention is not.

### Cold prefill

*Median of 3 runs, each with its own UUID in the prompt*

| Tokens | NVFP4 | INT4 | Delta | NVFP4 relative |
|---|---|---|---|---|
| ≈ 4,100 | 7161 | 5170 | −28 % | — |
| ≈ 16,300 | 5238 | 4108 | −22 % | 100 % |
| ≈ 64,600 | 2396 | 2109 | −12 % | 46 % |
| ≈ 131,600 | 1416 | 1326 | −6 % | 27 % |

NVFP4 stays ahead throughout, but the gap halves with every lengthening. At very long prompts the attention computation dominates and the difference in the matrix-multiplication path carries less weight.

> [!IMPORTANT]
> **Steep drop at long prompts**
>
> From 16k to 65k tokens Gemma-4 loses 54 % of its prefill rate, and another 19 percentage points by 132k. At comparable points Laguna still held 80 and 64 %, Qwen3.6 76 and 58 %. A plausible cause is the window size: Gemma-4 uses 1024-token sliding windows against Laguna's 512, and every sixth of 30 layers computes full attention. The quadratic share is therefore larger. This attribution is derived from the configuration, not measured directly.

## Memory

| Metric | NVFP4 | INT4 | Difference |
|---|---|---|---|
| Weights | 15.88 GiB | 16.63 GiB | +4.7 % |
| KV cache | 3,912,140 | 3,875,400 | −0.9 % |
| Concurrency at 262k | 14.92× | 14.78× | −0.9 % |

The difference in KV cache corresponds exactly to the 0.75 GiB of additional weights. The memory aspect is therefore **neutral** in this comparison — unlike with Laguna, where NVFP4 at 99.7 against 71.9 GB shrank the cache by a factor of ten and dominated the result.

Both variants also reach the **highest cache capacity of the entire test series**: 3.9 M tokens against 3.4 M for Qwen3.6 and 1.0 M for Laguna.

## Pitfalls

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| `prefill.py` writes only at the end | Aborting the last stage discards all previous measurements | Reconstruct values from the run log; incremental writing would be better |
| Load times appear different | 39.2 against 101.9 s for almost the same amount of data | Warm page cache on the second run — do not read as a model property |
| Format advantage assumed wholesale | “NVFP4 is native, therefore faster” applies only at low load | Measure across several concurrency levels, not just at c=1 |
| No MTP available | Decode without a speculative multiplier, unlike Laguna and Qwen3.6 | Take into account when comparing across models |
| Multimodality unexpected | Vision encoder occupies memory even in text-only operation | Check `config.json` for `vision_config` before loading |
| QAT quality invisible | The INT4 checkpoint is retrained, the measurement captures speed only | Assess separately when choosing the format |

## Placing it in the series

Three models, same hardware, same engine, same harness.

*The best measured variant of each model*

| Metric | Laguna-S-2.1 | Qwen3.6-35B | Gemma-4-26B |
|---|---|---|---|
| Active parameters | 8 B | 3 B | 4 B |
| Weights | 69.34 GiB | 23.45 GiB | 15.88 GiB |
| Decode c=1 | 20.6 | 60.7 | 53.0 |
| Decode c=4 | 52.8 | 180.7 | 177.1 |
| Best TTFT | 0.260 s | 0.094 s | 0.036 s |
| KV cache | 1,001,532 | 3,366,051 | 3,912,140 |
| Prefix cache factor | ≈ 30 | 1.0 | ≈ 34 |
| Cold prefill at 16k | 2301 | 5301 | 5238 |
| Speculation | DFlash | MTP | none |

Gemma-4 achieves the second-best decode speed *without* speculative decoding, while Qwen3.6 and Laguna both use a speculation multiplier. On the latency and memory metrics it leads the series.

## Recommendation

### For Gemma-4 on this box

- **Single user or low load** → NVFP4, roughly 17 % faster

- **From four concurrent streams** → equivalent, slight edge to INT4 on prose

- **Heavy prefill with short prompts** → NVFP4, up to 28 % faster

- **When answer quality matters** → the QAT INT4 has an unmeasured advantage

### Correcting an earlier statement

From the Laguna test I concluded that INT4 beats NVFP4 on this hardware. The result was correct there, the generalisation was not: it was driven by memory pressure (99.7 against 71.9 GB on a 121 GB machine), not by the compute path. Where memory plays no role, NVFP4 wins at low load and draws level at high load.

### Open optimisation

The measurement stops at four streams. INT4 still holds 103 % of its single-stream rate there, and the KV cache would allow 14.8 concurrent full-context requests. Whether INT4 extends its lead at eight or sixteen streams is open, and would be the obvious next test.

## Limitations

- **The 215k stage is missing from both series.** The NVFP4 run was aborted there; the INT4 run was stopped at the same point to preserve symmetry.

- **Prefill data reconstructed.** Both `prefill_gemma4_*.json` were generated from the run logs, because `prefill.py` writes only at the end.

- **QAT quality advantage not measured.** The INT4 checkpoint is quantization-aware retrained; only speed was captured.

- **One engine only.** vLLM 0.25.1 exclusively; SGLang was not tested for Gemma-4.

- **Two of nine variants.** AutoRound, modelopt NVFP4, FP8 and BF16 went unmeasured.

- **Concurrency only up to 4.** The scaling potential of both formats is not exhausted.

- **Multimodality unused.** All measurements with text-only requests.

- **One measurement series each.** Three repeats per point, no averaging across sessions.

## Glossary

**A4B**
Naming convention for “4 B active parameters”. In MoE models this number determines decode speed, while the total parameter count determines memory footprint.

**MARLIN**
CUDA kernel for weight-quantized formats with unquantized activations (WNA16). Unpacks 4-bit weights and multiplies at higher precision; the overhead amortises over larger batches.

**FLASHINFER_CUTLASS**
Kernel path that addresses the FP4 tensor cores of the Blackwell generation directly. On sm_121 the native route for NVFP4.

**QAT**
Quantization-aware training. The model is retrained with simulated quantization, which makes the quality loss smaller than with post-hoc quantization.

**Sliding-window attention**
Attention over a bounded window of preceding tokens. Gemma-4 uses windows of 1024 in five of six layers. Unlike linear attention, the associated KV cache can be reused piecewise.

**Prefix cache**
Reuse of the KV tensors of an already-processed prompt prefix. Automatic Prefix Caching in vLLM, RadixAttention in SGLang.

**compressed-tensors**
Quantization container that vLLM and SGLang read directly. Can hold both NVFP4 and INT4 variants — which is what made the comparison drawn here, with an identical container, possible.

**TTFT**
Time to first token. Determines perceived responsiveness; dominated by prefill for long prompts and by scheduling for short ones.

## Artefacts

**start-vllm-gemma4.sh** — Launch script, variant selected via the argument `nvfp4` or `int4`

**bench.py** — Throughput and TTFT across 4 scenarios × 3 concurrency levels

**prefill.py** — Cold prefill with a unique UUID prefix per request

**ergebnisse_gemma4_nvfp4.json** — 12 measurement points, NVFP4

**ergebnisse_gemma4_int4.json** — 12 measurement points, INT4

**prefill_gemma4_*.json** — 4 prefill stages each, reconstructed from the logs

**gemma4-*.log** — Startup logs of both runs, including the backend lines

> [!NOTE]
> All files in `~/bench/`. Both models remain in the HuggingFace cache at 33 GB combined, and the FlashInfer kernel cache is warm — a restart takes a few minutes.

---

*Test report Gemma-4-26B-A4B on DGX Spark GB10 · 2 configurations, 12 measurement points and 4 prefill stages each · Part 3 of the series after Laguna-S-2.1 and Qwen3.6-35B-A3B · all values measured.*
