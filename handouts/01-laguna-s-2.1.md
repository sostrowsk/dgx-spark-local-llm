# Laguna-S-2.1 on DGX Spark GB10

*Test report · Inference benchmark*

> vLLM against SGLang, NVFP4 against INT4 — full documentation of every measurement run, including the six failed server starts, two OOM kills and four misdiagnoses.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB unified memory **Model** poolside/Laguna-S-2.1 · 118B MoE, 8B active **Server starts** 10 · 4 of them successful **Measurement series** 3 configurations × 12 scenarios

## Contents

1. [At a glance](#at-a-glance)
2. [Key findings](#key-findings)
3. [Test setup](#test-setup)
4. [Methodology](#methodology)
5. [Chronicle of the start attempts](#chronicle-of-the-start-attempts)
6. [Pitfalls](#pitfalls)
7. [Throughput](#throughput)
8. [Response latency](#response-latency)
9. [Prefill scaling](#prefill-scaling)
10. [Memory and startup](#memory-and-startup)
11. [Misdiagnoses](#misdiagnoses-along-the-way)
12. [Refuted assumptions](#refuted-assumptions)
13. [Pinned revisions](#pinned-revisions)
14. [Recommendation](#recommendation)
15. [Limitations](#limitations)
16. [Glossary](#glossary)
17. [Artefacts](#artefacts)

## At a glance

Three server configurations were measured in full. Both starting assumptions — that NVFP4 was the right quantization, and that SGLang would fail on sm_121 for lack of kernels — turned out to be wrong.

### INT4 beats NVFP4 everywhere

19–59 % more throughput on an identical engine, a fivefold shorter load time, a tenfold KV cache. The native FP4 compute path is worth nothing when memory bandwidth is the limit.

### vLLM leads on throughput

13–17 % ahead of SGLang, with a slightly growing lead at higher concurrency — the opposite of the common expectation.

### SGLang leads on latency

TTFT two to three times better (0.19–0.27 s against 0.48–0.69 s) and 79 % more throughput on a recurring prefix — that is where RadixAttention takes effect.

### sm_121a is present

`common_ops.abi3.so` contains sm_121a kernels in the prebuilt cu130 aarch64 wheel. SGLang ran without any source build.

- **64.3** — tok/s peak throughput vLLM+INT4, code, c=4

- **1,001,532** — tokens KV cache vLLM+INT4 at 262k context

- **O(n1.25)** — prefill scaling near-linear instead of quadratic

- **6** — failed server starts out of 10 attempts

## Key findings

### 1. Bandwidth beats compute format

NVFP4 computes natively in FP4 on Blackwell tensor cores, INT4 has to dequantize for the matrix multiplication. On a machine with 273 GB/s of memory bandwidth that is irrelevant: what matters is how many bytes are moved per token. INT4 moves 26 GiB fewer and therefore wins even where NVFP4 could play its native path.

### 2. The NVFP4 checkpoint uses twice as much KV memory per token

77.7 KiB against 38.8 KiB for INT4 — with an identical model and an identical engine. Together with the 26 GiB larger weights, that leads to 94,135 instead of 950,420 tokens of KV cache, a factor of ten. This points to a fault in this build, not to a property of the format.

### 3. The bottleneck at startup is the compiler, not the model

Both engines compile FlashInfer kernels at runtime. A single `cicc` process occupies up to 7.5 GB. At 95 GiB of weights in 121 GB of total memory, the number of parallel compiler processes decides between success and an OOM kill.

### 4. Prefix caching is the single strongest lever

16,913 tokens take 9.32 s cold and 0.30 s warm — a factor of 30. For agents with a growing history that is more effective than any choice of engine.

### 5. Hybrid attention keeps long contexts affordable

At 11.9× the prompt length, prefill time rises only 22.3×. Laguna's split — 12 of 48 layers with full attention, 36 with a 512-token sliding window — dampens the quadratic term far enough that 222,000 tokens stay practically manageable.

## Test setup

### Hardware

**Product** — GX10 (DGX Spark platform)

**GPU** — NVIDIA GB10, compute capability 12.1 (sm_121a)

**CPU** — 20 cores, aarch64

**Memory** — 121 GB unified memory (124,546 MB), 16 GB swap

**Bandwidth** — approx. 273 GB/s

**Storage** — NVMe, 916 GB

**Operating system** — Ubuntu 24.04.4 LTS, kernel 6.17.0-1029-nvidia

**Driver** — NVIDIA 580.173.02, CUDA 13.0, nvcc 13.0.88

### Model

`poolside/Laguna-S-2.1` — 118 B parameters in total, 8 B active per token (mixture-of-experts), released on 22 July 2026 under OpenMDW-1.1.

**Layers** — 48 — of which 12 full attention, 36 sliding-window (window 512)

**Experts** — 256 routed (top-10) plus 1 shared

**Attention** — grouped-query, 8 KV heads, head dimension 128

**Context** — `max_position_embeddings` 262,144, YaRN scaling factor 32 from 8192

**Quantization** — `compressed-tensors`

#### Measured file sizes of the variants

*Queried via the HuggingFace API, sum of all weight files*

| Variant | Size | usable in 121 GB? |
|---|---|---|
| BF16 | 235.1 GB | no |
| FP8 | 131.3 GB | no |
| NVFP4 | 99.7 GB | yes, but with no headroom |
| INT4 | 71.9 GB | yes, with headroom |
| DFlash-NVFP4 / -INT4 | 2.23 GB each | draft model |

### Software

**vLLM** — 0.25.1, aarch64 wheel from PyPI

**SGLang** — 0.5.16 with `sgl-kernel 0.3.21+cu130`

**PyTorch** — 2.11.0+cu130 in both environments

**FlashInfer** — 0.6.13 (vLLM environment)

**Python** — 3.12.13, uv-managed

<details>
<summary>Why native instead of Docker — and why without sudo</summary>

The original plan called for containers for both engines, because that keeps the comparison symmetric. Two obstacles forced the change:

- **No Docker access.** The user is not a member of the `docker` group; every invocation ended in `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`. The first pull attempt nevertheless reported exit code 0, because `| tail -5` in the pipeline swallowed the exit code of the first stage — the failure only surfaced later.

- **No passwordless sudo.** That made it impossible to install `python3.12-dev`, which is needed for the Triton JIT headers.

Solved with `uv python install 3.12`: uv's own CPython distribution ships the development headers, so neither `sudo` nor the system package was needed. Both engines then ran in separate venvs on an identical PyTorch base — for the comparison, even cleaner than two containers with possibly diverging dependencies.

</details>

<details>
<summary>Evidence for the sm_121a kernels</summary>

The claim that `sgl_kernel` does not cover sm_121a and only runs after a rebuild from source was checked in two stages.

#### Source: sgl-kernel/CMakeLists.txt at v0.5.16

```bash
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
    ...
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        list(APPEND SGL_KERNEL_CUDA_FLAGS
            "-gencode=arch=compute_110a,code=sm_110a"
            "-gencode=arch=compute_121a,code=sm_121a")
```

The gencode is therefore emitted exactly when CUDA ≥ 13.0 *and* aarch64 apply — which is precisely the `-cu130` build on this machine. In the `else` branch aarch64 instead gets only `sm_101a`; for cu12x wheels the claim therefore holds.

#### Binary: cuobjdump over the installed libraries

| Library | Size | SM architectures contained |
|---|---|---|
| common_ops.abi3.so | 80 M | sm_90 · sm_90a · sm_100a · sm_103a · sm_110a · sm_120a · **sm_121a** |
| spatial_ops.abi3.so | 196 K | sm_90 · sm_90a · sm_100a · sm_103a · sm_110a · sm_120a · **sm_121a** |
| flashmla_ops.abi3.so | 12 M | sm_90a · sm_100a · sm_103a — no sm_121 |
| flash_ops.abi3.so | 309 M | sm_90a only (FlashAttention-3, Hopper-exclusive) |

The main kernel library contains sm_121a. That FA3 and FlashMLA are missing is correct and expected — neither is usable on GB10 anyway. Those are exactly the two targets the circulating community patch disables; its diagnosis was right, but the conclusion “will not run without a source build” went too far.

</details>

## Methodology

Two purpose-built measurement scripts, both using only the Python standard library, both against the OpenAI-compatible API that vLLM and SGLang alike provide. This gives both engines exactly the same requests.

### bench.py — throughput and response latency

- **4 scenarios** × **3 concurrency levels** (1, 2, 4) × **3 repeats**

- `stream: true` to measure time to first token

- `ignore_eos: true` forces exactly `max_tokens` output tokens and thereby eliminates EOS noise from the tok/s calculation

- `temperature: 0`

- One warm-up run before measuring, so that graph capture and JIT do not enter as outliers

- Medians across all repeats are reported

#### The four scenarios

| Name | Output | Purpose |
|---|---|---|
| prosa_decode | 256 tokens | decode rate at the bandwidth ceiling |
| code_decode | 256 tokens | decode on code, higher DFlash acceptance expected |
| prefill_8k | 64 tokens | prefill throughput on a long prompt |
| prefill_8k_wiederholt | 64 tokens | identical prefix — measures the prefix cache |

### prefill.py — cold prefill

Every request gets a unique prefix: a UUID is woven into every code block of the generated prompt, so that neither vLLM's Automatic Prefix Caching nor SGLang's RadixAttention can take effect. Only TTFT at `max_tokens: 8` is measured; the pure prefill rate follows from it.

> [!IMPORTANT]
> **Two methodological flaws that only surfaced during the run**
>
> **The prefix comparison failed.** `prefill_8k` runs three times per concurrency level — after the very first request the prefix is already in the cache. The median over three runs therefore already measures the warm case, and `prefill_8k_wiederholt` measures that same warm case again. The intended cold-versus-warm contrast did not materialise. Fixed after the fact by `prefill.py`.

> [!IMPORTANT]
> **The size labels were off by a factor of two.** The estimate of 68 tokens per generated code block was wrong; the real figure is about 154. The prompts labelled “8k” actually contained 16,909 to 18,820 tokens. In `prefill.py` the stages were later renamed after the measured token counts.

## Chronicle of the start attempts

Ten server starts were needed to measure three configurations. The order carries information here: every failure supplied the correction for the next attempt. Expanding an entry shows cause, log excerpt and consequence.

<details>
<summary>01 vLLM + NVFP4 — memory reservation too high 17:35 · aborted after < 1 min failed</summary>

Started with `--gpu-memory-utilization 0.92`. The engine aborts before loading.

```bash
ValueError: Free memory on device cuda:0 (110.73/121.63 GiB) on startup is
less than desired GPU memory utilization (0.92, 111.9 GiB). Decrease ...
```

**Cause:** The operating system occupied roughly 11 GiB, so only 110.73 of the 121.63 GiB were free — just under the 111.9 GiB demanded.

**Correction:** `--gpu-memory-utilization 0.89`.

</details>

<details>
<summary>02 vLLM + NVFP4 — OOM killer during JIT compilation 17:36–18:01 · 25 min lost failed</summary>

The run got far: weights loaded (95.63 GiB in 595.6 s), `torch.compile` completed (30.61 s). After that, twelve minutes of silence in the application log, then the EngineCore process was a zombie.

```bash
Aug 01 18:01:30 dgx-spark kernel: systemd invoked oom-killer: ...
Aug 01 18:01:30 dgx-spark kernel: Out of memory: Killed process 21394 (cicc)
    total-vm:10886376kB, anon-rss:7529332kB, UID:1000
```

**Cause:** About 26 GiB remained free after the model. `MAX_JOBS=4` started four parallel `cicc` processes (the CUDA compiler frontend) at roughly 7.5 GB each — 30 GB together. The OOM killer stepped in. There was no error in the application log; the cause was only to be found in the kernel log.

**Correction:** `MAX_JOBS=1`. The widely circulated recommendation `MAX_JOBS=4` is set too high for the NVFP4 variant on this box.

</details>

<details>
<summary>03 vLLM + NVFP4 — terminated along with the parent shell 18:02 · 26 log lines aborted</summary>

Not a technical failure: the process was cleaned up when the session ended. No OOM entry in the kernel log, and the run never got past initialisation.

**Correction:** in future start via `setsid nohup … < /dev/null &`, so that the server survives a session change.

> [!NOTE]
> Incidental finding: a `pkill -f "vllm serve"` twice killed the invoking shell with exit code 143 and 144 respectively, because the search pattern appeared in its own command line. Better to identify processes via `ps` and kill them by PID.

</details>

<details>
<summary>04 vLLM + NVFP4 — successful, first measurement series 21:06–21:23 · 17 min to ready successful</summary>

`MAX_JOBS=1`, `--gpu-memory-utilization 0.89`, `--max-model-len 65536`.

```bash
Model loading took 95.63 GiB memory and 595.56 seconds
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
Available KV cache memory: 6.97 GiB
GPU KV cache size: 94,135 tokens
Maximum concurrency for 65,536 tokens per request: 1.44x
```

Incidentally confirms the native FP4 path: `FLASHINFER_CUTLASS` instead of a Marlin fallback. The smoke test returned 150 tokens in 8.2 s.

**Critical metric:** 1.44× concurrency means that not even one and a half full-context requests fit into memory at the same time.

</details>

<details>
<summary>05 SGLang + NVFP4 — graph capturing stuck at 0/58 21:40–22:01 · aborted after 10 min failed</summary>

`MAX_JOBS=1`, `--mem-fraction-static 0.88`. The weights loaded cleanly (618.01 s, 93.16 GB), after which the progress bar of the CUDA graph capturing sat at step 0 of 58 for ten minutes.

```bash
Capture target prefill CUDA graph begin. backend=breakable
Capturing num tokens (num_tokens=8192 avail_mem=11.78 GB): 0%| | 0/58
```

A `cicc` process ran at 99.9 % CPU and the FlashInfer cache grew from 80 to 179 MB — so it was compiling, just very slowly.

**Diagnosis at the time:** “hung”. **In fact:** the first capture step contains the entire kernel compilation and takes 114 s on its own; the following 57 run orders of magnitude faster. Aborting was premature (see Misdiagnoses).

</details>

<details>
<summary>06 SGLang + NVFP4 — memory down to 1 GB, aborted pre-emptively 22:06–22:25 · MAX_JOBS=3 failed</summary>

Assumption: more parallel compiler jobs would speed up the capturing. Weights loaded in 549 s, then two `cicc` processes — and free system memory fell to **1 GB**. The bar still sat at 0/58.

**Aborted** before the OOM killer could strike. A second OOM would have cost the ten minutes of load time all over again.

</details>

<details>
<summary>07 SGLang + NVFP4 — aborted on instruction 22:25–22:35 · mem-fraction 0.78 aborted</summary>

Third attempt with inverted priorities: `--mem-fraction-static 0.78` instead of 0.88, that is, 26.8 instead of 14.6 GiB of system headroom for the compiler, at the expense of the KV cache. The run was aborted while the weights were loading.

**Interim conclusion at this point:** SGLang fails with NVFP4 on this box not for lack of kernels but on the startup path. That assessment later turned out to be wrong as well — see attempt 09.

</details>

<details>
<summary>08 vLLM + INT4 — successful, quantization comparison 23:19–23:30 · 11 min to ready successful</summary>

All performance parameters identical to attempt 04, only the quantization differs.

```bash
Model loading took 69.34 GiB memory and 113.64 seconds
Available KV cache memory: 35.2 GiB
GPU KV cache size: 950,420 tokens
Maximum concurrency for 65,536 tokens per request: 14.50x
```

Against NVFP4: 26.3 GiB fewer weights, a fivefold shorter load time, a tenfold KV cache. The factor of ten is composed of five times more memory *and* twice as efficient use per token (38.8 instead of 77.7 KiB).

</details>

<details>
<summary>09 SGLang + INT4 — successful, capturing in 3:33 23:37–23:43 · 6 min to ready successful</summary>

`MAX_JOBS=2`, `--mem-fraction-static 0.85`. The first capture step again took 114.6 s, and tqdm extrapolated from it:

```bash
Capturing num tokens (num_tokens=8192): 2%|▏ | 1/58 [01:54<1:48:53, 114.62s/it]
```

In fact the rate fell steeply afterwards — 51 s, 31 s, 21 s, 16 s … down to 6 steps per second at the end:

```bash
Capturing num tokens (num_tokens=4): 100%|██████████| 58/58 [03:33<00:00, 3.68s/it]
[23:43:42] max_total_num_tokens=185897, max_running_requests=48
[23:43:47] The server is fired up and ready to roll!
```

**That refuted the diagnosis from attempts 05 to 07:** SGLang never hung, it was merely slow in the first step. The aborted NVFP4 attempts would probably have completed as well.

</details>

<details>
<summary>10 vLLM + INT4 at 262k context — successful 00:21–00:24 · 3 min to ready successful</summary>

`--max-model-len 262144` instead of 65,536, otherwise unchanged.

```bash
Model loading took 69.34 GiB memory and 115.06 seconds
GPU KV cache size: 1,001,532 tokens
Maximum concurrency for 262,144 tokens per request: 3.82x
```

**Finding:** Quadrupling the context window cost no KV cache — it even yielded 5 % more (1,001,532 instead of 950,420 tokens), presumably through a more favourable block layout. There is no reason to stay below the checkpoint's maximum.

</details>

## Pitfalls

Ordered by cause, each with the countermeasure that worked.

### Memory and compiler

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Parallel JIT fan-out | OOM kill of `cicc`, no error in the application log | `MAX_JOBS=1` with a cold FlashInfer cache |
| Reservation above what is free | `ValueError` before loading | `--gpu-memory-utilization` ≤ 0.89 |
| mem-fraction-static misunderstood | a smaller model creates *no* system headroom | The parameter reserves independently of model size; the saving goes into the KV cache |
| Delayed memory release | restart fails immediately after stopping | Wait for the release, here 5–15 s until 118 GB were free |
| Page cache eviction | shard load rate collapses from 2.7 to 30 s per shard | Unavoidable at 95 GiB of weights; swap activity stays moderate |

### Installation

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Torch ABI break | `undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v` | `torch==2.11.0` — the version pinned by sglang 0.5.16 |
| Outdated recipes | `--force-reinstall torch` pulls the newest version (2.13.0) | Omit the step; the cu130 wheels bring the matching version with them |
| Wrong CUDA wheel | missing sm_121a kernels on cu12x builds | On the Spark use `-cu130` on aarch64 exclusively |
| Missing Python headers | Triton JIT fails without `python3.12-dev` | `uv python install 3.12` — ships the headers, no sudo needed |

### Operation and tooling

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Exit code swallowed by a pipeline | `docker pull … \| tail -5` reports success despite `permission denied` | Check the exit code of the first pipeline stage, not the last |
| `pkill -f` hits your own shell | command ends with exit 143/144, the target keeps running | Determine PIDs via `ps`, kill them specifically |
| Server dies with the session | log breaks off after a few lines, no error | `setsid nohup … < /dev/null &` |
| tqdm remaining time read as fact | displayed 1:48:53, actually 3:33 | For processes with a one-off start-up hurdle, watch the rate over several steps |
| Orphaned `.incomplete` files | 13 GB occupied after an aborted download | Clean up after an abort; otherwise `hf download` resumes correctly |

### Model and API

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Reasoning parser does not initialise | `Auto-initialization of reasoning token IDs failed`; reasoning ends up in the answer text | Unsolved — devalues the code-versus-prose separation |
| Prompt beyond `max_model_len` | `HTTP 400 Bad Request` with no further explanation | Raise the context window; 262,144 costs nothing here |
| Size fields from the HF API | missing `size` fields are counted as 0 | Check for completeness, otherwise totals are 28 GB too low |
| Several models at once | two models over 60 GB do not fit side by side | Strictly sequential operation |
| Model revision not pinned | upstream moves `main`; 93 GB are re-downloaded in silence, then the start aborts on the KV cache | Pin `--revision` for the model and the draft — see below |

## Throughput

*Total throughput in tok/s across all concurrent requests · median of 3 runs*

| Scenario | c | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|---|
| Prose | 1 | 17.3 | 20.6 | 18.2 |
| Prose | 2 | 25.5 | 31.7 | 27.8 |
| Prose | 4 | 42.7 | 52.8 | 45.3 |
| Code | 1 | 15.0 | 23.8 | 21.9 |
| Code | 2 | 27.4 | 39.0 | 33.2 |
| Code | 4 | 42.4 | 64.3 | 56.2 |
| Prefill 18k | 1 | 5.8 | 13.1 | 6.3 |
| Prefill 18k | 2 | 11.6 | 38.9 | 25.1 |
| Prefill 18k | 4 | 18.7 | 64.6 | 38.0 |
| Prefill 18k repeated | 1 | 7.5 | 24.0 | 25.4 |
| Prefill 18k repeated | 2 | 11.8 | 31.1 | 41.1 |
| Prefill 18k repeated | 4 | 17.6 | 39.5 | 70.6 |

### What the numbers show

**INT4 beats NVFP4 throughout** — on an identical engine by between 19 and 59 % in decode. The extreme values in the prefill scenarios (up to +246 %) are not a measurement error but a consequence of the KV cache: four concurrent requests at 16,913 tokens each occupy 67,652 tokens. With NVFP4's capacity of 94,135 tokens that leads to eviction and queueing; with INT4's 950,420 tokens it is uncritical.

**vLLM beats SGLang on decode** — by 13 to 17 %, with a slightly growing lead at higher concurrency. That contradicts the widespread assumption that SGLang scales better over concurrency.

> [!NOTE]
> **The most revealing row**
>
> On the repeated prefix at c=4 the relationship inverts: SGLang reaches **70.6** against vLLM's **39.5** tok/s. In the scenario before it, on first contact with that same prefix, it was the other way round (38.0 against 64.6). SGLang therefore gets better the more often the same context recurs, while vLLM drops off — exactly the profile of agentic coding with a growing history, and exactly the effect RadixAttention promises.

<details>
<summary>Decode rate per individual stream</summary>

While total throughput rises with concurrency, the rate of the individual stream falls — the expected behaviour of a bandwidth-limited system.

*Median decode rate per request in tok/s*

| Scenario | c | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|---|
| Prose | 1 | 18.11 | 21.20 | 18.56 |
| Prose | 2 | 13.29 | 16.81 | 14.25 |
| Prose | 4 | 11.14 | 13.72 | 11.40 |
| Code | 1 | 17.03 | 25.35 | 22.13 |
| Code | 2 | 14.15 | 20.40 | 17.10 |
| Code | 4 | 11.16 | 17.60 | 14.48 |
| Prefill 18k | 1 | 7.90 | 29.04 | 20.27 |
| Prefill 18k | 2 | 6.20 | 22.82 | 14.19 |
| Prefill 18k | 4 | 4.93 | 19.18 | 10.76 |
| Prefill 18k repeated | 1 | 7.77 | 27.25 | 27.94 |
| Prefill 18k repeated | 2 | 6.22 | 17.64 | 22.31 |
| Prefill 18k repeated | 4 | 4.66 | 14.27 | 19.08 |

The low values in the prefill scenarios are an artefact of the short output: at only 64 tokens the fixed-cost share dominates, which is why these numbers are not comparable with the 256-token scenarios.

</details>

## Response latency

*Time to first token in seconds · median*

| Scenario | c | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|---|
| Prose | 1 | 0.352 | 0.276 | 0.314 |
| Prose | 2 | 0.443 | 0.635 | 0.201 |
| Prose | 4 | 0.427 | 0.513 | 0.193 |
| Code | 1 | 0.374 | 0.260 | 0.194 |
| Code | 2 | 0.355 | 0.266 | 0.196 |
| Code | 4 | 0.374 | 0.480 | 0.194 |
| Prefill 18k | 1 | 0.307 | 0.309 | 0.200 |
| Prefill 18k | 2 | 0.468 | 0.458 | 0.217 |
| Prefill 18k | 4 | 0.686 | 0.551 | 0.269 |
| Prefill 18k repeated | 1 | 0.301 | 0.328 | 0.204 |
| Prefill 18k repeated | 2 | 0.460 | 0.484 | 0.255 |
| Prefill 18k repeated | 4 | 0.781 | 0.630 | 0.285 |

SGLang wins in eleven of twelve measurement points. What stands out is the **stability**: the values lie between 0.19 and 0.29 s almost independently of concurrency and prompt length, while vLLM scatters up to 0.78 s. For interactive use, where perceived responsiveness counts, that is a real advantage — it just does not show up in throughput.

## Prefill scaling

Measured with a unique prefix per request, so that no cache takes effect. The three short series come from the main runs, the long series from the follow-up measurement at a 262,144-token context.

*Cold prefill in tok/s · median of 3 runs, each with its own UUID in the prompt*

| Prompt tokens | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|
| ≈ 4,700 | 2051 | 2288 | 1620 |
| ≈ 18,500 | 2019 | 2243 | 1686 |

vLLM is clearly ahead here — by 33 % over SGLang. Together with the throughput advantage, that is the second axis on which vLLM wins.

### Behaviour over prompt length

*vLLM + INT4 at a 262,144-token context window*

| Prompt tokens | TTFT | Prefill | relative |
|---|---|---|---|
| 4,689 | 2.1 s | 2255 tok/s | — |
| 18,699 | 8.1 s | 2301 tok/s | 100 % |
| 72,443 | 39.2 s | 1850 tok/s | 80 % |
| 149,057 | 100.4 s | 1484 tok/s | 64 % |
| 221,996 | 181.3 s | 1224 tok/s | 53 % |

From 18,699 to 221,996 tokens the length is **11.9×**, but the time rises only **22.3×**. From that follows a complexity of about **O(n1.25)** — remarkably close to linear.

For context: with pure full attention one would expect O(n²), so the rate at 222,000 tokens would have collapsed to around 8 % instead of the 53 % measured. Laguna's split — only 12 of the 48 layers with full attention, 36 with a 512-token sliding window — absorbs most of the quadratic term.

> [!NOTE]
> **Practical consequence**
>
> The full 262k context costs an extrapolated **3.7 minutes** of cold prefill. Tolerable once, not per request. With a prefix cache that falls to a fraction: at 16,913 tokens, 0.30 s warm was measured against 9.32 s cold, a **factor of 30**. For agents with a growing history the long context is thus paid for once and reused thereafter.

## Memory and startup

*From the startup logs of the successful runs*

| Metric | vLLM+NVFP4 | vLLM+INT4 (65k) | vLLM+INT4 (262k) | SGLang+INT4 |
|---|---|---|---|---|
| Weights | 95.63 GiB | 69.34 GiB | 69.34 GiB | 67.56 GB |
| Load time | 595.6 s | 113.6 s | 115.1 s | 109.3 s |
| KV memory | 6.97 GiB | 35.2 GiB | — | — |
| KV cache | 94,135 | 950,420 | 1,001,532 | 185,897 |
| KV per token | 77.7 KiB | 38.8 KiB | — | — |
| Max. concurrency | 1.44× (65k) | 14.50× (65k) | 3.82× (262k) | 48 requests |
| Total startup time | ≈ 17 min | ≈ 11 min | ≈ 3 min | ≈ 6 min |

The row **KV per token** is the most unexpected single finding: with an identical model and an identical engine, the NVFP4 checkpoint consumes exactly twice as much KV memory per token as the INT4 checkpoint. Together with the 26 GiB larger weights, that produces the factor of ten in usable cache.

**vLLM against SGLang on KV cache:** 950,420 against 185,897 tokens at comparable memory settings — more than five times. In fairness: `--mem-fraction-static 0.85` and `--gpu-memory-utilization 0.89` are not exactly equivalent, and part of the gap follows from that. But it does not explain a factor of five.

<details>
<summary>Startup behaviour of the two engines compared</summary>

The engines differ structurally in *when* they compile FlashInfer kernels:

- **vLLM** does the JIT in a phase of its own after `torch.compile` and before graph capturing. The OOM kill in attempt 02 hit precisely that phase.

- **SGLang** compiles *during* graph capturing. The first capture step (`num_tokens=8192`) contains the entire compilation and took 114.6 s; the remaining 57 steps together took only 99 s.

The kernel cache in `~/.cache/flashinfer` grew from 80 MB to 234 MB across the attempts and is persistent. Every later start with the same model benefits — SGLang was ready after 6 minutes in the last run, instead of the nearly 25 of the first vLLM success.

</details>

## Misdiagnoses along the way

Four assessments made during the work turned out to be wrong and influenced how it proceeded. They are documented here because they explain the timeline and several aborts.

| Statement | In fact | Consequence |
|---|---|---|
| NVFP4 is 71.9 GB in size | 99.7 GB — for some of the files the HF API returned no `size` field, which was counted as 0 | Memory planning and context recommendation initially too optimistic |
| SGLang's graph capturing “hangs” | It was compiling; the first of 58 steps takes 114 s, after which the rate falls steeply | Attempts 05 and 06 aborted prematurely |
| Capturing takes 1 h 50 min | 3 min 33 s — the figure was a tqdm extrapolation from the first, compile-heavy step | An unnecessary question about how to proceed, while the server was already running |
| INT4 gives SGLang more system headroom | `mem-fraction-static` reserves a share of total memory independently of model size; the saving goes into the KV cache | Wrong expectation of attempt 09, which succeeded for a different reason |

> [!IMPORTANT]
> The common pattern: a **snapshot** was taken for a **state**. A stationary progress bar did not mean standstill, an extrapolated remaining time was not a measurement, an incomplete API response was not completeness. In all four cases a second observation over time would have exposed the error.

## Refuted assumptions

The tests started from circulating recommendations for running Laguna-S-2.1 on the DGX Spark. These were checked against the measurements.

| Claim | Finding | Evidence |
|---|---|---|
| `sgl_kernel` lacks sm_121a, runs only after a rebuild | false | cuobjdump shows sm_121a in `common_ops.abi3.so`; SGLang ran without a source build |
| NVFP4 is the variant for the Spark | false | INT4 wins in all twelve measurement points |
| NVFP4 occupies about 71 GB | false | 99.7 GB per the HF API, 95.63 GiB per the load log |
| INT4 occupies about 59 GB | false | 71.9 GB per the HF API, 69.34 GiB per the load log |
| Prefill 600–800 tok/s | too low | 1224–2301 tok/s depending on prompt length |
| BF16 about 236 GB | confirmed | 235.1 GB |
| Decode without speculation 13–14 tok/s | plausible | with DFlash 18.1–25.3 tok/s on a single stream |
| `MAX_JOBS=4` is necessary | confirmed, but not strict enough | Led to the OOM kill; what was needed was `MAX_JOBS=1` |

## Pinned revisions

Added on 4 August 2026, after a restart of the recommended configuration failed.

On 3 August, poolside moved `main` of `Laguna-S-2.1-INT4` to a checkpoint that is **INT4 in name only**. Starting the unchanged launch script pulled 93 GB in silence for forty minutes and then aborted.

### What changed upstream

The `quantization_config` is byte-for-byte the same in both revisions — 4 bit, `type: int`, `pack-quantized`, group size 32, 48 layers. The difference is in what is *exempted* from it. The new revision adds one line to the ignore list:

```bash
re:^model\.layers\.(?:40|41|42|43|44|45|46|47)\.mlp\.experts\.[0-9]+\.(?:gate_proj|up_proj|down_proj)$
```

That is 8 layers × 256 experts × 3 projections = **6144 weight matrices moved from INT4 to bfloat16**. The tensor inventory confirms it exactly: `weight_packed` drops by 6144, plain `weight` rises by the same number.

*Measured from `config.json` and `model.safetensors.index.json` of both revisions*

| Property | pinned `67dbeda4` | upstream `main` (3 Aug) |
|---|---|---|
| Checkpoint size | 66.96 GiB | 92.84 GiB |
| Shards | 15 | 19 |
| INT4-packed tensors | 36,096 | 29,952 |
| Unquantized weights | 626 | 6,770 |
| `k_scale` / `v_scale` | 96 | 0 |
| KV cache per token | 38.8 KiB | 73.4 KiB |

### Why the server refuses to start

The two changes compound. The larger weights leave less memory, and the missing KV scales then make every token cost twice as much cache:

```bash
ValueError: To serve at least one request with the model's max seq len (262144),
18.35 GiB KV cache is needed, which is larger than the available KV cache
memory (8.72 GiB). Based on the available memory, the estimated
maximum model length is 121840.
```

`--gpu-memory-utilization` was already at 0.89, the ceiling established in attempt 01. There is no headroom left to raise.

### The fix

```bash
MODELL_REV=67dbeda456e68139f281c40831f9d12049d8fc11
DRAFT_REV=f6b32f4fb7ef2fb2ad481bb4c05433a2bf8b0ed1

exec vllm serve poolside/Laguna-S-2.1-INT4 \
  --revision "$MODELL_REV" \
  --speculative-config "{\"model\":\"poolside/Laguna-S-2.1-DFlash-INT4\",\"revision\":\"$DRAFT_REV\",…}" \
```

The draft model is pinned as well. It currently has only one revision, but leaving it unpinned just moves the same failure downstream.

> [!IMPORTANT]
> **All measurements in this report come from the pinned revision.** Values taken from upstream `main` are not comparable: different weights, half the KV cache, and a context window capped at roughly 121,840 instead of 262,144 tokens.

> [!NOTE]
> **Recognising it early**
>
> The symptom is deceptive — forty minutes with no log output, no error, memory filling up. What identifies it is not the log but the process: `write_bytes` in `/proc/<pid>/io` growing into the tens of gigabytes while `read_bytes` stays small. That is a download, not a load. A second snapshot directory under `~/.cache/huggingface/hub/models--*/snapshots/` confirms it.

## Recommendation

### Default: vLLM with INT4

Best throughput, five times the KV cache of SGLang, fastest cold prefill, 3.82 concurrent requests at the full 262k context.

```bash
export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=1

vllm serve poolside/Laguna-S-2.1-INT4 \
  --revision 67dbeda456e68139f281c40831f9d12049d8fc11 \
  --speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash-INT4","revision":"f6b32f4fb7ef2fb2ad481bb4c05433a2bf8b0ed1","num_speculative_tokens":15}' \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --enable-auto-tool-choice \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.89 \
  --host 0.0.0.0 --port 8000
```

### Alternative for interactive agent work: SGLang with INT4

Three times better response latency and 79 % more throughput on a recurring prefix. If the workload has long shared contexts, that can outweigh the throughput disadvantage.

```bash
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=2

python3 -m sglang.launch_server \
  --model-path poolside/Laguna-S-2.1-INT4 \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-INT4 \
  --page-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 65536 \
  --cuda-graph-max-bs 8 \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --host 0.0.0.0 --port 30000
```

> [!NOTE]
> **Do not set `--attention-backend`.** On sm_121 the auto-selection picks `flashinfer`, because `trtllm_mha` does not support SM120 per the source and `fa3` is Hopper-exclusive. Triton is expressly unsuitable for Laguna's sliding-window attention.

## Limitations

- **Reasoning not separated out.** The `poolside_v1` parser could not be initialised. Both text scenarios consist predominantly of reasoning prose, which devalues the separation between code and prose. The engine comparison remains valid, because both engines received identical prompts.

- **Memory parameters not exactly equivalent.** `--mem-fraction-static 0.85` and `--gpu-memory-utilization 0.89` are not the same thing; part of the KV cache difference between the engines follows from that.

- **SGLang measured only with INT4.** The NVFP4 runs were abandoned after three failed attempts — prematurely, as we now know.

- **One measurement series per configuration.** Three repeats per point, but no averaging across sessions. Reproducibility was spot-checked: the 4.7k prefill stage gave 2288 and 2255 tok/s in two runs (a deviation of 1.4 %).

- **The 32k stage failed in the main run.** The prompt exceeded `max_model_len 65536` and was only captured in the follow-up measurement at 262,144.

- **Concurrency only up to 4.** Higher levels were not measured, even though vLLM+INT4 allows 14.5× and SGLang 48 concurrent requests.

## Glossary

**sm_121a**
Compute capability 12.1 of the GB10 GPU. CUDA kernels must be compiled for this architecture; the gencode directive emits them only for CUDA ≥ 13.0 on aarch64.

**NVFP4 / INT4**
Four-bit quantizations. NVFP4 is NVIDIA's microscaling format (E2M1 with FP8 block scales), which Blackwell tensor cores process natively. INT4 stores integers and dequantizes before the matrix multiplication.

**KV cache**
Cache of the key and value tensors of already-processed tokens. Its size limits how many requests at what context length can run concurrently.

**TTFT**
Time to first token — the time from the request to the first token emitted. Determines perceived responsiveness, independently of throughput.

**Prefill / decode**
Prefill processes the input prompt (compute-heavy, parallelisable), decode produces the output token by token (bandwidth-heavy, sequential).

**DFlash**
Draft model trained by poolside for speculative decoding, paired with the base model. Produces candidate tokens cheaply and has them verified in parallel.

**RadixAttention**
SGLang's prefix cache as a radix tree: shared prompt prefixes across several requests are reused automatically. The counterpart in vLLM is called Automatic Prefix Caching.

**Sliding-window attention**
Attention over a bounded window of preceding tokens instead of the whole sequence. Laguna uses a window of 512 in 36 of 48 layers, which makes the KV cache and prefill costs grow much more slowly than quadratically.

**cicc**
Compiler frontend of the CUDA toolchain, invoked by `nvcc`. In these tests it occupied up to 7.5 GB per process and was the trigger of the OOM kill.

**CUDA graph capturing**
Recording fixed execution graphs in advance for recurring batch shapes. Costs startup time and memory, but is indispensable on this hardware — disabling it costs around 55 % throughput per the vendor.

**MoE / active parameters**
Mixture-of-experts: only some of the experts are evaluated per token. Laguna has 118 B parameters in total but only 8 B active — the second number governs decode speed, the first governs the memory footprint.

**Unified memory**
On GB10, CPU and GPU share one physical memory pool. There is no separate VRAM; model weights, KV cache, operating system and compiler processes compete for the same 121 GB.

## Artefacts

#### Measurement tools

**bench.py** — Throughput and TTFT across 4 scenarios × 3 concurrency levels, standard library only

**prefill.py** — Cold prefill across 5 length stages up to 222,000 tokens, unique prefix per request

#### Launch scripts

**start-vllm-int4.sh** — Recommended configuration: INT4, 262k context, DFlash, `MAX_JOBS=1`

**start-sglang-int4.sh** — Latency alternative: INT4, DFlash, `MAX_JOBS=2`

**start-vllm.sh · start-sglang.sh** — NVFP4 variants for reproduction

#### Raw data

**ergebnisse_vllm.json** — vLLM + NVFP4, all 12 measurement points

**ergebnisse_int4.json** — vLLM + INT4, all 12 measurement points

**ergebnisse_sglang.json** — SGLang + INT4, all 12 measurement points

**prefill_*.json** — four prefill series including the long measurement

***.log** — startup logs of all ten attempts, including the failed ones

> [!NOTE]
> After the tests concluded, the NVFP4 weights (93 GB) and the associated draft model (2.1 GB) were removed. What remains in the cache is `Laguna-S-2.1-INT4` (67 GB) and `Laguna-S-2.1-DFlash-INT4` (2.1 GB). The launch scripts `start-vllm.sh` and `start-sglang.sh` still point at the deleted model and would download it again on a start.

> [!NOTE]
> On 4 August 2026 the upstream `main` revision (92.84 GiB, 22 blobs) was deleted again as well, after it had been pulled in by an unpinned start. Only the pinned revision `67dbeda4` remains — 26 files, 66.96 GiB, verified complete.

---

*Test report Laguna-S-2.1 on DGX Spark GB10 · 10 server starts, 3 complete measurement series, 36 measurement points plus 9 prefill stages · all values measured, no extrapolations except the one expressly marked as such to 262,144 tokens.*
