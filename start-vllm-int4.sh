#!/usr/bin/env bash
# vLLM 0.25.1 mit Laguna-S-2.1-INT4 + DFlash auf GB10 (sm_121a).
# Gegenstueck zu start-vllm.sh (NVFP4): nur die Quantisierung unterscheidet sich,
# alle Leistungsparameter bleiben identisch, damit der Vergleich sauber ist.
#
# Revisionen sind gepinnt. poolside hat main am 03.08.2026 auf einen Checkpoint
# umgestellt, der trotz des Namens INT4 nur teilweise quantisiert ist: die Experten
# der Layer 40-47 liegen in bfloat16 (92,84 statt 66,96 GiB), und die k_scale/v_scale
# fehlen, was den KV-Cache von 38,8 auf 73,4 KiB je Token verdoppelt. Damit passen
# 262144 Token nicht mehr in den Speicher -- der Start bricht mit
# "18.35 GiB KV cache is needed ... available 8.72 GiB" ab.
# Alle Messwerte dieses Repos stammen von der hier gepinnten Revision.
set -euo pipefail

MODELL_REV=67dbeda456e68139f281c40831f9d12049d8fc11
DRAFT_REV=f6b32f4fb7ef2fb2ad481bb4c05433a2bf8b0ed1

export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
# Vier parallele cicc-Prozesse haben den ersten NVFP4-Lauf per OOM-Killer gerissen.
export MAX_JOBS=1
export VLLM_LOGGING_LEVEL=INFO

source ~/bench/venv-vllm/bin/activate

exec vllm serve poolside/Laguna-S-2.1-INT4 \
  --revision "$MODELL_REV" \
  --speculative-config "{\"model\":\"poolside/Laguna-S-2.1-DFlash-INT4\",\"revision\":\"$DRAFT_REV\",\"num_speculative_tokens\":15}" \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --enable-auto-tool-choice \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.89 \
  --host 0.0.0.0 --port 8000 \
  "$@"
