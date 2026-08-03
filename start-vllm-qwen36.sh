#!/usr/bin/env bash
# vLLM 0.25.1 mit Qwen3.6-35B-A3B-NVFP4 + MTP auf GB10 (sm_121a).
# Leistungsparameter bewusst identisch zu start-vllm-int4.sh, damit die
# Messwerte mit den Laguna-Laeufen vergleichbar bleiben.
set -euo pipefail

export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
# Vier parallele cicc-Prozesse haben den ersten Laguna-Lauf per OOM-Killer
# gerissen. Hier ist zwar reichlich Luft, aber die Regel bleibt.
export MAX_JOBS=1
export VLLM_LOGGING_LEVEL=INFO

source ~/bench/venv-vllm/bin/activate

exec vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000 \
  "$@"
