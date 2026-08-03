#!/usr/bin/env bash
# vLLM 0.25.1 mit Laguna-S-2.1-NVFP4 + DFlash auf GB10 (sm_121a).
set -euo pipefail

export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
# Ungebremster nvcc-Fan-out bei kaltem FlashInfer-Cache kann die 128 GB
# Unified Memory ausschoepfen und die Maschine mitreissen.
export MAX_JOBS=1
export VLLM_LOGGING_LEVEL=INFO

source ~/bench/venv-vllm/bin/activate

exec vllm serve poolside/Laguna-S-2.1-NVFP4 \
  --speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash-NVFP4","num_speculative_tokens":15}' \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --enable-auto-tool-choice \
  --max-num-seqs 32 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.89 \
  --host 0.0.0.0 --port 8000 \
  "$@"
