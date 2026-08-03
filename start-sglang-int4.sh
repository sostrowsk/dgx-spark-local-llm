#!/usr/bin/env bash
# SGLang 0.5.16 mit Laguna-S-2.1-INT4 + DFlash auf GB10 (sm_121a).
set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# Vier parallele cicc-Prozesse haben vLLM per OOM-Killer gerissen.
export MAX_JOBS=2

source ~/bench/venv-sglang/bin/activate

# Kein --attention-backend: die Auto-Auswahl trifft auf sm_121 flashinfer,
# und Triton ist fuer Lagunas Sliding-Window-Attention ausdruecklich falsch.
exec python3 -m sglang.launch_server \
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
  --host 0.0.0.0 --port 30000 \
  "$@"
