#!/usr/bin/env bash
# vLLM 0.25.1 mit Qwen3.8-27B-INT4 + MTP auf GB10 (sm_121a).
# Gegenstueck zu start-vllm-qwen38-nvfp4.sh: nur die Quantisierung unterscheidet
# sich, alle Leistungsparameter bleiben identisch, damit der Vergleich sauber ist.
#
# INT4 ist hier W4A16 (group_size 128, compressed-tensors pack-quantized): die
# Gewichte liegen in 4 Bit, die Aktivierungen bleiben bf16. vLLM waehlt dafuer
# den MARLIN-Kernel, waehrend die NVFP4-Variante ueber FLASHINFER_CUTLASS geht.
# Genau dieser Kernel-Unterschied hat bei Gemma-4 den Lastverlauf gedreht:
# CUTLASS vorn bei c=1, Marlin ab c=4.
#
# Mit 18,1 GiB gegen 21,8 GiB liegen beide Varianten dicht genug beieinander,
# dass -- anders als bei Laguna -- kein Speicherdruck den Vergleich verfaelscht.
set -euo pipefail

# Gepinnt nach der Laguna-Erfahrung: poolside hat main unter uns umgestellt und
# damit die Messreihe entwertet. Fuer Qwen3.8 gilt dieselbe Vorsicht.
MODELL_REV=2fb0debc365fb6c1683d7d3ad7722470919627a8

export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
# Vier parallele cicc-Prozesse haben den ersten Laguna-Lauf per OOM-Killer gerissen.
export MAX_JOBS=1
export VLLM_LOGGING_LEVEL=INFO

source ~/bench/venv-vllm/bin/activate

exec vllm serve RedHatAI/Qwen3.8-27B-INT4 \
  --revision "$MODELL_REV" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000 \
  "$@"
