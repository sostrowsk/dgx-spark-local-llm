#!/usr/bin/env bash
# vLLM 0.25.1 mit Qwen3.8-27B-NVFP4 + MTP auf GB10 (sm_121a).
# Gegenstueck zu start-vllm-qwen38-int4.sh: nur die Quantisierung unterscheidet
# sich, alle Leistungsparameter bleiben identisch, damit der Vergleich sauber ist.
#
# Anders als Qwen3.6-35B-A3B ist Qwen3.8-27B ein dense Modell -- 27 B aktive
# Parameter statt 3 B. Bandbreite ist damit der Flaschenhals: 21,8 GiB muessen
# bei jedem Token durch die ~273 GB/s des Spark, was das Decode-Dach rechnerisch
# bei ~12 tok/s deckelt. Die MTP-Schicht (mtp_num_hidden_layers=1) ist der einzige
# Hebel dagegen und daher eingeschaltet.
#
# "NVFP4" ist hier mixed-precision, nicht durchgaengig 4 Bit: die MLP-Layer
# liegen in NVFP4 (W4A4, tensor_group gs=16), die Attention-Projektionen in
# FP8 (W8A8). Daher 21,8 GiB statt der naiv erwarteten ~14.
# Die linear_attn-Layer bleiben komplett unquantisiert (303 ignore-Eintraege).
set -euo pipefail

# Gepinnt nach der Laguna-Erfahrung: poolside hat main unter uns umgestellt und
# damit die Messreihe entwertet. Fuer Qwen3.8 gilt dieselbe Vorsicht.
MODELL_REV=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108

export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
# Vier parallele cicc-Prozesse haben den ersten Laguna-Lauf per OOM-Killer gerissen.
export MAX_JOBS=1
export VLLM_LOGGING_LEVEL=INFO

source ~/bench/venv-vllm/bin/activate

exec vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --revision "$MODELL_REV" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000 \
  "$@"
