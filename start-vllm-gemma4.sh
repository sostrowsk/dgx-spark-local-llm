#!/usr/bin/env bash
# vLLM 0.25.1 mit Gemma-4-26B-A4B auf GB10 (sm_121a).
# Variante ueber $1 waehlbar: nvfp4 (Vorgabe) oder int4.
# Leistungsparameter identisch zu den Laguna- und Qwen3.6-Laeufen, damit die
# Messwerte vergleichbar bleiben. Kein MTP: die Checkpoints liefern keines mit.
set -euo pipefail

VARIANTE="${1:-nvfp4}"
shift || true

case "$VARIANTE" in
  nvfp4) MODELL="RedHatAI/gemma-4-26B-A4B-it-NVFP4" ;;
  int4)  MODELL="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4" ;;
  *) echo "Unbekannte Variante '$VARIANTE' — erlaubt: nvfp4, int4" >&2; exit 2 ;;
esac

export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
# Vier parallele cicc-Prozesse haben den ersten Laguna-Lauf per OOM-Killer
# gerissen. Hier ist reichlich Luft, die Regel bleibt trotzdem.
export MAX_JOBS=1
export VLLM_LOGGING_LEVEL=INFO

source ~/bench/venv-vllm/bin/activate

echo "Starte $MODELL"
exec vllm serve "$MODELL" \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000 \
  "$@"
