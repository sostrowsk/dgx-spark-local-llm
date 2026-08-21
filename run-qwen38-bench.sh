#!/usr/bin/env bash
# Wartet, bis der vLLM-Server auf Port 8000 bereit ist, und fahrt dann
# bench.py (warmer Fall) und prefill.py (kalter Fall) gegen ihn.
#
# Aufruf: run-qwen38-bench.sh <modellname> <label>
set -uo pipefail

MODELL="$1"
LABEL="$2"
cd ~/bench
source ~/bench/venv-vllm/bin/activate

echo "== warte auf Server ($LABEL) =="
# 40 min Deckel: Qwen3.6 brauchte ~14 min bis ready, dense 27B darf laenger.
for i in $(seq 1 2400); do
  if curl -s --max-time 3 http://localhost:8000/health > /dev/null 2>&1; then
    echo "bereit nach ${i}s"
    break
  fi
  if [ "$i" -eq 2400 ]; then echo "ABBRUCH: Server nach 40 min nicht bereit"; exit 1; fi
  sleep 1
done

echo
echo "== bench.py =="
python3 bench.py http://localhost:8000 "$MODELL" "$LABEL" "ergebnisse_${LABEL}.json"

echo
echo "== prefill.py =="
python3 prefill.py http://localhost:8000 "$MODELL" "$LABEL" "prefill_${LABEL}.json"

echo
echo "== SpecDecoding-Metriken (letzte Zeile) =="
grep 'SpecDecoding metrics' "qwen38-${LABEL#qwen38-}.log" 2>/dev/null | tail -1

echo "== FERTIG $LABEL =="
