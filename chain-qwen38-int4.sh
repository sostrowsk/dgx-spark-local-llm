#!/usr/bin/env bash
# Wartet, bis der NVFP4-Lauf durch ist, schaltet dann auf INT4 um und
# benchmarkt dieselbe Reihe. Zweck: kein Leerlauf zwischen den Laeufen.
set -uo pipefail
cd ~/bench

NVFP4_WRAPPER_PID="$1"

echo "== warte auf Ende des NVFP4-Laufs (PID $NVFP4_WRAPPER_PID) =="
while kill -0 "$NVFP4_WRAPPER_PID" 2>/dev/null; do sleep 15; done
echo "NVFP4-Lauf beendet $(date +%H:%M:%S)"

echo "== stoppe NVFP4-Server =="
pkill -f 'vllm serve unsloth/Qwen3.8-27B-NVFP4' || true
for i in $(seq 1 60); do
  ss -tln 2>/dev/null | grep -q ':8000 ' || { echo "Port 8000 frei nach ${i}s"; break; }
  sleep 1
done
# EngineCore-Reste einsammeln, sonst blockieren sie den Speicher.
pkill -f 'VLLM::EngineCore' 2>/dev/null || true
sleep 5

echo "== starte INT4-Server $(date +%H:%M:%S) =="
nohup ./start-vllm-qwen38-int4.sh > qwen38-int4.log 2>&1 &
echo "Server-PID $!"

exec ./run-qwen38-bench.sh "RedHatAI/Qwen3.8-27B-INT4" "qwen38-int4"
