#!/usr/bin/env python3
"""
Kalte Prefill-Messung: jeder Request bekommt einen einmaligen Praefix,
damit weder vLLMs APC noch SGLangs RadixAttention greifen kann.

Ergaenzt bench.py, das den warmen Fall misst. Erst beide zusammen erlauben
die Aussage, wie viel der Praefix-Cache tatsaechlich bringt.
"""

import json
import statistics
import sys
import time
import urllib.request
import uuid

BAUSTEIN = (
    "def modul_{i}_verarbeite(daten, konfiguration=None):\n"
    '    """Verarbeitet Datensaetze fuer Teilsystem {i} im Lauf {nonce}."""\n'
    "    if konfiguration is None:\n"
    "        konfiguration = {{'schwelle': {s}, 'modus': 'strikt'}}\n"
    "    ergebnis = []\n"
    "    for eintrag in daten:\n"
    "        if eintrag.get('wert', 0) > konfiguration['schwelle']:\n"
    "            ergebnis.append({{'id': eintrag['id'], 'stufe': {i}}})\n"
    "    return ergebnis\n"
)

# Bausteine pro Groessenstufe; gemessen ~154 Token je Baustein.
# Namen nach der tatsaechlichen Tokenzahl, nicht nach der urspruenglichen Schaetzung.
STUFEN = [
    ("~5k", 30),
    ("~18k", 120),
    ("~72k", 470),
    ("~145k", 940),
    ("~215k", 1400),
]
WIEDERHOLUNGEN = 3


def einmaliger_prompt(bausteine):
    """Baut einen Prompt, der sich in jedem Block vom vorigen Lauf unterscheidet."""
    nonce = uuid.uuid4().hex[:12]
    koerper = "".join(
        BAUSTEIN.format(i=i, s=i * 7 + len(nonce), nonce=nonce) for i in range(bausteine)
    )
    return (
        f"Code-Review-Kontext (Lauf {nonce}):\n\n{koerper}\n\n"
        "Frage: Nenne in einem Satz, was diese Module gemeinsam haben."
    )


def messen(base_url, modell, prompt, timeout=900):
    payload = {
        "model": modell,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,          # minimal, wir wollen nur die TTFT
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for roh in resp:
            z = roh.decode("utf-8").strip()
            if not z.startswith("data: "):
                continue
            rest = z[6:]
            if rest == "[DONE]":
                break
            try:
                obj = json.loads(rest)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for w in obj.get("choices", []):
                delta = w.get("delta", {})
                if ttft is None and any(
                    delta.get(f) for f in ("content", "reasoning", "reasoning_content")
                ):
                    ttft = time.perf_counter() - start
    return ttft, (usage or {}).get("prompt_tokens")


def main():
    if len(sys.argv) < 4:
        print("Aufruf: prefill.py <base_url> <modell> <label> [ausgabe.json]")
        sys.exit(2)
    base_url, modell, label = sys.argv[1], sys.argv[2], sys.argv[3]
    ziel = sys.argv[4] if len(sys.argv) > 4 else f"prefill_{label}.json"

    print(f"== Kaltes Prefill, {label} ==", flush=True)
    bericht = {"engine": label, "modell": modell, "stufen": []}

    for name, bausteine in STUFEN:
        ttfts, tokens = [], None
        for _ in range(WIEDERHOLUNGEN):
            try:
                t, n = messen(base_url, modell, einmaliger_prompt(bausteine))
                if t:
                    ttfts.append(t)
                    tokens = n or tokens
            except Exception as exc:
                print(f"  {name}: Fehler {exc!r}", flush=True)
        if not ttfts:
            continue
        med = statistics.median(ttfts)
        rate = (tokens / med) if tokens else None
        bericht["stufen"].append({
            "stufe": name, "prompt_tokens": tokens,
            "ttft_s_median": round(med, 3),
            "prefill_tps": round(rate) if rate else None,
        })
        print(f"  {name:4s}  {tokens} Token  ttft={med:.2f}s  "
              f"-> {rate:.0f} tok/s Prefill" if rate else f"  {name}: -", flush=True)

    with open(ziel, "w") as fh:
        json.dump(bericht, fh, indent=2, ensure_ascii=False)
    print(f"geschrieben: {ziel}")


if __name__ == "__main__":
    main()
