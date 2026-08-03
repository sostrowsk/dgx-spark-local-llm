#!/usr/bin/env python3
"""
Engine-Benchmark fuer Laguna-S-2.1 auf DGX Spark (GB10).
Nur stdlib. Spricht die OpenAI-kompatible API, die vLLM und SGLang beide anbieten.

Gemessen wird pro Request:
  ttft_s      Zeit bis zum ersten Token (Streaming)
  decode_tps  Output-Tokens / (Gesamtzeit - ttft)
  total_s     Gesamtdauer

Szenarien decken die Achsen ab, auf denen sich die Engines unterscheiden koennen:
Decode-Speed (bandbreitenlimitiert), Prefill, und Praefix-Wiederverwendung.
"""

import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- Szenarien

# ~8k Token gemeinsamer Praefix. Zweimal gefahren: der zweite Lauf zeigt,
# was Praefix-Caching bringt (RadixAttention bei SGLang, APC bei vLLM).
LONG_PREFIX = (
    "Du bist ein Code-Review-Assistent. Hier ist der relevante Auszug aus der "
    "Codebasis, den du fuer alle folgenden Fragen heranziehen sollst.\n\n"
    + "\n".join(
        f"def modul_{i}_verarbeite(daten, konfiguration=None):\n"
        f'    """Verarbeitet Datensaetze fuer Teilsystem {i}."""\n'
        f"    if konfiguration is None:\n"
        f"        konfiguration = {{'schwelle': {i * 7}, 'modus': 'strikt'}}\n"
        f"    ergebnis = []\n"
        f"    for eintrag in daten:\n"
        f"        if eintrag.get('wert', 0) > konfiguration['schwelle']:\n"
        f"            ergebnis.append({{'id': eintrag['id'], 'stufe': {i}}})\n"
        f"    return ergebnis\n"
        for i in range(120)
    )
)

SCENARIOS = [
    {
        "name": "prosa_decode",
        "prompt": "Erklaere ausfuehrlich, wie Speicherbandbreite die Decode-"
                  "Geschwindigkeit von Sprachmodellen begrenzt.",
        "max_tokens": 256,
    },
    {
        "name": "code_decode",
        "prompt": "Schreibe eine vollstaendige Python-Implementierung eines "
                  "LRU-Caches mit Typannotationen und Docstrings.",
        "max_tokens": 256,
    },
    {
        "name": "prefill_8k",
        "prompt": LONG_PREFIX + "\n\nFrage: Welche Funktion nutzt die hoechste Schwelle?",
        "max_tokens": 64,
    },
    {
        "name": "prefill_8k_wiederholt",  # identischer Praefix -> Cache-Treffer
        "prompt": LONG_PREFIX + "\n\nFrage: Wie viele Module sind definiert?",
        "max_tokens": 64,
    },
]

CONCURRENCIES = [1, 2, 4]
REPEATS = 3  # Requests pro Szenario und Concurrency-Stufe


# ---------------------------------------------------------------- Messkern

def eine_anfrage(base_url, modell, prompt, max_tokens, timeout=600):
    """Streamt einen Request und misst TTFT sowie Decode-Rate."""
    payload = {
        "model": modell,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Erzwingt exakt max_tokens Ausgabe -> saubere tok/s ohne EOS-Rauschen.
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for roh in resp:
            zeile = roh.decode("utf-8").strip()
            if not zeile.startswith("data: "):
                continue
            rest = zeile[6:]
            if rest == "[DONE]":
                break
            try:
                obj = json.loads(rest)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for wahl in obj.get("choices", []):
                delta = wahl.get("delta", {})
                # Je nach Reasoning-Parser landet der Text in content,
                # reasoning oder reasoning_content. Alle drei zaehlen als Ausgabe.
                if any(delta.get(f) for f in ("content", "reasoning", "reasoning_content")):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    chunks += 1

    gesamt = time.perf_counter() - start
    aus_tokens = (usage or {}).get("completion_tokens") or chunks
    ein_tokens = (usage or {}).get("prompt_tokens")
    decode_zeit = gesamt - (ttft or 0)
    return {
        "ttft_s": ttft,
        "total_s": gesamt,
        "out_tokens": aus_tokens,
        "in_tokens": ein_tokens,
        "decode_tps": (aus_tokens / decode_zeit) if decode_zeit > 0 else None,
    }


def parallel_fahren(base_url, modell, szenario, nebenlaeufig):
    """Startet `nebenlaeufig` Requests gleichzeitig, sammelt alle Messwerte."""
    ergebnisse = []
    sperre = threading.Lock()
    fehler = []

    def arbeiter():
        try:
            r = eine_anfrage(base_url, modell, szenario["prompt"], szenario["max_tokens"])
            with sperre:
                ergebnisse.append(r)
        except Exception as exc:  # Netz-, Timeout- oder 4xx-Fehler festhalten
            with sperre:
                fehler.append(repr(exc))

    wand_start = time.perf_counter()
    faeden = [threading.Thread(target=arbeiter) for _ in range(nebenlaeufig)]
    for t in faeden:
        t.start()
    for t in faeden:
        t.join()
    wand = time.perf_counter() - wand_start

    return ergebnisse, fehler, wand


def median(werte):
    sauber = [w for w in werte if w is not None]
    return round(statistics.median(sauber), 3) if sauber else None


def main():
    if len(sys.argv) < 4:
        print("Aufruf: bench.py <base_url> <modellname> <engine-label> [ausgabe.json]")
        sys.exit(2)
    base_url, modell, label = sys.argv[1], sys.argv[2], sys.argv[3]
    ziel = sys.argv[4] if len(sys.argv) > 4 else f"ergebnisse_{label}.json"

    print(f"== Benchmark {label} gegen {base_url} ==", flush=True)

    # Aufwaermen: erste Anfrage triggert Graph-Capture/JIT und waere sonst Ausreisser.
    print("Aufwaermen ...", flush=True)
    try:
        eine_anfrage(base_url, modell, "Sag kurz Hallo.", 16)
    except Exception as exc:
        print(f"  Aufwaermen fehlgeschlagen: {exc!r}")

    bericht = {"engine": label, "base_url": base_url, "modell": modell, "laeufe": []}

    for szenario in SCENARIOS:
        for nebenlaeufig in CONCURRENCIES:
            alle, fehler, wand = [], [], 0.0
            for _ in range(REPEATS):
                r, f, w = parallel_fahren(base_url, modell, szenario, nebenlaeufig)
                alle.extend(r)
                fehler.extend(f)
                wand += w

            if not alle:
                print(f"{szenario['name']:24s} c={nebenlaeufig}  FEHLER: {fehler[:1]}", flush=True)
                bericht["laeufe"].append({
                    "szenario": szenario["name"], "concurrency": nebenlaeufig,
                    "fehler": fehler[:3], "ok": 0,
                })
                continue

            summe_out = sum(r["out_tokens"] or 0 for r in alle)
            eintrag = {
                "szenario": szenario["name"],
                "concurrency": nebenlaeufig,
                "ok": len(alle),
                "fehler": fehler[:3],
                "ttft_s_median": median([r["ttft_s"] for r in alle]),
                "decode_tps_median": median([r["decode_tps"] for r in alle]),
                "gesamt_tps": round(summe_out / wand, 2) if wand > 0 else None,
                "in_tokens": alle[0]["in_tokens"],
                "out_tokens": alle[0]["out_tokens"],
            }
            bericht["laeufe"].append(eintrag)
            print(
                f"{szenario['name']:24s} c={nebenlaeufig}  "
                f"ttft={eintrag['ttft_s_median']}s  "
                f"decode={eintrag['decode_tps_median']} tok/s  "
                f"gesamt={eintrag['gesamt_tps']} tok/s"
                + (f"  [{len(fehler)} Fehler]" if fehler else ""),
                flush=True,
            )

    with open(ziel, "w") as fh:
        json.dump(bericht, fh, indent=2, ensure_ascii=False)
    print(f"\ngeschrieben: {ziel}")


if __name__ == "__main__":
    main()
