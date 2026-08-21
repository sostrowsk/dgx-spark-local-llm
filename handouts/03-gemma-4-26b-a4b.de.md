# Gemma-4-26B-A4B auf DGX Spark GB10

*Testprotokoll · Inferenz-Benchmark · Teil 3*

> NVFP4 gegen INT4 unter Bedingungen, die erstmals fast ausschließlich den Kernel-Pfad messen — gleiches Modell, gleiche Engine, praktisch gleicher Speicher. Das Ergebnis hängt von der Last ab.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB Unified Memory **Modell** Gemma-4-26B-A4B · 26B MoE, 4B aktiv **Engine** vLLM 0.25.1 **Läufe** 2 Konfigurationen · beide beim ersten Versuch erfolgreich

## Inhalt

1. [Auf einen Blick](#auf-einen-blick)
2. [Kernaussagen](#kernaussagen)
3. [Modell und Varianten](#modell-und-varianten)
4. [Testaufbau](#testaufbau)
5. [Startverlauf](#startverlauf)
6. [Die zwei Kernel-Pfade](#die-zwei-kernel-pfade)
7. [Decode-Durchsatz](#decode-durchsatz)
8. [Antwortlatenz](#antwortlatenz)
9. [Prefill](#prefill)
10. [Speicher](#speicher)
11. [Fallstricke](#fallstricke)
12. [Einordnung in die Testreihe](#einordnung-in-die-testreihe)
13. [Empfehlung](#empfehlung)
14. [Limitationen](#limitationen)
15. [Glossar](#glossar)
16. [Artefakte](#artefakte)

## Auf einen Blick

Zwei Varianten desselben Modells, deren Gewichte sich um 0,75 GiB und deren KV-Cache sich um 1 % unterscheiden. Was übrig bleibt, ist der Unterschied zwischen zwei CUDA-Kernel-Pfaden auf sm_121 — und der kehrt sich mit steigender Last um.

### NVFP4 führt bei niedriger Last

Bei einem einzelnen Stream 17 % schneller, beim kalten Prefill kurzer Prompts sogar 28 %. Dort spielt der native FP4-Pfad seinen Vorteil aus.

### INT4 zieht bei vier Streams gleich

Marlin liefert konstante 44–47 tok/s je Stream über alle Laststufen, während NVFP4 von 53,5 auf 43,2 abfällt. Bei c=4 liegt INT4 im Prosa-Szenario 4 % vorn.

### Beste Latenz der ganzen Reihe

0,036 s bis zum ersten Token — ein Achtel von Laguna, weniger als die Hälfte von Qwen3.6.

### Präfix-Cache funktioniert

16k Token in 0,093 s warm gegen 3,13 s kalt, also Faktor 34. Die Gegenprobe zu Qwen3.6, wo der Effekt vollständig ausblieb.

- **177,1** — tok/s Spitze INT4, Prosa, c=4

- **0,036** — Sekunden TTFT Bestwert der Reihe

- **3 912 140** — Token KV-Cache Bestwert der Reihe

- **7161** — tok/s kaltes Prefill NVFP4 bei 4k

## Kernaussagen

### 1. Der Formatvorteil ist lastabhängig, nicht absolut

NVFP4 gewinnt bei einem Stream um 17 %, bei zwei um 6 %, und bei vier Streams liegt INT4 im Prosa-Szenario 4 % vorn. Die verbreitete Aussage „NVFP4 ist auf Blackwell nativ und daher schneller" gilt also nur für den Teillastbereich.

### 2. Marlin skaliert über den Batch, FlashInfer-CUTLASS nicht

Die Decode-Rate je Stream zeigt zwei unterschiedliche Charakteristiken: INT4 bleibt über alle Laststufen bei 44–47 tok/s, NVFP4 startet bei 53,5 und fällt auf 43,2. Der FlashInfer-Pfad arbeitet schon bei einem Stream nahe seinem Optimum und hat entsprechend weniger Reserve.

### 3. Beim Prefill bleibt NVFP4 klar vorn — mit schrumpfendem Abstand

28 % Vorsprung bei 4k Token, 22 % bei 16k, 12 % bei 65k, 6 % bei 132k. Je rechenintensiver und kürzer, desto wertvoller der native FP4-Pfad.

### 4. Sliding-Window-Attention lässt sich cachen, lineare Attention nicht

Gemma-4 zeigt den Präfix-Cache-Effekt mit Faktor 34, den Qwen3.6 im vorigen Test vermissen ließ. Beide Modelle dämpfen den quadratischen Attention-Anteil, aber nur der Sliding-Window-Ansatz erlaubt die Wiederverwendung — die im Qwen-Test aufgestellte Hypothese wird damit gestützt.

### 5. Erstmals ein aussagekräftiger Code-gegen-Prosa-Vergleich

Gemma-4 arbeitet ohne Reasoning-Modus und liefert echte Antworten. Ergebnis: 53,1 gegen 53,0 tok/s — praktisch identisch. Ohne spekulative Dekodierung ist der Decode rein bandbreitengetrieben und vom Inhalt unabhängig. Bei Laguna und Qwen3.6 war dieser Vergleich durch den Reasoning-Anteil entwertet.

## Modell und Varianten

### Architektur

**Architektur** — `Gemma4ForConditionalGeneration` — multimodal, mit `vision_config`

**Parameter** — 26 Mrd. gesamt, 4 Mrd. aktiv pro Token

**Experten** — 128

**Layer** — 30 — Muster 5 × `sliding_attention` zu 1 × `full_attention`

**Sliding Window** — 1024

**Kontext** — `max_position_embeddings` 262 144

**MTP** — in keinem der geprüften Checkpoints enthalten

### Vermessene Varianten

| Rolle | Repo | Größe | Format | Kernel-Pfad |
|---|---|---|---|---|
| NVFP4 | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 16,5 GB | compressed-tensors | `FLASHINFER_CUTLASS` |
| INT4 | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` | 17,2 GB | compressed-tensors, QAT | `MARLIN` |

Beide im selben Container-Format, damit vLLM sie identisch behandelt und sich ausschließlich das Zahlenformat unterscheidet. Das INT4 ist ein **QAT**-Checkpoint, also quantisierungsbewusst nachtrainiert — ein Qualitätsvorteil, den diese Geschwindigkeitsmessung nicht sichtbar macht.

### Weitere verfügbare Varianten

*Größen über die HuggingFace-API gemessen, GGUF und MLX ausgelassen*

| Repo | Größe | Format |
|---|---|---|
| Intel/…-int4-AutoRound | 15,4 GB | auto-round |
| Intel/…-int4-mixed-AutoRound | 16,2 GB | auto-round |
| RedHatAI/…-NVFP4 | 16,5 GB | compressed-tensors |
| unsloth/…-NVFP4 | 16,9 GB | compressed-tensors |
| cyankiwi/…-AWQ-4bit | 17,2 GB | compressed-tensors |
| cyankiwi/…-qat-AWQ-INT4 | 17,2 GB | compressed-tensors, QAT |
| nvidia/Gemma-4-26B-A4B-NVFP4 | 18,8 GB | modelopt |
| RedHatAI/…-FP8-dynamic | 28,7 GB | compressed-tensors |
| google/gemma-4-26B-A4B-it | 51,6 GB | BF16 |

Alle Repos sind frei zugänglich, kein Gating — bei Google-Modellen keine Selbstverständlichkeit.

## Testaufbau

Hardware, Engine und Messmethodik identisch zu den Laguna- und Qwen3.6-Tests. Beide Varianten über dasselbe Startskript mit Variantenargument, sodass abweichende Parameter ausgeschlossen sind.

```bash
#!/usr/bin/env bash
VARIANTE="${1:-nvfp4}"
case "$VARIANTE" in
  nvfp4) MODELL="RedHatAI/gemma-4-26B-A4B-it-NVFP4" ;;
  int4)  MODELL="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4" ;;
esac

export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=1

exec vllm serve "$MODELL" \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

Kein `--speculative-config`: die Checkpoints bringen keinen MTP-Kopf mit. Gemma-4 misst damit die reine Decode-Leistung ohne spekulativen Multiplikator, den Laguna (DFlash) und Qwen3.6 (MTP) beide hatten.

#### Messwerkzeuge

- `bench.py` — 4 Szenarien × 3 Concurrency-Stufen × 3 Wiederholungen, Streaming für die TTFT, `ignore_eos`, `temperature 0`, Aufwärmlauf vorab, Mediane

- `prefill.py` — kaltes Prefill mit einmaligem UUID-Präfix je Request

## Startverlauf

Beide Läufe gelangen beim ersten Versuch. Die aus den Laguna-Tests erarbeiteten Einstellungen trugen unverändert.

| Kennzahl | NVFP4 | INT4 |
|---|---|---|
| Gewichte | 15,88 GiB | 16,63 GiB |
| Ladezeit | 101,9 s | 39,2 s |
| KV-Cache | 3 912 140 | 3 875 400 |
| Concurrency bei 262k | 14,92× | 14,78× |
| MoE-Backend | FLASHINFER_CUTLASS | MARLIN |
| Ergebnis | erfolgreich | erfolgreich |

> [!IMPORTANT]
> Die Ladezeiten sind **nicht vergleichbar**: der INT4-Lauf folgte unmittelbar auf den NVFP4-Lauf, sodass der Seitencache des Betriebssystems warm war. Die 39,2 s messen also Cache-Wärme, nicht Modelleigenschaften.

## Die zwei Kernel-Pfade

vLLM protokolliert beim Start, welchen MoE-Backend es wählt. Damit ist der Unterschied, der diesem Vergleich zugrunde liegt, direkt belegt statt aus der Literatur zitiert.

```bash
NVFP4: INFO [nvfp4.py:285] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
INT4:  INFO [int_wna16.py:197] Using 'MARLIN' WNA16 MoE backend
```

### Was die beiden Pfade unterscheidet

**FLASHINFER_CUTLASS** nutzt die FP4-Tensor-Cores der Blackwell-Generation direkt. Die Gewichte liegen im E2M1-Layout mit FP8-Blockskalen vor und werden ohne Zwischenschritt verrechnet.

**MARLIN** ist ein Kernel für gewichtsquantisierte Formate mit unquantisierten Aktivierungen (WNA16). Die 4-Bit-Gewichte werden entpackt und in höherer Präzision multipliziert. Der Mehraufwand amortisiert sich über größere Batches — genau das zeigt die Messung.

### Der Effekt in Zahlen

*Decode-Rate je einzelnem Stream, Prosa-Szenario*

| Concurrency | NVFP4 | INT4 | Charakteristik |
|---|---|---|---|
| 1 | 53,5 | 44,1 | NVFP4 nahe Optimum |
| 2 | 50,9 | 46,7 | INT4 gewinnt durch Batching |
| 4 | 43,2 | 45,5 | INT4 überholt |

INT4 bleibt über den gesamten Lastbereich in einem Band von 3 tok/s, NVFP4 verliert 19 %. Die Marlin-Kernel holen sich ihren Anfangsnachteil über den Batch zurück.

## Decode-Durchsatz

*Gesamtdurchsatz in tok/s über alle gleichzeitigen Requests · Median aus 3 Läufen*

| Szenario | c | NVFP4 | INT4 | Delta |
|---|---|---|---|---|
| Prosa | 1 | 53,0 | 43,9 | −17 % |
| Prosa | 2 | 101,0 | 94,9 | −6 % |
| Prosa | 4 | 170,6 | 177,1 | +4 % |
| Code | 1 | 53,1 | 44,1 | −17 % |
| Code | 2 | 101,5 | 95,0 | −6 % |
| Code | 4 | 173,6 | 172,5 | −1 % |
| Prefill 16k | 1 | 27,6 | 22,8 | −18 % |
| Prefill 16k | 2 | 82,3 | 75,9 | −8 % |
| Prefill 16k | 4 | 142,5 | 140,0 | −2 % |
| Prefill 16k wdh. | 1 | 44,8 | 38,3 | −15 % |
| Prefill 16k wdh. | 2 | 85,5 | 77,8 | −9 % |
| Prefill 16k wdh. | 4 | 141,1 | 136,2 | −3 % |

Über alle vier Szenarien dasselbe Muster: bei c=1 liegt NVFP4 15–18 % vorn, bei c=2 noch 6–9 %, bei c=4 ist der Abstand auf −1 bis +4 % geschrumpft.

### Skalierungsverhalten

*Prosa-Szenario · wie viel der Einzelstream-Rate bei höherer Last erhalten bleibt*

| c | NVFP4 erhalten | INT4 erhalten | NVFP4 gesamt | INT4 gesamt |
|---|---|---|---|---|
| 1 | 100 % | 100 % | 53,0 | 43,9 |
| 2 | 95 % | 106 % | 101,0 | 94,9 |
| 4 | 81 % | 103 % | 170,6 | 177,1 |

INT4 wird bei zwei und vier Streams pro Stream sogar *schneller* als bei einem — ein klares Batching-Signal der Marlin-Kernel.

## Antwortlatenz

*Zeit bis zum ersten Token in Sekunden · Median*

| Szenario | c | NVFP4 | INT4 |
|---|---|---|---|
| Prosa | 1 | 0,036 | 0,038 |
| Prosa | 2 | 0,036 | 0,037 |
| Prosa | 4 | 0,069 | 0,064 |
| Code | 1 | 0,042 | 0,042 |
| Code | 2 | 0,045 | 0,044 |
| Code | 4 | 0,078 | 0,076 |
| Prefill 16k | 1 | 0,093 | 0,084 |
| Prefill 16k | 2 | 0,112 | 0,114 |
| Prefill 16k | 4 | 0,153 | 0,180 |
| Prefill 16k wdh. | 1 | 0,077 | 0,078 |
| Prefill 16k wdh. | 2 | 0,097 | 0,098 |
| Prefill 16k wdh. | 4 | 0,142 | 0,128 |

Die TTFT unterscheidet sich zwischen den Formaten praktisch nicht — der Kernel-Pfad wirkt auf den Durchsatz, nicht auf die Anlaufzeit. Bemerkenswert ist das absolute Niveau: **0,036 s** ist der beste Wert der gesamten Testreihe, gegenüber 0,094 s bei Qwen3.6 und 0,276 s bei Laguna.

## Prefill

### Der Präfix-Cache funktioniert

Anders als bei Qwen3.6 greift die Wiederverwendung: 16 386 Token brauchen kalt **3,13 s**, im Hauptlauf mit warmem Cache **0,093 s** — **Faktor 34**. Das stützt die im Qwen-Test aufgestellte Hypothese, dass Sliding-Window-Attention cachebar ist und lineare Attention nicht.

### Kaltes Prefill

*Median aus 3 Läufen mit je eigener UUID im Prompt*

| Token | NVFP4 | INT4 | Delta | NVFP4 relativ |
|---|---|---|---|---|
| ≈ 4 100 | 7161 | 5170 | −28 % | — |
| ≈ 16 300 | 5238 | 4108 | −22 % | 100 % |
| ≈ 64 600 | 2396 | 2109 | −12 % | 46 % |
| ≈ 131 600 | 1416 | 1326 | −6 % | 27 % |

NVFP4 bleibt durchgehend vorn, aber der Abstand halbiert sich mit jeder Verlängerung. Bei sehr langen Prompts dominiert die Attention-Berechnung, und der Unterschied im Matrixmultiplikations-Pfad fällt weniger ins Gewicht.

> [!IMPORTANT]
> **Steiler Abfall bei langen Prompts**
>
> Gemma-4 verliert von 16k auf 65k Token 54 % seiner Prefill-Rate und bis 132k weitere 19 Prozentpunkte. Laguna hielt an vergleichbaren Stellen noch 80 und 64 %, Qwen3.6 76 und 58 %. Eine plausible Ursache ist die Fenstergröße: Gemma-4 nutzt 1024er Sliding Windows gegen Lagunas 512, und jede sechste von 30 Layern rechnet volle Attention. Der quadratische Anteil ist damit größer. Diese Zurückführung ist aus der Konfiguration abgeleitet, nicht direkt gemessen.

## Speicher

| Kennzahl | NVFP4 | INT4 | Differenz |
|---|---|---|---|
| Gewichte | 15,88 GiB | 16,63 GiB | +4,7 % |
| KV-Cache | 3 912 140 | 3 875 400 | −0,9 % |
| Concurrency bei 262k | 14,92× | 14,78× | −0,9 % |

Der Unterschied im KV-Cache entspricht exakt den 0,75 GiB mehr Gewichten. Damit ist der Speicheraspekt in diesem Vergleich **neutral** — anders als bei Laguna, wo NVFP4 mit 99,7 gegen 71,9 GB den Cache um Faktor zehn schrumpfen ließ und das Ergebnis dominierte.

Beide Varianten erreichen zudem die **höchste Cache-Kapazität der gesamten Testreihe**: 3,9 Mio. Token gegen 3,4 Mio. bei Qwen3.6 und 1,0 Mio. bei Laguna.

## Fallstricke

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| `prefill.py` schreibt erst am Ende | Abbruch der letzten Stufe verwirft alle vorherigen Messwerte | Werte aus dem Laufprotokoll rekonstruieren; besser wäre inkrementelles Schreiben |
| Ladezeiten scheinbar verschieden | 39,2 gegen 101,9 s bei fast gleicher Datenmenge | Warmer Seitencache beim zweiten Lauf — nicht als Modelleigenschaft lesen |
| Formatvorteil pauschal angenommen | „NVFP4 ist nativ, also schneller" trifft nur bei niedriger Last zu | Über mehrere Concurrency-Stufen messen, nicht nur bei c=1 |
| Kein MTP verfügbar | Decode ohne spekulativen Multiplikator, anders als bei Laguna und Qwen3.6 | Beim Vergleich zwischen Modellen berücksichtigen |
| Multimodalität unerwartet | Vision-Encoder belegt Speicher auch im reinen Textbetrieb | Vor dem Laden `config.json` auf `vision_config` prüfen |
| QAT-Qualität unsichtbar | Der INT4-Checkpoint ist nachtrainiert, die Messung erfasst nur Tempo | Für die Formatwahl getrennt bewerten |

## Einordnung in die Testreihe

Drei Modelle, dieselbe Hardware, dieselbe Engine, derselbe Harness.

*Jeweils die beste gemessene Variante je Modell*

| Kennzahl | Laguna-S-2.1 | Qwen3.6-35B | Gemma-4-26B |
|---|---|---|---|
| Aktive Parameter | 8 Mrd. | 3 Mrd. | 4 Mrd. |
| Gewichte | 69,34 GiB | 23,45 GiB | 15,88 GiB |
| Decode c=1 | 20,6 | 60,7 | 53,0 |
| Decode c=4 | 52,8 | 180,7 | 177,1 |
| TTFT bestenfalls | 0,260 s | 0,094 s | 0,036 s |
| KV-Cache | 1 001 532 | 3 366 051 | 3 912 140 |
| Präfix-Cache-Faktor | ≈ 30 | 1,0 | ≈ 34 |
| Kaltes Prefill bei 16k | 2301 | 5301 | 5238 |
| Spekulation | DFlash | MTP | keine |

Gemma-4 erreicht das zweitbeste Decode-Tempo *ohne* spekulative Dekodierung, während Qwen3.6 und Laguna beide einen Spekulationsmultiplikator nutzen. Bei den Latenz- und Speicherkennzahlen führt es die Reihe an.

## Empfehlung

### Für Gemma-4 auf dieser Box

- **Einzelnutzer oder niedrige Last** → NVFP4, rund 17 % schneller

- **Ab vier parallelen Streams** → gleichwertig, leichter Vorteil für INT4 bei Prosa

- **Viel Prefill mit kurzen Prompts** → NVFP4, bis zu 28 % schneller

- **Wenn Antwortqualität zählt** → das QAT-INT4 hat einen nicht gemessenen Vorteil

### Korrektur einer früheren Aussage

Aus dem Laguna-Test hatte ich abgeleitet, INT4 schlage NVFP4 auf dieser Hardware. Das Ergebnis war dort korrekt, die Verallgemeinerung nicht: es wurde vom Speicherdruck getrieben (99,7 gegen 71,9 GB auf einer 121-GB-Maschine), nicht vom Rechenpfad. Wo der Speicher keine Rolle spielt, gewinnt NVFP4 bei niedriger Last und liegt bei hoher Last gleichauf.

### Offene Optimierung

Die Messung endet bei vier Streams. INT4 hält dort noch 103 % seiner Einzelstream-Rate, und der KV-Cache ließe 14,8 parallele Vollkontext-Requests zu. Ob INT4 seinen Vorsprung bei acht oder sechzehn Streams ausbaut, ist offen und wäre der naheliegende nächste Test.

## Limitationen

- **215k-Stufe fehlt in beiden Reihen.** Der NVFP4-Lauf wurde dort abgebrochen; der INT4-Lauf wurde zur Wahrung der Symmetrie an derselben Stelle beendet.

- **Prefill-Daten rekonstruiert.** Beide `prefill_gemma4_*.json` wurden aus den Laufprotokollen erzeugt, weil `prefill.py` erst am Ende schreibt.

- **QAT-Qualitätsvorteil nicht gemessen.** Der INT4-Checkpoint ist quantisierungsbewusst nachtrainiert; erfasst wurde ausschließlich Geschwindigkeit.

- **Nur eine Engine.** Ausschließlich vLLM 0.25.1; SGLang wurde für Gemma-4 nicht getestet.

- **Zwei von neun Varianten.** AutoRound, modelopt-NVFP4, FP8 und BF16 blieben ungemessen.

- **Concurrency nur bis 4.** Das Skalierungspotenzial beider Formate ist nicht ausgereizt.

- **Multimodalität ungenutzt.** Alle Messungen mit reinen Textanfragen.

- **Je eine Messreihe.** Drei Wiederholungen je Messpunkt, kein Mittel über mehrere Sitzungen.

## Glossar

**A4B**
Namenskonvention für „4 Mrd. aktive Parameter". Bei MoE-Modellen bestimmt diese Zahl das Decode-Tempo, die Gesamtparameterzahl den Speicherbedarf.

**MARLIN**
CUDA-Kernel für gewichtsquantisierte Formate mit unquantisierten Aktivierungen (WNA16). Entpackt 4-Bit-Gewichte und multipliziert in höherer Präzision; der Mehraufwand amortisiert sich über größere Batches.

**FLASHINFER_CUTLASS**
Kernel-Pfad, der die FP4-Tensor-Cores der Blackwell-Generation direkt anspricht. Auf sm_121 der native Weg für NVFP4.

**QAT**
Quantization-Aware Training. Das Modell wird mit simulierter Quantisierung nachtrainiert, wodurch der Qualitätsverlust gegenüber nachträglicher Quantisierung geringer ausfällt.

**Sliding-Window-Attention**
Attention über ein begrenztes Fenster zurückliegender Token. Gemma-4 nutzt Fenster von 1024 in fünf von sechs Layern. Anders als lineare Attention lässt sich der zugehörige KV-Cache stückweise wiederverwenden.

**Präfix-Cache**
Wiederverwendung der KV-Tensoren eines bereits verarbeiteten Prompt-Anfangs. Bei vLLM Automatic Prefix Caching, bei SGLang RadixAttention.

**compressed-tensors**
Quantisierungs-Container, den vLLM und SGLang direkt lesen. Kann sowohl NVFP4 als auch INT4-Varianten enthalten — dadurch war der hier gezogene Vergleich mit identischem Container möglich.

**TTFT**
Time To First Token. Bestimmt die gefühlte Reaktionszeit; bei langen Prompts vom Prefill dominiert, bei kurzen vom Scheduling.

## Artefakte

**start-vllm-gemma4.sh** — Startskript, Variante über Argument `nvfp4` oder `int4`

**bench.py** — Durchsatz und TTFT über 4 Szenarien × 3 Concurrency-Stufen

**prefill.py** — Kaltes Prefill mit einmaligem UUID-Präfix je Request

**ergebnisse_gemma4_nvfp4.json** — 12 Messpunkte NVFP4

**ergebnisse_gemma4_int4.json** — 12 Messpunkte INT4

**prefill_gemma4_*.json** — je 4 Prefill-Stufen, aus den Protokollen rekonstruiert

**gemma4-*.log** — Startprotokolle beider Läufe, mit den Backend-Zeilen

> [!NOTE]
> Alle Dateien in `~/bench/`. Beide Modelle verbleiben mit zusammen 33 GB im HuggingFace-Cache, der FlashInfer-Kernel-Cache ist warm — ein Neustart dauert wenige Minuten.

---

*Testprotokoll Gemma-4-26B-A4B auf DGX Spark GB10 · 2 Konfigurationen, je 12 Messpunkte und 4 Prefill-Stufen · Teil 3 der Reihe nach Laguna-S-2.1 und Qwen3.6-35B-A3B · alle Werte gemessen.*
