# Qwen3.6-35B-A3B auf DGX Spark GB10

*Testprotokoll · Inferenz-Benchmark · Teil 2*

> NVFP4 mit Multi-Token-Prediction, gemessen mit demselben Harness wie Laguna-S-2.1 — dreifaches Decode-Tempo bei einem Drittel des Speicherbedarfs, aber ohne wirksame Präfix-Wiederverwendung.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB Unified Memory **Modell** RedHatAI/Qwen3.6-35B-A3B-NVFP4 · 35B MoE, 3B aktiv **Engine** vLLM 0.25.1 **Serverstarts** 1 · beim ersten Versuch erfolgreich

## Inhalt

1. [Auf einen Blick](#auf-einen-blick)
2. [Kernaussagen](#kernaussagen)
3. [Modell und Varianten](#modell-und-varianten)
4. [Testaufbau](#testaufbau)
5. [Startverlauf](#startverlauf)
6. [Decode-Durchsatz](#decode-durchsatz)
7. [Antwortlatenz](#antwortlatenz)
8. [Der fehlende Präfix-Cache](#der-fehlende-präfix-cache)
9. [Prefill-Skalierung](#prefill-skalierung)
10. [Speicher](#speicher)
11. [Fallstricke](#fallstricke)
12. [Vergleich mit Laguna](#vergleich-mit-laguna-s-21)
13. [Empfehlung](#empfehlung)
14. [Limitationen](#limitationen)
15. [Glossar](#glossar)
16. [Artefakte](#artefakte)

## Auf einen Blick

Eine Messreihe, ein einziger Serverstart, keine gescheiterten Versuche — im Gegensatz zu den zehn Anläufen der Laguna-Tests. Das Ergebnis ist auf zwei Achsen eindeutig und auf einer dritten überraschend.

### Dreifaches Decode-Tempo

60,7 bis 201,2 tok/s gegen Lagunas 20,6 bis 64,3 — Faktor 2,7 bis 3,4 über alle Decode-Szenarien. Entspricht dem Verhältnis der aktiven Parameter, 3 zu 8 Mrd.

### Ein Drittel des Speichers

23,45 GiB Gewichte statt 69,34. Dadurch 3 366 051 Token KV-Cache und 12,84× Concurrency bei vollem 262k-Kontext.

### Kein Präfix-Cache-Effekt

Warmer und kalter Lauf identisch: 2,822 gegen 2,806 s. Laguna löste denselben Präfix in 0,30 s auf — Faktor 30 durch Caching, der hier vollständig entfällt.

### Doppelt so schnelles Prefill

5301 tok/s gegen 2301 bei 16k Token, und 2429 gegen 1224 bei knapp 200k. Der Rohdurchsatz ist durchweg gut doppelt so hoch.

- **201,2** — tok/s Spitzendurchsatz Code, c=4

- **0,094** — Sekunden TTFT bestes Ergebnis

- **3 366 051** — Token KV-Cache bei 262k Kontext

- **1,00×** — Cache-Gewinn bei wiederholtem Präfix

## Kernaussagen

### 1. Aktive Parameter bestimmen das Tempo

Qwen3.6 aktiviert 3 Mrd. Parameter pro Token, Laguna 8 Mrd. Das gemessene Verhältnis von 2,7 bis 3,4 im Durchsatz deckt sich fast exakt damit. Auf einer Maschine mit 273 GB/s Speicherbandbreite ist das die dominierende Größe — Gesamtparameterzahl und Quantisierungsformat treten dahinter zurück.

### 2. Die Maschine ist bei diesem Modell nicht am Limit

Von einem auf vier parallele Streams fällt der Einzelstream nur auf 77 % (Laguna: 61 %), während sich der Gesamtdurchsatz fast verdreifacht. Höhere Concurrency als die gemessenen vier Stufen würde vermutlich weiter tragen.

### 3. Lineare Attention verhindert die Präfix-Wiederverwendung

Qwen3.6 nutzt in etwa drei von vier Layern lineare Attention mit sequenziell fortgeschriebenem Zustand. Ein solcher Zustand lässt sich nicht wie ein KV-Cache stückweise wiederverwenden. Die Messung bestätigt das Verhalten eindeutig; die architektonische Erklärung ist aus der Konfiguration abgeleitet, nicht direkt nachgewiesen.

### 4. Das Modell ist multimodal — was der Name nicht verrät

Architektur `Qwen3_5MoeForConditionalGeneration` mit eigener `model_visual.safetensors` (0,89 GB) und `vision_config`. Der Vision-Encoder ist von der Quantisierung ausgenommen und belegt Speicher, auch wenn ausschließlich Text verarbeitet wird.

### 5. Der Harness musste korrigiert werden

Qwen3.6 liefert seine Ausgabe im Streaming-Feld `reasoning` statt `content`. Ohne Anpassung wäre die Zeit bis zum ersten Token nie gesetzt und die Decode-Rate über die Gesamtzeit statt über die Decode-Phase gerechnet worden.

## Modell und Varianten

> [!IMPORTANT]
> **Zur Bezeichnung:** Ein Modell namens *Qwen3.6-36B-A3B* existiert nicht. Die MoE-Variante der Qwen3.6-Reihe heißt **Qwen3.6-35B-A3B**, daneben gibt es das dichte **Qwen3.6-27B**.

### Architektur des vermessenen Checkpoints

**Repo** — `RedHatAI/Qwen3.6-35B-A3B-NVFP4`

**Architekturen** — `Qwen3_5MoeForConditionalGeneration` · Spekulationskopf `Qwen3_5MoeMTP`

**model_type** — `qwen3_5_moe`

**Parameter** — 35 Mrd. gesamt, 3 Mrd. aktiv pro Token

**Layer** — 40 — lineare Attention in etwa drei von vier Positionen, volle Attention in jeder vierten

**Kontext** — `max_position_embeddings` 262 144

**Quantisierung** — `compressed-tensors`, Format `nvfp4-pack-quantized`, Gruppengröße 16, FP8-E4M3-Skalen

**Ausgenommen** — Vision-Encoder, `lm_head`, MTP-Gewichte (`re:^mtp.*`), Gate- und Linear-Attention-Projektionen

#### Dateien im Repo

| Datei | Größe | Funktion |
|---|---|---|
| model.safetensors | 22,46 GB | Sprachmodell, NVFP4-quantisiert |
| model_mtp.safetensors | 1,69 GB | Multi-Token-Prediction, unquantisiert |
| model_visual.safetensors | 0,89 GB | Vision-Encoder, unquantisiert |

### Verfügbare 4-Bit-Varianten

Ein reines `INT4` im Sinne von compressed-tensors gibt es für Qwen3.6 nicht — die Integer-Pfade heißen AWQ, GPTQ und AutoRound. Alle Größen über die HuggingFace-API gemessen.

*Sortiert nach Format, dann nach Downloads*

| Repo | Größe | Format | MTP |
|---|---|---|---|
| nvidia/Qwen3.6-35B-A3B-NVFP4 | 23,5 GB | modelopt | nein |
| RedHatAI/Qwen3.6-35B-A3B-NVFP4 | 25,1 GB | compressed-tensors | **ja** |
| unsloth/Qwen3.6-35B-A3B-NVFP4-Fast | 23,7 GB | compressed-tensors | nein |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | 26,5 GB | compressed-tensors | nein |
| Intel/…-int4-mixed-AutoRound | 21,5 GB | auto-round | nein |
| palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 | 24,5 GB | gptq | **ja** |
| cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | 25,0 GB | awq | nein |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | 25,5 GB | awq | nein |
| Qwen/Qwen3.6-35B-A3B-FP8 | 37,5 GB | fp8 | **ja** |
| Qwen/Qwen3.6-35B-A3B | 71,9 GB | BF16 | nein |

Gewählt wurde `RedHatAI/…-NVFP4` als einzige NVFP4-Variante mit MTP-Gewichten. `compressed-tensors` ist zudem das Format, das vLLM und SGLang direkt verarbeiten. Die MLX-Varianten sind für Apple Silicon und auf GB10 nutzlos.

> [!NOTE]
> **Warum NVFP4 und nicht AWQ oder GPTQ**
>
> Im Laguna-Test wurde nachgewiesen, dass NVFP4 auf sm_121 über `FLASHINFER_CUTLASS` nativ läuft, ohne Marlin-Rückfall. AWQ und GPTQ gehen auf dieser Architektur durch Marlin-Kernel — den langsameren Pfad. Das dortige Ergebnis *INT4 schlägt NVFP4* überträgt sich nicht, weil es dort vom Speicherdruck bei 99,7 GB Gewichten getrieben war. Bei 23–26 GB ist Speicher schlicht kein Thema.

## Testaufbau

Hardware, Engine und Messmethodik identisch zum Laguna-Test, damit die Werte vergleichbar bleiben.

**GPU** — NVIDIA GB10, Compute Capability 12.1 (sm_121a)

**Speicher** — 121 GB Unified Memory

**Betriebssystem** — Ubuntu 24.04.4 LTS, aarch64

**Treiber** — NVIDIA 580.173.02, CUDA 13.0

**Engine** — vLLM 0.25.1, torch 2.11.0+cu130, FlashInfer 0.6.13

#### Startkonfiguration

```bash
export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=1

vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

`MAX_JOBS=1` wurde aus dem Laguna-Test übernommen, wo vier parallele `cicc`-Prozesse den OOM-Killer auslösten. Bei 23 GB Gewichten wäre hier deutlich mehr Spielraum gewesen, aber die Regel kostet nichts.

### Messmethodik

- `bench.py` — 4 Szenarien × 3 Concurrency-Stufen (1, 2, 4) × 3 Wiederholungen, Streaming für die TTFT, `ignore_eos` für exakte Tokenzahlen, `temperature 0`, Aufwärmlauf vorab, Mediane als Ergebnis

- `prefill.py` — kaltes Prefill über 5 Längenstufen, pro Request eine eigene UUID im Prompt, damit kein Cache greifen kann

## Startverlauf

Ein einziger Versuch, keine Korrekturen nötig — im Gegensatz zu den zehn Anläufen der Laguna-Tests. Die dort erarbeiteten Einstellungen trugen unverändert.

*Zeitverlauf des Starts*

| Zeit | Ereignis | Dauer |
|---|---|---|
| 09:08–09:19 | Download, 24 GB | ≈ 10 min |
| 09:19:34 | Architektur erkannt: `Qwen3_5MoeForConditionalGeneration` | — |
| 09:19:39 | Spekulationskopf erkannt: `Qwen3_5MoeMTP` | — |
| 09:22:17 | Gewichte geladen, 23,45 GiB | 125,7 s |
| 09:33:53 | KV-Cache alloziert, 3 366 051 Token | — |
| ≈ 09:34 | Server bereit | ≈ 14 min |

Die Methode `mtp` war korrekt — vLLM 0.25.1 kennt daneben auch `qwen3_next_mtp`, das hier nicht nötig war. Kein OOM, kein Kernel-Fehler, kein Neustart.

> [!IMPORTANT]
> **Die erste Anfrage ist keine Messung**
>
> Der Smoke-Test lieferte 120 Token in 31,4 s, also **3,8 tok/s**. Die zweite Anfrage lieferte 60 Token in 0,797 s, also **75 tok/s** — ein Unterschied um Faktor 20. Die erste Anfrage nach dem Start enthält Aufwärmkosten, die in keine Messung einfließen dürfen. `bench.py` fährt deshalb einen expliziten Aufwärmlauf, bevor gemessen wird.

## Decode-Durchsatz

*Gesamtdurchsatz in tok/s über alle gleichzeitigen Requests · Median aus 3 Läufen*

| Szenario | c | pro Stream | gesamt | Laguna INT4 | Faktor |
|---|---|---|---|---|---|
| Prosa | 1 | 62,21 | 60,7 | 20,6 | 2,94× |
| Prosa | 2 | 56,15 | 101,3 | 31,7 | 3,19× |
| Prosa | 4 | 47,81 | 180,7 | 52,8 | 3,42× |
| Code | 1 | 65,99 | 63,5 | 23,8 | 2,67× |
| Code | 2 | 67,01 | 121,2 | 39,0 | 3,11× |
| Code | 4 | 52,84 | 201,2 | 64,3 | 3,13× |

### Skalierungsverhalten

*Prosa-Szenario · Verlust je Stream gegenüber Gewinn insgesamt*

| c | pro Stream | davon erhalten | gesamt | Skalierung |
|---|---|---|---|---|
| 1 | 62,2 | 100 % | 60,7 | — |
| 2 | 56,2 | 90 % | 101,3 | 1,67× |
| 4 | 47,8 | 77 % | 180,7 | 2,98× |

Zum Vergleich behielt Laguna bei c=4 nur 61 % der Einzelstream-Rate. Qwen3.6 ist mit 3 Mrd. aktiven Parametern weiter von der Bandbreitendecke entfernt und kann Parallelität besser ausnutzen — die gemessenen vier Stufen schöpfen das Potenzial vermutlich nicht aus.

## Antwortlatenz

*Zeit bis zum ersten Token in Sekunden · Median*

| Szenario | c | Qwen3.6 | Laguna INT4 |
|---|---|---|---|
| Prosa | 1 | 0,094 | 0,276 |
| Prosa | 2 | 0,259 | 0,635 |
| Prosa | 4 | 0,184 | 0,513 |
| Code | 1 | 0,128 | 0,260 |
| Code | 2 | 0,121 | 0,266 |
| Code | 4 | 0,130 | 0,480 |
| Prefill 16k | 1 | 2,806 | 0,309 |
| Prefill 16k | 2 | 4,360 | 0,458 |
| Prefill 16k | 4 | 7,226 | 0,551 |
| Prefill 16k wiederholt | 1 | 2,822 | 0,328 |
| Prefill 16k wiederholt | 2 | 4,362 | 0,484 |
| Prefill 16k wiederholt | 4 | 7,265 | 0,630 |

Bei kurzen Prompts ist Qwen3.6 durchweg zwei- bis dreimal schneller am ersten Token. Bei langen Prompts kehrt sich das Bild um — dazu der folgende Abschnitt.

## Der fehlende Präfix-Cache

Der auffälligste Befund dieser Messreihe. Das Szenario `prefill_16k_wiederholt` verwendet denselben langen Präfix wie das vorhergehende Szenario und sollte daher aus dem Cache bedient werden.

*TTFT beim ersten Kontakt gegen die Wiederholung desselben Präfix*

| c | erster Kontakt | wiederholt | Differenz | Laguna zum Vergleich |
|---|---|---|---|---|
| 1 | 2,806 s | 2,822 s | +0,6 % | 0,309 → 0,328 s |
| 2 | 4,360 s | 4,362 s | +0,0 % | 0,458 → 0,484 s |
| 4 | 7,226 s | 7,265 s | +0,5 % | 0,551 → 0,630 s |

### Der Nachweis über die Kaltmessung

16 384 Token bei einer gemessenen Kaltrate von 5301 tok/s ergeben rechnerisch **3,09 s**. Gemessen wurden im Hauptlauf **2,806 s**. Die als „warm" bezeichneten Werte waren also Neuberechnungen zum vollen Preis — es gab nie einen Cache-Treffer.

Ein zweites Indiz liefert die Skalierung: die TTFT steigt mit der Concurrency nahezu linear (2,8 / 4,4 / 7,2 s). Bei einem funktionierenden Präfix-Cache bliebe sie weitgehend konstant, wie bei Laguna (0,31 / 0,46 / 0,55 s).

### Erklärungsansatz

Die Quantisierungs-Ignoreliste des Checkpoints zeigt `linear_attn`-Projektionen in etwa drei von vier Layern — Qwen3.6 mischt lineare mit voller Attention. Lineare Attention führt einen sequenziell fortgeschriebenen Zustand mit, der sich nicht wie ein KV-Cache stückweise adressieren und wiederverwenden lässt.

> [!IMPORTANT]
> Das Verhalten ist eindeutig gemessen. Die architektonische Begründung ist aus der Konfiguration abgeleitet und wurde nicht direkt nachgewiesen — etwa durch Vergleich mit einem rein voll-attentiven Modell derselben Größe.

### Praktische Bedeutung

Für einen Agenten, der über zwanzig Turns denselben 16k-Token-Kontext mitschleppt, zahlt Qwen3.6 jedes Mal etwa 2,8 s. Laguna zahlt beim ersten Mal 9,3 s und danach 0,3 s. Ab dem vierten Turn liegt Laguna vorn, und der Abstand wächst mit jedem weiteren.

## Prefill-Skalierung

Gemessen mit einmaligem Präfix je Request. Hier ist Qwen3.6 durchgehend gut doppelt so schnell.

*Kaltes Prefill · Median aus 3 Läufen mit je eigener UUID im Prompt*

| Prompt-Token | TTFT | Prefill | relativ | Laguna INT4 |
|---|---|---|---|---|
| 4 053 | 0,80 s | 5064 tok/s | — | 2255 tok/s |
| 16 384 | 3,09 s | 5301 tok/s | 100 % | 2301 tok/s |
| 65 362 | 16,21 s | 4033 tok/s | 76 % | 1850 tok/s |
| 129 280 | 42,24 s | 3060 tok/s | 58 % | 1484 tok/s |
| 198 163 | 81,58 s | 2429 tok/s | 46 % | 1224 tok/s |

Von 16 384 auf 198 163 Token ist die Länge 12,1-fach, die Zeit steigt 26,4-fach. Das entspricht einer Komplexität von etwa **O(n1,29)** — praktisch identisch zu Lagunas O(n1,25), nur auf doppeltem Niveau. Beide Modelle dämpfen den quadratischen Attention-Anteil erfolgreich, Qwen3.6 über lineare Attention, Laguna über Sliding Windows.

> [!NOTE]
> **Tokenisierung beachten**
>
> Derselbe Prompttext ergibt bei Qwen3.6 **16 384** Token, bei Laguna **18 699**. Qwen3.6 tokenisiert rund 12 % dichter. Beim Vergleich absoluter TTFT-Werte ist das mitzudenken; die tok/s-Raten sind davon nicht betroffen.

## Speicher

| Kennzahl | Qwen3.6 NVFP4 | Laguna INT4 |
|---|---|---|
| Gewichte | 23,45 GiB | 69,34 GiB |
| Ladezeit | 125,7 s | 115,1 s |
| KV-Cache | 3 366 051 | 1 001 532 |
| KV je Token | ≈ 25 KiB | 38,8 KiB |
| Concurrency bei 262k | 12,84× | 3,82× |
| Startdauer gesamt | ≈ 14 min | ≈ 3 min |

Der dreifache KV-Cache speist sich aus zwei Quellen: 46 GiB weniger Gewichte und ein um ein Drittel niedrigerer Verbrauch pro Token. Letzterer folgt aus der linearen Attention — nur etwa ein Viertel der 40 Layer skaliert mit der Sequenzlänge.

Dass die Ladezeit trotz eines Drittels der Datenmenge nicht kürzer ausfällt, liegt am kalten Seitencache: Laguna war beim Vergleichslauf bereits mehrfach geladen worden.

## Fallstricke

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Ausgabe im Feld `reasoning` | Streaming-Deltas enthalten kein `content`; TTFT bleibt ungesetzt, Decode-Rate wird über die Gesamtzeit gerechnet | Harness auf `content`, `reasoning` und `reasoning_content` prüfen lassen |
| Erste Anfrage als Messung genommen | 3,8 statt 75 tok/s — Faktor 20 daneben | Expliziter Aufwärmlauf vor der Messung |
| Präfix-Cache stillschweigend wirkungslos | „warme" Werte identisch zu kalten, ohne Fehlermeldung | Gegen eine echte Kaltmessung mit einmaligem Präfix prüfen |
| Multimodalität unerwartet | `model_visual.safetensors` belegt Speicher auch bei reinem Textbetrieb | Vor dem Laden `config.json` auf `vision_config` prüfen |
| Falscher Modellname | Suche nach *36B-A3B* liefert null Treffer | Die MoE-Variante heißt **35B**-A3B |
| Spekulationsmethode raten | vLLM kennt sowohl `mtp` als auch `qwen3_next_mtp` | Für Qwen3.6 ist `mtp` korrekt; die erkannte Architektur im Log bestätigt es |
| Tokenisierung unterschiedlich | derselbe Text ergibt 12 % weniger Token als bei Laguna | Absolute TTFT nur bei gleicher Tokenzahl vergleichen, sonst über tok/s normieren |

<details>
<summary>Die Harness-Korrektur im Detail</summary>

Der Streaming-Delta von Qwen3.6 sieht so aus:

```bash
data: {"choices":[{"delta":{"reasoning":"Here"},...}]}
data: {"choices":[{"delta":{"reasoning"":"'s a thinking process"},...}]}
```

Die ursprüngliche Auswertung prüfte ausschließlich `delta["content"]`. Folge: `ttft` blieb `None`, der Chunk-Zähler bei null, und die Decode-Zeit wurde als `gesamt - 0` berechnet — also inklusive Prefill.

Die Korrektur ist additiv und ändert das Verhalten für Antworten im `content`-Feld nicht:

```bash
# vorher
if delta.get("content"):

# nachher
if any(delta.get(f) for f in ("content", "reasoning", "reasoning_content")):
```

Damit bleiben die Laguna-Messungen unverändert gültig — dort schlug die Initialisierung des Reasoning-Parsers fehl, sodass die gesamte Ausgabe ohnehin als `content` kam.

</details>

## Vergleich mit Laguna-S-2.1

*Beide auf vLLM 0.25.1, identischer Harness, identische Prompts*

| Szenario | c | Laguna INT4 | Qwen3.6 NVFP4 | Faktor |
|---|---|---|---|---|
| Prosa | 1 | 20,6 | 60,7 | 2,94× |
| Prosa | 2 | 31,7 | 101,3 | 3,19× |
| Prosa | 4 | 52,8 | 180,7 | 3,42× |
| Code | 1 | 23,8 | 63,5 | 2,67× |
| Code | 2 | 39,0 | 121,2 | 3,11× |
| Code | 4 | 64,3 | 201,2 | 3,13× |
| Prefill 16k | 1 | 13,1 | 16,0 | 1,23× |
| Prefill 16k | 2 | 38,9 | 18,8 | 0,48× |
| Prefill 16k | 4 | 64,6 | 20,3 | 0,31× |
| Prefill 16k wdh. | 1 | 24,0 | 16,5 | 0,69× |
| Prefill 16k wdh. | 2 | 31,1 | 19,2 | 0,62× |
| Prefill 16k wdh. | 4 | 39,5 | 20,3 | 0,51× |

Das Profil ist klar getrennt: **Decode geht dreifach an Qwen3.6, langer wiederkehrender Kontext geht an Laguna.** Die Prefill-Zeilen messen dabei nicht die Rechenleistung — dort ist Qwen3.6 doppelt so schnell — sondern den Cache-Effekt, den nur Laguna hat.

## Empfehlung

### Qwen3.6-35B-A3B-NVFP4 für Durchsatz und Interaktivität

Dreifaches Decode-Tempo, dreifacher KV-Cache, ein Drittel des Speicherbedarfs, TTFT unter 0,2 s. Für Chat, Codevervollständigung und alles, wo pro Anfrage ein überschaubarer Kontext anfällt, ist es auf dieser Box die klar bessere Wahl.

### Laguna-S-2.1-INT4 für lange wiederkehrende Kontexte

Ab etwa dem vierten Turn mit demselben 16k-Kontext liegt Laguna vorn, weil sein Präfix-Cache greift. Für Agenten, die eine wachsende Historie mitschleppen, kann das den dreifachen Rechenvorteil von Qwen3.6 aufwiegen.

### Offene Optimierung

Die Concurrency-Messung endet bei vier Streams, obwohl Qwen3.6 dort noch 77 % der Einzelstream-Rate hält und der KV-Cache 12,84 parallele Vollkontext-Requests zuließe. Höhere Stufen wurden nicht gemessen und dürften weiter skalieren.

## Limitationen

- **Reasoning dominiert die Ausgabe.** Qwen3.6 denkt sichtbar; bei 256 Token Limit besteht die Ausgabe fast vollständig aus Reasoning. Gemessen wurde die Token-Erzeugungsrate, nicht die Zeit bis zu einer fertigen Antwort. Für Hardware- und Engine-Vergleiche ist das die richtige Größe, für Wartezeiten aus Nutzersicht nicht.

- **Nur eine Engine.** Ausschließlich vLLM 0.25.1 vermessen; SGLang wurde für Qwen3.6 nicht getestet.

- **Nur eine Quantisierung.** Von den acht verfügbaren 4-Bit-Varianten wurde nur `RedHatAI/…-NVFP4` gemessen. Ob AWQ, GPTQ oder AutoRound auf sm_121 nennenswert abweichen, ist offen.

- **Erklärung des fehlenden Cache-Effekts unbelegt.** Das Verhalten ist gemessen, die Zurückführung auf lineare Attention aus der Konfiguration abgeleitet.

- **Concurrency nur bis 4.** Das Skalierungspotenzial wurde nicht ausgereizt.

- **Multimodalität ungenutzt.** Alle Messungen mit reinen Textanfragen; der Vision-Encoder belegte Speicher, wurde aber nie beansprucht.

- **Je eine Messreihe.** Drei Wiederholungen je Messpunkt, kein Mittel über mehrere Sitzungen.

## Glossar

**A3B**
Namenskonvention für „3 Mrd. aktive Parameter". Bei MoE-Modellen entscheidet diese Zahl über das Decode-Tempo, die Gesamtparameterzahl über den Speicherbedarf.

**MTP**
Multi-Token-Prediction. Ein mitgelieferter Kopf sagt mehrere Token auf einmal voraus, die das Hauptmodell parallel verifiziert. Wirkt wie Speculative Decoding ohne separates Draft-Modell.

**Lineare Attention**
Attention-Variante mit sequenziell fortgeschriebenem Zustand konstanter Größe statt eines mit der Sequenzlänge wachsenden KV-Caches. Spart Speicher, verhindert aber die stückweise Wiederverwendung eines Präfix.

**NVFP4**
NVIDIAs Microscaling-Format: FP4-Werte im E2M1-Layout mit FP8-E4M3-Skalen je 16 Werte. Blackwell-Tensor-Cores verarbeiten es nativ.

**compressed-tensors**
Quantisierungs-Container, den vLLM und SGLang direkt lesen. Alternative zu NVIDIAs `modelopt`-Format; beide können NVFP4 enthalten.

**Präfix-Cache**
Wiederverwendung der KV-Tensoren eines bereits verarbeiteten Prompt-Anfangs. Heißt bei vLLM Automatic Prefix Caching, bei SGLang RadixAttention.

**TTFT**
Time To First Token. Bestimmt die gefühlte Reaktionszeit und wird bei langen Prompts vom Prefill dominiert.

**sm_121a**
Compute Capability 12.1 der GB10-GPU. CUDA-Kernel müssen dafür übersetzt sein; auf dieser Architektur läuft NVFP4 über FlashInfer-CUTLASS nativ.

## Artefakte

**start-vllm-qwen36.sh** — Startskript, NVFP4 mit MTP, 262k Kontext

**bench.py** — Durchsatz und TTFT, mit der Korrektur für das `reasoning`-Feld

**prefill.py** — Kaltes Prefill über 5 Längenstufen bis 198 163 Token

**ergebnisse_qwen36.json** — alle 12 Messpunkte

**prefill_qwen36.json** — 5 Prefill-Stufen

**qwen36.log** — Startprotokoll

> [!NOTE]
> Alle Dateien in `~/bench/`. Das Modell verbleibt mit 24 GB im HuggingFace-Cache, der FlashInfer-Kernel-Cache ist warm — ein Neustart dauert rund vier Minuten.

---

*Testprotokoll Qwen3.6-35B-A3B-NVFP4 auf DGX Spark GB10 · 1 Serverstart, 12 Messpunkte, 5 Prefill-Stufen · Begleitdokument zum Testprotokoll Laguna-S-2.1 · alle Werte gemessen.*
