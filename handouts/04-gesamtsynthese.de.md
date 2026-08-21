# LLM-Inferenz auf DGX Spark GB10

*Testprotokoll · Inferenz-Benchmark · Teil 4 — Synthese*

> Sechs Serverkonfigurationen, drei Modelle, zwei Engines, zwei Quantisierungsformate — auf derselben Hardware, mit demselben Harness und denselben Prompts. Was sich daraus über die Maschine lernen lässt.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB Unified Memory · 273 GB/s **Konfigurationen** 6 **Messpunkte** 72 plus 22 Prefill-Stufen **Serverstarts** 14 · davon 6 gescheitert

## Inhalt

1. [Auf einen Blick](#auf-einen-blick)
2. [Die fünf Kernaussagen](#die-fünf-kernaussagen)
3. [Die sechs Konfigurationen](#die-sechs-konfigurationen)
4. [Rangliste](#rangliste)
5. [Aktive Parameter](#aktive-parameter)
6. [vLLM gegen SGLang](#vllm-gegen-sglang)
7. [INT4 gegen NVFP4](#int4-gegen-nvfp4)
8. [Der Präfix-Cache](#der-präfix-cache)
9. [Prefill-Skalierung](#prefill-skalierung)
10. [Speicher als Hauptvariable](#speicher-als-hauptvariable)
11. [Vollständige Messdaten](#vollständige-messdaten)
12. [Übertragbare Fallstricke](#übertragbare-fallstricke)
13. [Widerlegte Annahmen](#widerlegte-annahmen)
14. [Empfehlungsmatrix](#empfehlungsmatrix)
15. [Methodik](#methodik)
16. [Offene Fragen](#offene-fragen)
17. [Glossar](#glossar)
18. [Einzelprotokolle](#einzelprotokolle)

## Auf einen Blick

Die Testreihe begann mit zwei Annahmen, die beide falsch waren: NVFP4 sei die richtige Quantisierung für diese Hardware, und SGLang scheitere dort an fehlenden Kerneln. Was am Ende zählt, ist etwas Drittes — die Zahl der aktiven Parameter und die Frage, ob sich der Kontext wiederverwenden lässt.

### Aktive Parameter entscheiden

8 Mrd. → 20,6 tok/s, 4 Mrd. → 53,0, 3 Mrd. → 60,7. Die Zahl hinter dem „A" im Modellnamen sagt mehr über das Tempo als Format, Engine und Gesamtgröße zusammen.

### Der Präfix-Cache hängt an der Architektur

Sliding Window: Faktor 30 bis 34. Lineare Attention: Faktor 1,0 — gar kein Effekt. Für Agenten mit wachsender Historie ist das der größte Einzelhebel.

### INT4 gegen NVFP4 ist lastabhängig

Bei Speicherdruck gewinnt das kleinere Format. Ohne Speicherdruck gewinnt NVFP4 bei niedriger Last um 17 %, und INT4 zieht ab vier Streams gleich.

### vLLM und SGLang teilen sich die Disziplinen

vLLM: +17 % Durchsatz, fünffacher KV-Cache. SGLang: dreifach bessere Latenz, +79 % bei wiederkehrendem Präfix.

- **201,2** — tok/s Bestwert Qwen3.6, Code, c=4

- **0,036** — Sekunden TTFT Gemma-4

- **3,9 Mio** — Token KV-Cache Gemma-4

- **9,8×** — Spanne zwischen bester und schlechtester Konfiguration

## Die fünf Kernaussagen

### 1. Auf 273 GB/s zählt, wie viele Bytes pro Token bewegt werden

Alles Weitere folgt daraus. Aktive Parameter bestimmen das Decode-Tempo, die Gewichtsgröße bestimmt, wie viel Speicher für den KV-Cache übrig bleibt, und der KV-Cache bestimmt, wie viele Nutzer gleichzeitig bedient werden können. Rechenformate und Kernel-Pfade wirken erst in zweiter Ordnung.

### 2. Der Präfix-Cache ist die zweitwichtigste Eigenschaft

Ein Agent, der über zwanzig Turns denselben 16k-Kontext mitschleppt, zahlt bei einem Modell mit linearer Attention jedes Mal den vollen Preis. Bei Sliding-Window-Modellen fällt der Aufwand nach dem ersten Mal um Faktor 30 bis 34. Kein Engine-Wechsel und kein Quantisierungsformat gleicht das aus.

### 3. Speicherdruck verzerrt jeden Formatvergleich

Der Laguna-Test schien zu zeigen, dass INT4 NVFP4 schlägt — tatsächlich zeigte er, dass ein Modell, das 95,63 von 121 GiB belegt, kaum noch KV-Cache übrig lässt. Erst der Gemma-4-Test mit praktisch gleich großen Varianten misst die Formate selbst.

### 4. Die Engine-Wahl ist kein Performance-, sondern ein Profilthema

vLLM und SGLang liegen beim Durchsatz 13 bis 17 % auseinander — spürbar, aber klein gegenüber dem Faktor 3 zwischen den Modellen. Interessanter ist, dass sie unterschiedliche Disziplinen gewinnen: Durchsatz und Speicher gegen Latenz und Präfix-Wiederverwendung.

### 5. Die meisten Fallstricke sind Messfehler, keine Hardwareprobleme

Von den dokumentierten Stolperstellen betrafen die folgenschwersten nicht die Maschine, sondern die Beobachtung: eine hochgerechnete Restzeit für eine Messung gehalten, die erste Anfrage als repräsentativ genommen, ein Ausgabefeld übersehen, unvollständige API-Daten als vollständig verrechnet.

## Die sechs Konfigurationen

| # | Modell | Format | Engine | aktiv | Gewichte | Spekulation |
|---|---|---|---|---|---|---|
| 1 | Laguna-S-2.1 | NVFP4 | vLLM | 8 Mrd. | 95,63 GiB | DFlash |
| 2 | Laguna-S-2.1 | INT4 | vLLM | 8 Mrd. | 69,34 GiB | DFlash |
| 3 | Laguna-S-2.1 | INT4 | SGLang | 8 Mrd. | 67,56 GB | DFlash |
| 4 | Qwen3.6-35B-A3B | NVFP4 | vLLM | 3 Mrd. | 23,45 GiB | MTP |
| 5 | Gemma-4-26B-A4B | NVFP4 | vLLM | 4 Mrd. | 15,88 GiB | keine |
| 6 | Gemma-4-26B-A4B | INT4 (QAT) | vLLM | 4 Mrd. | 16,63 GiB | keine |

Alle drei Modelle sind Mixture-of-Experts-Architekturen mit hybrider Attention und multimodalen Fähigkeiten, die in diesen Tests ungenutzt blieben. Alle unterstützen 262 144 Token Kontext.

> [!IMPORTANT]
> **Zur Vergleichbarkeit:** Laguna und Qwen3.6 nutzen spekulative Dekodierung (DFlash bzw. MTP), Gemma-4 nicht — die Checkpoints liefern keinen Spekulationskopf mit. Gemma-4 erreicht seine Werte also ohne diesen Multiplikator, was seine Position in der Rangliste zusätzlich aufwertet.

## Rangliste

*Bester gemessener Wert je Disziplin*

| Disziplin | Bestwert | Konfiguration | schlechtester Wert | Spanne |
|---|---|---|---|---|
| Durchsatz c=4 | 201,2 tok/s | Qwen3.6 NVFP4 | 42,4 | 4,7× |
| Durchsatz c=1 | 63,5 tok/s | Qwen3.6 NVFP4 | 15,0 | 4,2× |
| TTFT | 0,036 s | Gemma-4 NVFP4 | 0,352 s | 9,8× |
| KV-Cache | 3 912 140 | Gemma-4 NVFP4 | 94 135 | 41,6× |
| Kaltes Prefill 16k | 5301 tok/s | Qwen3.6 NVFP4 | 1686 | 3,1× |
| Präfix-Cache-Faktor | ≈ 34 | Gemma-4 | 1,0 | 34× |
| Wiederholter Präfix c=4 | 141,1 tok/s | Gemma-4 NVFP4 | 17,6 | 8,0× |

Die Spanne zwischen bester und schlechtester Konfiguration beträgt bei der Latenz fast das Zehnfache, beim KV-Cache über das Vierzigfache. **Laguna belegt in keiner Disziplin den ersten Platz** — was die Modellqualität nicht berührt, die diese Reihe nicht misst.

## Aktive Parameter

*Decode bei einem einzelnen Stream, jeweils beste Variante des Modells*

| Modell | aktiv | Decode c=1 | c=4 gesamt | Spekulation |
|---|---|---|---|---|
| Laguna-S-2.1 | 8 Mrd. | 20,6 | 52,8 | DFlash |
| Gemma-4-26B | 4 Mrd. | 53,0 | 177,1 | keine |
| Qwen3.6-35B | 3 Mrd. | 60,7 | 201,2 | MTP |

Der Zusammenhang ist deutlich, aber nicht exakt proportional. Bemerkenswert ist die mittlere Zeile: **Gemma-4 erreicht mit halb so vielen aktiven Parametern wie Laguna das 2,6-fache Tempo — und das ohne jede spekulative Dekodierung**, während Laguna mit DFlash arbeitet. Rechnet man den Spekulationsvorteil heraus, ist der Zusammenhang zwischen aktiven Parametern und Tempo noch enger, als die Rohzahlen zeigen.

### Skalierung über Concurrency

*Anteil der Einzelstream-Rate, der bei vier parallelen Streams erhalten bleibt*

| Konfiguration | c=1 | c=4 | erhalten |
|---|---|---|---|
| Laguna INT4 | 21,2 | 13,7 | 65 % |
| Qwen3.6 NVFP4 | 62,2 | 47,8 | 77 % |
| Gemma-4 NVFP4 | 53,5 | 43,2 | 81 % |
| Gemma-4 INT4 | 44,1 | 45,5 | 103 % |

Je weniger aktive Parameter, desto weiter ist die Maschine von der Bandbreitendecke entfernt und desto besser skaliert sie über Parallelität. Gemma-4 INT4 wird bei vier Streams pro Stream sogar schneller als bei einem — ein Batching-Effekt der Marlin-Kernel.

## vLLM gegen SGLang

Getestet auf Laguna-S-2.1-INT4, beide mit DFlash-Spekulation, beide auf torch 2.11.0+cu130.

| Disziplin | vLLM 0.25.1 | SGLang 0.5.16 | Vorsprung |
|---|---|---|---|
| Prosa c=1 | 20,6 | 18,2 | vLLM +13 % |
| Prosa c=2 | 31,7 | 27,8 | vLLM +14 % |
| Prosa c=4 | 52,8 | 45,3 | vLLM +17 % |
| Code c=4 | 64,3 | 56,2 | vLLM +14 % |
| TTFT c=2 | 0,635 s | 0,201 s | SGLang 3,2× |
| TTFT c=4 | 0,513 s | 0,193 s | SGLang 2,7× |
| Wiederholter Präfix c=4 | 39,5 | 70,6 | SGLang +79 % |
| KV-Cache | 950 420 | 185 897 | vLLM 5,1× |
| Kaltes Prefill 16k | 2243 | 1686 | vLLM +33 % |

### Was die Zahlen bedeuten

**vLLM gewinnt beim Durchsatz, und der Vorsprung wächst mit der Concurrency** — das Gegenteil der verbreiteten Erwartung, SGLang skaliere über Parallelität besser.

**SGLang gewinnt bei der Latenz deutlich und stabil.** Die TTFT liegt fast unabhängig von Last und Prompt-Länge zwischen 0,19 und 0,29 s, während vLLM bis 0,78 s streut.

**Bei wiederkehrendem Präfix dreht sich das Bild.** SGLang erreicht 70,6 gegen vLLMs 39,5 tok/s — im Szenario davor, beim ersten Kontakt mit demselben Präfix, war es umgekehrt (38,0 gegen 64,6). RadixAttention liefert also genau im vorhergesagten Fall.

<details>
<summary>Die widerlegte Behauptung zu sm_121a</summary>

Ausgangspunkt war die Aussage, SGLang scheitere auf GB10, weil die vorkompilierten `sgl_kernel`-Wheels keine sm_121a-Kernels enthielten und nur ein Neubau aus dem Quellcode helfe. Zweistufig geprüft:

**Quellcode** — in `sgl-kernel/CMakeLists.txt` wird der Gencode erzeugt, wenn CUDA ≥ 13.0 *und* aarch64 vorliegen. Genau der cu130-Build auf dieser Maschine.

**Binary** — `cuobjdump` über die installierten Bibliotheken:

```bash
common_ops.abi3.so   80M   sm_90 sm_90a sm_100a sm_103a sm_110a sm_120a sm_121a
spatial_ops.abi3.so  196K  sm_90 sm_90a sm_100a sm_103a sm_110a sm_120a sm_121a
flashmla_ops.abi3.so 12M   sm_90a sm_100a sm_103a          — kein sm_121
flash_ops.abi3.so    309M  sm_90a                          — FA3, Hopper-exklusiv
```

Die Hauptkernel-Bibliothek enthält sm_121a. Dass FA3 und FlashMLA fehlen, ist korrekt und irrelevant — beide sind auf GB10 ohnehin nicht nutzbar. SGLang lief ohne jeden Eigenbau.

</details>

## INT4 gegen NVFP4

Zwei Tests, zwei gegensätzliche Ergebnisse — und der Unterschied liegt nicht am Format, sondern an den Randbedingungen.

| Testfall | Gewichte NVFP4 / INT4 | KV-Cache NVFP4 / INT4 | Ergebnis |
|---|---|---|---|
| Laguna-S-2.1 | 95,63 / 69,34 GiB | 94k / 950k | **INT4**, +19 bis +246 % |
| Gemma-4-26B | 15,88 / 16,63 GiB | 3,91 / 3,88 Mio | **lastabhängig** |

### Der Laguna-Fall: Speicher entscheidet

NVFP4 belegte 95,63 von 121 GiB, sodass nur 6,97 GiB für den KV-Cache blieben — rechnerisch **1,44 parallele Requests** bei vollem Kontext. Der Checkpoint verbrauchte zudem doppelt so viel KV-Speicher pro Token (77,7 gegen 38,8 KiB), was auf einen Build-Fehler hindeutet. INT4 gewann dort nicht wegen des Rechenpfads, sondern weil es 26 GiB kleiner war.

### Der Gemma-4-Fall: der Kernel-Pfad wird sichtbar

Beide Varianten unterscheiden sich um 0,75 GiB Gewichte und 0,9 % KV-Cache. Was übrig bleibt, protokolliert vLLM beim Start:

```bash
NVFP4: Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
INT4:  Using 'MARLIN' WNA16 MoE backend
```

*Gemma-4 · Gesamtdurchsatz und Decode-Rate je Stream, Prosa*

| c | NVFP4 gesamt | INT4 gesamt | NVFP4 je Stream | INT4 je Stream |
|---|---|---|---|---|
| 1 | 53,0 | 43,9 | 53,5 | 44,1 |
| 2 | 101,0 | 94,9 | 50,9 | 46,7 |
| 4 | 170,6 | 177,1 | 43,2 | 45,5 |

**Marlin amortisiert seinen Entpack-Aufwand über den Batch** und hält 44–47 tok/s über alle Laststufen. **FlashInfer-CUTLASS arbeitet schon bei einem Stream nahe seinem Optimum** und verliert entsprechend, wenn die Last steigt.

Beim kalten Prefill — dem rechenintensivsten Fall — bleibt NVFP4 durchgehend vorn, mit schrumpfendem Abstand: 28 % bei 4k Token, 22 % bei 16k, 12 % bei 65k, 6 % bei 132k.

> [!NOTE]
> **Faustregel**
>
> Ist der Speicher knapp, entscheidet die Dateigröße — unabhängig vom Format. Ist er es nicht, gewinnt NVFP4 bei niedriger Last um rund 17 % und liegt ab vier parallelen Streams gleichauf. Die pauschale Aussage „NVFP4 ist auf Blackwell nativ und daher schneller" gilt nur im Teillastbereich.

## Der Präfix-Cache

Der folgenreichste Befund der Reihe, und er hängt weder an der Engine noch am Format, sondern an der Attention-Architektur des Modells.

*Zeit bis zum ersten Token bei ~16k Token Prompt*

| Modell | Attention | kalt | warm | Faktor |
|---|---|---|---|---|
| Laguna-S-2.1 | Sliding Window 512 | 9,32 s | 0,30 s | ≈ 30 |
| Gemma-4-26B | Sliding Window 1024 | 3,13 s | 0,093 s | ≈ 34 |
| Qwen3.6-35B | lineare Attention | 3,09 s | 2,81 s | 1,0 |

Bei Qwen3.6 sind warmer und kalter Lauf **identisch**. Der Nachweis erfolgte zweifach: die gemessene Kaltrate von 5301 tok/s ergibt für 16 384 Token rechnerisch 3,09 s — im Hauptlauf gemessen wurden 2,81 s. Es gab nie einen Cache-Treffer. Zusätzlich stieg die TTFT dort linear mit der Concurrency (2,8 / 4,4 / 7,2 s), während sie bei den anderen beiden nahezu konstant blieb.

**Erklärungsansatz:** Lineare Attention führt einen sequenziell fortgeschriebenen Zustand mit, der sich nicht wie ein KV-Cache stückweise adressieren und wiederverwenden lässt. Die Hypothese entstand im Qwen-Test und wurde im Gemma-4-Test durch die Gegenprobe gestützt — direkt nachgewiesen ist sie nicht.

> [!IMPORTANT]
> **Praktische Tragweite**
>
> Ein Agent, der über zwanzig Turns denselben 16k-Kontext mitschleppt, zahlt bei Qwen3.6 jedes Mal 2,8 s. Bei Gemma-4 sind es nach dem ersten Mal 0,093 s. Über zwanzig Turns summiert sich das auf 56 gegen 5 Sekunden — ein Unterschied, den kein Durchsatzvorteil ausgleicht.

## Prefill-Skalierung

Alle drei Modelle dämpfen den quadratischen Attention-Anteil erfolgreich — Laguna und Gemma-4 über Sliding Windows, Qwen3.6 über lineare Attention.

*Kaltes Prefill in tok/s, einmaliger Präfix je Request*

| Prompt-Token | Laguna INT4 | Qwen3.6 | Gemma-4 NVFP4 |
|---|---|---|---|
| ≈ 4 000 | 2255 | 5064 | 7161 |
| ≈ 16 000 | 2301 | 5301 | 5238 |
| ≈ 65 000 | 1850 | 4033 | 2396 |
| ≈ 132 000 | 1484 | 3060 | 1416 |
| ≈ 200 000 | 1224 | 2429 | — |

*Komplexität, abgeleitet aus dem Verhältnis Längenzuwachs zu Zeitzuwachs*

| Modell | Komplexität | Rate bei 132k | Architektur |
|---|---|---|---|
| Laguna | O(n1,25) | 64 % | Fenster 512, 12 von 48 voll |
| Qwen3.6 | O(n1,29) | 58 % | lineare Attention, 3 von 4 |
| Gemma-4 | steiler | 27 % | Fenster 1024, 5 von 30 voll |

Bei reiner Voll-Attention wäre ein Einbruch auf unter 10 % zu erwarten gewesen. Gemma-4 fällt am steilsten ab — plausibel wegen der doppelt so großen Fenster und des höheren Anteils voller Attention. Es ist damit bei kurzen Prompts das schnellste und bei sehr langen das langsamste Modell der Reihe.

## Speicher als Hauptvariable

*121 GB Unified Memory, geteilt zwischen Betriebssystem, Gewichten, KV-Cache und Compiler*

| Konfiguration | Gewichte | KV-Cache | KV je Token | Concurrency |
|---|---|---|---|---|
| Laguna NVFP4 | 95,63 GiB | 94 135 | 77,7 KiB | 1,44× bei 65k |
| Laguna INT4 (65k) | 69,34 GiB | 950 420 | 38,8 KiB | 14,50× bei 65k |
| Laguna INT4 (262k) | 69,34 GiB | 1 001 532 | 38,8 KiB | 3,82× bei 262k |
| Laguna INT4 SGLang | 67,56 GB | 185 897 | — | 48 Requests |
| Qwen3.6 NVFP4 | 23,45 GiB | 3 366 051 | ≈ 25 KiB | 12,84× bei 262k |
| Gemma-4 NVFP4 | 15,88 GiB | 3 912 140 | — | 14,92× bei 262k |
| Gemma-4 INT4 | 16,63 GiB | 3 875 400 | — | 14,78× bei 262k |

Die Spanne beim KV-Cache beträgt **Faktor 41,6**. Laguna NVFP4 ist mit 1,44 parallelen Vollkontext-Requests für Mehrbenutzerbetrieb praktisch unbrauchbar, Gemma-4 mit 14,92 komfortabel.

> [!NOTE]
> **Ein Befund nebenbei: das Kontextfenster ist gratis**
>
> Die Anhebung von `--max-model-len` bei Laguna INT4 von 65 536 auf 262 144 kostete keinen KV-Cache — sie brachte sogar 5 % mehr (1 001 532 statt 950 420 Token), vermutlich durch günstigere Blockaufteilung. Es gibt keinen Grund, unter dem Maximum des Checkpoints zu bleiben.

## Vollständige Messdaten

<details>
<summary>Gesamtdurchsatz aller sechs Konfigurationen (tok/s)</summary>

| Szenario | c | Lag NVFP4 | Lag INT4 | Lag SGLang | Qwen3.6 | Gem NVFP4 | Gem INT4 |
|---|---|---|---|---|---|---|---|
| Prosa | 1 | 17,3 | 20,6 | 18,2 | 60,7 | 53,0 | 43,9 |
| Prosa | 2 | 25,5 | 31,7 | 27,8 | 101,3 | 101,0 | 94,9 |
| Prosa | 4 | 42,7 | 52,8 | 45,3 | 180,7 | 170,6 | 177,1 |
| Code | 1 | 15,0 | 23,8 | 21,9 | 63,5 | 53,1 | 44,1 |
| Code | 2 | 27,4 | 39,0 | 33,2 | 121,2 | 101,5 | 95,0 |
| Code | 4 | 42,4 | 64,3 | 56,2 | 201,2 | 173,6 | 172,5 |
| Prefill 16k | 1 | 5,8 | 13,1 | 6,3 | 16,0 | 27,6 | 22,8 |
| Prefill 16k | 2 | 11,6 | 38,9 | 25,1 | 18,8 | 82,3 | 75,9 |
| Prefill 16k | 4 | 18,7 | 64,6 | 38,0 | 20,3 | 142,5 | 140,0 |
| Prefill wdh. | 1 | 7,5 | 24,0 | 25,4 | 16,5 | 44,8 | 38,3 |
| Prefill wdh. | 2 | 11,8 | 31,1 | 41,1 | 19,2 | 85,5 | 77,8 |
| Prefill wdh. | 4 | 17,6 | 39,5 | 70,6 | 20,3 | 141,1 | 136,2 |

</details>

<details>
<summary>Zeit bis zum ersten Token (Sekunden, Median)</summary>

| Szenario | c | Lag NVFP4 | Lag INT4 | Lag SGLang | Qwen3.6 | Gem NVFP4 | Gem INT4 |
|---|---|---|---|---|---|---|---|
| Prosa | 1 | 0,352 | 0,276 | 0,314 | 0,094 | 0,036 | 0,038 |
| Prosa | 2 | 0,443 | 0,635 | 0,201 | 0,259 | 0,036 | 0,037 |
| Prosa | 4 | 0,427 | 0,513 | 0,193 | 0,184 | 0,069 | 0,064 |
| Code | 4 | 0,374 | 0,480 | 0,194 | 0,130 | 0,078 | 0,076 |
| Prefill 16k | 1 | 0,307 | 0,309 | 0,200 | 2,806 | 0,093 | 0,084 |
| Prefill 16k | 4 | 0,686 | 0,551 | 0,269 | 7,226 | 0,153 | 0,180 |
| Prefill wdh. | 4 | 0,781 | 0,630 | 0,285 | 7,265 | 0,142 | 0,128 |

</details>

## Übertragbare Fallstricke

Aus vierzehn Serverstarts, von denen sechs scheiterten.

### Speicher und Compiler

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| `MAX_JOBS` zu hoch | OOM-Kill von `cicc` (7,5 GB je Prozess), **kein Fehler im Anwendungslog** | `MAX_JOBS=1` beim ersten Start; Kernel-Cache ist danach warm |
| `gpu-memory-utilization` > 0,89 | Abbruch vor dem Laden | Das OS belegt ~11 GiB, die frei bleiben müssen |
| `mem-fraction-static` missverstanden | kleineres Modell schafft *keine* Systemluft | Der Parameter reserviert unabhängig von der Modellgröße; Ersparnis wandert in den KV-Cache |
| Verzögerte Speicherfreigabe | Neustart scheitert direkt nach dem Stoppen | Auf Freigabe warten, hier 5–15 s |

### Messung und Beobachtung

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Erste Anfrage als Messung | 3,8 statt 75 tok/s — Faktor 20 daneben | Expliziter Aufwärmlauf vor der Messung |
| Ausgabe im Feld `reasoning` | TTFT bleibt ungesetzt, Decode-Rate über die Gesamtzeit gerechnet | Auf `content`, `reasoning` und `reasoning_content` prüfen |
| Hochgerechnete Restzeit als Tatsache | tqdm zeigte 1:48:53, real waren es 3:33 | Bei ungleichmäßiger Arbeit die Rate über mehrere Schritte beobachten |
| Stehender Fortschrittsbalken | als „hängt" gedeutet, tatsächlich lief die Kompilierung | Nebenindikatoren prüfen: CPU-Last, wachsender Kernel-Cache |
| Unvollständige API-Daten | fehlende `size`-Felder als 0 verrechnet, 28 GB zu niedrig | Vollständigkeit der Antwort prüfen |
| Präfix-Cache stillschweigend wirkungslos | „warme" Werte identisch zu kalten, ohne Fehlermeldung | Gegen eine echte Kaltmessung mit einmaligem Präfix prüfen |
| Ergebnisdatei erst am Ende | Abbruch verwirft alle bereits gemessenen Stufen | Inkrementell schreiben oder aus dem Protokoll rekonstruieren |

### Werkzeuge und Umgebung

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Exit-Code in Pipeline verschluckt | `docker pull … \| tail -5` meldet Erfolg trotz `permission denied` | Exit-Code der ersten Stufe prüfen |
| `pkill -f` trifft die eigene Shell | Exit 143/144, Ziel läuft weiter | PIDs über `ps` ermitteln, gezielt beenden |
| Server stirbt mit der Sitzung | Log bricht nach wenigen Zeilen ab, kein Fehler | `setsid nohup … < /dev/null &` |
| Torch-ABI-Bruch | `undefined symbol: _ZNK2at10TensorBase…` | Die von der Bibliothek gepinnte Torch-Version verwenden |
| Falsches CUDA-Wheel | fehlende sm_121a-Kernels bei cu12x | Auf dem Spark ausschließlich `-cu130` auf aarch64 |
| Fehlende Python-Header | Triton-JIT scheitert ohne `python3.12-dev` | `uv python install 3.12` bringt sie mit, kein sudo nötig |
| Modell-Revision nicht gepinnt | Upstream verschiebt `main`; 93 GB stillschweigend nachgeladen, danach bricht der Start am KV-Cache ab | `--revision` für Modell und Draft pinnen; ein Checkpoint kann „INT4“ heißen und trotzdem weniger quantisieren |

## Widerlegte Annahmen

| Annahme | Befund | Beleg |
|---|---|---|
| `sgl_kernel` ohne sm_121a, nur mit Neubau lauffähig | falsch | cuobjdump zeigt sm_121a; SGLang lief ohne Eigenbau |
| NVFP4 ist generell die Wahl für den Spark | falsch | hängt vom Speicherdruck ab; bei Laguna verlor NVFP4 klar |
| INT4 schlägt NVFP4 auf dieser Hardware | nur unter Speicherdruck | bei Gemma-4 gewinnt NVFP4 im Teillastbereich |
| SGLang skaliert besser über Concurrency | falsch | vLLMs Vorsprung wächst von 13 auf 17 % |
| Prefill 600–800 tok/s | deutlich zu niedrig | 1224 bis 7161 tok/s je nach Modell und Länge |
| `MAX_JOBS=4` genügt | zu schwach | führte zum OOM-Kill; nötig war `MAX_JOBS=1` |
| SGLangs Graph-Capturing hängt auf sm_121 | falsch | 3 min 33 s, nicht die hochgerechneten 1 h 50 min |

## Empfehlungsmatrix

| Anwendungsfall | Empfehlung | Begründung |
|---|---|---|
| Maximaler Durchsatz | **Qwen3.6-35B-A3B NVFP4** auf vLLM | 201 tok/s bei c=4 |
| Minimale Antwortlatenz | **Gemma-4-26B-A4B NVFP4** auf vLLM | 0,036 s TTFT |
| Agent mit wachsender Historie | **Gemma-4**, alternativ **Laguna auf SGLang** | Cache-Faktor 34 bzw. +79 % bei wiederholtem Präfix |
| Viele parallele Nutzer | **Gemma-4** (beide Formate) | 14,9 Vollkontext-Requests gleichzeitig |
| Sehr lange Prompts | **Qwen3.6** | 3060 tok/s noch bei 132k Token |
| Größtes Modell, das noch läuft | **Laguna-S-2.1 INT4** | aber 3× langsamer als die Alternativen |
| nicht nehmen | Laguna NVFP4, FP8- und BF16-Varianten | 1,44× Concurrency bzw. passen nicht in 121 GB |

> [!NOTE]
> **Wenn nur eine Konfiguration gewählt werden dürfte**
>
> **Gemma-4-26B-A4B NVFP4 auf vLLM.** Es gewinnt bei Latenz, KV-Cache und Speicherbedarf, liegt beim Durchsatz nur knapp hinter Qwen3.6 — und hat als einziges der drei Modelle sowohl einen funktionierenden Präfix-Cache *als auch* hohes Decode-Tempo, ohne dafür auf spekulative Dekodierung angewiesen zu sein.

#### Startkonfiguration, die sich über alle Läufe bewährt hat

```bash
export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=1          # beim ersten Start; danach Cache warm

vllm serve <modell> \
  --max-num-seqs 32 \
  --max-model-len 262144 \  # kostet keinen KV-Cache
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

Kein `--attention-backend` setzen: die Auto-Auswahl trifft auf sm_121 `flashinfer`, weil `trtllm_mha` SM120 nicht unterstützt und `fa3` Hopper-exklusiv ist.

## Methodik

**Hardware** — NVIDIA GB10, sm_121a, 121 GB Unified Memory, ~273 GB/s, aarch64

**System** — Ubuntu 24.04.4, Kernel 6.17.0-1029-nvidia, Treiber 580.173.02, CUDA 13.0

**Engines** — vLLM 0.25.1 und SGLang 0.5.16, beide torch 2.11.0+cu130, native venvs

### Zwei eigene Messskripte, nur Standardbibliothek

- `bench.py` — 4 Szenarien × 3 Concurrency-Stufen (1, 2, 4) × 3 Wiederholungen. Streaming für die TTFT, `ignore_eos` für exakte Tokenzahlen, `temperature 0`, expliziter Aufwärmlauf, Mediane als Ergebnis.

- `prefill.py` — kaltes Prefill über bis zu 5 Längenstufen. Jeder Request erhält eine eigene UUID im Prompt, sodass weder Automatic Prefix Caching noch RadixAttention greifen kann.

Beide sprechen die OpenAI-kompatible API, die vLLM und SGLang gleichermaßen anbieten — dadurch bekommen alle Konfigurationen exakt dieselben Anfragen.

> [!IMPORTANT]
> **Zwei Mängel, die erst im Verlauf auffielen**
>
> **Der Warm-Kalt-Vergleich in `bench.py` misslang.** Beide Prefill-Szenarien messen den warmen Fall, weil der Präfix schon nach dem ersten der drei Wiederholungsläufe im Cache liegt. Nachträglich behoben durch `prefill.py`.

> [!IMPORTANT]
> **Die Größenlabels lagen um Faktor zwei daneben.** Die als „8k" bezeichneten Prompts enthielten real 16 000 bis 18 800 Token. In `prefill.py` wurden die Stufen später nach gemessenen Tokenzahlen benannt.

## Offene Fragen

- **Concurrency über vier Streams.** Gemma-4 INT4 hält dort noch 103 % seiner Einzelstream-Rate, der KV-Cache ließe 14,8 Vollkontext-Requests zu. Das Skalierungspotenzial ist bei keinem Modell ausgereizt.

- **SGLang für Qwen3.6 und Gemma-4.** Der Engine-Vergleich beruht auf einem einzigen Modell; ob sich das Profil überträgt, ist offen.

- **Antwortqualität.** Vollständig ungemessen. Der QAT-Vorteil des Gemma-4-INT4-Checkpoints taucht in keiner Zahl auf, ebenso wenig der Qualitätsunterschied zwischen den Modellen.

- **Reasoning sauber abgetrennt.** Bei Laguna und Qwen3.6 bestand die Ausgabe überwiegend aus Reasoning, wodurch der Code-gegen-Prosa-Vergleich entwertet ist. Nur Gemma-4 lieferte echte Antworten.

- **Die 200k-Prefill-Stufe** fehlt bei Gemma-4 in beiden Varianten.

- **Multimodalität.** Alle drei Modelle bringen Vision-Encoder mit, die Speicher belegen und in keinem Test beansprucht wurden.

- **Reproduzierbarkeit über Sitzungen.** Je drei Wiederholungen pro Messpunkt, aber kein Mittel über mehrere Sitzungen. Eine Stichprobe zeigte 1,4 % Abweichung zwischen zwei Läufen derselben Konfiguration.

## Glossar

**Aktive Parameter (A3B, A4B)**
Bei Mixture-of-Experts-Modellen die Zahl der pro Token tatsächlich ausgewerteten Parameter. Bestimmt das Decode-Tempo; die Gesamtparameterzahl bestimmt den Speicherbedarf.

**KV-Cache**
Zwischenspeicher der Key- und Value-Tensoren verarbeiteter Token. Seine Größe begrenzt, wie viele Requests mit welcher Kontextlänge gleichzeitig laufen.

**Präfix-Cache**
Wiederverwendung der KV-Tensoren eines bereits verarbeiteten Prompt-Anfangs. Bei vLLM Automatic Prefix Caching, bei SGLang RadixAttention.

**Sliding-Window-Attention**
Attention über ein begrenztes Fenster zurückliegender Token. Dämpft den quadratischen Aufwand und bleibt dabei cachebar.

**Lineare Attention**
Attention mit sequenziell fortgeschriebenem Zustand konstanter Größe. Spart Speicher, verhindert aber die stückweise Wiederverwendung eines Präfix.

**NVFP4**
NVIDIAs Microscaling-Format: FP4-Werte im E2M1-Layout mit FP8-Blockskalen. Blackwell-Tensor-Cores verarbeiten es nativ über `FLASHINFER_CUTLASS`.

**MARLIN**
CUDA-Kernel für gewichtsquantisierte Formate mit unquantisierten Aktivierungen. Entpackt 4-Bit-Gewichte und multipliziert in höherer Präzision; amortisiert sich über größere Batches.

**Spekulative Dekodierung**
Ein günstiges Verfahren schlägt mehrere Token vor, das Hauptmodell verifiziert sie parallel. Varianten hier: DFlash (eigenes Draft-Modell), MTP (mitgelieferter Kopf).

**TTFT**
Time To First Token. Bestimmt die gefühlte Reaktionszeit; bei langen Prompts vom Prefill dominiert, bei kurzen vom Scheduling.

**Unified Memory**
Auf GB10 teilen sich CPU und GPU einen physischen Speicherpool. Gewichte, KV-Cache, Betriebssystem und Compilerprozesse konkurrieren um dieselben 121 GB.

**sm_121a**
Compute Capability 12.1 der GB10-GPU. Kernel müssen dafür übersetzt sein; der Gencode entsteht nur bei CUDA ≥ 13.0 auf aarch64.

**cicc**
Compiler-Frontend der CUDA-Toolchain. Belegte bis zu 7,5 GB je Prozess und war Auslöser des OOM-Kills beim ersten Laguna-Start.

## Einzelprotokolle

Diese Synthese fasst drei ausführliche Testprotokolle zusammen. Dort finden sich die vollständigen Verläufe, Logauszüge und Detailbefunde.

| Teil | Inhalt | Datei |
|---|---|---|
| 1 | Laguna-S-2.1 — vLLM gegen SGLang, 10 Startversuche, 2 OOM-Kills, 4 Fehldiagnosen | `01-laguna-s-2.1.de.html` |
| 2 | Qwen3.6-35B-A3B — dreifaches Decode-Tempo, fehlender Präfix-Cache | `02-qwen3.6-35b-a3b.de.html` |
| 3 | Gemma-4-26B-A4B — NVFP4 gegen INT4, dokumentierte Kernel-Pfade | `03-gemma-4-26b-a4b.de.html` |
| 4 | Diese Synthese | `04-gesamtsynthese.de.html` |

#### Werkzeuge und Rohdaten

**bench.py** — Durchsatz und TTFT über 4 Szenarien × 3 Concurrency-Stufen

**prefill.py** — Kaltes Prefill mit einmaligem UUID-Präfix je Request

**start-vllm-*.sh** — Startskripte je Modell und Variante

**start-sglang-*.sh** — SGLang-Startskripte

**ergebnisse_*.json** — 6 Messreihen à 12 Punkte

**prefill_*.json** — 6 Prefill-Reihen

***.log** — Startprotokolle aller 14 Versuche, auch der gescheiterten

Alles in `~/bench/`.

---

*Gesamtsynthese · 6 Serverkonfigurationen, 3 Modelle, 2 Engines, 2 Quantisierungsformate · 72 Durchsatz- und Latenzmesspunkte, 22 Prefill-Stufen, 14 Serverstarts · alle Werte gemessen, Hochrechnungen ausdrücklich gekennzeichnet.*
