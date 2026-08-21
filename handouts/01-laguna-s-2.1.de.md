# Laguna-S-2.1 auf DGX Spark GB10

*Testprotokoll · Inferenz-Benchmark*

> vLLM gegen SGLang, NVFP4 gegen INT4 — vollständige Dokumentation aller Messläufe einschließlich der sechs gescheiterten Serverstarts, zweier OOM-Kills und vier Fehldiagnosen.

**Hardware** NVIDIA GB10 · sm_121a · 121 GB Unified Memory **Modell** poolside/Laguna-S-2.1 · 118B MoE, 8B aktiv **Serverstarts** 10 · davon 4 erfolgreich **Messreihen** 3 Konfigurationen × 12 Szenarien

## Inhalt

1. [Auf einen Blick](#auf-einen-blick)
2. [Kernaussagen](#kernaussagen)
3. [Testaufbau](#testaufbau)
4. [Methodik](#methodik)
5. [Chronik der Startversuche](#chronik-der-startversuche)
6. [Fallstricke](#fallstricke)
7. [Durchsatz](#durchsatz)
8. [Antwortlatenz](#antwortlatenz)
9. [Prefill-Skalierung](#prefill-skalierung)
10. [Speicher und Start](#speicher-und-start)
11. [Fehldiagnosen](#fehldiagnosen-im-verlauf)
12. [Widerlegte Annahmen](#widerlegte-annahmen)
13. [Gepinnte Revisionen](#gepinnte-revisionen)
14. [Empfehlung](#empfehlung)
15. [Limitationen](#limitationen)
16. [Glossar](#glossar)
17. [Artefakte](#artefakte)

## Auf einen Blick

Drei Serverkonfigurationen wurden vollständig vermessen. Beide Ausgangsannahmen — NVFP4 sei die richtige Quantisierung, und SGLang scheitere auf sm_121 an fehlenden Kerneln — erwiesen sich als falsch.

### INT4 schlägt NVFP4 überall

19–59 % mehr Durchsatz bei identischer Engine, fünffach kürzere Ladezeit, zehnfacher KV-Cache. Der native FP4-Rechenpfad nützt nichts, wenn die Speicherbandbreite limitiert.

### vLLM führt beim Durchsatz

13–17 % vor SGLang, mit leicht wachsendem Vorsprung bei höherer Concurrency — das Gegenteil der verbreiteten Erwartung.

### SGLang führt bei der Latenz

TTFT zwei- bis dreifach besser (0,19–0,27 s gegen 0,48–0,69 s) und 79 % mehr Durchsatz bei wiederkehrendem Präfix — dort wirkt RadixAttention.

### sm_121a ist vorhanden

`common_ops.abi3.so` enthält sm_121a-Kernels im fertigen cu130-aarch64-Wheel. SGLang lief ohne jeden Eigenbau.

- **64,3** — tok/s Spitzendurchsatz vLLM+INT4, Code, c=4

- **1 001 532** — Token KV-Cache vLLM+INT4 bei 262k Kontext

- **O(n1,25)** — Prefill-Skalierung nahezu linear statt quadratisch

- **6** — gescheiterte Serverstarts von 10 Versuchen

## Kernaussagen

### 1. Bandbreite schlägt Rechenformat

NVFP4 rechnet auf Blackwell nativ in FP4 auf den Tensor Cores, INT4 muss für die Matrixmultiplikation dequantisieren. Auf einer Maschine mit 273 GB/s Speicherbandbreite ist das irrelevant: entscheidend ist, wie viele Bytes pro Token bewegt werden. INT4 bewegt 26 GiB weniger und gewinnt deshalb auch dort, wo NVFP4 seinen nativen Pfad ausspielen könnte.

### 2. Der NVFP4-Checkpoint verbraucht doppelt so viel KV-Speicher pro Token

77,7 KiB gegen 38,8 KiB bei INT4 — bei identischem Modell und identischer Engine. Zusammen mit den 26 GiB größeren Gewichten führt das zu 94 135 statt 950 420 Token KV-Cache, also Faktor zehn. Das deutet auf einen Fehler in diesem Build hin, nicht auf eine Eigenschaft des Formats.

### 3. Der Engpass beim Start ist der Compiler, nicht das Modell

Beide Engines übersetzen FlashInfer-Kernel zur Laufzeit. Ein einzelner `cicc`-Prozess belegt bis zu 7,5 GB. Bei 95 GiB Gewichten auf 121 GB Gesamtspeicher entscheidet die Zahl paralleler Compilerprozesse über Erfolg oder OOM-Kill.

### 4. Präfix-Caching ist der stärkste Einzelhebel

16 913 Token brauchen kalt 9,32 s, warm 0,30 s — Faktor 30. Für Agenten mit wachsender Historie ist das wirksamer als jede Engine-Wahl.

### 5. Hybride Attention hält lange Kontexte bezahlbar

Bei 11,9-facher Prompt-Länge steigt die Prefill-Zeit nur 22,3-fach. Lagunas Aufteilung — 12 von 48 Layern volle Attention, 36 mit 512er Sliding Window — dämpft den quadratischen Anteil so weit, dass 222 000 Token praktisch handhabbar bleiben.

## Testaufbau

### Hardware

**Produkt** — GX10 (DGX-Spark-Plattform)

**GPU** — NVIDIA GB10, Compute Capability 12.1 (sm_121a)

**CPU** — 20 Kerne, aarch64

**Speicher** — 121 GB Unified Memory (124 546 MB), 16 GB Swap

**Bandbreite** — ca. 273 GB/s

**Massenspeicher** — NVMe, 916 GB

**Betriebssystem** — Ubuntu 24.04.4 LTS, Kernel 6.17.0-1029-nvidia

**Treiber** — NVIDIA 580.173.02, CUDA 13.0, nvcc 13.0.88

### Modell

`poolside/Laguna-S-2.1` — 118 Mrd. Parameter gesamt, 8 Mrd. aktiv pro Token (Mixture-of-Experts), veröffentlicht am 22. Juli 2026 unter OpenMDW-1.1.

**Layer** — 48 — davon 12 volle Attention, 36 Sliding-Window (Fenster 512)

**Experten** — 256 geroutet (Top-10) plus 1 geteilter

**Attention** — Grouped-Query, 8 KV-Köpfe, Kopfdimension 128

**Kontext** — `max_position_embeddings` 262 144, YaRN-Skalierung Faktor 32 von 8192

**Quantisierung** — `compressed-tensors`

#### Gemessene Dateigrößen der Varianten

*Abgefragt über die HuggingFace-API, Summe aller Gewichtsdateien*

| Variante | Größe | auf 121 GB nutzbar? |
|---|---|---|
| BF16 | 235,1 GB | nein |
| FP8 | 131,3 GB | nein |
| NVFP4 | 99,7 GB | ja, aber ohne Reserve |
| INT4 | 71,9 GB | ja, mit Reserve |
| DFlash-NVFP4 / -INT4 | je 2,23 GB | Draft-Modell |

### Software

**vLLM** — 0.25.1, aarch64-Wheel von PyPI

**SGLang** — 0.5.16 mit `sgl-kernel 0.3.21+cu130`

**PyTorch** — 2.11.0+cu130 in beiden Umgebungen

**FlashInfer** — 0.6.13 (vLLM-Umgebung)

**Python** — 3.12.13, uv-verwaltet

<details>
<summary>Warum nativ statt Docker — und warum ohne sudo</summary>

Der ursprüngliche Plan sah Container für beide Engines vor, weil das den Vergleich symmetrisch hält. Zwei Hindernisse erzwangen den Wechsel:

- **Kein Docker-Zugriff.** Der Benutzer ist nicht Mitglied der Gruppe `docker`; jeder Aufruf endete mit `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`. Der erste Pull-Versuch meldete trotzdem Exit-Code 0, weil `| tail -5` in der Pipeline den Exit-Code der ersten Stufe verschluckte — der Fehlschlag fiel erst später auf.

- **Kein passwortloses sudo.** Damit ließ sich `python3.12-dev` nicht nachinstallieren, das für die Triton-JIT-Header benötigt wird.

Gelöst durch `uv python install 3.12`: die uv-eigene CPython-Distribution bringt die Entwicklungs-Header mit, wodurch weder `sudo` noch das Systempaket nötig waren. Beide Engines liefen anschließend in getrennten venvs auf identischer PyTorch-Basis — für den Vergleich sogar sauberer als zwei Container mit womöglich abweichenden Abhängigkeiten.

</details>

<details>
<summary>Nachweis der sm_121a-Kernels</summary>

Die Behauptung, `sgl_kernel` decke sm_121a nicht ab und sei nur nach einem Neubau aus dem Quellcode lauffähig, wurde zweistufig geprüft.

#### Quellcode: sgl-kernel/CMakeLists.txt auf v0.5.16

```bash
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
    ...
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        list(APPEND SGL_KERNEL_CUDA_FLAGS
            "-gencode=arch=compute_110a,code=sm_110a"
            "-gencode=arch=compute_121a,code=sm_121a")
```

Der Gencode wird also genau dann erzeugt, wenn CUDA ≥ 13.0 *und* aarch64 vorliegen — exakt der `-cu130`-Build auf dieser Maschine. Im `else`-Zweig erhält aarch64 stattdessen nur `sm_101a`; für cu12x-Wheels trifft die Behauptung also zu.

#### Binary: cuobjdump über die installierten Bibliotheken

| Bibliothek | Größe | Enthaltene SM-Architekturen |
|---|---|---|
| common_ops.abi3.so | 80 M | sm_90 · sm_90a · sm_100a · sm_103a · sm_110a · sm_120a · **sm_121a** |
| spatial_ops.abi3.so | 196 K | sm_90 · sm_90a · sm_100a · sm_103a · sm_110a · sm_120a · **sm_121a** |
| flashmla_ops.abi3.so | 12 M | sm_90a · sm_100a · sm_103a — kein sm_121 |
| flash_ops.abi3.so | 309 M | nur sm_90a (FlashAttention-3, Hopper-exklusiv) |

Die Hauptkernel-Bibliothek enthält sm_121a. Dass FA3 und FlashMLA fehlen, ist korrekt und erwartbar — beide sind auf GB10 ohnehin nicht nutzbar. Genau diese zwei Targets deaktiviert der kursierende Community-Patch; seine Diagnose war richtig, die Schlussfolgerung „läuft nicht ohne Eigenbau" jedoch zu weitreichend.

</details>

## Methodik

Zwei eigene Messskripte, beide nur mit der Python-Standardbibliothek, beide gegen die OpenAI-kompatible API, die vLLM und SGLang gleichermaßen anbieten. Dadurch bekommen beide Engines exakt dieselben Anfragen.

### bench.py — Durchsatz und Antwortlatenz

- **4 Szenarien** × **3 Concurrency-Stufen** (1, 2, 4) × **3 Wiederholungen**

- `stream: true` zur Messung der Zeit bis zum ersten Token

- `ignore_eos: true` erzwingt exakt `max_tokens` Ausgabetoken und eliminiert damit EOS-Rauschen aus der tok/s-Rechnung

- `temperature: 0`

- Ein Aufwärmlauf vor der Messung, damit Graph-Capture und JIT nicht als Ausreißer einfließen

- Berichtet werden Mediane über alle Wiederholungen

#### Die vier Szenarien

| Name | Ausgabe | Zweck |
|---|---|---|
| prosa_decode | 256 Token | Decode-Rate an der Bandbreitendecke |
| code_decode | 256 Token | Decode auf Code, höhere DFlash-Akzeptanz erwartet |
| prefill_8k | 64 Token | Prefill-Durchsatz bei langem Prompt |
| prefill_8k_wiederholt | 64 Token | identischer Präfix — misst Präfix-Cache |

### prefill.py — kaltes Prefill

Jeder Request erhält einen einmaligen Präfix: eine UUID wird in jeden Codeblock des generierten Prompts eingewoben, sodass weder vLLMs Automatic Prefix Caching noch SGLangs RadixAttention greifen kann. Gemessen wird ausschließlich die TTFT bei `max_tokens: 8`; daraus folgt die reine Prefill-Rate.

> [!IMPORTANT]
> **Zwei Mängel der Methodik, die erst im Lauf auffielen**
>
> **Der Präfix-Vergleich misslang.** `prefill_8k` läuft dreimal pro Concurrency-Stufe — schon nach dem ersten Request liegt der Präfix im Cache. Der Median über drei Läufe misst also bereits den warmen Fall, und `prefill_8k_wiederholt` misst denselben warmen Fall erneut. Die beabsichtigte Gegenüberstellung kalt gegen warm entfiel. Nachträglich behoben durch `prefill.py`.

> [!IMPORTANT]
> **Die Größenlabels lagen um Faktor zwei daneben.** Die Schätzung von 68 Token pro generiertem Codeblock war falsch, real sind es etwa 154. Die als „8k" bezeichneten Prompts enthielten tatsächlich 16 909 bis 18 820 Token. In `prefill.py` wurden die Stufen später nach den gemessenen Tokenzahlen umbenannt.

## Chronik der Startversuche

Zehn Serverstarts waren nötig, um drei Konfigurationen zu vermessen. Die Reihenfolge trägt hier Information: jeder Fehlschlag lieferte die Korrektur für den nächsten Versuch. Aufklappen zeigt Ursache, Logauszug und Konsequenz.

<details>
<summary>01 vLLM + NVFP4 — Speicherreservierung zu hoch 17:35 · abgebrochen nach < 1 min gescheitert</summary>

Gestartet mit `--gpu-memory-utilization 0.92`. Die Engine bricht vor dem Laden ab.

```bash
ValueError: Free memory on device cuda:0 (110.73/121.63 GiB) on startup is
less than desired GPU memory utilization (0.92, 111.9 GiB). Decrease ...
```

**Ursache:** Das Betriebssystem belegte rund 11 GiB, sodass nur 110,73 der 121,63 GiB frei waren — knapp unter den geforderten 111,9 GiB.

**Korrektur:** `--gpu-memory-utilization 0.89`.

</details>

<details>
<summary>02 vLLM + NVFP4 — OOM-Killer während der JIT-Kompilierung 17:36–18:01 · 25 min verloren gescheitert</summary>

Der Lauf kam weit: Gewichte geladen (95,63 GiB in 595,6 s), `torch.compile` durchgelaufen (30,61 s). Danach zwölf Minuten Stille im Anwendungslog, dann war der EngineCore-Prozess ein Zombie.

```bash
Aug 01 18:01:30 dgx-spark kernel: systemd invoked oom-killer: ...
Aug 01 18:01:30 dgx-spark kernel: Out of memory: Killed process 21394 (cicc)
    total-vm:10886376kB, anon-rss:7529332kB, UID:1000
```

**Ursache:** Nach dem Modell blieben etwa 26 GiB frei. `MAX_JOBS=4` startete vier parallele `cicc`-Prozesse (CUDA-Compiler-Frontend) zu je rund 7,5 GB — zusammen 30 GB. Der OOM-Killer griff ein. Im Anwendungslog stand kein Fehler; die Ursache war nur im Kernel-Log zu finden.

**Korrektur:** `MAX_JOBS=1`. Die verbreitete Empfehlung `MAX_JOBS=4` ist für die NVFP4-Variante auf dieser Box zu hoch angesetzt.

</details>

<details>
<summary>03 vLLM + NVFP4 — mit der Elternshell beendet 18:02 · 26 Logzeilen abgebrochen</summary>

Kein technischer Fehlschlag: der Prozess wurde abgeräumt, als die Sitzung endete. Kein OOM-Eintrag im Kernel-Log, der Lauf kam nicht über die Initialisierung hinaus.

**Korrektur:** Start künftig über `setsid nohup … < /dev/null &`, damit der Server einen Sitzungswechsel überlebt.

> [!NOTE]
> Nebenbefund: Ein `pkill -f "vllm serve"` beendete zweimal die eigene Shell mit Exit-Code 143 bzw. 144, weil das Suchmuster in deren Kommandozeile stand. Prozesse besser über `ps` ermitteln und per PID beenden.

</details>

<details>
<summary>04 vLLM + NVFP4 — erfolgreich, erste Messreihe 21:06–21:23 · 17 min bis bereit erfolgreich</summary>

`MAX_JOBS=1`, `--gpu-memory-utilization 0.89`, `--max-model-len 65536`.

```bash
Model loading took 95.63 GiB memory and 595.56 seconds
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
Available KV cache memory: 6.97 GiB
GPU KV cache size: 94,135 tokens
Maximum concurrency for 65,536 tokens per request: 1.44x
```

Bestätigt nebenbei den nativen FP4-Pfad: `FLASHINFER_CUTLASS` statt eines Marlin-Rückfalls. Der Smoke-Test lieferte 150 Token in 8,2 s.

**Kritische Kennzahl:** 1,44× Concurrency bedeutet, dass nicht einmal anderthalb Requests mit vollem Kontext gleichzeitig in den Speicher passen.

</details>

<details>
<summary>05 SGLang + NVFP4 — Graph-Capturing bei 0/58 21:40–22:01 · nach 10 min abgebrochen gescheitert</summary>

`MAX_JOBS=1`, `--mem-fraction-static 0.88`. Die Gewichte luden sauber (618,01 s, 93,16 GB), danach blieb der Fortschrittsbalken des CUDA-Graph-Capturings zehn Minuten auf Schritt 0 von 58 stehen.

```bash
Capture target prefill CUDA graph begin. backend=breakable
Capturing num tokens (num_tokens=8192 avail_mem=11.78 GB): 0%| | 0/58
```

Ein `cicc`-Prozess lief mit 99,9 % CPU, der FlashInfer-Cache wuchs von 80 auf 179 MB — es wurde also kompiliert, nur sehr langsam.

**Damalige Diagnose:** „hängt". **Tatsächlich:** der erste Capture-Schritt enthält die gesamte Kernel-Übersetzung und dauert allein 114 s; die folgenden 57 laufen um Größenordnungen schneller. Der Abbruch war voreilig (siehe Fehldiagnosen).

</details>

<details>
<summary>06 SGLang + NVFP4 — Speicher auf 1 GB, präventiv abgebrochen 22:06–22:25 · MAX_JOBS=3 gescheitert</summary>

Annahme: mehr parallele Compilerjobs beschleunigen das Capturing. Gewichte in 549 s geladen, danach zwei `cicc`-Prozesse — und der freie Systemspeicher fiel auf **1 GB**. Der Balken stand weiterhin bei 0/58.

**Abgebrochen,** bevor der OOM-Killer zuschlagen konnte. Ein zweiter OOM hätte die zehn Minuten Ladezeit erneut gekostet.

</details>

<details>
<summary>07 SGLang + NVFP4 — auf Anweisung abgebrochen 22:25–22:35 · mem-fraction 0.78 abgebrochen</summary>

Dritter Anlauf mit umgekehrter Priorität: `--mem-fraction-static 0.78` statt 0,88, also 26,8 statt 14,6 GiB Systemluft für den Compiler, zulasten des KV-Caches. Der Lauf wurde während des Ladens der Gewichte abgebrochen.

**Zwischenfazit an dieser Stelle:** SGLang scheitert mit NVFP4 auf dieser Box nicht an fehlenden Kerneln, sondern am Startpfad. Diese Einschätzung erwies sich später als ebenfalls falsch — siehe Versuch 09.

</details>

<details>
<summary>08 vLLM + INT4 — erfolgreich, Quantisierungsvergleich 23:19–23:30 · 11 min bis bereit erfolgreich</summary>

Alle Leistungsparameter identisch zu Versuch 04, nur die Quantisierung unterscheidet sich.

```bash
Model loading took 69.34 GiB memory and 113.64 seconds
Available KV cache memory: 35.2 GiB
GPU KV cache size: 950,420 tokens
Maximum concurrency for 65,536 tokens per request: 14.50x
```

Gegenüber NVFP4: 26,3 GiB weniger Gewichte, fünffach kürzere Ladezeit, zehnfacher KV-Cache. Der Faktor zehn setzt sich zusammen aus fünffach mehr Speicher *und* doppelt so effizienter Nutzung pro Token (38,8 statt 77,7 KiB).

</details>

<details>
<summary>09 SGLang + INT4 — erfolgreich, Capturing in 3:33 23:37–23:43 · 6 min bis bereit erfolgreich</summary>

`MAX_JOBS=2`, `--mem-fraction-static 0.85`. Der erste Capture-Schritt dauerte erneut 114,6 s, und tqdm extrapolierte daraus:

```bash
Capturing num tokens (num_tokens=8192): 2%|▏ | 1/58 [01:54<1:48:53, 114.62s/it]
```

Tatsächlich fiel die Rate danach steil ab — 51 s, 31 s, 21 s, 16 s … bis auf 6 Schritte pro Sekunde am Ende:

```bash
Capturing num tokens (num_tokens=4): 100%|██████████| 58/58 [03:33<00:00, 3.68s/it]
[23:43:42] max_total_num_tokens=185897, max_running_requests=48
[23:43:47] The server is fired up and ready to roll!
```

**Damit war die Diagnose aus den Versuchen 05 bis 07 widerlegt:** SGLang hing nie, es war nur im ersten Schritt langsam. Die abgebrochenen NVFP4-Versuche wären wahrscheinlich ebenfalls durchgelaufen.

</details>

<details>
<summary>10 vLLM + INT4 bei 262k Kontext — erfolgreich 00:21–00:24 · 3 min bis bereit erfolgreich</summary>

`--max-model-len 262144` statt 65 536, sonst unverändert.

```bash
Model loading took 69.34 GiB memory and 115.06 seconds
GPU KV cache size: 1,001,532 tokens
Maximum concurrency for 262,144 tokens per request: 3.82x
```

**Befund:** Das vervierfachte Kontextfenster kostete keinen KV-Cache — es brachte sogar 5 % mehr (1 001 532 statt 950 420 Token), vermutlich durch günstigere Blockaufteilung. Es gibt keinen Grund, unter dem Maximum des Checkpoints zu bleiben.

</details>

## Fallstricke

Geordnet nach Ursache, mit der jeweils wirksamen Gegenmaßnahme.

### Speicher und Compiler

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Paralleler JIT-Fan-out | OOM-Kill von `cicc`, kein Fehler im Anwendungslog | `MAX_JOBS=1` bei kaltem FlashInfer-Cache |
| Reservierung über dem Freibestand | `ValueError` vor dem Laden | `--gpu-memory-utilization` ≤ 0,89 |
| mem-fraction-static missverstanden | kleineres Modell schafft *keine* Systemluft | Der Parameter reserviert unabhängig von der Modellgröße; Ersparnis wandert in den KV-Cache |
| Verzögerte Speicherfreigabe | Neustart scheitert direkt nach dem Stoppen | Auf Freigabe warten, hier 5–15 s bis 118 GB frei |
| Page-Cache-Verdrängung | Shard-Laderate bricht von 2,7 auf 30 s/Shard ein | Unvermeidlich bei 95 GiB Gewichten; Swap-Aktivität bleibt moderat |

### Installation

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Torch-ABI-Bruch | `undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v` | `torch==2.11.0` — die von sglang 0.5.16 gepinnte Version |
| Veraltete Rezepte | `--force-reinstall torch` zieht die neueste Version (2.13.0) | Schritt weglassen; die cu130-Wheels bringen die passende Version mit |
| Falsches CUDA-Wheel | fehlende sm_121a-Kernels bei cu12x-Builds | Auf dem Spark ausschließlich `-cu130` auf aarch64 verwenden |
| Fehlende Python-Header | Triton-JIT scheitert ohne `python3.12-dev` | `uv python install 3.12` — bringt Header mit, kein sudo nötig |

### Betrieb und Werkzeuge

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Exit-Code in Pipeline verschluckt | `docker pull … \| tail -5` meldet Erfolg trotz `permission denied` | Exit-Code der ersten Pipeline-Stufe prüfen, nicht der letzten |
| `pkill -f` trifft die eigene Shell | Kommando endet mit Exit 143/144, Ziel läuft weiter | PIDs über `ps` ermitteln, gezielt beenden |
| Server stirbt mit der Sitzung | Log bricht nach wenigen Zeilen ab, kein Fehler | `setsid nohup … < /dev/null &` |
| tqdm-Restzeit als Tatsache gelesen | angezeigt 1:48:53, real 3:33 | Bei Verläufen mit einmaliger Anlaufhürde die Rate über mehrere Schritte beobachten |
| Verwaiste `.incomplete`-Dateien | 13 GB belegt nach abgebrochenem Download | Nach Abbruch aufräumen; `hf download` setzt sonst korrekt fort |

### Modell und API

| Fallstrick | Symptom | Gegenmaßnahme |
|---|---|---|
| Reasoning-Parser initialisiert nicht | `Auto-initialization of reasoning token IDs failed`; Reasoning landet im Antworttext | Ungelöst — entwertet die Trennung Code gegen Prosa |
| Prompt über `max_model_len` | `HTTP 400 Bad Request` ohne weitere Erklärung | Kontextfenster anheben; 262 144 kostet hier nichts |
| Größenangaben der HF-API | fehlende `size`-Felder werden als 0 verrechnet | Vollständigkeit prüfen, sonst um 28 GB zu niedrige Summen |
| Mehrere Modelle gleichzeitig | zwei Modelle über 60 GB passen nicht nebeneinander | Strikt sequenzieller Betrieb |
| Modell-Revision nicht gepinnt | Upstream verschiebt `main`; 93 GB werden stillschweigend nachgeladen, danach bricht der Start am KV-Cache ab | `--revision` für Modell und Draft pinnen — siehe unten |

## Durchsatz

*Gesamtdurchsatz in tok/s über alle gleichzeitigen Requests · Median aus 3 Läufen*

| Szenario | c | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|---|
| Prosa | 1 | 17,3 | 20,6 | 18,2 |
| Prosa | 2 | 25,5 | 31,7 | 27,8 |
| Prosa | 4 | 42,7 | 52,8 | 45,3 |
| Code | 1 | 15,0 | 23,8 | 21,9 |
| Code | 2 | 27,4 | 39,0 | 33,2 |
| Code | 4 | 42,4 | 64,3 | 56,2 |
| Prefill 18k | 1 | 5,8 | 13,1 | 6,3 |
| Prefill 18k | 2 | 11,6 | 38,9 | 25,1 |
| Prefill 18k | 4 | 18,7 | 64,6 | 38,0 |
| Prefill 18k wiederholt | 1 | 7,5 | 24,0 | 25,4 |
| Prefill 18k wiederholt | 2 | 11,8 | 31,1 | 41,1 |
| Prefill 18k wiederholt | 4 | 17,6 | 39,5 | 70,6 |

### Was die Zahlen zeigen

**INT4 gewinnt gegen NVFP4 durchgängig** — bei identischer Engine zwischen 19 und 59 % im Decode. Die extremen Werte bei den Prefill-Szenarien (bis +246 %) sind kein Messfehler, sondern Folge des KV-Caches: vier parallele Requests zu je 16 913 Token belegen 67 652 Token. Bei NVFP4 mit 94 135 Token Kapazität führt das zu Verdrängung und Warteschlangen, bei INT4 mit 950 420 Token ist es unkritisch.

**vLLM gewinnt gegen SGLang beim Decode** — 13 bis 17 %, mit leicht wachsendem Vorsprung bei höherer Concurrency. Das widerspricht der verbreiteten Annahme, SGLang skaliere über Concurrency besser.

> [!NOTE]
> **Die aufschlussreichste Zeile**
>
> Beim wiederholten Präfix mit c=4 dreht sich das Verhältnis: SGLang erreicht **70,6** gegen vLLMs **39,5** tok/s. Im Szenario davor, beim ersten Kontakt mit demselben Präfix, war es umgekehrt (38,0 gegen 64,6). SGLang wird also besser, je öfter derselbe Kontext wiederkehrt, während vLLM dort einbricht — genau das Profil agentischen Codierens mit wachsender Historie, und genau die Wirkung, die RadixAttention verspricht.

<details>
<summary>Decode-Rate pro einzelnem Stream</summary>

Während der Gesamtdurchsatz mit der Concurrency steigt, fällt die Rate des einzelnen Streams — das erwartbare Verhalten eines bandbreitenlimitierten Systems.

*Median der Decode-Rate je Request in tok/s*

| Szenario | c | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|---|
| Prosa | 1 | 18,11 | 21,20 | 18,56 |
| Prosa | 2 | 13,29 | 16,81 | 14,25 |
| Prosa | 4 | 11,14 | 13,72 | 11,40 |
| Code | 1 | 17,03 | 25,35 | 22,13 |
| Code | 2 | 14,15 | 20,40 | 17,10 |
| Code | 4 | 11,16 | 17,60 | 14,48 |
| Prefill 18k | 1 | 7,90 | 29,04 | 20,27 |
| Prefill 18k | 2 | 6,20 | 22,82 | 14,19 |
| Prefill 18k | 4 | 4,93 | 19,18 | 10,76 |
| Prefill 18k wiederholt | 1 | 7,77 | 27,25 | 27,94 |
| Prefill 18k wiederholt | 2 | 6,22 | 17,64 | 22,31 |
| Prefill 18k wiederholt | 4 | 4,66 | 14,27 | 19,08 |

Die niedrigen Werte in den Prefill-Szenarien sind ein Artefakt der kurzen Ausgabe: bei nur 64 Token dominiert der Fixkostenanteil, weshalb diese Zahlen nicht mit den 256-Token-Szenarien vergleichbar sind.

</details>

## Antwortlatenz

*Zeit bis zum ersten Token in Sekunden · Median*

| Szenario | c | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|---|
| Prosa | 1 | 0,352 | 0,276 | 0,314 |
| Prosa | 2 | 0,443 | 0,635 | 0,201 |
| Prosa | 4 | 0,427 | 0,513 | 0,193 |
| Code | 1 | 0,374 | 0,260 | 0,194 |
| Code | 2 | 0,355 | 0,266 | 0,196 |
| Code | 4 | 0,374 | 0,480 | 0,194 |
| Prefill 18k | 1 | 0,307 | 0,309 | 0,200 |
| Prefill 18k | 2 | 0,468 | 0,458 | 0,217 |
| Prefill 18k | 4 | 0,686 | 0,551 | 0,269 |
| Prefill 18k wiederholt | 1 | 0,301 | 0,328 | 0,204 |
| Prefill 18k wiederholt | 2 | 0,460 | 0,484 | 0,255 |
| Prefill 18k wiederholt | 4 | 0,781 | 0,630 | 0,285 |

SGLang gewinnt in elf von zwölf Messpunkten. Auffällig ist die **Stabilität**: die Werte liegen fast unabhängig von Concurrency und Prompt-Länge zwischen 0,19 und 0,29 s, während vLLM bis 0,78 s streut. Für interaktive Nutzung, wo die gefühlte Reaktionszeit zählt, ist das ein realer Vorteil — er geht nur nicht in den Durchsatz ein.

## Prefill-Skalierung

Gemessen mit einmaligem Präfix je Request, sodass kein Cache greift. Die drei kurzen Reihen stammen aus den Hauptläufen, die lange Reihe aus der Nachmessung bei 262 144 Token Kontext.

*Kaltes Prefill in tok/s · Median aus 3 Läufen mit je eigener UUID im Prompt*

| Prompt-Token | vLLM+NVFP4 | vLLM+INT4 | SGLang+INT4 |
|---|---|---|---|
| ≈ 4 700 | 2051 | 2288 | 1620 |
| ≈ 18 500 | 2019 | 2243 | 1686 |

vLLM liegt hier deutlich vorn — gegenüber SGLang um 33 %. Zusammen mit dem Durchsatzvorteil ist das die zweite Achse, auf der vLLM gewinnt.

### Verhalten über die Prompt-Länge

*vLLM + INT4 bei 262 144 Token Kontextfenster*

| Prompt-Token | TTFT | Prefill | relativ |
|---|---|---|---|
| 4 689 | 2,1 s | 2255 tok/s | — |
| 18 699 | 8,1 s | 2301 tok/s | 100 % |
| 72 443 | 39,2 s | 1850 tok/s | 80 % |
| 149 057 | 100,4 s | 1484 tok/s | 64 % |
| 221 996 | 181,3 s | 1224 tok/s | 53 % |

Von 18 699 auf 221 996 Token ist die Länge **11,9-fach**, die Zeit steigt aber nur **22,3-fach**. Daraus folgt eine Komplexität von etwa **O(n1,25)** — bemerkenswert nah am linearen Verlauf.

Zur Einordnung: bei reiner Voll-Attention wäre O(n²) zu erwarten, die Rate bei 222 000 Token also auf rund 8 % statt der gemessenen 53 % eingebrochen. Lagunas Aufteilung — nur 12 der 48 Layer mit voller Attention, 36 mit 512er Sliding Window — fängt den quadratischen Anteil weitgehend ab.

> [!NOTE]
> **Praktische Konsequenz**
>
> Der volle 262k-Kontext kostet hochgerechnet rund **3,7 Minuten** kaltes Prefill. Einmalig erträglich, pro Anfrage nicht. Mit Präfix-Cache fällt das auf Bruchteile: bei 16 913 Token wurden 0,30 s warm gegen 9,32 s kalt gemessen, also **Faktor 30**. Für Agenten mit wachsender Historie wird der lange Kontext damit einmal bezahlt und danach wiederverwendet.

## Speicher und Start

*Aus den Startlogs der jeweils erfolgreichen Läufe*

| Kennzahl | vLLM+NVFP4 | vLLM+INT4 (65k) | vLLM+INT4 (262k) | SGLang+INT4 |
|---|---|---|---|---|
| Gewichte | 95,63 GiB | 69,34 GiB | 69,34 GiB | 67,56 GB |
| Ladezeit | 595,6 s | 113,6 s | 115,1 s | 109,3 s |
| KV-Speicher | 6,97 GiB | 35,2 GiB | — | — |
| KV-Cache | 94 135 | 950 420 | 1 001 532 | 185 897 |
| KV je Token | 77,7 KiB | 38,8 KiB | — | — |
| Max. Concurrency | 1,44× (65k) | 14,50× (65k) | 3,82× (262k) | 48 Requests |
| Startdauer gesamt | ≈ 17 min | ≈ 11 min | ≈ 3 min | ≈ 6 min |

Die Zeile **KV je Token** ist der unerwartetste Einzelbefund: bei identischem Modell und identischer Engine verbraucht der NVFP4-Checkpoint genau doppelt so viel KV-Speicher pro Token wie der INT4-Checkpoint. Zusammen mit den 26 GiB größeren Gewichten ergibt das den Faktor zehn beim nutzbaren Cache.

**vLLM gegen SGLang beim KV-Cache:** 950 420 gegen 185 897 Token bei vergleichbarer Speichereinstellung — mehr als das Fünffache. Zur Fairness: `--mem-fraction-static 0.85` und `--gpu-memory-utilization 0.89` sind nicht exakt äquivalent, ein Teil der Lücke geht darauf. Fünffach erklärt das aber nicht.

<details>
<summary>Startverhalten der beiden Engines im Vergleich</summary>

Die Engines unterscheiden sich strukturell darin, *wann* sie FlashInfer-Kernel übersetzen:

- **vLLM** erledigt den JIT in einer eigenen Phase nach `torch.compile` und vor dem Graph-Capturing. Der OOM-Kill in Versuch 02 traf genau diese Phase.

- **SGLang** kompiliert *während* des Graph-Capturings. Der erste Capture-Schritt (`num_tokens=8192`) enthält die gesamte Übersetzung und dauerte 114,6 s; die verbleibenden 57 Schritte zusammen nur 99 s.

Der Kernel-Cache in `~/.cache/flashinfer` wuchs über die Versuche hinweg von 80 MB auf 234 MB und ist persistent. Jeder spätere Start mit demselben Modell profitiert davon — SGLang war im letzten Lauf nach 6 Minuten bereit statt nach den knapp 25 des ersten vLLM-Erfolgs.

</details>

## Fehldiagnosen im Verlauf

Vier Einschätzungen während der Durchführung erwiesen sich als falsch und beeinflussten das Vorgehen. Sie sind hier dokumentiert, weil sie den Zeitverlauf und einige Abbrüche erklären.

| Aussage | Tatsächlich | Folge |
|---|---|---|
| NVFP4 sei 71,9 GB groß | 99,7 GB — die HF-API lieferte für einen Teil der Dateien kein `size`-Feld, das als 0 verrechnet wurde | Speicherplanung und Kontextempfehlung zunächst zu optimistisch |
| SGLangs Graph-Capturing „hängt" | Es kompilierte; der erste von 58 Schritten dauert 114 s, danach fällt die Rate steil ab | Versuche 05 und 06 vorzeitig abgebrochen |
| Capturing dauert 1 h 50 min | 3 min 33 s — die Zahl war eine tqdm-Hochrechnung aus dem ersten, compile-lastigen Schritt | Unnötige Rückfrage zum weiteren Vorgehen, während der Server bereits lief |
| INT4 verschaffe SGLang mehr Systemluft | `mem-fraction-static` reserviert einen Anteil des Gesamtspeichers unabhängig von der Modellgröße; die Ersparnis wandert in den KV-Cache | Falsche Erwartung an Versuch 09, der aus anderem Grund gelang |

> [!IMPORTANT]
> Das gemeinsame Muster: Eine **Momentaufnahme** wurde für einen **Zustand** gehalten. Ein stehender Fortschrittsbalken bedeutete nicht Stillstand, eine extrapolierte Restzeit war keine Messung, eine unvollständige API-Antwort keine Vollständigkeit. In allen vier Fällen hätte eine zweite Beobachtung über die Zeit den Irrtum aufgedeckt.

## Widerlegte Annahmen

Ausgangspunkt der Tests waren kursierende Empfehlungen zum Betrieb von Laguna-S-2.1 auf der DGX Spark. Diese wurden gegen die Messungen geprüft.

| Behauptung | Befund | Beleg |
|---|---|---|
| `sgl_kernel` ohne sm_121a, nur mit Neubau lauffähig | falsch | cuobjdump zeigt sm_121a in `common_ops.abi3.so`; SGLang lief ohne Eigenbau |
| NVFP4 ist die Variante für die Spark | falsch | INT4 gewinnt in allen zwölf Messpunkten |
| NVFP4 belegt etwa 71 GB | falsch | 99,7 GB laut HF-API, 95,63 GiB laut Ladeprotokoll |
| INT4 belegt etwa 59 GB | falsch | 71,9 GB laut HF-API, 69,34 GiB laut Ladeprotokoll |
| Prefill 600–800 tok/s | zu niedrig | 1224–2301 tok/s je nach Prompt-Länge |
| BF16 etwa 236 GB | bestätigt | 235,1 GB |
| Decode ohne Spekulation 13–14 tok/s | plausibel | mit DFlash 18,1–25,3 tok/s im Einzelstream |
| `MAX_JOBS=4` ist nötig | bestätigt, aber zu schwach | Führte zum OOM-Kill; nötig war `MAX_JOBS=1` |

## Gepinnte Revisionen

Nachgetragen am 4. August 2026, nachdem ein Neustart der empfohlenen Konfiguration scheiterte.

poolside hat am 3. August `main` von `Laguna-S-2.1-INT4` auf einen Checkpoint umgestellt, der **nur dem Namen nach INT4** ist. Der Start mit dem unveränderten Skript zog vierzig Minuten lang stillschweigend 93 GB nach und brach dann ab.

### Was sich upstream geändert hat

Die `quantization_config` ist in beiden Revisionen identisch — 4 Bit, `type: int`, `pack-quantized`, Gruppengröße 32, 48 Layer. Der Unterschied liegt darin, was davon *ausgenommen* wird. Die neue Revision ergänzt die Ignore-Liste um eine Zeile:

```bash
re:^model\.layers\.(?:40|41|42|43|44|45|46|47)\.mlp\.experts\.[0-9]+\.(?:gate_proj|up_proj|down_proj)$
```

Das sind 8 Layer × 256 Experten × 3 Projektionen = **6144 Gewichtsmatrizen, die von INT4 nach bfloat16 gewandert sind**. Das Tensor-Inventar bestätigt es exakt: `weight_packed` sinkt um 6144, einfaches `weight` steigt um dieselbe Zahl.

*Gemessen aus `config.json` und `model.safetensors.index.json` beider Revisionen*

| Eigenschaft | gepinnt `67dbeda4` | Upstream `main` (3. Aug.) |
|---|---|---|
| Checkpoint-Größe | 66,96 GiB | 92,84 GiB |
| Shards | 15 | 19 |
| INT4-gepackte Tensoren | 36 096 | 29 952 |
| Unquantisierte Gewichte | 626 | 6 770 |
| `k_scale` / `v_scale` | 96 | 0 |
| KV-Cache je Token | 38,8 KiB | 73,4 KiB |

### Warum der Server den Dienst verweigert

Beide Änderungen verstärken sich: Die größeren Gewichte lassen weniger Speicher übrig, und die fehlenden KV-Skalen lassen dann jedes Token doppelt so viel Cache kosten:

```bash
ValueError: To serve at least one request with the model's max seq len (262144),
18.35 GiB KV cache is needed, which is larger than the available KV cache
memory (8.72 GiB). Based on the available memory, the estimated
maximum model length is 121840.
```

`--gpu-memory-utilization` stand bereits auf 0,89, der in Versuch 01 ermittelten Obergrenze. Nach oben ist kein Spielraum mehr.

### Die Korrektur

```bash
MODELL_REV=67dbeda456e68139f281c40831f9d12049d8fc11
DRAFT_REV=f6b32f4fb7ef2fb2ad481bb4c05433a2bf8b0ed1

exec vllm serve poolside/Laguna-S-2.1-INT4 \
  --revision "$MODELL_REV" \
  --speculative-config "{\"model\":\"poolside/Laguna-S-2.1-DFlash-INT4\",\"revision\":\"$DRAFT_REV\",…}" \
```

Das Draft-Modell ist mitgepinnt. Es hat derzeit nur eine Revision, aber es ungepinnt zu lassen verschiebt denselben Fehler nur nach hinten.

> [!IMPORTANT]
> **Alle Messwerte dieses Protokolls stammen von der gepinnten Revision.** Werte von Upstream-`main` sind nicht vergleichbar: andere Gewichte, halber KV-Cache und ein Kontextfenster, das bei rund 121 840 statt 262 144 Token endet.

> [!NOTE]
> **Früh erkennen**
>
> Das Symptom täuscht — vierzig Minuten ohne Logausgabe, kein Fehler, der Speicher füllt sich. Erkennbar ist es nicht am Log, sondern am Prozess: `write_bytes` in `/proc/<pid>/io` wächst in den zweistelligen Gigabytebereich, während `read_bytes` klein bleibt. Das ist ein Download, kein Ladevorgang. Ein zweites Snapshot-Verzeichnis unter `~/.cache/huggingface/hub/models--*/snapshots/` bestätigt es.

## Empfehlung

### Standard: vLLM mit INT4

Bester Durchsatz, fünffacher KV-Cache gegenüber SGLang, schnellstes Kaltprefill, 3,82 parallele Requests bei vollem 262k-Kontext.

```bash
export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=1

vllm serve poolside/Laguna-S-2.1-INT4 \
  --speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash-INT4","num_speculative_tokens":15}' \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --enable-auto-tool-choice \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.89 \
  --host 0.0.0.0 --port 8000
```

### Alternative bei interaktivem Agentenbetrieb: SGLang mit INT4

Dreifach bessere Antwortlatenz und 79 % mehr Durchsatz bei wiederkehrendem Präfix. Wenn die Arbeitslast lange geteilte Kontexte hat, kann das den Durchsatznachteil überwiegen.

```bash
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=2

python3 -m sglang.launch_server \
  --model-path poolside/Laguna-S-2.1-INT4 \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path poolside/Laguna-S-2.1-DFlash-INT4 \
  --page-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 65536 \
  --cuda-graph-max-bs 8 \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --host 0.0.0.0 --port 30000
```

> [!NOTE]
> **Kein `--attention-backend` setzen.** Die Auto-Auswahl trifft auf sm_121 `flashinfer`, weil `trtllm_mha` laut Quellcode SM120 nicht unterstützt und `fa3` Hopper-exklusiv ist. Triton ist für Lagunas Sliding-Window-Attention ausdrücklich nicht geeignet.

## Limitationen

- **Reasoning nicht abgetrennt.** Der `poolside_v1`-Parser ließ sich nicht initialisieren. Beide Textszenarien enthalten überwiegend Reasoning-Prosa, wodurch die Trennung zwischen Code und Prosa entwertet ist. Der Engine-Vergleich bleibt gültig, weil beide Engines identische Prompts erhielten.

- **Speicherparameter nicht exakt äquivalent.** `--mem-fraction-static 0.85` und `--gpu-memory-utilization 0.89` sind nicht dasselbe; ein Teil der KV-Cache-Differenz zwischen den Engines geht darauf zurück.

- **SGLang nur mit INT4 vermessen.** Die NVFP4-Läufe wurden nach drei Fehlversuchen abgebrochen — nach heutigem Kenntnisstand vorzeitig.

- **Je eine Messreihe pro Konfiguration.** Drei Wiederholungen je Messpunkt, aber kein Mittel über mehrere Sitzungen. Reproduzierbarkeit wurde punktuell geprüft: die 4,7k-Prefill-Stufe lieferte in zwei Läufen 2288 und 2255 tok/s (Abweichung 1,4 %).

- **32k-Stufe im Hauptlauf gescheitert.** Der Prompt überschritt `max_model_len 65536` und wurde erst in der Nachmessung bei 262 144 erfasst.

- **Concurrency nur bis 4.** Höhere Stufen wurden nicht gemessen, obwohl vLLM+INT4 mit 14,5× und SGLang mit 48 gleichzeitigen Requests deutlich mehr zulassen.

## Glossar

**sm_121a**
Compute Capability 12.1 der GB10-GPU. CUDA-Kernel müssen für diese Architektur übersetzt sein; die Gencode-Direktive erzeugt sie nur bei CUDA ≥ 13.0 auf aarch64.

**NVFP4 / INT4**
Vier-Bit-Quantisierungen. NVFP4 ist NVIDIAs Microscaling-Format (E2M1 mit FP8-Blockskalen), das Blackwell-Tensor-Cores nativ verarbeiten. INT4 speichert Ganzzahlen und dequantisiert vor der Matrixmultiplikation.

**KV-Cache**
Zwischenspeicher der Key- und Value-Tensoren bereits verarbeiteter Token. Seine Größe begrenzt, wie viele Requests mit welcher Kontextlänge gleichzeitig laufen können.

**TTFT**
Time To First Token — Zeit von der Anfrage bis zum ersten ausgegebenen Token. Bestimmt die gefühlte Reaktionszeit, unabhängig vom Durchsatz.

**Prefill / Decode**
Prefill verarbeitet den Eingabe-Prompt (rechenlastig, parallelisierbar), Decode erzeugt die Ausgabe Token für Token (bandbreitenlastig, sequenziell).

**DFlash**
Von poolside trainiertes Draft-Modell für Speculative Decoding, gepaart mit dem Basismodell. Erzeugt Kandidaten-Token günstig und lässt sie parallel verifizieren.

**RadixAttention**
SGLangs Präfix-Cache als Radix-Baum: gemeinsame Prompt-Anfänge mehrerer Requests werden automatisch wiederverwendet. Das Gegenstück bei vLLM heißt Automatic Prefix Caching.

**Sliding-Window-Attention**
Attention über ein begrenztes Fenster zurückliegender Token statt über die gesamte Sequenz. Laguna nutzt in 36 von 48 Layern ein Fenster von 512, wodurch der KV-Cache und die Prefill-Kosten deutlich langsamer als quadratisch wachsen.

**cicc**
Compiler-Frontend der CUDA-Toolchain, von `nvcc` aufgerufen. Belegte in diesen Tests bis zu 7,5 GB pro Prozess und war Auslöser des OOM-Kills.

**CUDA-Graph-Capturing**
Vorabaufzeichnung fester Ausführungsgraphen für wiederkehrende Batch-Formen. Kostet Startzeit und Speicher, ist auf dieser Hardware aber unverzichtbar — Deaktivierung kostet laut Herstellerangaben rund 55 % Durchsatz.

**MoE / aktive Parameter**
Mixture-of-Experts: nur ein Teil der Experten wird pro Token ausgewertet. Laguna hat 118 Mrd. Parameter gesamt, aber nur 8 Mrd. aktiv — für die Decode-Geschwindigkeit zählt die zweite Zahl, für den Speicherbedarf die erste.

**Unified Memory**
Auf GB10 teilen sich CPU und GPU einen physischen Speicherpool. Es gibt keinen separaten VRAM; Modellgewichte, KV-Cache, Betriebssystem und Compilerprozesse konkurrieren um dieselben 121 GB.

## Artefakte

#### Messwerkzeuge

**bench.py** — Durchsatz und TTFT über 4 Szenarien × 3 Concurrency-Stufen, nur Standardbibliothek

**prefill.py** — Kaltes Prefill über 5 Längenstufen bis 222 000 Token, einmaliger Präfix je Request

#### Startskripte

**start-vllm-int4.sh** — Empfohlene Konfiguration: INT4, 262k Kontext, DFlash, `MAX_JOBS=1`

**start-sglang-int4.sh** — Latenz-Alternative: INT4, DFlash, `MAX_JOBS=2`

**start-vllm.sh · start-sglang.sh** — NVFP4-Varianten zur Reproduktion

#### Rohdaten

**ergebnisse_vllm.json** — vLLM + NVFP4, alle 12 Messpunkte

**ergebnisse_int4.json** — vLLM + INT4, alle 12 Messpunkte

**ergebnisse_sglang.json** — SGLang + INT4, alle 12 Messpunkte

**prefill_*.json** — vier Prefill-Reihen einschließlich der Langmessung

***.log** — Startprotokolle aller zehn Versuche, auch der gescheiterten

> [!NOTE]
> Nach Abschluss der Tests wurden die NVFP4-Gewichte (93 GB) und das zugehörige Draft-Modell (2,1 GB) entfernt. Im Cache verbleiben `Laguna-S-2.1-INT4` (67 GB) und `Laguna-S-2.1-DFlash-INT4` (2,1 GB). Die Startskripte `start-vllm.sh` und `start-sglang.sh` verweisen weiterhin auf das gelöschte Modell und würden es bei einem Start erneut herunterladen.

---

*Testprotokoll Laguna-S-2.1 auf DGX Spark GB10 · 10 Serverstarts, 3 vollständige Messreihen, 36 Messpunkte plus 9 Prefill-Stufen · alle Werte gemessen, keine Hochrechnungen außer der ausdrücklich als solche gekennzeichneten Extrapolation auf 262 144 Token.*
