# Donor-Implantate für den 959,5-M-Seed — Branch `delivery/donor-paket-20261111`

**Was hier liegt:** das komplette Lieferpaket vom 11.11.2026 als einzelne Dateien — einsetzfertige Implantate aus
Qwen3-Coder (Code) und Mistral Large 4 (Reasoning, adaptives Denken, Lernen im Kontext), alle Manifeste, Messungen,
Routing-Tabellen, Prüfnachweise. Dieser Branch hat keine gemeinsame Historie mit dem Code; die Werkzeuge liegen auf
`titan-x1/platform` (Kopie unter `code/`).

| Was | Wo |
|---|---|
| Implantate (einsetzfertig, Layout `accel/moe.py`) | [`implants/`](implants/) — Abschnitt 11 unten |
| Lieferdokument | diese Datei ab „Donor-Transplantation für den 959,5-M-Seed — Lieferung v2" |
| Mistral: Auswahl, Layer-Profil, Messungen | [`mistral/`](mistral/), [`runs/mistral/`](runs/mistral/) |
| Qwen3-Coder: Auswahl, Messungen | [`qwen3coder/`](qwen3coder/), [`runs/qwen3coder/`](runs/qwen3coder/) |
| Prüfnachweise mit eurem `upcycle()`/`forward()` | [`checks/`](checks/) |

## Herunterladen

GitHub nimmt keine Dateien über 100 MB an. Alle Dateien über 95 MiB liegen deshalb in Teilen (`*.part01`, `*.part02`, …);
`join_parts.py` setzt sie zusammen und prüft danach jede Datei gegen `MANIFEST.sha256`.

```bash
git clone --depth 1 --single-branch -b delivery/donor-paket-20261111 \
    https://github.com/clarityosbaerbelwesterop-gif/Ezhdhhz.git donor-paket
cd donor-paket
python join_parts.py implants          # nur die Implantate (≈ 1,1 GB), dann prüfen
python join_parts.py                   # alles, dann jede Datei prüfen; --delete-parts entfernt danach die Teile
```

Ohne Python: `cat implants/implant_qwen3coder_fe640_code.safetensors.part0* > implants/implant_qwen3coder_fe640_code.safetensors`
und `sha256sum -c MANIFEST.sha256` (Zeilen der noch nicht zusammengesetzten Dateien melden dann „FAILED open").
Einzelne Teile lassen sich auch im Browser über „Download raw file" laden.

## Transplantation durchführen

Ein Befehl vom dichten Champion bis zum fertigen Checkpoint mit Implantaten. Die Werkzeuge liegen auf Branch
`titan-x1/platform` dieses Repos; euer eigener Code (`accel/` mit `scripts/rebase.py`) wird nur aufgerufen, nicht verändert.

```bash
git clone --depth 1 -b titan-x1/platform https://github.com/clarityosbaerbelwesterop-gif/Ezhdhhz.git titan
python join_parts.py implants
python titan/tools/donor/transplant_seed.py --accel /workspace/run2-prep/moe \
    --dense <checkpoint step-007500> \
    --implant implants/implant_qwen3coder_fe640_code.safetensors implants/implant_mistral_l4_fe640_*.safetensors \
    --layers 16,17,18,19,20,21,22,23 --check-ids <echte Token-IDs als .npy> --out <arbeitsordner>/transplant-01
```

Was der Befehl tut:

1. lädt den Champion mit eurem `forge_io.load_checkpoint` und prüft dessen `model_sha256`;
2. macht ihn mit eurem `moe.upcycle` zum MoE (fe 640, 2 Shared, Top-6 — damit exakt dieselbe Funktion) und schreibt `upcycled/`;
3. setzt alle Implantate ein (`insert_implant.py`) und schreibt `implanted/` — `state.json` mit eurem `forge_io.dumps`;
4. vergleicht euren `forward()` von Champion, MoE und Modell mit Implantaten auf den Token-IDs und bricht ab, wenn sie
   sich unterscheiden; alles steht in `transplant_report.json`.

Beim echten Champion ergibt das: 24 MoE-Layer mit je 13 statt 6 Routed-Experten (+7 Slots je Layer, auf den 8 Ziel-Layern
mit Implantaten, sonst Füll-Slots), ≈ 1,58 Mrd. Parameter (6,3 GB float32), aktive Parameter je Token praktisch
unverändert. Speicher: etwa 8 GB für Schritt 1–3, die Vorwärtsprüfung hält jeweils ein Modell (bis ≈ 13 GB); bei
knappem Speicher `--no-forward-check` und die Prüfung auf einer größeren Maschine nachholen.

**Generalprobe** mit eurem Code bei echter Breite (dim 1920, FFN 5120, GQA 30/10, Vokabular 8192, 2 Layer, echte
Seed-Tensoren aus dem Subset, alle vier fe-640-Implantate, echter Text als Token-IDs):
Champion → MoE größte Logit-Abweichung 9,5e-6, MoE → mit Implantaten **0,0**, Verlust bei allen drei gleich,
neue Slots ohne Last, euer `moe.load()` lädt das Ergebnis — siehe [`checks/result_pipeline_real_width.json`](checks/result_pipeline_real_width.json).
Am echten Champion ist der Befehl noch nicht gelaufen: dessen 218 Tensoren liegen nur bei euch.

Danach: euer Training. Eingesetzt rechnet das Modell exakt wie vorher; die neuen Slots sind abgeschaltet (Bias −4) und
werden über euren Router-Bias-Abgleich und das Training eingebunden (Abschnitt 11.3 unten, Gates in Abschnitt 8).

`PAKET.md` beschreibt die ZIP-Variante desselben Pakets; hier liegen die Dateien bereits entpackt.

---

# Donor-Transplantation für den 959,5-M-Seed — Lieferung v2

Stand: 10.11.2026. Antwort auf „Auftrag an Fable 5.2 — v2“ (`FABLE_PROMPT_DE_v2_SPLIT.md`).
Werkzeuge: `titan.donor` und `tools/donor/` im Repo (Branch `titan-x1/platform`), Tests in `tests/test_donor*.py`.

Lesehinweis: Jede Zahl in diesem Dokument ist **gemessen** (mit Datei, aus der sie stammt), **abgeleitet**
(Arithmetik aus Metadaten) oder ausdrücklich als **nicht ausgeführt** markiert. Es gibt keine Aussage über die
Qualität des Seeds oder eines gewachsenen Modells — dafür fehlen 210 von 218 Seed-Tensoren.
Aus den Messdateien generierte Tabellen (Abschnitt 5.4–5.9) verwenden den Dezimalpunkt, der übrige Text das Komma.

---

## 0. Kurzfassung

| # | Befund | Art |
|---|--------|-----|
| 1 | Paket bitgenau verifiziert (ZIP-SHA256 `4722cd77…57a4`), Subset-Datei und alle 8 Tensor-Hashes stimmen mit der Provenienz überein, Inventar 218 Tensoren = 959 541 120 Parameter = Konfigurationsformel. | gemessen |
| 2 | Alle drei Donors live auf dem Hub geprüft (Repo, Revision, Lizenz, Dateien, LFS-SHA256), vollständige Tensor-Inventare aus den Safetensors-Headern. | gemessen |
| 3 | Qwen3-Coder komplett gestreamt (59,8 GB, 48 Layer) und gegen `transformers` validiert: nativ **0,2279 bit/Byte** auf 201 KB Code + Prosa. Über Seed-Tokens mit transplantierten Zeilen **1,0234 bit/Byte**. | gemessen |
| 4 | „Hohe Layer = die guten Experten“ ist **nicht belegt**: Code/Prosa-Routing unterscheidet sich in allen 48 Layern etwa gleich stark (TV 0,20–0,38 bei Rauschboden 0,06–0,12). | gemessen |
| 5 | Der Qwen-Ausgangskopf kann Layer-Ausgänge unterhalb von ~L39 nicht lesen (Logit-Lens ≈ Zufall bis L38). Das ist konsistent mit dem Fehlschlag des alten Stitch an L28–L29; als Erfolg wird hier nichts davon gewertet. | gemessen |
| 6 | Datenfreie Anker-Brücken aus den Token-Tabellen sind **schwache Initialisierungen** (Holdout-R² ≤ 0,10 bzw. negativ). Eine nur auf der Brücke trainierte Ausgangsbrücke erreicht durch den echten Seed-Ausgangspfad 0,6368 bit/Byte (Referenz mit transplantiertem Kopf 0,7253). | gemessen |
| 7 | Slot-Breite: Qwen-Experten (768 Einheiten) passen exakt in einen 768er-Slot; im Team-Standard 640 kostet die Einheitenauswahl 1,9–6,4 % der Ausgangsenergie (Median 3,5 %), gemessen auf echten Experten-Eingängen. | gemessen |
| 8 | Mistral Large 4 komplett gestreamt (1,047 TB, 72 Layer), eigener Vorwärtspfad gegen `transformers` validiert. Experten gewählt für Reasoning (311 Kandidaten, Kern L41–L63), Effort (462, wirksam ab L37, Top-Experten L63–L71), Denkblock (69, L48–L70) und Lernen im Kontext (64 mit Zuwachs, L20–L58). 40 Experten als COPY-Manifest und Projektion. | gemessen |
| 8a | Mistral-Experten (6144 Einheiten) passen **nicht** verlustarm in 640er-Slots: Median 46 % Ausgangsfehler; die Ausgangsbrücke Mistral → Seed-Vokabular ist mit den vorhandenen Daten **nicht** belegt (Heal schlechter als Referenz). | gemessen |
| 9 | Kein Tensor des Seeds wurde verändert. Nichts wurde auf Modellausgaben trainiert. Kein 2–4-bit, kein QLoRA, kein Destillat. | Regel |
| 10 | **Einsetzfertige Implantate** je Donor im Tensor-Layout eures `accel/moe.py` plus Einsetz-Werkzeug (Abschnitt 11). Mit eurem eigenen `upcycle()`/`forward()` geprüft: nach dem Einsetzen identische Logits (0,0 bzw. ≤ 8e-6), neue Slots abgeschaltet und trainierbar. Am echten Checkpoint nicht gelaufen. | gemessen |

Was **nicht** geliefert wird, weil es ohne das volle Seed-Modell nicht existieren kann: ein gewachsenes
Modell, eine Baseline, eine kalibrierte Brücke an einem inneren Seed-Layer, ein Nachweis, dass ein
eingesetzter Experte dem Seed nützt.

---

## 1. Drei Klassen von Operationen

| Klasse | Bedeutung | Nachweis |
|--------|-----------|----------|
| **COPY** (Direktkopie) | Nutzlast-Bytes des Donors, unverändert | SHA-256 = Hub-Bytebereich (Shard + Offset im Manifest) |
| **PROJECTION** (Projektion/Bridge) | deterministische Arithmetik: Zeilen-Transplantat, Brücken in Gewichte gefaltet, Null-Auffüllung, Einheitenauswahl | Formel + SHA-256 von Ein- und Ausgabe; Test „Faltung = Komposition“ |
| **HEAL** (Training) | alles, was Gradienten sieht | Trainingslog, Holdout-Metrik, Liste der trainierbaren Tensoren |

In dieser Lieferung gibt es genau **eine** HEAL-Messung (Ausgangsbrücke Qwen → Seed-Vokabular, 3,93 M
Brückenparameter, Seed-Tensoren nur gelesen). Sie ist ein Machbarkeitsmaß für die Brücke, kein Modell.

---

## 2. Paket und Seed

- ZIP aus drei Teilen bytegenau zusammengesetzt, SHA-256 `4722cd771b65d3d098a7d5b8f7f09ca2a46f5a6b082ba23d3f210ca28a3357a4` = Vorgabe.
- `weights_subset/champion_step007500_WEIGHT_SUBSET.safetensors`: Datei-SHA `da5626cd…a388` = Provenienz; 8 Tensoren, alle Hashes = Provenienz (`seed/seed_verification.json`).
- Vorhanden: `tok_emb.weight` [8192, 1920], `norm.weight`, `blocks.{0,11,23}.{attn_norm,ffn_norm}.weight`. **Fehlend: 210 von 218 Tensoren** (alle Attention- und FFN-Gewichte).
- Inventar (nur Formen): 218 Tensoren, 959 541 120 Parameter; identisch zur Formel aus der Konfiguration (24 Layer, d = 1920, GQA 30/10, FFN 5120 SwiGLU, Vokabular 8192, gebundene Ein-/Ausgabe).
- `tok_emb`-Statistik: Std 0,0397; Zeilennorm Median 1,74; effektiver Rang 583 (exp. Entropie), 640 Dimensionen tragen 90 % der Varianz, 1456 tragen 99 %.
- Tokenizer: Byte-BPE (256 Bytes + 7933 Merges + `<bos>` 8189, `<eos>` 8190, `<pad>` 8191), Datei-SHA `fe1ee90a…8012`. Eigener Leser (`titan.donor.vocab.SeedBPE`) = Paket-`bpe.py` auf allen Prüftexten.
- Ziel-MoE-Format (aus `run2-prep/moe/accel/moe.py`): `blocks.{i}.moe.experts.w1/w3 [E, fe, d]`, `w2 [E, d, fe]`, `moe.router.weight [E, d]`, Puffer `moe.router.bias [E]`; Sigmoid-Scores, Top-k auf Score + Bias, Gates normiert × `route_scale`; **fe = 640** als Standard. Ziel ≈ 25,7 B gesamt / ≈ 5,9 B aktiv: **nicht gebaut**.

Der Seed bleibt unverändert: alle hier erzeugten Tensoren sind **zusätzliche** Slots bzw. Brücken.

---

## 3. Verfahren in einem Absatz

Donor-Gewichte werden per HTTP-Range aus dem Hub gelesen (Revision festgenagelt, SHA-256 je Tensor), der
Donor läuft Layer für Layer auf Text mit bekannter Herkunft, pro Token wird die Routing-Tabelle gespeichert.
Experten werden über **vorab festgelegte Kontraste** zwischen Token-Gruppen gewählt und müssen in zwei
disjunkten Hälften der Daten bestehen (Split-Half). Ein gewählter Experte wird (a) als Byte-Kopie mit Hash
dokumentiert und (b) mit Brücken `P_in` (Seed-Slot-Eingang → Donor-Experteneingang) und `P_out`
(Donor-Update → Seed-Update) in das Slot-Format gefaltet:
`w1' = gate·P_inᵀ`, `w3' = up·P_inᵀ`, `w2' = P_outᵀ·down`. Eingesetzt wird mit Auswahl-Bias ≤ −1: der Slot
ist am Schritt 0 nie unter den Top-k: jedes Token behält seine Experten, die alten Slot-Tensoren bleiben
bitgleich (getestet; Ausgaben gleich bis auf Gleitkomma-Rundung der Matrixprodukte, auf CI ≤ 2e-6 relativ). Ob er danach nützt,
entscheidet Training (HEAL) gegen die Gates in Abschnitt 8.

---

## 4. Donor Code: Qwen3-Coder-30B-A3B-Instruct

### 4.1 Provenienz

| | |
|---|---|
| Repo | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Revision | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` (lastModified 2025-12-03) |
| Lizenz | Apache-2.0 (LICENSE-SHA256 `832dd9e0…`) |
| Gewichte | 16 Shards, 61 066 575 656 B; 18 867 Tensoren, 30 532 122 624 Parameter, alle BF16 |
| Architektur | 48 Layer, d = 2048, 128 Experten Top-8, Experten-Breite 768, 32/4 Köpfe, RoPE θ = 1e7, Softmax-Router, Kopf nicht gebunden |

Dateien: `qwen3coder/provenance.json` (alle Dateien mit LFS-SHA256), `qwen3coder/tensor_inventory.tsv`
(jeder Tensor: Name, dtype, Form, Shard, Bytebereich), `qwen3coder/inventory_summary.json`.
Gegenprobe: 48/48 Tensoren der Team-Phase C sind bitgleich zu den Team-Hashes.

### 4.2 Vokabularvertrag (`qwen3coder/vocab_contract.json`)

Jedes der 8189 regulären Seed-Tokens wird als Bytefolge durch den Donor-Tokenizer geschickt:
**6757 exakt** (ein Donor-Token), **1432 zusammengesetzt** (2–14 Stücke), **0 unbekannt**, 3 Spezial.
Identisch zur Team-Datei `vocab_map.json`. Zeilen-Transplantat: exakt → Kopie der Zeile; zusammengesetzt →
Mittel der Stück-Zeilen (PROJECTION).

### 4.3 Donor-Messungen (voller Forward, 3283 s, `qwen3coder/measurements/final.json`)

Kalibriertext: 112 Abschnitte, je zur Hälfte menschengeschriebener Code (CPython-Standardbibliothek) und Prosa
(deren Dokumentation), 201 KB; keine Modellausgaben.

| Regime | Verlust | bit/Byte |
|--------|---------|----------|
| A nativ (Donor-Tokens, Donor-Kopf) | 0,6856 nats/Token | **0,2279** (Code 0,2166 · Prosa 0,2377) |
| B Seed-Tokens → transplantierte Embeddings → transplantierter Kopf | 2,674 nats/Seed-Token | **1,0234** (Code 1,1869 · Prosa 0,8807) |
| C Donor-Eingang nativ, Ausgabe im Seed-Vokabular (42 261 bytegenau ausgerichtete Ziele) | 1,8917 | 0,7109 |
| B auf denselben Zielen | 1,6617 | 0,6245 |

Lesart: der größere Teil des Verlusts von A nach B entsteht **auf der Ausgangsseite** (C ≈ B).
Der Forward ist gegen die `transformers`-Referenz getestet (`tests/test_donor_reference.py`).

Logit-Lens (nativer Kopf auf Zwischenschichten, Zufall = ln 151 936 = 11,93 nats): L0–L38 ≈ 10–12,5;
L39 10,09; L40 9,59; L41 8,38; L42 6,51; L43 5,19; L44 3,74; L45 2,97; L46 1,45; L47 0,68.
Ein Donor-Kopf direkt hinter L29 arbeitet also auf Zufallsniveau — passend zur Vor-Heal-CE 9,19–9,61 des
alten Stitch. Das alte Stitch bleibt ein Rollback.

### 4.4 Layer- und Expertenwahl

Kontrast: Gate-Masse eines Experten auf Code-Tokens gegen Prosa-Tokens, je Layer; stabil nur, wenn der
Experte in beiden Datenhälften unter den Top-8 des Layers nach Überschussmasse liegt.

- 315 stabile Kandidaten über alle Layer (`qwen3coder/measurements/selection.json`).
- Code/Prosa-Abstand der Routing-Verteilungen: TV 0,20–0,38 in **allen** Tiefen, Split-Half-Rauschen
  0,06–0,12, Rangkorrelation zwischen den Hälften 0,79–0,96. Es gibt keinen Tiefenbereich, in dem die
  Spezialisierung sprunghaft zunimmt. Die Hypothese „hohe Layer = die guten Experten“ ist damit weder
  bestätigt noch widerlegt, sondern **durch Routing nicht gestützt**.
- Die 16 Team-Picks aus Phase C: Median-Rang 5 von 128, 11 in den Top-8 ihres Layers, 2 in der unteren
  Hälfte (L29 e110 Rang 86, L41 e110 Rang 102).
- **Warum trotzdem L41–L47 als Standard-Satz:** nicht wegen „besserer“ Experten, sondern weil nur dort der
  Donor-Ausgangsraum lesbar ist (Logit-Lens) — und die datenfreie Ausgangsbrücke ist am Donor-**Kopf**
  verankert. Für tiefere Layer gibt es ohne gepaarte Aktivierungen keine begründbare Brücke.
- Standard-Satz (stabil, Verhältnis ≥ 2, Top 16 nach Überschussmasse in L41–L47):
  L41 e25 · L46 e86 · L45 e44 · L41 e91 · L45 e100 · L42 e85 · L42 e29 · L47 e60 · L41 e43 · L46 e40 ·
  L46 e85 · L44 e32 · L41 e41 · L43 e84 · L44 e41 · L47 e25.

### 4.5 Brücken (`qwen3coder/measurements/bridge_tests.json`, `qwen3coder/measurements/output_bridge.json`)

Datenfreie Anker über das Seed-Vokabular (Ridge 1e-2, ohne Bias):

| Brücke | R² Fit | R² Holdout-Tokens |
|--------|--------|-------------------|
| Q: Seed-Vor-Kopf → Donor-Vor-Kopf (1920 → 2048) | 0,533 | 0,096 |
| P: Donor-Update → Seed-Update (2048 → 1920) | 0,395 | −0,074 |

Ausgangsbrücke durch den **echten Seed-Ausgangspfad** (finale RMSNorm-Gains + gebundenes Embedding),
ausgewertet auf der zurückgehaltenen Hälfte (21 215 Positionen; Gleichverteilung 9,011 nats):

| Variante | Klasse | nats | bit/Byte |
|----------|--------|------|----------|
| Referenz: transplantierter Donor-Kopf | PROJECTION | 1,9171 | 0,7253 |
| Anker-Brücke, datenfrei | PROJECTION | 4,5063 | 1,7050 |
| Anker ohne Seed-Norm | PROJECTION | 2,6434 | 1,0001 |
| nur die Brücke trainiert (400 Schritte, 3,93 M Parameter) | **HEAL** | **1,6830** | **0,6368** |

Die geheilte Brücke überanpasst die Fit-Hälfte (0,0074 nats) — die Zahl oben ist die Holdout-Hälfte.
Konsequenz: Der Seed-Ausgangsraum **kann** Donor-Information tragen, aber nur nach Training der Brücke;
die datenfreie Projektion allein ist eine Startlage, kein Ergebnis.

### 4.6 Artefakte (`qwen3coder/implant/`)

| Datei | Klasse | Inhalt | SHA-256 |
|-------|--------|--------|---------|
| `donor_experts_COPY.bf16.safetensors` (150 999 760 B) | COPY | 16 Experten × gate/up/down, BF16-Bytes des Hubs | `3ac98b765271453d7bbfc94cdac2cf775b79191cc9a54fbdd09c52513d335786` |
| `seed_slots_PROJECTION.f32.safetensors` (283 244 232 B) | PROJECTION | Slots fe = 768 (Donor-Breite, exakt gefaltet) + Router-Zeilen | `c74f96c7505972def586c232e6cf7c42167cba089dc072012424e9c8a0e79f31` |
| `seed_slots_PROJECTION_fe640.f32.safetensors` (236 058 304 B) | PROJECTION (verlustbehaftet) | Slots fe = 640 (Team-Standard), Einheitenauswahl nach gemessener Wichtigkeit | `457f412244389f24ffaf59a3580da8aeb52fa64ec4327806cdd58ffbba9cdf2f` |
| `implant_manifest.json` | — | je Tensor: Quelle, Shard, Bytebereich, Form, SHA-256; Brücken-Metadaten; Auswahlfehler je Experte und Breite | |
| `expert_activation_report.json` | Messung | Zahl der Experten-Eingänge je Experte, Restart-Abweichung | |

Die 768er-Datei wurde zweimal unabhängig erzeugt und ist bitgleich (deterministisch).

Brücken im Slot (Formeln in `titan/donor/adapt.py::slot_bridges`):
`P_in = diag(g_seed/g_slot)·Q·diag(w_post(L)/w_final)`,
`P_out = diag(w_final)·P·diag(1/g_seed)·(rms_seed/rms_donor(L))`.
**`g_slot` (ffn_norm-Gain des empfangenden Seed-Layers) und `rms_seed` fehlen** und stehen auf 1,0 — beides
sind reine Skalierungen, die mit dem vollen Seed in Sekunden nachgetragen werden.

Einheitenauswahl 768 → fe, relativer quadratischer Ausgangsfehler auf echten Experten-Eingängen
(2048 Zeilen je Experte, 16 Experten; min / Median / max):

| fe | Fehler (Donor-Raum) |
|----|---------------------|
| 768 | 0 (exakt) |
| 640 | 0,019 / **0,035** / 0,064 |
| 512 | 0,053 / 0,097 / 0,166 |
| 384 | 0,097 / 0,192 / 0,292 |

(Die Experten-Eingänge stammen aus einem Neustart ab dem gespeicherten L35-Residuum; Abweichung zum
Originallauf 0,2 % an L41, 0,6 % an L47.)

### 4.7 Tests

Ausgeführt und bestanden: Forward = `transformers`-Referenz; Hash jeder Kopie = Hub; Faltung = Komposition
(`slot(x) == P_out(expert(x·P_in))`); Null-Auffüllung fügt exakte Nullen an; eingesetzte Slots ändern weder
das Routing noch die alten Slots (Ausgaben gleich bis auf Rundung); Auswahlfehler-Formel; Anker finden eine
gepflanzte lineare Beziehung wieder.

**Nicht ausgeführt** (Grund: volles Seed-Modell fehlt):
Identität des gewachsenen Modells auf echten Seed-Logits · Brücken-R² an einem inneren Seed-Layer mit
gepaarten Aktivierungen · Nutzen eines eingesetzten Experten nach Heal · jede Baseline.

---

## 5. Donor Reasoning / adaptives Denken / Lernen: Mistral Large 4

### 5.1 Provenienz

| | |
|---|---|
| Repo | `mistralai/Mistral-Large-4-1T-A52B` |
| Revision | `af26e7869af601441d5cdf75a1272469ddc3d059` |
| Lizenz | „Mistral Ecosystem Open-Weight License“ (Model-Card `other`; modifizierte MIT; LICENSE-SHA256 `d242ebed…`). Interne Nutzung frei; **wer das Modell oder ein Derivat Dritten als Model-as-a-Service anbietet, braucht eine separate kommerzielle Lizenz.** Ein Seed mit Mistral-Experten ist ein Derivat. |
| Gewichte | 54 Shards, 1 060 554 587 912 B; 56 563 Tensoren: 27 668 FP8-E4M3 + 27 668 F32-Blockskalen [48, 48] + 1 227 BF16 |
| Präzision | FP8 mit einer F32-Skala je 128×128-Block (`W = fp8 · qscale`). Das ist 8 bit — **kein** 2–4-bit; die NVFP4-Schwester wurde nicht angefasst. |
| Architektur | 72 Layer (L0 dicht, 71 MoE), d = 6144, 128 Experten Top-4 + 1 Shared-Expert, Experten-Breite 6144, latente Attention (MLA), LeRoPE, jeder 4. Layer ohne Positionssignal, Schlüssel-Sharing (L39, 43, …, 71 nutzen die latenten Schlüssel von L35), Vokabular 204 800 |

Dateien: `mistral/provenance.json`, `mistral/tensor_inventory.tsv`, `mistral/inventory_summary.json`.
`params.json`/`config.json` des Hubs sind identisch zu den Paket-Kopien.

### 5.2 Vokabularvertrag (`mistral/vocab_contract.json`)

**6580 exakt**, **1609 zusammengesetzt** (2–7 Stücke), **0 unbekannt**, 3 Spezial (`<bos>`→1, `<eos>`→2, `<pad>`→11).
Tekken-Leser gegen den Repo-Tokenizer geprüft: 0 Abweichungen auf 8040 Tokens.

### 5.3 Vorwärtspfad und Sondensatz

- Eigene Implementierung des Textstapels (`titan/donor/mistral_l4.py`), **gleich der `transformers`-Referenz**
  auf Zufallsgewichten inklusive NoPE-Layern, Schlüssel-Sharing, LeRoPE/YaRN, dichtem L0 und Shared-Expert;
  FP8-Dekodierung = `torch.float8_e4m3fn`.
- Experten werden einzeln gestreamt (113 MB je Experte), angewandt und verworfen; ein Layer sind 14,7 GB.
- Sondensatz (`tools/donor/mistral_probes.py`, 154 Sequenzen, 23 210 Tokens), nur menschengeschriebene bzw.
  programmerzeugte Inhalte, **keine Modellausgaben**:

| Gruppe | Quelle | Tokens |
|--------|--------|--------|
| Prosa | WikiText-2 Test (CC-BY-SA) | 2 699 |
| Code | CPython-Standardbibliothek (PSF) | 4 985 |
| Aufgabe + Lösungsweg als Fließtext | GSM8K Test (MIT), 28 Aufgaben | 4 048 |
| Chat-Format, dieselbe Aufgabe unter `reasoning_effort` low/medium/high, mit und ohne `[THINK]`-Block | GSM8K, 14 Aufgaben × 6 | 9 408 |
| Regel-Lernen im Kontext: 8 Regeln × 2 Instanzen × 10 Beispiele auf Zufallszeichenketten | Programm, Seed 20261110 | 2 070 |

Die GSM8K-Lösungswege sind die menschlichen Referenzlösungen des Datensatzes, keine Modell-Ketten.

### 5.4 Vorwärtslauf (gemessen)

- Alle 72 Layer vollständig gestreamt: **1 046 711 691 264 B** (Summe der Layer-Protokolle; 14,73 GB je MoE-Layer, 128 von 128 Experten je Layer getroffen), 23 056 Zielpositionen. Der Lauf wurde nach drei Unterbrechungen jeweils aus dem gesicherten Zustand fortgesetzt (`state.pt` nach jedem Layer).
- Verlust des Donors auf den Sonden (nats/Token, eigene Tokens): gesamt **2.588**; wiki 1.743 · code 0.703 · gsm_raw 1.233 · chat 4.300 · icl 3.140.
- Nach Zielgruppe: wiki 1.743 · code 0.703 · gsm_question 1.785 · gsm_steps 0.858 · gsm_answer 0.846 · icl_input 3.694 · icl_output 0.541.
- **Lernen im Kontext, gemessen am Ausgang:** Verlust auf den Regel-Ausgaben nach Beispielnummer — 1: 1.32 · 2: 0.54 · 3: 0.42 · 4: 0.29 · 5: 0.58 · 6: 0.98 · 7: 0.25 · 8: 0.20 · 9: 0.56 · 10: 0.19 nats. (Beispiel 1 ist ohne Vorwissen über die Regel; fällt die Kurve, hat das Modell die Regel aus dem Kontext gelernt.)
- `reasoning_effort` high gegen low, Lösungsschritte (dieselben 944 Zielpositionen): 1.340 gegen 1.288 nats.
- `reasoning_effort` high gegen low, Antwort (dieselben 168 Zielpositionen): 5.240 gegen 4.808 nats.
- medium gegen low, Lösungsschritte (dieselben 944 Zielpositionen): 1.326 gegen 1.288 nats.
- medium gegen low, Antwort (dieselben 168 Zielpositionen): 4.852 gegen 4.808 nats.
- mit gegen ohne `[THINK]`-Block, Antwort (dieselben 210 Zielpositionen): 1.671 gegen 3.477 nats.
- Bit/Byte nativ auf den Klartext-Sonden (`mistral/native_loss.json`): Prosa **0.564** · Code **0.252** · GSM8K-Fließtext **0.505** · Regelaufgaben 1.934 (deren Eingaben sind Zufallszeichenketten).
- Chat-Format nach Zielsegment: Steuer-Tokens 12,94 · Nutzerfrage 3,55 · Schritte im Denkblock 1,32 · Antwort 2,35 mit und 7,59 ohne Denkblock (alle Antwort-Tokens einschließlich des ersten; der gepaarte Vergleich oben betrifft nur das gemeinsame Ende). Der hohe Chat-Mittelwert kommt also von Steuer- und Nutzer-Tokens, die ein Chat-Modell nicht vorherzusagen gelernt hat — kein Hinweis auf einen Fehler im Vorwärtspfad.
- Lesart: Diese Verluste sind plausibel für ein großes Sprachmodell und damit die zweite, unabhängige Bestätigung des eigenen Vorwärtspfads auf **echten** Gewichten (die erste ist der Referenztest auf Zufallsgewichten). Sie sind keine Benchmark-Aussage.
- Lesart Effort: Unter `high` werden die kurzen menschlichen Referenzschritte etwas *unwahrscheinlicher* (1,340 gegen 1,288). Gemessen ist damit, dass die Einstellung das Modell kausal verschiebt — nicht, dass es „besser denkt“.

### 5.5 Layer-Profil (`mistral/selection.json`, Feld `layer_profile`)

TV = halbe L1-Distanz der Experten-Masseverteilungen zweier Gruppen; „Rauschen“ = dieselbe Größe zwischen den zwei Datenhälften der Zielgruppe. „Residual hi/lo“ = relativer Abstand des Residualstroms derselben Tokens unter `reasoning_effort` high und low.

| Layer | Reasoning TV / Rauschen | Lernen TV / Rauschen | ICL spät–früh TV / Rauschen | Residual hi/lo | Residual med/lo | Residual Think/ohne | Lens Antwort | Lens ICL spät |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.164 / 0.118 | 0.502 / 0.191 | 0.183 / 0.236 | 0.001 | 0.002 | 0.015 | 12.26 | 13.46 |
| 3 | 0.187 / 0.105 | 0.391 / 0.200 | 0.257 / 0.252 | 0.010 | 0.003 | 0.046 | 11.95 | 13.63 |
| 7 | 0.177 / 0.111 | 0.347 / 0.219 | 0.226 / 0.299 | 0.010 | 0.007 | 0.077 | 12.05 | 14.08 |
| 11 | 0.192 / 0.101 | 0.318 / 0.190 | 0.306 / 0.249 | 0.011 | 0.008 | 0.083 | 12.00 | 14.24 |
| 15 | 0.173 / 0.119 | 0.367 / 0.174 | 0.326 / 0.289 | 0.016 | 0.014 | 0.135 | 13.04 | 14.12 |
| 19 | 0.257 / 0.175 | 0.369 / 0.222 | 0.332 / 0.280 | 0.024 | 0.020 | 0.188 | 13.23 | 13.85 |
| 23 | 0.206 / 0.145 | 0.332 / 0.221 | 0.358 / 0.252 | 0.034 | 0.031 | 0.279 | 13.13 | 13.63 |
| 27 | 0.309 / 0.157 | 0.288 / 0.136 | 0.378 / 0.160 | 0.043 | 0.040 | 0.352 | 11.87 | 13.29 |
| 31 | 0.326 / 0.118 | 0.405 / 0.139 | 0.344 / 0.138 | 0.045 | 0.045 | 0.334 | 11.17 | 12.61 |
| 35 | 0.303 / 0.139 | 0.398 / 0.113 | 0.189 / 0.157 | 0.047 | 0.047 | 0.353 | 10.84 | 13.15 |
| 39 | 0.288 / 0.126 | 0.347 / 0.138 | 0.281 / 0.209 | 0.102 | 0.083 | 0.449 | 9.77 | 14.09 |
| 43 | 0.276 / 0.133 | 0.387 / 0.089 | 0.201 / 0.116 | 0.130 | 0.092 | 0.508 | 9.34 | 13.67 |
| 47 | 0.347 / 0.090 | 0.347 / 0.118 | 0.237 / 0.149 | 0.181 | 0.120 | 0.594 | 9.24 | 12.79 |
| 51 | 0.390 / 0.142 | 0.365 / 0.129 | 0.217 / 0.184 | 0.229 | 0.152 | 0.656 | 9.12 | 12.85 |
| 55 | 0.365 / 0.146 | 0.326 / 0.099 | 0.224 / 0.136 | 0.256 | 0.174 | 0.693 | 8.66 | 12.71 |
| 59 | 0.342 / 0.128 | 0.380 / 0.106 | 0.264 / 0.158 | 0.289 | 0.201 | 0.712 | 7.38 | 13.06 |
| 63 | 0.402 / 0.121 | 0.412 / 0.132 | 0.285 / 0.170 | 0.306 | 0.229 | 0.772 | 7.73 | 13.10 |
| 67 | 0.304 / 0.136 | 0.408 / 0.151 | 0.242 / 0.240 | 0.334 | 0.269 | 0.837 | 6.57 | 6.46 |
| 71 | 0.350 / 0.165 | 0.614 / 0.135 | 0.149 / 0.207 | 0.477 | 0.311 | 0.937 | 3.93 | 0.43 |

- Reasoning-Kontrast über dem Rauschboden in 71 von 71 Layern; größter Abstand zum Rauschen in L42, L45, L50, L57, L58, L60, L62, L63.
- Lern-Kontrast (Regel-Ausgabe gegen Vergleichsgruppen) über dem Rauschboden in 71 von 71 Layern; spät gegen früh über dem Rauschboden in 60 von 71; größter Abstand dort in L21, L22, L24, L27, L31, L32, L44, L46.
- Effort-Wirkung auf den Residualstrom: von 0.0009 (L1) auf 0.4770 (L71); Anteil der Tokens mit anderem Top-1-Experten unter high statt low: 0.001–0.297.

### 5.6 Expertenwahl (`mistral/selection.json`, Feld `selected`)

Kriterium: Masse in der Zielgruppe ≥ 2 × Masse in **jeder** Vergleichsgruppe, in beiden Datenhälften getrennt, mindestens 12 geroutete Ziel-Tokens je Hälfte; Rang nach Überschussmasse. Gleichverteilung = 0.0078. Effort: gepaarter Vergleich, |t| >= 4 over item groups, same sign in both halves, |delta| >= 0.25/E.

**Reasoning** — Lösungsschritte (3168 Tokens) gegen Aufgabentext (3174), Prosa (2685), Code (4973). Kandidaten gesamt: 311 in 61 Layern.

| Rang | Layer | Experte | Masse Ziel | größte Vergleichsmasse | Verhältnis Hälfte 1 / 2 | Ziel-Tokens geroutet |
|---|---|---|---|---|---|---|
| 1 | 50 | 72 | 0.1624 | 0.0280 | 6.5 / 5.1 | 1626 von 3168 |
| 2 | 52 | 91 | 0.1241 | 0.0072 | 15.4 / 18.7 | 1260 von 3168 |
| 3 | 37 | 17 | 0.1800 | 0.0637 | 2.8 / 2.8 | 1427 von 3168 |
| 4 | 40 | 74 | 0.1429 | 0.0313 | 4.3 / 4.9 | 1242 von 3168 |
| 5 | 46 | 69 | 0.1281 | 0.0192 | 5.7 / 7.9 | 1324 von 3168 |
| 6 | 60 | 8 | 0.1021 | 0.0107 | 8.8 / 10.3 | 1112 von 3168 |
| 7 | 27 | 116 | 0.1068 | 0.0199 | 4.7 / 6.4 | 968 von 3168 |
| 8 | 58 | 18 | 0.1100 | 0.0314 | 3.8 / 3.2 | 1097 von 3168 |

**Adaptives Denken (a): `[THINK]`-Modus** — Lösungsschritte im Denkblock (944) gegen dieselbe Textart ohne Block (2224) und die übrigen Vergleichsgruppen. Kandidaten gesamt: 69 in 35 Layern.

| Rang | Layer | Experte | Masse Ziel | größte Vergleichsmasse | Verhältnis Hälfte 1 / 2 | Ziel-Tokens geroutet |
|---|---|---|---|---|---|---|
| 1 | 52 | 21 | 0.0916 | 0.0220 | 4.9 / 3.7 | 216 von 944 |
| 2 | 56 | 32 | 0.0807 | 0.0192 | 3.9 / 4.6 | 280 von 944 |
| 3 | 63 | 118 | 0.0690 | 0.0128 | 8.0 / 3.8 | 275 von 944 |
| 4 | 70 | 8 | 0.0800 | 0.0240 | 3.5 / 3.2 | 313 von 944 |
| 5 | 68 | 83 | 0.0585 | 0.0056 | 10.5 / 8.0 | 196 von 944 |
| 6 | 48 | 14 | 0.0679 | 0.0189 | 5.3 / 2.7 | 201 von 944 |
| 7 | 56 | 112 | 0.0730 | 0.0308 | 2.4 / 2.4 | 316 von 944 |
| 8 | 54 | 65 | 0.0534 | 0.0112 | 5.8 / 4.1 | 229 von 944 |

**Adaptives Denken (b): `reasoning_effort`** — dieselben 2828 Tokens unter high und low. Kandidaten gesamt: 462 in 67 Layern.

| Rang | Layer | Experte | Masse high | Masse low | Δ (Mittel über Aufgaben) | t | Vorzeichen gleich |
|---|---|---|---|---|---|---|---|
| 1 | 66 | 79 | 0.0281 | 0.0042 | +0.0294 | +11.3 | 100 % von 28 |
| 2 | 69 | 14 | 0.0443 | 0.0689 | -0.0293 | -12.4 | 100 % von 28 |
| 3 | 63 | 85 | 0.0246 | 0.0020 | +0.0275 | +11.9 | 100 % von 28 |
| 4 | 70 | 80 | 0.0210 | 0.0000 | +0.0260 | +11.6 | 100 % von 28 |
| 5 | 63 | 11 | 0.0040 | 0.0249 | -0.0254 | -11.9 | 100 % von 28 |
| 6 | 65 | 65 | 0.0304 | 0.0093 | +0.0248 | +11.2 | 100 % von 28 |
| 7 | 69 | 70 | 0.0198 | 0.0001 | +0.0244 | +11.9 | 100 % von 28 |
| 8 | 71 | 65 | 0.0194 | 0.0000 | +0.0241 | +11.4 | 100 % von 28 |

**Lernen im Kontext** — Positionen, die eine Regel-Ausgabe erzeugen (361) gegen Eingabe-Positionen (1693), Aufgabentext, Prosa, Code. Kandidaten gesamt: 158 in 68 Layern.

| Rang | Layer | Experte | Masse Ziel | größte Vergleichsmasse | Verhältnis Hälfte 1 / 2 | Ziel-Tokens geroutet | Masse spät / früh |
|---|---|---|---|---|---|---|---|
| 1 | 41 | 102 | 0.4421 | 0.1585 | 2.9 / 2.7 | 341 von 361 | 0.400 / 0.516 |
| 2 | 71 | 57 | 0.2836 | 0.0513 | 5.6 / 5.5 | 270 von 361 | 0.282 / 0.304 |
| 3 | 71 | 64 | 0.1997 | 0.0088 | 23.9 / 21.4 | 265 von 361 | 0.208 / 0.196 |
| 4 | 66 | 94 | 0.2922 | 0.1026 | 2.8 / 2.9 | 324 von 361 | 0.288 / 0.303 |
| 5 | 59 | 66 | 0.3336 | 0.1442 | 2.3 / 2.3 | 277 von 361 | 0.320 / 0.387 |
| 6 | 65 | 63 | 0.2909 | 0.1109 | 2.6 / 2.6 | 313 von 361 | 0.287 / 0.315 |
| 7 | 35 | 105 | 0.2827 | 0.1138 | 2.4 / 2.5 | 287 von 361 | 0.273 / 0.246 |
| 8 | 70 | 11 | 0.1940 | 0.0345 | 4.9 / 6.5 | 262 von 361 | 0.205 / 0.176 |

**Lernen im Kontext, nur Experten mit Zuwachs** — Kandidaten von oben, deren Masse bei späten Beispielen (6–10) in beiden Datenhälften größer ist als bei frühen (1–2); Rang nach Zuwachs. Kandidaten gesamt: 64 in 46 Layern.

| Rang | Layer | Experte | Masse Ziel | größte Vergleichsmasse | Verhältnis Hälfte 1 / 2 | Ziel-Tokens geroutet | Masse spät / früh |
|---|---|---|---|---|---|---|---|
| 1 | 22 | 71 | 0.2006 | 0.0868 | 2.4 / 2.2 | 272 von 361 | 0.233 / 0.120 |
| 2 | 24 | 49 | 0.1043 | 0.0407 | 2.6 / 2.4 | 142 von 361 | 0.138 / 0.035 |
| 3 | 33 | 93 | 0.1289 | 0.0593 | 2.1 / 2.3 | 175 von 361 | 0.158 / 0.071 |
| 4 | 46 | 52 | 0.2195 | 0.0744 | 3.0 / 2.9 | 253 von 361 | 0.236 / 0.154 |
| 5 | 53 | 19 | 0.2210 | 0.0861 | 2.4 / 2.8 | 252 von 361 | 0.235 / 0.160 |
| 6 | 33 | 5 | 0.0677 | 0.0266 | 2.7 / 2.3 | 94 von 361 | 0.075 / 0.013 |
| 7 | 58 | 122 | 0.1870 | 0.0741 | 2.5 / 2.6 | 232 von 361 | 0.199 / 0.140 |
| 8 | 27 | 35 | 0.1194 | 0.0514 | 2.4 / 2.3 | 170 von 361 | 0.135 / 0.079 |

Einordnung der Auswahl:

- **Reasoning** ist die klarste Spezialisierung: die Top-Experten erhalten 10–18 % der Gate-Masse auf Lösungsschritten, auf jeder Vergleichsgruppe höchstens 6,4 % (Gleichverteilung 0,8 %). 311 Kandidaten, davon 198 in L37–L63.
- **Effort** schaltet einzelne Experten nahezu ein und aus: L70 e80, L69 e70 und L71 e65 haben unter `low` praktisch keine Masse (≤ 0,0001) und 2 % unter `high`; L69 e14 und L63 e11 verhalten sich umgekehrt. Das Vorzeichen stimmt in allen 28 Aufgaben-Gruppen überein. Alle acht Top-Experten liegen in L63–L71.
- **Lernen:** Das erste Kriterium findet Experten der *Ausgabeposition* (z. B. L41 e102 mit 44 % Masse) — deren Nutzung wächst aber nicht mit der Zahl der Beispiele (0,400 spät gegen 0,516 früh). Die zweite Liste behält nur Experten mit stabilem Zuwachs; sie liegt deutlich tiefer im Stapel (L22–L58). Dieses Zusatzkriterium wurde nach Sichtung der Layer 1–47 ergänzt, bevor L48–L71 vorlagen; die Schwellen der übrigen Kontraste standen vor dem Lauf fest.
- **Grenze der Aussage:** „Reasoning“ heißt hier *ausformulierte Rechenschritte zu Textaufgaben* (GSM8K). Logik, Planung, Code-Reasoning und andere Sprachen wurden nicht sondiert. Routing zeigt, **wo** ein Experte benutzt wird, nicht, was er rechnet; der kausale Nachweis (Experte abschalten, Verlust je Gruppe messen) ist nicht ausgeführt.

### 5.7 Layerwahl

| Funktion | Layer | Messgrundlage |
|---|---|---|
| Reasoning | **L41–L63** | Reasoning-Kontrast minus Rauschen ≥ 0,20 in 19 von 23 Layern dieses Bands (Mittel 0,246); in L1–L26 in keinem (Mittel 0,065, Maximum 0,111), in L27–L40 in 3 von 14, in L64–L71 in 2 von 8. 134 der 311 Kandidaten liegen in L48–L63. |
| Adaptives Denken: Effort | **L37–L71** | Residual-Abstand high/low: 0,001 (L1) · 0,045 (L30) · 0,050 (L36) · 0,100 (L38) · 0,223 (L50) · 0,283 (L60) · 0,477 (L71) — der Anstieg setzt bei L37/L38 ein. Anderer Top-1-Experte: unter 5 % der Tokens bis L35, 12 % bei L42, 24 % bei L58, 30 % bei L70. |
| Adaptives Denken: Denkblock | **L48–L70** | alle acht Top-Experten; 49 der 69 Kandidaten in L48–L71, nur 7 unterhalb von L37. Residual-Abstand der Antwort mit/ohne Block: 0,34 (L30) · 0,67 (L50) · 0,94 (L70). |
| Lernen im Kontext (Zuwachs) | **L20–L33**, schwächer **L44–L63** | Kontrast spät–früh minus Rauschen ≥ 0,10 in 12 von 14 Layern in L20–L33 (Mittel 0,134), in 11 von 20 in L44–L63 (Mittel 0,106); in L1–L19 in keinem (Mittel −0,004), in L64–L71 in keinem (Mittel 0,021). |
| Auslesen in den Token-Raum | **L66–L71** | Logit-Lens auf Regel-Ausgaben: 12,2 (L64) · 9,4 (L66) · 4,3 (L68) · 0,43 (L71); auf Antworten 7,2 (L64) · 3,9 (L71). Davor ist der Residualstrom für den Kopf nicht lesbar — wie bei Qwen. |

„Hohe Layer = die guten Experten“ trifft für Mistral **je nach Funktion unterschiedlich** zu: Effort- und Denkblock-Experten sitzen spät, Reasoning in der oberen Mitte, die Experten mit Lernzuwachs früh bis mittig.

Was ein Layer-Transplantat aus Mistral strukturell bedeutet (abgeleitet aus `config.json` und dem Inventar, nicht gebaut):

1. **Schlüssel-Sharing:** L39, 43, 47, …, 71 verwenden die latenten Schlüssel aus L35. Ein Block, der einen dieser Layer enthält, braucht L35s Schlüsselpfad und das Residuum am Eingang von L35 — er ist nur **ab L35** eigenständig lauffähig. Für das Reasoning-Band wäre das L35–L63 (29 Layer).
2. **Größe:** ein MoE-Layer = 14,73 GB FP8 (128 Experten à 113 MB plus ≈ 0,3 GB Attention, Shared-Expert, Router); L35–L63 = 427 GB. Je Token aktiv: 4 + 1 Experten, 1,353 GFLOP je Token und Layer.
3. **Format:** Mistrals Attention (latente Schlüssel/Werte, 64 Köpfe, LeRoPE, jeder 4. Layer ohne Positionssignal, skalierbarer Softmax) hat kein Gegenstück im Seed-Block (GQA 30/10, RoPE θ = 1e4). Ein Mistral-Layer lässt sich **nicht in einen Seed-Layer projizieren**; er kann nur als Fremdblock mit Ein- und Ausgangsbrücke (1920 ↔ 6144) eingehängt werden. Das ist die Bauart des alten Stitch und ohne HEAL wertlos.

Empfehlung: **Experten zuerst** (Slots, Abschnitt 5.8), Layer-Block erst danach und nur gegen die Gates G2–G6. Ein Layer-Artefakt ist in dieser Lieferung bewusst nicht enthalten; für jeden Layer-Tensor stehen Shard und Bytebereich in `mistral/tensor_inventory.tsv`, die Shard-Hashes in `mistral/provenance.json`.

### 5.8 Brücken und Implant (`mistral/implant/implant_manifest.json`)

| Anker-Brücke (datenfrei, Ridge 1e-02) | R² Fit | R² Holdout-Tokens |
|---|---|---|
| Q: Seed-Vor-Kopf → Donor-Vor-Kopf (1920 → 6144) | 0.818 | -1.374 |
| P: Donor-Update → Seed-Update (6144 → 1920) | 0.331 | -0.211 |

- 40 Experten (Vereinigung der Top-Ränge je Funktion). Je Experte im Manifest: Shard, Bytebereich und SHA-256 der FP8-Nutzlast und der Blockskalen (COPY), Formen und SHA-256 der Slot-Tensoren (PROJECTION), Liste der behaltenen Einheiten.
- Pfadprüfung: die im Lauf gespeicherten Beitragsnormen werden aus der Eingangs-Stichprobe reproduziert, mediane relative Abweichung 1.1e-04–2.3e-04 (float16-Speicherung).
- Einheitenauswahl 6144 → fe, relativer quadratischer Ausgangsfehler auf echten Experten-Eingängen (min / Median / max über 40 Experten):

| fe | behaltener Wichtigkeitsanteil | Fehler Donor-Raum | Fehler nach Ausgangsbrücke |
|---|---|---|---|
| 320 | 0.085 / **0.148** / 0.338 | 0.228 / **0.595** / 0.825 | 0.170 / **0.595** / 0.822 |
| 640 (Team-Standard, geliefert) | 0.152 / **0.237** / 0.484 | 0.118 / **0.461** / 0.733 | 0.076 / **0.444** / 0.730 |
| 768 | 0.177 / **0.268** / 0.526 | 0.098 / **0.420** / 0.701 | 0.058 / **0.411** / 0.700 |
| 1280 | 0.271 / **0.370** / 0.652 | 0.051 / **0.308** / 0.591 | 0.026 / **0.306** / 0.589 |
| 1920 | 0.382 / **0.479** / 0.752 | 0.026 / **0.221** / 0.478 | 0.013 / **0.215** / 0.475 |
| 3072 | 0.568 / **0.650** / 0.864 | 0.010 / **0.114** / 0.309 | 0.005 / **0.120** / 0.305 |

| Datei | Klasse | SHA-256 |
|---|---|---|
| `seed_slots_PROJECTION_fe640.f32.safetensors` (791 578 576 B) | PROJECTION (verlustbehaftet: Einheitenauswahl) | `22407fde6fc7e079094440353cba24c918d1229f01cfb53e2e3a08060c1b50a0` |
| COPY der FP8-Experten | COPY | nicht abgelegt (113 MB je Experte); Shard + Bytebereich + SHA-256 je Tensor im Manifest, `--store-copy` legt die Datei an |
| Kopf-Zeilen auf dem Seed-Vokabular (`transplant.head_rows` in der Projektionsdatei) | PROJECTION | `ca892576af5adc2698a7765e45345f0effafefcc8c62f4c10d90b3fab947a540` |

Folgerungen:

- Die datenfreien Anker sind für Mistral **nur eine Startlage**: Q interpoliert die 8192 Vokabelzeilen in 6144 Dimensionen (Fit 0,82) und verallgemeinert nicht (Holdout negativ).
- **Der Team-Standard fe = 640 passt nicht zu Mistral-Experten.** Ein Experte hat 6144 Einheiten; auf 640 gekürzt gehen im Median 46 % der Ausgangsenergie verloren (bester Experte 12 %, schlechtester 73 %). Erst ab fe = 3072 liegt der Median bei 11 %. Optionen: (a) breitere Slots für Mistral-Experten, (b) ein Experte verteilt auf mehrere 640er-Slots mit gekoppeltem Routing (im Team-Router nicht vorhanden), (c) 640 als verlustbehaftete Startlage plus Heal. Geliefert ist (c) als Datei und die Messkurve für die Entscheidung; `--slot-width` erzeugt jede andere Breite.
- Kein Experte wurde für zwei Funktionen gleichzeitig gewählt (40 verschiedene Experten aus 5 Listen à 8).

### 5.9 Ausgangsbrücke Mistral → Seed-Vokabular (`mistral/output_bridge.json`)

Mistral-Endresiduum auf den Klartext-Sonden, gelesen durch den echten Seed-Ausgangspfad; 12546 bytegenau ausgerichtete Zielpositionen, Auswertung auf der zurückgehaltenen Hälfte (6372); Gleichverteilung 9.011 nats.

| Variante | Klasse | nats | bit/Byte |
|---|---|---|---|
| Referenz: transplantierter Donor-Kopf | PROJECTION | 3.0940 | 1.4928 |
| Anker-Brücke, datenfrei | PROJECTION | 4.4009 | 2.1234 |
| Anker ohne Seed-Norm | PROJECTION | 3.7295 | 1.7994 |
| nur die Brücke trainiert | **HEAL** | 3.4214 | 1.6508 |

Heal: 400 Schritte, 11 796 480 Brückenparameter, Seed-Tensoren nur gelesen; Fit-Hälfte 0.0206 nats (Überanpassung, deshalb zählt nur die Holdout-Hälfte).

Lesart: Anders als bei Qwen **schlägt die trainierte Brücke die Referenz auf der Holdout-Hälfte nicht** (3,42 gegen 3,09 nats). 11,8 M Brückenparameter auf 6 174 Positionen überanpassen vollständig. Für Mistral ist eine tragfähige Ausgangsbrücke damit **nicht belegt**; nötig wären deutlich mehr ausgerichtete Positionen (bei Qwen waren es 21 046 für 3,9 M Parameter) oder eine Brücke mit weniger Freiheitsgraden. Die Positionen enthalten auch die Regelaufgaben mit Zufallszeichenketten, was alle Zeilen der Tabelle anhebt.

### 5.10 Tests

Ausgeführt und bestanden: Vorwärtspfad = `transformers`-Referenz (NoPE, Schlüssel-Sharing, LeRoPE/YaRN, dichter L0, Shared-Expert); FP8-Dekodierung = `torch.float8_e4m3fn`; plausible Verluste auf echten Gewichten; Beitragsnormen aus der Eingangs-Stichprobe reproduziert (Abweichung ≈ 2e-4); SHA-256 der FP8-Nutzlast und der Skalen aller 40 Experten; Faltung = Komposition; Auswahlfehler-Formel.

**Nicht ausgeführt:**
- Regime B (Seed-Tokens über transplantierte Embeddings durch den ganzen Stapel): ein zweiter 1-TB-Durchlauf.
- Kausaltest der gewählten Experten (Ablation je Experte und Gruppe).
- Attention-Köpfe: Lernen im Kontext läuft wesentlich über Attention; betrachtet wurden nur MoE-Experten und Layer.
- Andere Reasoning-Arten, andere Sprachen, Kontexte über 2048 Tokens (dort greift Mistrals Sparse-Attention-Index, der hier nicht gebraucht und nicht implementiert ist).
- Alles am Seed: Identität des gewachsenen Modells, Brücken an inneren Layern, Nutzen nach Heal, jede Baseline.

---

## 6. Optionaler Donor Sprache/Wissen: Qwen3.6-35B-A3B

| | |
|---|---|
| Repo | `Qwen/Qwen3.6-35B-A3B` |
| Revision | `345d467b117d1fd5eda9b879a5988de9ae5f47a7` |
| Lizenz | Apache-2.0 |
| Gewichte | 26 Shards, 71 903 776 776 B; 1 045 Tensoren, 35 951 822 704 Parameter (inkl. Bild-Encoder), alle BF16 |
| Architektur (Text) | 40 Layer, d = 2048, 256 Experten Top-8 + Shared-Expert, Experten-Breite 512, Vokabular 248 320; 30 Layer mit linearer Attention, jeder 4. Layer mit voller Attention; Experten als gestapelte Tensoren (`gate_up_proj [256, 1024, 2048]`, `down_proj [256, 2048, 512]`) |

Geliefert: `qwen36/provenance.json`, `qwen36/tensor_inventory.tsv`, `qwen36/inventory_summary.json`,
`qwen36/vocab_contract.json` (**6807 exakt**, **1382 zusammengesetzt**, **0 unbekannt**).

Abgeleitet (nicht gemessen): Ein Experte hat 512 Einheiten und passt damit per Null-Auffüllung **ohne
Einheitenauswahl** in einen 640er-Slot; die Residualbreite ist wie bei Qwen3-Coder 2048.

**Nicht ausgeführt:** Forward, Expertenwahl, Brücken, Implant. Grund: Die lineare Attention dieses Modells
ist in `titan.donor` nicht implementiert und daher nicht gegen eine Referenz geprüft; ohne geprüften
Vorwärtspfad wähle ich keine Experten. Aufwand: ein Vorwärtsmodul plus Referenztest, danach derselbe Ablauf
wie bei Qwen3-Coder (72 GB Stream).

---

## 7. Fehlende Eingaben

| # | Was fehlt | Wofür es gebraucht wird |
|---|-----------|-------------------------|
| 1 | Die übrigen 210 Seed-Tensoren (Champion step-007500) | jede Aussage über den Seed; Identitätstest des gewachsenen Modells; Baseline |
| 2 | Seed-Aktivierungen auf dem Kalibriertext an den Slot-Layern (Eingang der FFN-Norm, Residual-RMS) | `g_slot`, `rms_seed`; gepaarte Brücken an inneren Layern statt datenfreier Anker |
| 3 | Ziel-MoE-Konfiguration: welche Layer, E, Top-k, `route_scale`, fe (640 bestätigt?), Anzahl Shared-Experten | Slot-Tensoren in endgültiger Form; Parameterzahl ≈ 25,7 B / 5,9 B aktiv nachrechnen |
| 4 | Eval-Protokoll des Teams: Holdout-Dateien, Metrik, Baseline-Zahl des Seeds | Gates 3–5 |
| 5 | Entscheidung, welche Tensoren beim Heal trainierbar sind (Vorschlag: nur neue Slots, Router-Zeilen der neuen Slots, Brücken) | „Seed unverändert“ prüfbar machen |
| 6 | Trainingsdaten-Freigabe für Heal (nur der freigegebene Train-Mix, keine Modellausgaben) | HEAL |
| 7 | Lizenzentscheidung Mistral (Derivat; MaaS nur mit kommerzieller Lizenz) | ob Mistral-Experten überhaupt in ein auslieferbares Modell dürfen |
| 8 | GPU-Zeit | Heal und Eval eines ≈ 25-B-Modells laufen nicht auf 4 CPU-Kernen |

---

## 8. Akzeptanz-Gates

Jedes Gate hat eine Messgröße und eine Schwelle. Schwellen mit „Vorschlag“ legt das Team fest.

| Gate | Prüfung | Schwelle | Stand |
|------|---------|----------|-------|
| G0 Integrität | SHA-256 jeder COPY = Hub-Bytebereich; Paket-Hashes | exakt gleich | **bestanden** (alle Manifeste) |
| G1 Donor-Pfad korrekt | eigener Forward = `transformers`-Referenz | Testtoleranz 2e-4 | **bestanden** (Qwen3-MoE, Mistral Large 4) |
| G2 Identität beim Einsetzen | gewachsenes Modell gegen Seed auf einem Eval-Batch: gleiche Experten je Token, alte Tensoren bitgleich, Logits | max. Abweichung ≤ 1e-5 relativ | Mechanik **bestanden** (NumPy-Nachbau des Team-Routers); am echten Seed **nicht ausgeführt** |
| G3 Brücke trägt | Holdout-R² der Brücke am Slot-Layer mit gepaarten Aktivierungen | ≥ 0,5 (Vorschlag); darunter kein Aktivieren ohne Heal | **nicht ausgeführt** (Eingabe 2); datenfreie Anker liegen bei ≤ 0,10 |
| G4 Seed unverändert | SHA-256 aller Seed-Tensoren vor und nach Heal | exakt gleich | Regel eingehalten; Prüfung nach Heal offen |
| G5 Nutzen | Holdout-Verlust des gewachsenen Modells gegen Seed-Baseline, getrennt nach Domäne (Code / Reasoning / allgemein) | Zieldomäne besser, allgemeine Domäne nicht schlechter als Rauschband zweier Seeds (Vorschlag) | **nicht ausgeführt** |
| G6 Rollback-Regel | wenn G5 nach dem vereinbarten Schritt-Budget nicht erfüllt ist | Slots per Bias wieder abschalten (= Seed), Ergebnis als Rollback dokumentieren | — |
| G7 Datenregeln | keine Closed-Teacher-Ausgaben, keine Fable-Ausgaben, kein 2–4-bit, kein QLoRA | Liste der Datenquellen und dtypes | **eingehalten** (Quellen in Abschnitt 4.3 und 5.3; dtypes BF16/FP8/F32) |
| G8 Lizenz | Mistral-Derivat nur mit geklärter Nutzung | schriftliche Entscheidung | offen (Eingabe 7) |

---

## 9. Reproduktion

Alle Befehle aus dem Repo-Wurzelverzeichnis; `$W` ist ein Arbeitsverzeichnis, `$C` ein Cache je Donor.
Netz: `huggingface.co` und `us.aws.cdn.hf.co`.

```bash
# Tests
python -m pytest -q tests/test_donor.py tests/test_donor_reference.py

# Qwen3-Coder: voller Forward (59,8 GB Stream), Routing je Token, Logit-Lens
python tools/donor/qwen_forward.py --seed-tokenizer <paket>/tokenizer.json --chunks $W/calib_chunks.json \
    --out $W/qwen_full --cache $C/qwen3coder
# Experten-Aktivierungen der gewählten Experten (ab gespeichertem L35-Residuum)
python tools/donor/qwen_expert_acts.py --cache $C/qwen3coder --run $W/qwen_full --start-after 35 \
    --selection $W/qwen_tail16.json --out $W/qwen_acts
# COPY + PROJECTION
python tools/donor/qwen_implant.py --cache $C/qwen3coder --selection $W/qwen_tail16.json \
    --seed-emb $W/seed_tok_emb.npy --seed-norm $W/seed_norm.npy --donor-head-rows $W/qwen_full/transplant_head.npy \
    --donor-norm $W/qwen_norm.npy --donor-rms $W/qwen_rms.json --exact-rows $W/qwen_exact_rows.npy \
    --slot-widths 768,640 --acts $W/qwen_acts --out qwen3coder/implant
# Ausgangsbrücke (Referenz / Anker / nur-Brücke-Heal)
python tools/donor/output_bridge.py --help

# Mistral Large 4: Sondensatz, voller Forward (≈ 1,05 TB Stream), Auswahl, Implant
python tools/donor/mistral_probes.py --tokenizer-dir $C/mistral_tok --gsm8k $W/gsm8k_test.parquet \
    --wikitext $W/wikitext2_test.parquet --code-chunks $W/calib_chunks.json --out $W/mistral_probes.json
python tools/donor/mistral_forward.py --probes $W/mistral_probes.json --out $W/mistral_full --cache $C/mistral
python tools/donor/mistral_select.py --run $W/mistral_full --out mistral/selection.json
python tools/donor/mistral_implant.py --cache $C/mistral --run $W/mistral_full --selection mistral/selection.json \
    --contract mistral/vocab_contract.json --seed-emb $W/seed_tok_emb.npy --seed-norm $W/seed_norm.npy \
    --out mistral/implant        # --store-copy legt zusätzlich die FP8-Bytes ab (113 MB je Experte)

# Einsetzfertige Implantate (Abschnitt 11): stapeln und einsetzen
python tools/donor/build_implant.py --projection qwen3coder/implant/seed_slots_PROJECTION_fe640.f32.safetensors \
    --manifest qwen3coder/implant/implant_manifest.json --donor-layers 48 --group code=code \
    --prefix implant_qwen3coder_fe640 --out implants
python tools/donor/build_implant.py --projection mistral/implant/seed_slots_PROJECTION_fe640.f32.safetensors \
    --manifest mistral/implant/implant_manifest.json --donor-layers 72 --group reasoning=reasoning \
    --group adaptive_thinking=effort,think_mode --group learning=learning_gain,learning \
    --prefix implant_mistral_l4_fe640 --out implants
python tools/donor/insert_implant.py --checkpoint <ckpt> --implant implants/*_fe640_*.safetensors --layers 16,17,18,19,20,21,22,23 --out <ckpt_neu>
# Mistral-Projektion ohne das Laufverzeichnis des Forwards (andere Slot-Breite: --slot-width)
python tools/donor/mistral_implant.py --cache $C/mistral --rev <revision> --unit-act mistral/implant/unit_activation.safetensors \
    --layers-json mistral/measurements/layers.json --selection mistral/selection.json --contract mistral/vocab_contract.json \
    --seed-emb $W/seed_tok_emb.npy --seed-norm $W/seed_norm.npy --out mistral/implant
```

`seed_tok_emb.npy` und `seed_norm.npy` sind `tok_emb.weight` und `norm.weight` aus der Subset-Datei des Pakets.
Der Mistral-Forward ist nach jedem Layer fortsetzbar (`state.pt`).

Rechenaufwand, gemessen auf 4 CPU-Kernen (Xeon, AVX-512/AMX), ohne GPU: Qwen 3283 s für 99 783 Tokens;
Mistral ≈ 3–4 min je Layer für 23 210 Tokens (31,4 TFLOP und 14,7 GB je Layer), solange der Download
≥ 70 MB/s liefert.

---

## 10. Dateien

Alle Dateien liegen im Gesamtpaket, einschließlich der Gewichtsdateien, der Routing-Tabellen beider Läufe (`runs/`), der Eingabetexte (`inputs/`), der Prüfnachweise (`checks/`) und des Code-Stands (`code/`). Aufbau, Prüfung und Absicherung: `PAKET.md`; SHA-256 jeder Datei in `MANIFEST.sha256`. Die Dateien unter `*/implant/` enthalten dieselben Tensoren je Experte einzeln (Quelle der Implantate unter `implants/`).

| Gewichtsdatei | Größe | Klasse | SHA-256 |
|---|---|---|---|
| `implants/implant_mistral_l4_fe640_adaptive_thinking.safetensors` | 236.1 MB | PROJECTION (einsetzfertig) | `984f867d367441e3d66c7ccbf2019fa8d3b3ec7ee38cf9cfad225970104b4854` |
| `implants/implant_mistral_l4_fe640_learning.safetensors` | 236.1 MB | PROJECTION (einsetzfertig) | `fc8911deb8c87159cffa4c743fa54229e18d129c4e43c6745c9f5aa5810107b8` |
| `implants/implant_mistral_l4_fe640_reasoning.safetensors` | 118.0 MB | PROJECTION (einsetzfertig) | `1ae4d7b34d17b6ad39ce3b66bc0022ebd8536f2a07e1e283c9175a3986e003e0` |
| `implants/implant_qwen3coder_fe640_code.safetensors` | 236.1 MB | PROJECTION (einsetzfertig) | `1623d5a1362e1bb4d1183c143a9a63bfe17861fb72f034744d34bfd73c78112b` |
| `implants/implant_qwen3coder_fe768_code.safetensors` | 283.2 MB | PROJECTION (einsetzfertig) | `7ffc640762d160777368036ecea4ca80a5707aac29eb60303b81f22a206b08fa` |
| `mistral/implant/seed_slots_PROJECTION_fe640.f32.safetensors` | 791.6 MB | PROJECTION | `22407fde6fc7e079094440353cba24c918d1229f01cfb53e2e3a08060c1b50a0` |
| `qwen3coder/implant/donor_experts_COPY.bf16.safetensors` | 151.0 MB | COPY | `3ac98b765271453d7bbfc94cdac2cf775b79191cc9a54fbdd09c52513d335786` |
| `qwen3coder/implant/seed_slots_PROJECTION.f32.safetensors` | 283.2 MB | PROJECTION | `c74f96c7505972def586c232e6cf7c42167cba089dc072012424e9c8a0e79f31` |
| `qwen3coder/implant/seed_slots_PROJECTION_fe640.f32.safetensors` | 236.1 MB | PROJECTION | `457f412244389f24ffaf59a3580da8aeb52fa64ec4327806cdd58ffbba9cdf2f` |

| Ordner | Inhalt |
|---|---|
| `seed/` | Verifikation des Pakets und der Subset-Tensoren |
| `qwen3coder/` | Provenienz, Inventar, Vokabularvertrag, `measurements/` (Forward, Layer-Tabelle, Auswahl, Brücken), `implant/` (Manifest, Aktivierungsbericht) |
| `mistral/` | Provenienz, Inventar, Vokabularvertrag, `selection.json` (Layer-Profil, Kandidaten, Auswahl), `native_loss.json`, `output_bridge.json`, `measurements/`, `implant/` (Manifest mit Bytebereichen und Hashes der 40 Experten, Einheiten-Aktivierungen) |
| `qwen36/` | Provenienz, Inventar, Vokabularvertrag |
| `implants/` | einsetzfertige Implantate im Layout von `accel/moe.py` mit Index je Datei (Abschnitt 11) |

---

## 11. Einsetzfertige Implantate

Je Donor die gewählten Experten als **eine Datei im Tensor-Layout eures `accel/moe.py`**, dazu ein Werkzeug, das sie in
einen Checkpoint (`model.safetensors`, `optim.safetensors`, `state.json`) einsetzt. Klasse: PROJECTION. Die Reihen
sind Byte-Kopien der Slot-Tensoren aus Abschnitt 4.6 und 5.8 (per Hash geprüft).

| Datei | Donor | Funktion | Experten | fe | Größe | Verlust der Breitenkürzung, Median (min–max) | SHA-256 |
|---|---|---|---|---|---|---|---|
| `implants/implant_qwen3coder_fe640_code.safetensors` | Qwen3-Coder | Code | 16 | 640 | 236 MB | 3.5 % (1.9–6.4) | `1623d5a1362e1bb4d1183c143a9a63bfe17861fb72f034744d34bfd73c78112b` |
| `implants/implant_qwen3coder_fe768_code.safetensors` | Qwen3-Coder | Code (nur bei `expert_ffn = 768`) | 16 | 768 | 283 MB | 0 (exakt) | `7ffc640762d160777368036ecea4ca80a5707aac29eb60303b81f22a206b08fa` |
| `implants/implant_mistral_l4_fe640_reasoning.safetensors` | Mistral Large 4 | Reasoning | 8 | 640 | 118 MB | 47.9 % (19.8–69.4) | `1ae4d7b34d17b6ad39ce3b66bc0022ebd8536f2a07e1e283c9175a3986e003e0` |
| `implants/implant_mistral_l4_fe640_adaptive_thinking.safetensors` | Mistral Large 4 | adaptives Denken (8 Effort + 8 Denkblock) | 16 | 640 | 236 MB | 48.7 % (11.8–73.3) | `984f867d367441e3d66c7ccbf2019fa8d3b3ec7ee38cf9cfad225970104b4854` |
| `implants/implant_mistral_l4_fe640_learning.safetensors` | Mistral Large 4 | Lernen im Kontext (8 mit Zuwachs + 8 Ausgabeposition) | 16 | 640 | 236 MB | 41.6 % (20.1–64.9) | `fc8911deb8c87159cffa4c743fa54229e18d129c4e43c6745c9f5aa5810107b8` |

Inhalt je Datei: `implant.experts.w1 [N, fe, 1920]`, `implant.experts.w3 [N, fe, 1920]`, `implant.experts.w2 [N, 1920, fe]`,
`implant.router.weight [N, 1920]`. Daneben `<name>.json`: je Reihe Donor-Layer und -Experte, Auswahlgrund mit Messwerten,
Hashes der Quell-Tensoren, gemessener Kürzungsverlust.

### 11.1 Einsetzen

```bash
python tools/donor/insert_implant.py --checkpoint <euer_moe_checkpoint> \
    --implant implants/implant_qwen3coder_fe640_code.safetensors implants/implant_mistral_l4_fe640_reasoning.safetensors \
              implants/implant_mistral_l4_fe640_adaptive_thinking.safetensors implants/implant_mistral_l4_fe640_learning.safetensors \
    --layers 16,17,18,19,20,21,22,23 --out <neuer_checkpoint>          # optional: --seed-rms rms.json --off-bias -4
```

Was das Werkzeug tut:

1. Es verteilt die Reihen in Donor-Tiefenreihenfolge gleichmäßig auf die Ziel-Layer.
2. Euer Format kennt **eine** Expertenzahl `n_routed` für alle MoE-Layer. Deshalb wächst jeder MoE-Layer um dieselbe Zahl
   Slots; Layer mit weniger Implantat-Reihen erhalten Füll-Slots (kleine Zufallswerte in `w1`/`w3` mit festem Seed,
   `w2` = 0, Router-Zeile = 0 — Ausgabe exakt null, aber trainierbar; dieselbe Bauart wie in eurem `grow()`).
3. Es passt jede Reihe an ihren Layer an: `w1`, `w3` und die Router-Zeile werden durch den `ffn_norm`-Gain dieses Layers
   geteilt (das fehlende `g_slot` aus Abschnitt 4.6 wird damit automatisch gesetzt), `w2` wird mit dem Residual-RMS
   multipliziert, wenn ihr ihn mit `--seed-rms` angebt (sonst 1,0).
4. Alle neuen Slots erhalten Auswahl-Bias −4. Sigmoid-Scores liegen in (0, 1), also kommt kein neuer Slot in die Top-k:
   jedes Token behält seine Experten.
5. Alles andere wird Byte für Byte kopiert; die alten Reihen der gewachsenen Tensoren bleiben bytegleich (neue Reihen
   werden angehängt). Die Optimizer-Momente neuer Reihen sind null. In `state.json` ändern sich nur `n_routed`,
   `model_sha256` und `parent_sha256`. Der Quell-Checkpoint wird nie verändert.
6. Die Datei wird gestreamt; der Speicherbedarf ist unabhängig von der Checkpoint-Größe. `implant_report.json` hält
   Platzierung, Hashes und hinzugekommene Parameter fest.

### 11.2 Geprüft

| Prüfung | Ergebnis |
|---|---|
| Tests im Repo (`tests/test_donor_implant.py`) | Layout, Byte-Gleichheit, Gain-Anpassung, Füll-Slots, Optimizer, `state.json`, Ablehnung von `--off-bias` > −1 und von Einsetzen am Ort |
| **Euer Code**, Mini-Modell: Checkpoint aus eurem `upcycle()`, Tensorliste gegen eure `MoEConfig.all_specs()`, euer `forward()` vorher/nachher | Tensorliste und Formen exakt; größte Logit-Abweichung 1,2e-7 (`dense`) und 1,8e-7 (`ragged`); Last der neuen Slots 0 |
| **Generalprobe bei echter Breite:** 2-Layer-Modell (dim 1920, FFN 5120, GQA 30/10, Vokabular 8192) mit den echten Seed-Tensoren, die es gibt (`tok_emb`, `norm`, Norm-Gains der Blöcke 11 und 23), euer `upcycle()` (fe 640, 2 Shared, Top-6), **alle vier fe-640-Implantate** (56 Experten), euer `forward()` | `n_routed` 8 → 36; Logit-Abweichung **0,0** mit Null-Routern (Zustand direkt nach eurem Upcycle) und 8,1e-6 mit Zufalls-Routern (Logit-Maximum 11,9); Last der neuen Slots 0 |
| Neue Slots eingeschaltet (Bias +4), dieselbe Generalprobe | Logits endlich, ändern sich um bis zu 12,2; 26 der 56 neuen Slots bekommen Last — die Slots sind lebendig, nicht tot |
| Implantat ohne mein Laufverzeichnis neu erzeugt (`--unit-act`, `--layers-json`) | 25 von 25 geprüften Tensoren bitgleich zur Lieferung |

Euer Code lief dafür unverändert aus dem Paket (`run2-prep/moe/accel` plus `forge_io.py` aus `mgpu-trainer`); das im Paket
fehlende Modul `rebase` wurde durch eine leere Attrappe ersetzt, weil der Forward es nicht benutzt. Attention- und
FFN-Gewichte der Generalprobe sind Zufallswerte: geprüft sind **Format und Identität**, nicht Qualität.

### 11.3 Was ihr wissen und entscheiden müsst

1. **Einsetzbar heißt nicht wirksam.** Nach dem Einsetzen rechnet das Modell exakt wie vorher. Ob die Slots etwas
   beitragen, entscheidet erst euer Training (HEAL) gegen die Gates in Abschnitt 8. An eurem echten Checkpoint ist nichts
   davon gelaufen.
2. **Ziel-Layer** (`--layers`) sind nicht gemessen, weil der Seed fehlt. Die Tiefenreihenfolge der Donors bleibt erhalten;
   welche Seed-Layer sie aufnehmen, ist eure Entscheidung (Gate G3 mit gepaarten Aktivierungen wäre die Messung dazu).
3. **Kosten:** ein Slot hat 3 · 640 · 1920 + 1920 = 3 688 320 Parameter. 56 Experten auf 8 Layer verteilt heißt +7 Slots in
   jedem der 24 MoE-Layer: 619,6 M Parameter, davon 206,5 M Implantat und 413,1 M Füll-Slots. Aktive Parameter je Token
   ändern sich nicht, solange `top_k` gleich bleibt.
4. **Einschalten:** Euer `update_router_bias` hebt den Bias unterlasteter Slots um `gamma` je Schritt. Mit Null-Routern
   (alte Scores 0,5) wird ein neuer Slot frühestens wählbar, wenn sein Bias etwa −0,5 erreicht — bei Start −4 also nach
   rund 3,5 / `gamma` Schritten. `--off-bias -1` verkürzt das auf rund 0,5 / `gamma`, ohne die Identität am Schritt 0 zu
   verletzen.
5. **Mistral bei fe = 640 ist eine schwache Startlage** (Kürzungsverlust siehe Tabelle). Wenn Mistral-Experten ernsthaft
   tragen sollen, braucht es breitere Slots oder mehrere gekoppelte Slots je Experte (Abschnitt 5.8); beides ist mit
   `mistral_implant.py --slot-width` und den gelieferten Einheiten-Aktivierungen (`mistral/implant/unit_activation.safetensors`)
   ohne mein Laufverzeichnis erzeugbar, wurde aber nicht gebaut.
6. **`state.json`** wird mit Python geschrieben (sortierte Schlüssel, 2 Leerzeichen); die Zahlendarstellung kann von
   serde abweichen. Euer Python-Lader liest sie; den Rust-Lader konnte ich nicht prüfen.
7. **Byte-Kopien der Mistral-Experten** (COPY, 113 MB je Experte, 4,5 GB für 40) liegen nicht bei; Shard, Bytebereich und
   SHA-256 je Tensor stehen im Manifest, `mistral_implant.py --store-copy` legt sie an.
