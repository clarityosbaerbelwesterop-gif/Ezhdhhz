# Gesamtpaket Donor-Transplantation (Stand 11.11.2026)

Alles aus der Lieferung in einem Paket: einsetzfertige Implantate, Slot-Tensoren je Experte, Byte-Kopien der Qwen-Experten,
Manifeste, Inventare, Vokabularverträge, Messungen, Routing-Tabellen beider Läufe, Eingabetexte, Prüfnachweise mit eurem
Code und der Code-Stand der Werkzeuge. Das Lieferdokument ist `README.md`.

## Prüfen und entpacken

Das ZIP kommt in Teilen (`fable52-donor-paket-20261111.zip.001`, `.002`, …), dazu `SHA256SUMS.txt` mit den Prüfsummen jedes
Teils und des ganzen ZIP. Alle Teile und `SHA256SUMS.txt` in einen Ordner legen, dann:

```bash
python verify_package.py fable52-donor-paket-20261111.zip.0*                    # nur prüfen
python verify_package.py fable52-donor-paket-20261111.zip.0* --extract ziel/    # prüfen und entpacken
```

`verify_package.py` liegt auch einzeln bei, braucht nur Python 3.8+ und prüft: jeden Teil, das zusammengesetzte ZIP und jede
Datei im Paket gegen `MANIFEST.sha256` (keine fehlt, keine ist zu viel). Ohne Python:

```bash
cat fable52-donor-paket-20261111.zip.0* > fable52-donor-paket-20261111.zip
sha256sum -c SHA256SUMS.txt          # Teile und ganzes ZIP
unzip fable52-donor-paket-20261111.zip
cd fable52-donor-paket-20261111 && sha256sum -c MANIFEST.sha256
```

## Absichern

- **Integrität:** SHA-256 auf drei Ebenen (Teile, ZIP, jede Datei). `SHA256SUMS.txt` getrennt vom Paket aufbewahren (z. B. im
  Ticket oder Repo), dann fällt jede spätere Veränderung auf.
- **Vertraulichkeit:** Das Paket ist nicht verschlüsselt. Es enthält aus eurem privaten Seed abgeleitete Daten
  (Vokabularverträge aus eurem Tokenizer, Projektionen mit euren Embedding- und Norm-Gewichten). Verschlüsselt ablegen,
  z. B. `7z a -p -mhe=on paket.7z fable52-donor-paket-20261111.zip.0* SHA256SUMS.txt`, `age -p` oder `gpg -c`. Das Passwort
  nicht über denselben Kanal wie das Paket weitergeben.
- **Lizenz:** Mistral-Large-4-Teile unterliegen der Mistral Ecosystem Open-Weight License; ein Modell mit diesen Experten ist
  ein Derivat, als Dienst für Dritte nur mit kommerzieller Lizenz von Mistral. Qwen-Teile: Apache-2.0. Eingabetexte siehe
  `inputs/QUELLEN.md`.

## Aufbau

| Ordner / Datei | Inhalt |
|---|---|
| `README.md` | Lieferdokument: Verfahren, Provenienz, Messungen, Auswahl, Brücken, fehlende Eingaben, Akzeptanz-Gates, Reproduktion, Implantate |
| `implants/` | **einsetzfertige Implantate** im Tensor-Layout eures `accel/moe.py` + Index je Datei (README Abschnitt 11) |
| `qwen3coder/`, `mistral/`, `qwen36/` | je Donor: Provenienz, vollständiges Tensor-Inventar, Vokabularvertrag, Messungen, `implant/` (Slot-Tensoren je Experte, Manifest mit Shard, Bytebereich und SHA-256 jeder Donor-Quelle) |
| `seed/` | Verifikation eures Handoff-Pakets und der Subset-Tensoren |
| `runs/mistral/` | Mistral-Lauf: Routing je Token und Layer (`route_L*.npz`), Residual-Abstände und Logit-Lens je Token (`tok_L*.npz`), Token-Labels, Paare, Endverluste, Protokoll |
| `runs/qwen3coder/` | Qwen-Lauf: Routing je Token und Layer, Endverluste, transplantierte Embedding-/Kopf-Zeilen, Experten-Aktivierungen der 16 gewählten Experten, Protokoll |
| `inputs/` | Eingabetexte beider Läufe mit Quellen und Lizenzen |
| `checks/` | Prüfskripte und Ergebnisse mit eurem eigenen `upcycle()`/`forward()` |
| `code/Ezhdhhz-74cd23f.tar.gz` | Code-Stand des Repos (Branch `titan-x1/platform`, Commit 74cd23f): `titan/donor`, `tools/donor`, Tests, Doku |
| `verify_package.py` | Prüfwerkzeug |
| `MANIFEST.sha256` | SHA-256 jeder Datei im Paket außer sich selbst |

## Nicht enthalten

| Was | Warum | Wie zu bekommen |
|---|---|---|
| Byte-Kopien der 40 Mistral-Experten (FP8 + Skalen) | 4,5 GB, öffentlich auf dem Hub | Shard, Bytebereich, SHA-256 je Tensor in `mistral/implant/implant_manifest.json`; `mistral_implant.py --store-copy` |
| Experten-Eingänge des Mistral-Laufs (`xn_L*`, 4,2 GB) und Residuen des Qwen-Laufs (`h_after_L*`, 2,3 GB) | Zwischenzustände; Auswahl, Brücken und Implantate brauchen sie nicht mehr | erneuter Lauf (`mistral_forward.py`, `qwen_forward.py`) |
| eure Seed-Gewichte und euer Code | gehören euch, liegen im Handoff | — |

Was die Implantate können und was nicht: README Abschnitt 11.3. Kurz: eingesetzt rechnet das Modell exakt wie vorher;
ob die Slots etwas beitragen, entscheidet euer Training.
