# Eingabetexte der Messläufe

Keine Modellausgaben. Alle Texte sind menschengeschrieben oder programmerzeugt.

| Datei | Inhalt | Quelle | Lizenz |
|---|---|---|---|
| `calib_chunks.json` | 112 Abschnitte (56 Code, 56 Prosa), Kalibriertext des Qwen-Laufs | CPython-3.13-Standardbibliothek (Quelltext und `pydoc_data.topics`) | PSF-2.0 |
| `mistral_probes.json` | 154 Sequenzen, 23 210 Mistral-Tokens mit Labels je Token (Art, Segment, Effort, Denkblock, Aufgabe, Beispielnummer) | GSM8K Test, `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` | MIT |
| | | WikiText-2 Test, `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3` | CC-BY-SA-3.0 (Namensnennung, Weitergabe unter gleichen Bedingungen) |
| | | Code wie oben | PSF-2.0 |
| | | Regelaufgaben: erzeugt von `tools/donor/mistral_probes.py`, Seed 20261110 | — |
