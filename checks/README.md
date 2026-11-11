# Prüfnachweise mit eurem eigenen Code

`check.py` und `check_real.py` setzen Implantate mit `tools/donor/insert_implant.py` in einen Checkpoint ein und vergleichen
euren `forward()` vorher und nachher. Gebraucht wird euer Code aus dem Handoff: `run2-prep/moe/accel/{moe,model,config,attn}.py`
und `mgpu-trainer/accel/forge_io.py` in einem Ordner `accel/`, dazu `scripts/rebase.py` (der Forward benutzt es nicht; eine leere
Datei genügt). Aufruf siehe Kopf der Skripte; JAX auf CPU reicht.

| Datei | Prüfung | Ergebnis |
|---|---|---|
| `result_mini.json` | Mini-Modell aus eurem `upcycle()`, synthetisches Implantat, beide Dispatch-Pfade | Logit-Abweichung 1,2e-7 (`dense`) und 1,8e-7 (`ragged`); neue Slots ohne Last |
| `result_real_width.json` | dim 1920, FFN 5120, GQA 30/10, Vokabular 8192, echte Seed-Tensoren aus dem Subset, alle vier fe-640-Implantate (56 Experten) | Logit-Abweichung 0,0 (Null-Router wie nach eurem Upcycle) und 8,1e-6 (Zufalls-Router); eingeschaltet bekommen 26 von 56 neuen Slots Last |
| `result_pipeline_real_width.json` | `tools/donor/transplant_seed.py` von einem dichten 2-Layer-Champion in echter Breite (gespeichert mit eurem `save_checkpoint`, echte Seed-Tensoren, `make_dense.py`) über euer `upcycle()` bis zu allen vier fe-640-Implantaten, geprüft mit eurem `forward()` auf echtem Text | Champion → MoE 9,5e-6, MoE → Implantate 0,0, Verlust gleich, neue Slots ohne Last; euer `moe.load()` lädt das Ergebnis |

Geprüft sind Format und Identität beim Einsetzen. Attention- und FFN-Gewichte der Probe sind Zufallswerte; über die Qualität
eines gewachsenen Modells sagen die Läufe nichts.
