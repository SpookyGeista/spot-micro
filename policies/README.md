# Mitgelieferte Policies

| Verzeichnis | Zweck |
| --- | --- |
| `rough_student/` | Am 13. November 2026 abgenommene Sensor-Policy; Simulation, ONNX-Export und Messbericht |
| `early_teacher/` | Erster gespeicherter Checkpoint des dynamischen Teachers für die Vorher-Demo |

Der frühe Teacher verwendet privilegierte Simulationszustände. Er ist kein Pi-Modell und kein unverändertes Zufallsnetz. Die Student-Policy verarbeitet eine Sensorhistorie und wurde nach dem Teacher-Training durch DAgger übertragen. Der Vergleich in der Demo zeigt daher zwei Stufen des gesamten Entwicklungsablaufs, nicht zwei aufeinanderfolgende Updates desselben Netzes.

`rough_student/robot.json` ist das gespeicherte Geländeprofil. `flat.json` entfernt das Gelände für den ebenen Show-Vergleich. `validation.json` enthält die abschließenden Simulationsergebnisse. Darin genannte historische Laufnamen dienen der Nachvollziehbarkeit; die Trainingsläufe selbst werden nicht mitgeliefert.

Das ONNX-Modell wurde numerisch gegen den exportierten Actor geprüft; maximaler absoluter Fehler: 6,56 × 10⁻⁷. Das Modell hat 175.308 Parameter, 520 Eingaben und zwölf Ausgaben. Die Regelperiode beträgt 20 ms. Der aktuelle Export ist noch nicht am realen Roboter validiert.

Prüfsummen aus diesem Verzeichnis prüfen:

```bash
cd rough_student
sha256sum -c SHA256SUMS
```
