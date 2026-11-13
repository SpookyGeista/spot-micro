# Training und Auswertung

## Umgebung

Getestet unter Linux mit RTX 4090, Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.2, PyTorch 2.7 (CUDA 12.8) und RSL-RL 3.1.2. Eine funktionierende Isaac-Lab-Installation einschließlich NVIDIA-Treiber und der RSL-RL-Erweiterung ist Voraussetzung. `pip install -e .` allein installiert keinen Simulator.

Isaac Lab nach dessen [Installationsanleitung](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html) einrichten und dabei die genannten Versionen verwenden. Danach in der Simulator-Conda-Umgebung, aus dem Projektordner:

```bash
pip install -e .
```

Die Hilfsskripte verwenden standardmäßig `unitree_sim_env`. Bei abweichendem Namen:

```bash
export SPOT_CONDA_ENV=DEINE_UMGEBUNG
bash scripts/run.sh check
bash scripts/run.sh test
```

Die folgenden direkten Python-Aufrufe sind in der aktivierten Simulator-Umgebung auszuführen. Ein grafischer Desktop wird für die Vorführung benötigt; Training und Auswertung können mit `--headless` laufen.

## Teacher

Der Teacher sieht zusätzliche Simulationszustände. Sein Export gehört nicht auf den Pi. Ein neuer nominaler Lauf:

```bash
bash scripts/run.sh train --headless \
  --robot configs/spotmicro_teacher_dynamic.json \
  --num-envs 2048 --iterations 800 --no-randomization
```

Das Profil `configs/spotmicro_teacher_rough.json` ergänzt Wellen, Blöcke und Schrägen, Reibungsstreuung, Massevariation und zeitlich begrenzte physikalische Kräfte. Für einen kontrollierten Übergang von einem eigenen trainierten Teacher:

```bash
python scripts/launch_training.py \
  --robot configs/spotmicro_teacher_rough.json \
  --initialize-from logs/DEIN_TEACHER/model_final.pt \
  --reset-command-curriculum --adapt-environment \
  --num-envs 2048 --iterations 400 --run-name rough_teacher
```

`--adapt-environment` erlaubt Änderungen am Gelände und an der Randomisierung; Geometrie, Aktuatoren und Beobachtungsvertrag müssen weiterhin zusammenpassen. Die bewegte Neutralstellung wird nicht bestraft. Die Gewichte stehen vollständig im jeweiligen Roboterprofil. Kräfte und Gelände werden über Leistungs-Curricula gesteigert; die konfigurierten Maxima sind nicht automatisch die erreichte Trainingsschwierigkeit.

## Student

Die Student-Policy erhält ausschließlich die für die Pi-Laufzeit vorgesehenen Eingaben. DAgger überträgt die Teacher-Aktionen und lässt den Student während des Trainings zunehmend selbst handeln:

```bash
python scripts/distill_student.py \
  --teacher logs/DEIN_TEACHER/model_final.pt --headless \
  --num-envs 256 --rounds 150 --updates-per-round 64 \
  --history-length 20 --observe-acceleration --max-speed .6 \
  --randomize --seed 2052 --log-file work/student.log
```

Die abgenommene Policy liegt unter `policies/rough_student/`. `robot.json` und `ppo.json` gehören zum Checkpoint. `flat.json` entfernt ausschließlich das Gelände für die Vorführung auf der Ebene. Der ONNX-Export unter `export/` enthält Normalisierung und Metadaten für die Pi-Laufzeit.

## Unabhängige Tests

```bash
python scripts/evaluate_dynamic.py \
  --checkpoint policies/rough_student/model_final.pt \
  --flat --continuous --headless --num-envs 32 --steps 500 \
  --speeds 0 .4 .6 --yaw --seed 2050 --output outputs/nominal

python scripts/evaluate_dynamic.py \
  --checkpoint policies/rough_student/model_final.pt \
  --terrain-level 2 --randomize --push-strength .6 \
  --headless --num-envs 32 --steps 500 --speeds .4 .6 \
  --seed 2051 --output outputs/rough
```

`--continuous` deaktiviert Zeit-Resets. Stürze und Geländegrenzen werden separat gezählt. Auf unebenem Boden wird für die Teacher-Übertragung fortgesetztes Vorankommen von mindestens 0,1 m/s und eine Verringerung der Sturzereignisse gegenüber der Referenz verlangt, keine starre Quote der Sollgeschwindigkeit. Diese Auswahlkriterien ersetzen keinen realen Fahrversuch.

## Modelle und Messgrenzen

Die Simulationsmasse beträgt ungefähr 2,52 kg. Trägheiten und Servoantwort sind Schätzungen. Das MG996R-Modell berücksichtigt Verzögerung, Totzone und eine Drehmoment-Drehzahl-Kennlinie. Hardwarekalibrierung und die Messung der tatsächlichen Masse sind vor einer Übertragung auf einen abweichenden Aufbau erforderlich.

Die STL-Anpassungen unter `hardware/modified/` sind Fertigungsdateien; sie ändern nicht automatisch das URDF. Die Originalquelle der Druckteile ist Thingiverse 3445283. Historische Profile bleiben für ältere Experimente im Repository; neue Läufe sollten die ausdrücklich genannten Profile verwenden.
