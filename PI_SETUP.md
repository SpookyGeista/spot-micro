# Raspberry Pi 5: Inbetriebnahme

Die abgenommene Policy liegt unter `policies/rough_student/export/`. Sie nutzt MPU-6050-Daten und die bisher ausgegebenen Sollwinkel. Eine Prüfung dieser Policy am realen Roboter steht noch aus.

## 1. Pi und Python vorbereiten

Bei einer Neuinstallation eignet sich Raspberry Pi OS in der 64-Bit-Ausführung. Ein bereits eingerichtetes 64-Bit-Ubuntu muss dafür nicht ersetzt werden; wir prüfen zunächst die vorhandene Installation. Netzwerk, SSH und I2C werden benötigt. Unter Raspberry Pi OS lässt sich I2C über `sudo raspi-config` unter den Schnittstellen aktivieren. Bei Ubuntu prüfen wir die tatsächlich vorhandenen Boot- und Geräteeinstellungen. Die offizielle Anleitung beschreibt [Remote-Zugriff](https://www.raspberrypi.com/documentation/computers/remote-access.html) und die [I2C-Pins des Pi](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html).

Die Skripte verwenden NumPy, ONNX Runtime und smbus2. Der Trainingsrechner verwendet Python 3.11. Die separate Pi-Installation braucht dessen Isaac-Sim-Python nicht: Auf dem Pi prüfen wir seine System-Python-Version und die verfügbaren ARM64-Pakete und legen dafür eine eigene virtuelle Umgebung an.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-pi.txt
cp configs/hardware.example.json configs/hardware.json
mkdir -p work

# Im Pi-ZIP ist policy bereits enthalten. Im Repository einmal kopieren:
if [ ! -d policy ]; then cp -r policies/rough_student/export policy; fi
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json doctor
```

`doctor` prüft zunächst nur Dateien. Es öffnet ohne `--probe-i2c` keine Geräteschnittstelle. Für `/dev/i2c-1` braucht dein Benutzer die passende I2C-Gruppenberechtigung. Fehlende Geräte oder Zugriffsrechte werden nicht durch einen automatischen Root-Start umgangen.

## 2. Zunächst nur die MPU-6050 anschließen und auslesen

Der Pi verwendet für I2C üblicherweise physisch Pin 3 / GPIO2 für SDA und Pin 5 / GPIO3 für SCL. Die Signalleitungen müssen auf **3,3-V-Pegel** arbeiten. Die Spannungsversorgung und eventuelle Pull-ups deines konkreten MPU-Moduls müssen dazu passen. Servo-Versorgung und Pi-Versorgung sind getrennt zu dimensionieren; Servos werden nicht aus den Versorgungspins des Pi gespeist. [Pi-Hardwaredokumentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)

Die MPU-Adresse ist in der Beispielkonfiguration `104`, also `0x68`; bei gesetztem AD0 kann sie `105` / `0x69` sein. `WHO_AM_I` bleibt beim MPU-6050 entsprechend seiner Gerätekennung. Der Treiber verwendet ±4g, ±500°/s, 200-Hz-Sampling und den internen Tiefpass. Nach außen liefert er bereits **m/s²** und **rad/s**, passend zum Training. [MPU-6050-Registerbeschreibung](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map.pdf)

```bash
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json doctor --probe-i2c
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json imu \
  --seconds 10 --output work/imu_test.jsonl
```

Diese beiden Befehle erzeugen keine Servoausgabe. Der IMU-Befehl konfiguriert nur den Sensor und liest ihn aus.

Körperkoordinaten: X nach vorne, Y nach links, Z nach oben. Bei waagerecht ruhendem Körper soll ungefähr `accel=[0,0,9.81]` erscheinen. Drehen um eine positive Körperachse muss die entsprechende positive Drehrate liefern. Die Matrix `imu.rotation_sensor_to_body` bildet deine tatsächliche Sensormontage darauf ab. Sie muss eine räumliche Rotation sein; eine einzelne umgedrehte Achse ohne passende zweite Achse ist keine gültige Rotation.

Danach den Körper waagerecht und ruhig auflegen und den Bias bestimmen:

```bash
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json calibrate-imu \
  --samples 600 --output configs/hardware.json
```

Das speichert ungefähr drei Sekunden frischer Messungen als Offset-Kalibrierung. Bewegung, starke Vibration oder eine unplausible Lage führen zum Abbruch. Die Kalibrierung setzt voraus, dass der Körper tatsächlich waagerecht liegt und die Montagematrix stimmt; sie kann das nicht selbst beweisen. Es ist keine vollständige Sechs-Lagen-Skalierung des Beschleunigungssensors.

## 3. Servo-Treiber bestätigen und Kanäle zuordnen

Wenn dein Board ein PCA9685 ist, `servo_driver.type` in `configs/hardware.json` auf `"pca9685"` setzen. Standardadresse ist `64` / `0x40`. Frequenz 50 Hz und angenommener interner Takt 25 MHz stehen separat in der Datei. Bei einem anderen Board wird dessen Backend ergänzt; die Policy- und Kalibrierungslogik bleibt verwendbar.

Die zwölf Einträge unter `joints` ordnen jedem **Modellgelenk** einen Kanal zu. Die Startzuordnung ist lediglich Kanal 0…11 in der Reihenfolge:

| Bein | Hüfte | Oberschenkel | Knie |
| --- | --- | --- | --- |
| vorne links | 0 | 1 | 2 |
| vorne rechts | 3 | 4 | 5 |
| hinten links | 6 | 7 | 8 |
| hinten rechts | 9 | 10 | 11 |

Diese Zuordnung muss deiner Verkabelung entsprechen. Software erkennt vertauschte Servostecker nicht selbst.

Der PCA9685 hält nach einem Prozessabsturz gegebenenfalls seine letzten Pulse weiter. `finally`-Abschaltung und Signalbehandlung helfen bei normalen Fehlern und Stopps, ersetzen aber keine unabhängig wirkende Abschaltung. Zum Hardwareaufbau gehört daher ein erreichbarer Abschaltweg für die Servo-Versorgung beziehungsweise passend verdrahtetes OE. Der PCA9685-OE-Eingang ist aktiv High zum Deaktivieren; Details und Pegel am konkreten Breakout beachten. [NXP-Datenblatt](https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf)

## 4. Einzelne Gelenke beim Zusammenbau kalibrieren

Vor dem Festlegen der Servohörner und bei freier/abgestützter Mechanik jeweils nur ein Gelenk prüfen. Die erste Position ist ohne Encoder unbekannt: Auch ein einzelner gewählter Puls kann daher zunächst eine Bewegung auslösen. Es gibt keinen automatischen Sweep durch unbekannte Anschläge.

Beispiel für einen ausdrücklich gewählten Einzelpuls:

```bash
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json pulse \
  --joint motor_front_left_hip --pulse-us 1500 --seconds 2 --enable-servos
```

1500 µs ist eine übliche anfängliche Referenz zum Ausrichten eines ungekoppelten Servos; es ist **kein Nachweis** für einen sicheren Winkel in deinem fertig montierten Gelenk. Der Befehl schaltet alle Kanäle ab, gibt ausschließlich den ausgewählten Kanal frei und deaktiviert am Ende die Pulse wieder. Die Datei begrenzt den anfänglichen Pulsbereich auf 1000–2000 µs. Eine Anpassung braucht reale Messungen und mechanisch freie Bewegung.

Für jedes Gelenk mindestens zwei, besser drei oder mehr Paare bestimmen:

**Modell-Gelenkwinkel in rad ↔ tatsächlich ausgegebener Puls in µs.**

Die Punkte werden nach steigendem Modellwinkel gespeichert. Bei entgegengesetzter Servorichtung fallen die Pulswerte statt zu steigen. Damit sind Nullstellung und Richtung in den Messpunkten enthalten. Das Modell verwendet relative Gelenkwinkel: Ein Knie-Winkel ist relativ zum Oberschenkel, nicht die absolute Unterschenkelneigung zum Tisch. Die Referenzen stehen in der URDF und der Modellansicht des Trainingsprojekts; Horn-Mittelstellung ist nicht automatisch Modell-Nullstellung.

Zum Eintragen der **eigenen** Messungen folgende Platzhalter ersetzen:

```text
python scripts/pi_robot.py --hardware configs/hardware.json joint-points \
  --joint motor_front_left_hip \
  --point WINKEL_1_RAD PULS_1_US \
  --point WINKEL_2_RAD PULS_2_US \
  --output configs/hardware.json
```

`joint-points` hat keinen Gerätezugriff. Es prüft Messpunktreihenfolge und Pulsgrenzen und markiert nur dieses Gelenk als kalibriert. Der Policy-Betrieb prüft zusätzlich, ob die Messpunkte den gesamten erreichbaren Policy-Stellbereich abdecken. Er extrapoliert nicht über unbekannte Winkel hinaus.

## 5. Policy zunächst ohne Servobewegung testen

Die Software lässt sich schon vor Ende des Trainings vorbereiten. Für einen echten Policy-Test wird später der fertige Export unter `policy/` benötigt. Ein kurzer Testcheckpoint dient nur zum Prüfen der Software, nicht als fertige Laufsteuerung.

```bash
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json run \
  --dry-run --export-dir policy --seconds 10 --output work/policy_dry_run.jsonl
```

Ohne `--real-imu` wird ein synthetischer, waagerecht ruhender Sensor verwendet. Es wird kein I2C-Gerät geöffnet. Mit angeschlossener MPU:

```bash
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json run \
  --dry-run --real-imu --export-dir policy --seconds 10 \
  --output work/policy_with_imu.jsonl
```

So prüfen wir IMU, Beobachtungsverarbeitung, Netzausführung und Zeittakt, während die Servos unberührt bleiben.

## 6. Später: abgestützte Standversuche und Fahrbefehle

Echter Policy-Betrieb verlangt ausdrücklich `--enable-servos`, bestätigten PWM-Treiber, kalibrierte MPU und alle zwölf Gelenkkalibrierungen. Die Robotermodellwerte werden aus dem Export geladen, nicht aus einer inzwischen geänderten Trainingsdatei. Die erste Ausgabe fährt die kalibrierte Standpose an; das kann ohne Positionsrückmeldung nicht von einer beliebigen Ausgangsstellung garantiert ruckfrei erfolgen.

```bash
.venv/bin/python scripts/pi_robot.py --hardware configs/hardware.json run \
  --enable-servos --export-dir policy --seconds 10 \
  --command-file work/command.json --output work/first_stand.jsonl
```

Ohne Befehlsdatei oder bei ausgebliebenem Heartbeat wird Nullgeschwindigkeit angefordert. Der Actor darf zum Balancieren trotzdem Gelenkbewegungen ausgeben. Ein expliziter Stop beendet die Ausgabe vollständig. Befehle kommen vorerst über eine lokale Datei auf dem Pi, beispielsweise aus einem zweiten SSH-Terminal:

```bash
# Stillstand anfordern und den Heartbeat für zehn Sekunden senden:
.venv/bin/python scripts/pi_command.py --hardware configs/hardware.json \
  --file work/command.json --command 0 0 0 --seconds 10

# Expliziter Stop:
.venv/bin/python scripts/pi_command.py --file work/command.json --stop
```

Das Ende oder Ctrl-C im Befehlsgeber schreibt ebenfalls einen Stop. Erneutes Freigeben nach einem Stop verlangt einen neuen Prozessstart. Veraltete IMU-Werte, nicht-endliche Werte, überzogene Zykluszeiten und zu große geschätzte Körperneigung beenden den Controller. Sensor-/I2C-Fehler laufen ebenfalls durch die Abschaltung. `SIGTERM` und Ctrl-C werden behandelt.

Ein Stop wird zusätzlich als separate `command.json.stop`-Datei gespeichert. Ein noch laufender Heartbeat kann ihn dadurch nicht überschreiben. Vor einem neuen Controllerstart ausdrücklich mit `pi_command.py --file work/command.json --clear-stop` löschen; dieser Befehl schaltet selbst keine Motoren ein. Ein vorhandener Stop wird vor der ersten PWM-Freigabe geprüft. Beim Freigeben muss der Körper außerdem ungefähr waagerecht und ruhig abgestützt sein.

Diese Zyklusüberwachung arbeitet innerhalb des Prozesses: Bei einem blockierten Betriebssystem, blockierten I2C-Aufruf oder `SIGKILL` kann sie keine garantierte Reaktionszeit einhalten. Deshalb wird keine harte Echtzeit- oder unabhängige Watchdog-Garantie behauptet. Der Zielrechner muss erst auf Takt und Latenz gemessen werden.

## Software-Paket ohne fertige Policy erstellen

Auf dem Trainingsrechner:

```bash
python scripts/package_pi.py --output outputs/spot_micro_pi_software.zip
```

Das enthält Sensor-/Kalibrierungswerkzeuge und die Beispielkonfiguration. Nach dem Training lässt sich der Export hinzufügen:

```bash
python scripts/package_pi.py logs/DEIN_LAUF/export --output outputs/spot_micro_pi.zip
```

Nächster praktischer Meilenstein: Pi erreichbar, MPU-Werte plausibel, danach ein einzelnes korrekt zugeordnetes Servo. Erst danach folgen zwölf Gelenke, abgestütztes Stehen und die Lauf-Policy.
