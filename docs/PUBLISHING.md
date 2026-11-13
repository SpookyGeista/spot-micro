# Repository veröffentlichen

Der Ordner enthält Quellcode, dokumentierte Drittanbieter-Assets, die beiden angepassten STL-Dateien, Demo-Medien und zwei ausgewählte Checkpoints. Das öffentliche Repository liegt unter https://github.com/SpookyGeista/spot-micro. Der Remote-Name ist `origin`.

## Änderungen veröffentlichen

- Eigene Softwarelizenz festlegen, falls eine Open-Source-Freigabe gewünscht ist. Bestehende Asset-Lizenzen nicht ersetzen.
- `README.md`, `THIRD_PARTY_NOTICES.md` und `hardware/README.md` gegenlesen.
- `git status --short` prüfen. `work/`, `logs/`, `outputs/`, Zugangsschlüssel und `configs/hardware.json` bleiben lokal.
- `bash scripts/run.sh test` in der eingerichteten Trainingsumgebung ausführen.

Im Repository-Ordner:

```bash
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Describe the project changes"
git push origin main
```

Die kleinen Demo-Checkpoints und Medien liegen direkt im Repository; Git LFS ist für diesen Stand nicht erforderlich. Künftige große Trainingsläufe gehören in Releases oder einen separaten Datenspeicher, nicht in die Git-Historie. Die README verwendet eine GIF-Vorschau und relative MP4-Links, weil lokale MP4-Dateien in GitHub-Markdown nicht zuverlässig als eingebetteter Player erscheinen.

`policies/rough_student/SHA256SUMS` dokumentiert die ausgelieferten Modell- und Konfigurationsdateien. Checkpoints nur aus vertrauenswürdigen Quellen laden.
