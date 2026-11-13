# Quellen und bestehende Lizenzen

## Druckteile

**Spotmicro – robot dog**, KDY0523: https://www.thingiverse.com/thing:3445283

Die übernommenen Mesh-Lizenzdateien nennen Creative Commons Attribution 3.0:
- `assets/spot_micro_original/stl/LICENSE.txt`
- `assets/spot_micro/stl/LICENSE.txt`

Die beiden Dateien unter `hardware/modified/` enthalten die vom Projektteam angegebenen Änderungen an Befestigungs- und Anschlusslöchern. Sie sind Bearbeitungen des genannten Designs, keine unabhängig entworfenen Originalteile. Die Zuordnung und Änderungen stehen in `hardware/README.md`.

## Simulationsmodell

Repository: https://github.com/OpenQuadruped/spot_mini_mini

- `assets/spot_micro_original/`: Commit `bf91ab37620cbb036968952da92773522361aad7`; aktives Modell.
- `assets/spot_micro/`: Commit `c7e4905be176c63fa0e68a09c177b937e916fa60`; älteres Modell für historische Experimente.

Die MIT-Lizenz von Maurice Rahme bleibt in beiden Verzeichnissen erhalten. Die STL-Dateien haben separate CC-BY-Hinweise. Die Manifestdateien dokumentieren die importierten Dateien und Anpassungen, darunter Pfadauflösung, Kollisionsgeometrie und geschätzte Massen/Trägheiten. Die Bestätigung der Druckvorlage ist keine unabhängige Prüfung der geometrischen Identität jedes importierten Meshes.

## Laufzeit und Training

Isaac Sim, Isaac Lab, RSL-RL, PyTorch, NumPy, ONNX Runtime und smbus2 werden als externe Abhängigkeiten verwendet. Sie werden nicht mit diesem Repository neu lizenziert. Maßgeblich sind ihre jeweiligen Lizenzbedingungen.

Für die eigenen Softwareanteile dieses Projekts wurde noch keine Open-Source-Lizenz festgelegt. Die MIT-Lizenz der importierten Assets darf nicht als Lizenz für das gesamte Repository verstanden werden.
