#!/usr/bin/env python3
"""Extract and plot existing TensorBoard scalars without running the simulator."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
ea = EventAccumulator(str(args.run), size_guidance={"scalars": 0}).Reload()
series = {k: ea.Scalars(k) for k in ea.Tags()["scalars"] if not k.endswith("/time")}
summary = {}
for tag, events in series.items():
    y = np.array([e.value for e in events])
    summary[tag] = {"count": len(y), "first_100_mean": float(y[:100].mean()),
                    "last_100_mean": float(y[-100:].mean()),
                    "last_500_mean": float(y[-500:].mean()),
                    "last": float(y[-1]), "min": float(y.min()), "max": float(y.max())}
with (args.out / "scalars.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "iteration", "wall_time", "value"])
    for tag, events in series.items():
        writer.writerows((tag, e.step, e.wall_time, e.value) for e in events)
summary["notes"] = [
    "Episode_Reward values already include weights and are episode-integrated sums divided by fixed 15 s, including shortened episodes.",
    "These are stochastic training rollouts under changing commands and domain randomization, not deterministic evaluation errors.",
    "Episode_Termination/fallen is averaged from reset-batch fractions, not an exact aggregate fall count.",
    "Smoothing is a trailing 100-update average; raw data are preserved in scalars.csv.",
]
(args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
fig.suptitle("SpotMicro: Reward-Analyse des letzten Trainings\n2048 Envs · 3000 PPO-Updates · 196,6 Mio. Übergänge", fontsize=16)
curriculum_end = 100_000_000 / (2048 * 32)

def plot(ax, tag, label, color, factor=1):
    events = series[tag]
    x = np.array([e.step for e in events])
    y = np.array([e.value for e in events]) * factor
    ax.plot(x, y, color=color, alpha=.13, linewidth=.6)
    width = 100
    smooth = np.convolve(y, np.ones(width) / width, mode="valid")
    ax.plot(x[width-1:], smooth, color=color, label=label, linewidth=1.7)

plot(axs[0,0], "Train/mean_reward", "Episodenreturn", "#1565c0")
axs[0,0].set_title("Gesamtreward steigt, zuletzt weitgehend Plateau")
axs[0,0].set_ylabel("Return (anderes Maß als Einzelterme)")
plot(axs[0,1], "Episode_Reward/track_velocity", "Geschwindigkeit (Maximum 8)", "#1565c0")
plot(axs[0,1], "Episode_Reward/track_yaw", "Drehrate (Maximum 1,4)", "#e65100")
axs[0,1].set_title("Gewichtete Tracking-Beiträge")
axs[0,1].set_ylabel("Geloggter Reward-Beitrag")
plot(axs[1,0], "Episode_Reward/track_velocity", "Geschwindigkeit / 8", "#1565c0", 1/8)
plot(axs[1,0], "Episode_Reward/track_yaw", "Drehrate / 1,4", "#e65100", 1/1.4)
axs[1,0].set_title("Tracking relativ zum jeweiligen Maximum")
axs[1,0].set_ylabel("Normierter Beitrag (kein Tracking-Erfolg in %)")
for key, label, color in [("smoothness", "Glättung", "#c62828"), ("torque", "Drehmoment", "#6a1b9a"), ("posture", "Haltung", "#00897b"), ("vertical_motion", "Vertikal/Roll/Pitch", "#795548"), ("termination", "Sturzstrafe", "#455a64")]:
    plot(axs[1,1], "Episode_Reward/"+key, label, color)
axs[1,1].set_title("Glättung dominiert die Strafen")
for key, label, color in [("upright", "Aufrecht", "#00897b"), ("height", "Körperhöhe", "#6a1b9a"), ("contact_schedule", "Kontaktmuster", "#795548")]:
    plot(axs[2,0], "Episode_Reward/"+key, label, color)
axs[2,0].set_title("Weitere positive Beiträge")
plot(axs[2,1], "Policy/mean_noise_std", "Mittlere Aktions-Standardabweichung", "#c62828")
axs[2,1].axhline(.35, color="gray", linestyle=":", linewidth=1, label="Initialwert 0,35")
axs[2,1].set_title("Aktionsrauschen wächst spät stark")
plot(axs[3,0], "Train/mean_episode_length", "Mittlere Episodenlänge", "#1565c0", .02)
axs[3,0].axhline(15, color="gray", linestyle=":", linewidth=1)
axs[3,0].set_ylabel("Sekunden")
axs[3,0].set_title("Überleben verbessert sich")
plot(axs[3,1], "Episode_Termination/fallen", "Geloggter Sturzanteil je Reset-Batch", "#c62828")
axs[3,1].set_title("Stürze bleiben unter Trainingsbedingungen vorhanden")
for ax in axs.flat:
    ax.axvline(curriculum_end, color="gray", linestyle="--", linewidth=.9)
    ax.set_xlim(0, 3000)
    ax.set_xlabel("PPO-Update · gestrichelt: Ende Befehls-Curriculum")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, loc="best")
fig.savefig(args.out / "reward_curves.png", dpi=150)
fig.savefig(args.out / "reward_curves.pdf")
print(json.dumps({"output": str(args.out.resolve()), "last100": {k:v["last_100_mean"] for k,v in summary.items() if k != "notes"}}, indent=2))
