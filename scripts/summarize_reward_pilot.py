#!/usr/bin/env python3
"""Summarize physical evaluation metrics, never compare unlike reward totals."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("experiment", type=Path)
a = p.parse_args()
record = json.loads(a.experiment.read_text())
out = a.experiment.parent
results = {"source": record.get("source_results", {})}
results.update({k: v.get("results", {}) for k,v in record["arms"].items()})
if record["status"] != "complete":
    raise SystemExit("Wait for all experiment arms to complete")
names = {"source": "Ausgangspolicy", "control": "Kontrolle",
         "smooth_quarter": "Glättung ×0,25", "smooth_quarter_yaw3x": "Glättung ×0,25\nDrehreward ×3"}
rows = []
for arm, cases in results.items():
    for case, d in cases.items():
        rows.append({"arm": arm, "command": d["command_vx_vy_wz"],
                     "case": case, "mean_vx_vy_wz": d["mean_velocity_vx_vy_wz"],
                     "rmse_vx_vy_wz": d["velocity_rmse_vx_vy_wz"],
                     "fall_resets": d["fall_resets"], "diagnostics": d["diagnostics"]})
report = {"conditions": record["conditions"], "rows": rows,
          "limitations": ["100 updates per arm; one training seed; short warm-start pilot.",
                          "Same 32 nominal evaluation environments, seed 2027, 20 s per command; regular episode resets at 15 s.",
                          "Gaussian clipping probability is evaluated on deterministic evaluation states, not a measured stochastic training clipping fraction."]}
report["relative_to_control"] = {}
for arm in ("smooth_quarter", "smooth_quarter_yaw3x"):
    report["relative_to_control"][arm] = {}
    for case, d in results[arm].items():
        control = results["control"][case]
        report["relative_to_control"][arm][case] = {
            "rmse_change_percent_vx_vy_wz": [100*(v/b-1) if b else None for v,b in zip(d["velocity_rmse_vx_vy_wz"], control["velocity_rmse_vx_vy_wz"])],
            "mean_velocity_change_vx_vy_wz": [v-b for v,b in zip(d["mean_velocity_vx_vy_wz"], control["mean_velocity_vx_vy_wz"])],
            "fall_reset_difference": d["fall_resets"]-control["fall_resets"],
        }
(out / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)
keys = list(results)
x = np.arange(len(keys))
colors = ["#607d8b", "#1565c0", "#00897b", "#e65100"]

def bars(ax, values, title, unit, target=None):
    ax.bar(x, values, color=colors)
    ax.set_xticks(x, [names[k] for k in keys], fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=.2)
    if target is not None:
        ax.axhline(target, color="black", linestyle="--", linewidth=1, label="Sollwert")
        ax.legend(fontsize=8)
    for i, v in enumerate(values):
        text = f"{v:.0f}" if unit.startswith("Anzahl") else f"{v:.3f}"
        ax.annotate(text, (i,v), xytext=(0,3), textcoords="offset points", ha="center", fontsize=8)

bars(axs[0,0], [results[k]["forward_040"]["mean_velocity_vx_vy_wz"][0] for k in keys], "Vorwärtsgeschwindigkeit bei Soll 0,40 m/s", "m/s", .4)
bars(axs[0,1], [results[k]["forward_040"]["velocity_rmse_vx_vy_wz"][2] for k in keys], "Drehfehler beim Geradeauslauf (Soll 0)", "RMSE rad/s · kleiner besser")
bars(axs[1,0], [results[k]["yaw_060"]["mean_velocity_vx_vy_wz"][2] for k in keys], "Drehrate bei Soll +0,60 rad/s", "rad/s", .6)
bars(axs[1,1], [results[k]["yaw_060"]["velocity_rmse_vx_vy_wz"][2] for k in keys], "Drehfehler bei Soll +0,60 rad/s", "RMSE rad/s · kleiner besser")
bars(axs[2,0], [sum(v["fall_resets"] for v in results[k].values()) for k in keys], "Sturzresets über vier Testbefehle", "Anzahl · kleiner besser")
bars(axs[2,1], [np.linalg.norm(results[k]["stand"]["velocity_rmse_vx_vy_wz"][:2]) for k in keys], "Ungewollte Translation bei Stillstand", "RMS Betrag m/s · kleiner besser")
bars(axs[3,0], [results[k]["stand"]["velocity_rmse_vx_vy_wz"][2] for k in keys], "Ungewollte Drehbewegung bei Stillstand", "RMSE rad/s · kleiner besser")
bars(axs[3,1], [results[k]["forward_040"]["diagnostics"]["deterministic_action_clip_fraction"] for k in keys], "Abgeschnittene Aktionen bei Soll 0,40 m/s", "Anteil der Gelenkaktionen")
fig.suptitle("Reward-Gewichtung: kontrollierter Nachtrainings-Pilot\nJe 100 Updates / 6,55 Mio. Übergänge · gleiche Ausgangspolicy und PPO-Einstellungen", fontsize=14)
fig.savefig(out / "comparison.png", dpi=150)
fig.savefig(out / "comparison.pdf")
print(json.dumps(rows, indent=2))
