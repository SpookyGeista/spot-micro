#!/usr/bin/env python3
"""Summarize reproducible locomotion diagnostics and plot the observed gait."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path)
args = parser.parse_args()
folder = args.folder
summary = json.loads((folder/"summary.json").read_text())
calm_trace = json.loads((folder/"calm_040_trace.json").read_text())
fast_trace = json.loads((folder/"fast_040_trace.json").read_text())
zero = json.loads((folder/"zero_action_stand.json").read_text())

for label, case in summary["cases"].items():
    trace = json.loads((folder/f"{label}_trace.json").read_text())[100:]
    positions = np.array([x["positions"] for x in trace])
    frequency = np.fft.rfftfreq(len(trace), .02)
    power = abs(np.fft.rfft(positions-positions.mean(0), axis=0))**2
    power[0] = 0
    case["env0_joint_dominant_frequency_hz"] = frequency[power.argmax(0)].tolist()
    case["env0_trace_contains_resets"] = any(x["reset"] for x in trace)
    case["frequency_note"] = "One-robot spectrum after 2 s settling; interpret only reset-free traces."

rates = {}
for name in ("2026-11-11_14-52-14_walk_calm_trot_style", "2026-11-11_16-05-20_walk_speed_2mps_continue"):
    events = EventAccumulator(str(ROOT/"logs"/name), size_guidance={"scalars":0})
    events.Reload()
    values = np.array([x.value for x in events.Scalars("Loss/learning_rate")])
    rates[name] = {"updates":len(values), "min":float(values.min()), "max":float(values.max()),
                   "fraction_at_floor":float((values <= 1.00001e-5).mean())}
zero_trace = [x for x in zero["trace_env_0"] if x["time_s"] >= 2.]
zero_q = np.array([x["joint_positions_rad"] for x in zero_trace])
zero_v = np.array([x["measured_velocity"] for x in zero_trace])
facts = {
    "scope": "Code audit plus six fixed-policy tests, 32 nominal environments for 10 s each; additional zero-action stand test. No training or hardware actuation.",
    "learning_rate_history":rates,
    "zero_action_stand": {"fall_resets":zero["fall_resets"],
                          "env0_max_joint_range_after_2s_rad":float(np.ptp(zero_q,axis=0).max()),
                          "env0_velocity_rms_after_2s":np.sqrt((zero_v**2).mean(0)).tolist()},
    "tracking_reward_at_actual_vx_040":{str(v):8*math.exp(-((v-.4)/.15)**2) for v in (.4,.8,1.2,2.)},
    "findings": [
        {"priority":1,"file":"spot_micro/bounded_policy.py","line":22,
         "finding":"Warm-start mapping retains raw actor outputs in the tens/hundreds and maps them to latent means near +/-4. On fresh calm-policy states the median d(latent mean)/d(raw actor output) is about 3e-5; 81% of desired-angle exploration standard deviations are below the assumed servo deadband. Finite gradients and bounded latent std do not establish effective exploration.",
         "confidence":"Measured conditioning and noise collapse; exact share of training failure requires a retrained ablation."},
        {"priority":1,"file":"spot_micro/env.py","line":179,
         "finding":"Linear tracking width remains 0.15 m/s while command maximum rises to 2 m/s. Reward is nearly zero far below target, so improved speed can remain effectively unrewarded. Curriculum increases solely with transition count, without a performance gate.",
         "confidence":"Verified in code and by numerical calculation."},
        {"priority":1,"file":"spot_micro/control.py","line":53,
         "finding":"The action path imposes 2.5 rad/s target slew and 40 ms smoothing. About two thirds of commands hit the slew limiter. Actual gait is a small-amplitude 3.75-4.25 Hz oscillation, despite the 1 Hz reference. Simply raising the limiter on the frozen fast policy reduces speed, so this is not a stand-alone fix.",
         "confidence":"Measured; 7 rad/s sensitivity test is not a retraining experiment or hardware limit measurement."},
        {"priority":2,"file":"spot_micro/contract.py","line":8,
         "finding":"Actor observes commands, angular velocity, estimated gravity, target history and phase. It lacks actual joint position/velocity, base velocity and raw acceleration. Only the critic sees true state; that is asymmetric actor-critic training, not a teacher-student distillation pipeline.",
         "confidence":"Verified architecture; contribution to failure not yet isolated by a privileged-actor training control."},
        {"priority":2,"file":"scripts/play.py","line":89,
         "finding":"Evaluation does not restore style curriculum progress. For 32 environments and 1000 steps, the new style terms average only about 0.16% of full scale. Raw gait metrics and trajectories remain valid because evaluation is fixed-policy; reported weighted style rewards are not representative of full training weights.",
         "confidence":"Verified; diagnostic runner explicitly uses fully active style weights."},
        {"priority":2,"file":"spot_micro/control.py","line":64,
         "finding":"Servo gain, damping, armature, deadband and delays are provisional. There is no identified acceleration envelope. Imported masses/materials/joint ordering are internally consistent, but physical fidelity is unvalidated.",
         "confidence":"Inspected actual PhysX parameters and verified stable constant-target stand; hardware measurements still absent."}
    ],
    "recommended_order": [
        "Establish a clean privileged-state teacher baseline with a conventional learnable action distribution and fresh actor initialization; keep existing working checkpoint as reference.",
        "Use a tracking objective with useful signal across the command range and a performance-gated speed curriculum; measure actuator excitation, saturation and contact timing.",
        "Compare target slew/filter settings in separately trained controls within the same torque-speed model.",
        "Only after the teacher walks well, train and evaluate a student/state estimator with the actual Pi sensor contract; do not assume full-state performance transfers without feedback."
    ],
}
(folder/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(folder/"findings.json").write_text(json.dumps(facts,indent=2)+"\n")

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(1, 2, figsize=(12,4.2), constrained_layout=True)
for trace,label,color in [(calm_trace,"Bisherige Policy · 0,39 m/s","#2166ac"),
                           (fast_trace,"Nach 2-m/s-Training · 0,30 m/s","#d6604d")]:
    sample=[x for x in trace if 3 <= x["time_s"] <= 5]
    t=np.array([x["time_s"] for x in sample]);q=np.array([x["positions"][1] for x in sample])
    axes[0].plot(t-3,np.degrees(q-q.mean()),label=label,color=color,lw=1.7)
axes[0].set(title="Gemessene kurze Beinbewegungen",xlabel="Zeit im Ausschnitt [s]",ylabel="Oberes Beingelenk, relativ zum Mittel [°]")
axes[0].legend(fontsize=9,loc="upper right")
axes[0].grid(alpha=.2)
x=np.linspace(0,2.1,800)
for command,color in [(.4,"#2166ac"),(.8,"#7b3294"),(1.2,"#e08214"),(2.,"#d6604d")]:
    axes[1].plot(x,8*np.exp(-((x-command)/.15)**2),label=f"Soll {command:g} m/s",color=color)
axes[1].axvspan(.3,.5,color="#999999",alpha=.2,label="Bisher erreichter Bereich")
axes[1].set(title="Tracking-Reward verliert das Lernsignal",xlabel="Tatsächliche Vorwärtsgeschwindigkeit [m/s]",ylabel="Reward vor Zeitschritt-Skalierung",ylim=(0,8.5))
axes[1].legend(fontsize=9)
axes[1].grid(alpha=.2)
fig.savefig(folder/"diagnosis.png",dpi=180)
print(json.dumps({"summary":str(folder/"summary.json"),"findings":str(folder/"findings.json"),"figure":str(folder/"diagnosis.png")}))
