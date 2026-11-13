#!/usr/bin/env python3
"""Plot measured speed, foot travel and support patterns from saved rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--case',default='forward_0.60')
p.add_argument('--label',default='Neue Policy')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
folders=[a.baseline,a.candidate]
labels=['Bisherige Policy',a.label]
colors=['#667085','#007f70']
reports=[json.loads((d/'summary.json').read_text()) for d in folders]
metrics=[r['cases'][a.case] for r in reports]
data=[np.load(d/(a.case+'_trajectories.npz')) for d in folders]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for d,m,label,color in zip(data,metrics,labels,colors):
    t=np.arange(len(d['velocity']))*.02
    axes[0,0].plot(t,d['velocity'][:,:,0].mean(1),color=color,label=f'{label}: {m["mean_velocity"][0]:.2f} m/s; {m["fall_resets"]} Stürze',lw=1.5)
axes[0,0].axhline(metrics[0]['command'][0],color='#111827',ls='--',lw=1,label='Sollgeschwindigkeit')
axes[0,0].set(xlabel='Zeit [s]',ylabel='Vorwärtsgeschwindigkeit [m/s]',title='Gemessene Geschwindigkeit')
axes[0,0].legend(fontsize=8)
legs=['Vorn links','Vorn rechts','Hinten links','Hinten rechts']
for i,(m,label,color) in enumerate(zip(metrics,labels,colors)):
    axes[0,1].bar(np.arange(4)+(i-.5)*.36,np.array(m['gait']['foot_fore_aft_span_p95_minus_p05_m_by_leg'])*100,width=.36,color=color,label=label)
axes[0,1].set(xticks=np.arange(4),xticklabels=['VL','VR','HL','HR'],ylabel='Fußbewegung relativ zum Rumpf [cm]',title='Vorwärts-/Rückwärtsbewegung (P95–P05)')
axes[0,1].legend(fontsize=8)
for ax,d,label,color in zip(axes[1],data,labels,colors):
    ax.imshow(d['foot_contact'][100:200,0].T,aspect='auto',interpolation='nearest',
              extent=(2,4,3.5,-.5),cmap=ListedColormap(['#f2f4f7',color]),vmin=0,vmax=1)
    ax.set(yticks=np.arange(4),yticklabels=legs,xlabel='Zeit [s]',title=label+' – Bodenkontakt (farbig)')
fig.suptitle(f'SpotMicro: Vergleich bei {metrics[0]["command"][0]:.1f} m/s Soll – {reports[0]["num_envs"]} Simulationen',fontsize=14)
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=180)
plt.close(fig)
