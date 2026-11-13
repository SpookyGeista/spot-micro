#!/usr/bin/env python3
"""Compact actual progress from a training log and its TensorBoard events."""
import argparse
import json
from pathlib import Path
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--log',type=Path)
p.add_argument('--run',type=Path)
a=p.parse_args()
if a.log:
    text=a.log.read_text(errors='replace')
    matches=re.findall(r'\[SpotMicro\] Training run: (.+)',text)
    if not matches:
        print(json.dumps({'status':'initializing','last_log_lines':text.splitlines()[-4:]}))
        raise SystemExit()
    a.run=Path(matches[-1].strip())
if not a.run:
    p.error('Provide --run or --log')
events=EventAccumulator(str(a.run),size_guidance={'scalars':0}).Reload()
report={'run_dir':str(a.run),'latest':{}}
for tag in events.Tags()['scalars']:
    if tag in ('Loss/learning_rate','Policy/mean_noise_std','Train/mean_reward','Train/mean_episode_length') or tag.startswith(('Curriculum/','Episode_Reward/','Episode_Termination/','Episode_Control/')):
        values=events.Scalars(tag)
        report['latest'][tag]={'iteration':values[-1].step,'value':values[-1].value,
                               'mean_last_25':sum(x.value for x in values[-25:])/len(values[-25:])}
if a.log:
    report['finished']='[SpotMicro] Done.' in text
    errors=re.findall(r'(?:RuntimeError|ValueError|Traceback)[^\n]*',text)
    if errors:
        report['errors']=errors[-3:]
print(json.dumps(report,indent=2))
