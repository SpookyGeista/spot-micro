#!/usr/bin/env python3
"""One-shot continuation of a running rough-ground training job.

Wait for this exact training process, evaluate its teacher, and only transfer a
teacher that retains nominal command control. All candidates remain separate
from work/recommended_policy.json. This is not a recurring scheduler.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--training-pid',type=int,required=True)
p.add_argument('--output',type=Path,default=ROOT/'outputs/rough_policy_v1')
p.add_argument('--student-rounds',type=int,default=150)
p.add_argument('--wait-timeout-s',type=float,default=7200.)
a=p.parse_args()
a.run=a.run.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
status_path=a.output/'pipeline_status.json'
status={'pid':os.getpid(),'teacher_run':str(a.run),'training_pid':a.training_pid,
        'started':datetime.now().isoformat(),'stages':[], 'recommended_policy_replaced':False}


def save(phase,**extra):
    status.update(phase=phase,updated=datetime.now().isoformat(),**extra)
    tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
    active_path=ROOT/'work/active_training.json'
    if phase!='waiting_for_teacher' and active_path.is_file():
        active=json.loads(active_path.read_text())
        if active.get('pid')==a.training_pid:
            active.update(status='Teacher finished; rough pipeline: '+phase,running=False,
                          pipeline_status=str(status_path),pipeline_pid=os.getpid())
            temporary=active_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(active,indent=2)+'\n');temporary.replace(active_path)


def run_stage(label,script,arguments,artifact):
    command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,arguments)]
    record={'name':label,'command':command,'started':datetime.now().isoformat()}
    status['stages'].append(record)
    with (a.output/(label+'_launcher.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save(label,child_pid=process.pid)
        code=process.wait()
    record.update(exit_code=code,finished=datetime.now().isoformat())
    if code or not artifact.is_file():
        raise RuntimeError(f'{label} did not produce {artifact}; see its log')
    save(label+'_complete',child_pid=None)


def evaluate(label,checkpoint,extra):
    folder=a.output/label
    run_stage(label,'evaluate_dynamic.py',[
        '--checkpoint',checkpoint,'--environment-profile',ROOT/'configs/spotmicro_teacher_rough.json',
        '--output',folder,'--headless','--num-envs','32','--steps','500',*extra],folder/'summary.json')
    report=json.loads((folder/'summary.json').read_text())
    def values(flag):
        if flag not in extra:
            return []
        result=[]
        for item in extra[extra.index(flag)+1:]:
            if str(item).startswith('--'):
                break
            result.append(float(item))
        return result
    expected={f'forward_{v:.2f}' for v in values('--speeds')}
    expected.update(f'kick_{v:.2f}bw' for v in values('--kick-body-weights'))
    if '--yaw' in extra:
        expected.update(('yaw_positive','yaw_negative'))
    if set(report['cases'])!=expected or 'Traceback (most recent call last)' in (folder/'console.log').read_text():
        raise RuntimeError(f'{label} did not finish every requested evaluation case')
    return report


def compact(report):
    return {k:{'mean_vx':v['steady_mean_velocity'][0],'fall_resets':v['fall_resets'],
               'terrain_boundary_resets':v.get('terrain_boundary_resets',0),
               'gait':v['gait']} for k,v in report['cases'].items()}


def nominal_pass(report):
    cases=report['cases']
    return (all(c['fall_resets']==0 for c in cases.values())
            and cases['forward_0.40']['steady_mean_velocity'][0]>=.34
            and cases['forward_0.60']['steady_mean_velocity'][0]>=.51
            and abs(cases['forward_0.00']['steady_mean_velocity'][0])<.03
            and abs(cases['forward_0.00']['steady_mean_velocity'][1])<.03
            and abs(cases['yaw_positive']['steady_mean_velocity'][2]-.6)<.15
            and abs(cases['yaw_negative']['steady_mean_velocity'][2]+.6)<.15)


def rough_assessment(report, baseline):
    """Prefer fewer falls and sustained forward progress over command accuracy.

    This admits a teacher for student training, not deployment. Boundary resets
    remain visible in the report and are not treated as proof of stability.
    """
    falls=sum(c['fall_resets'] for c in report['cases'].values())
    baseline_falls=sum(c['fall_resets'] for c in baseline['cases'].values())
    progress=all(c['steady_mean_velocity'][0]>=.10 for c in report['cases'].values())
    return {'passed':falls<=.75*baseline_falls and progress,
            'fall_resets':falls,'baseline_fall_resets':baseline_falls,
            'minimum_forward_speed_m_s':.10,'forward_progress_passed':progress,
            'maximum_baseline_fall_ratio':.75,
            'terrain_boundary_resets':sum(c.get('terrain_boundary_resets',0) for c in report['cases'].values())}


def main():
    save('waiting_for_teacher')
    start=time.monotonic()
    checkpoint=a.run/'model_final.pt'
    while True:
        proc=Path(f'/proc/{a.training_pid}/cmdline')
        command=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
        if 'scripts/train.py' not in command:
            break
        if time.monotonic()-start>a.wait_timeout_s:
            raise TimeoutError('Training wait timed out')
        time.sleep(10)
    if not checkpoint.exists() or not (a.run/'export/verification.json').exists():
        raise RuntimeError('Teacher process ended without a finished checkpoint and verified export')
    nominal_args=['--flat','--continuous','--speeds','0','.4','.6','--yaw','--seed','2050']
    rough_args=['--terrain-level','2','--randomize','--push-strength','.6','--speeds','.4','.6','--seed','2051']
    teacher_nominal=evaluate('teacher_nominal',checkpoint,nominal_args)
    teacher_rough=evaluate('teacher_rough_level2',checkpoint,rough_args)
    baseline=ROOT/'logs/2026-11-12_00-28-09_sensor_student_dagger/model_final.pt'
    baseline_rough=evaluate('baseline_rough_level2',baseline,rough_args)
    save('teacher_evaluated',teacher_nominal=compact(teacher_nominal),teacher_rough=compact(teacher_rough),baseline_rough=compact(baseline_rough))
    if not nominal_pass(teacher_nominal):
        save('teacher_nominal_regression',result='Teacher retained separately; sensor transfer skipped because nominal command control regressed')
        return
    assessment=rough_assessment(teacher_rough,baseline_rough)
    save('teacher_rough_assessed',rough_assessment=assessment)
    if not assessment['passed']:
        save('teacher_rough_improvement_insufficient',result='Sensor transfer skipped: teacher did not sufficiently improve the same rough-ground test')
        return
    student_log=a.output/'student_distillation.log'
    run_stage('student_distillation','distill_student.py',[
        '--teacher',checkpoint,'--headless','--num-envs','256','--rounds',a.student_rounds,
        '--updates-per-round','64','--history-length','20','--observe-acceleration',
        '--max-speed','.6','--randomize','--seed','2052','--log-file',student_log],student_log)
    match=re.findall(r'\[SpotMicro\] Student done: (.+)',student_log.read_text())
    if not match:
        raise RuntimeError('Distillation has no completed checkpoint marker')
    student=Path(match[-1].strip())
    if not student.is_file():
        raise RuntimeError('Missing student checkpoint')
    save('student_trained',student_checkpoint=str(student))
    student_nominal=evaluate('student_nominal',student,nominal_args)
    student_rough=evaluate('student_rough_level2',student,rough_args)
    student_pushes=evaluate('student_pushes',student,[
        '--flat','--speeds','.4','--kick-body-weights','.25','.75','1.5','--kick-duration','.12','--seed','2053'])
    candidate={'checkpoint':str(student),'hardware_validated':False,
               'rough_assessment':rough_assessment(student_rough,baseline_rough),
               'nominal_passed':nominal_pass(student_nominal),'nominal':compact(student_nominal),
               'rough':compact(student_rough),'baseline_rough':compact(baseline_rough),'pushes':compact(student_pushes),
               'recommended_policy_replaced':False}
    validation=a.output/'candidate_validation.json';validation.write_text(json.dumps(candidate,indent=2)+'\n')
    run_stage('student_package','package_pi.py',[student.parent/'export','--output',a.output/'rough_student_candidate_pi.zip','--validation',validation],a.output/'rough_student_candidate_pi.zip')
    flat_robot=json.loads((student.parent/'robot.json').read_text());flat_robot.pop('terrain',None)
    flat_profile=a.output/'video_flat_profile.json';flat_profile.write_text(json.dumps(flat_robot,indent=2)+'\n')
    run_stage('student_video','play.py',[
        '--checkpoint',student,'--robot',flat_profile,'--headless','--num-envs','1','--steps','500',
        '--command','.6','0','0','--follow-camera','--video',a.output/'student_060.mp4',
        '--output',a.output/'student_video_metrics.json','--log-file',a.output/'student_video.log'],a.output/'student_video_metrics.json')
    save('complete_candidate_evaluated',candidate_validation=str(validation),nominal_passed=candidate['nominal_passed'])


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save('failed',error=str(error))
        raise
