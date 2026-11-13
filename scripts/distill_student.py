#!/usr/bin/env python3
"""DAgger transfer of a validated full-state teacher to the sensor-only actor.

The teacher labels states visited by the student. The student receives only
the deployment observation, and never simulator joints, velocity or contacts.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--teacher',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--rounds',type=int,default=100)
p.add_argument('--steps-per-round',type=int,default=64)
p.add_argument('--updates-per-round',type=int,default=32)
p.add_argument('--max-speed',type=float)
p.add_argument('--history-length',type=int)
p.add_argument('--observe-acceleration',action='store_true')
p.add_argument('--randomize',action='store_true')
p.add_argument('--smoke',action='store_true')
p.add_argument('--seed',type=int,default=2031)
p.add_argument('--log-file',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.smoke:
    a.num_envs=16;a.rounds=2;a.steps_per_round=16;a.updates_per_round=2
a.log_file.parent.mkdir(parents=True,exist_ok=True)
console=a.log_file.open('w',buffering=1)
os.dup2(console.fileno(),1);os.dup2(console.fileno(),2)
app=AppLauncher(a).app

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_micro.bounded_policy import DirectSquashedActorCritic
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg
from spot_micro.export import export_actor
from spot_micro.distillation import sensor_projection,initialize_student,teacher_observations
from spot_micro.runner import restore_robustness_state


def main():
    register()
    torch.manual_seed(a.seed)
    teacher_path=a.teacher.resolve()
    teacher_checkpoint=torch.load(teacher_path,map_location='cpu',weights_only=False)
    teacher_robot=json.loads((teacher_path.parent/'robot.json').read_text())
    robot=copy.deepcopy(teacher_robot)
    if robot.get('policy_role')!='teacher':
        raise ValueError('Expected a privileged teacher checkpoint')
    learned_limit=(teacher_checkpoint.get('infos') or {}).get('command_curriculum_state',{}).get('maximum',.4)
    max_speed=a.max_speed if a.max_speed is not None else learned_limit
    robot['policy_role']='student'
    if a.history_length is not None:
        robot['history_length']=a.history_length
    if a.observe_acceleration:
        # Matches the existing MPU driver: ACCEL_CONFIG=0x08, +/-4 g.
        robot.update(observe_acceleration=True,accel_scale=.1,accel_clip_m_s2=4*9.81)
    robot['description']='Sensor-only DAgger student of '+str(teacher_path)+'; requires independent rollout validation and hardware calibration.'
    robot.pop('performance_curriculum',None)
    for key in ('initial_vx_vy_wz','final_vx_vy_wz'):
        robot['command_curriculum'][key][0][1]=max_speed
    robot['teleop_speeds']['forward_m_s']=min(.6,max_speed)
    robot['teleop_speeds']['yaw_rad_s']=min(.6,robot['command_curriculum']['final_vx_vy_wz'][2][1])
    out=ROOT/'logs'/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_sensor_student_dagger'+('_smoke' if a.smoke else ''))
    out.mkdir(parents=True)
    (out/'robot.json').write_text(json.dumps(robot,indent=2)+'\n')
    cfg=make_env_cfg(out/'robot.json',a.num_envs,'cuda:0',a.seed,randomize=a.randomize)
    env=gym.make('SpotMicro-Velocity-Flat-v0',cfg=cfg)
    wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
    raw=env.unwrapped
    restore_robustness_state(raw,teacher_checkpoint.get('infos') or {})
    raw.terrain_curriculum_enabled=False
    ppo=json.loads((teacher_path.parent/'ppo.json').read_text())
    observations,_=wrapper.reset()
    indices=sensor_projection(teacher_robot,robot)
    teacher_kwargs=copy.deepcopy(ppo['policy'])
    if teacher_kwargs.pop('class_name')!='DirectSquashedActorCritic':
        raise ValueError('DAgger expects a direct-action teacher')
    teacher=DirectSquashedActorCritic(teacher_observations(observations,indices),ppo['obs_groups'],12,**teacher_kwargs).to('cuda:0')
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    ppo['obs_groups']={'policy':['policy'],'critic':['critic']}
    ppo['seed']=a.seed
    ppo['policy'].update(init_noise_std=.25,min_noise_std=.05,max_noise_std=.5)
    kwargs=copy.deepcopy(ppo['policy']);kwargs.pop('class_name')
    student=DirectSquashedActorCritic(observations,ppo['obs_groups'],12,**kwargs).to('cuda:0')
    n_obs=observations['policy'].shape[-1]
    # Initialize common features and all later layers; missing privileged inputs
    # are zero in the teacher's normalized coordinates, not fabricated sensors.
    indices=initialize_student(student,teacher,teacher_robot,robot)
    student.train()
    optimizer=torch.optim.Adam(student.actor.parameters(),lr=3e-4)
    capacity=262144
    replay_x=torch.empty(capacity,n_obs,device='cuda:0')
    replay_y=torch.empty(capacity,12,device='cuda:0')
    cursor=size=0
    history=[]
    train_envs=max(1,round(a.num_envs*.75))
    raw.style_curriculum_offset=100_000_000
    raw.curriculum_offset=100_000_000
    (out/'ppo.json').write_text(json.dumps(ppo,indent=2)+'\n')
    run={'policy_role':'student','teacher':str(teacher_path),'training_method':'DAgger',
         'smoke_test':a.smoke,
         'sensor_observation_size':n_obs,'simulator_inputs_in_actor':False,
         'seed':a.seed,'num_envs':a.num_envs,'randomization':a.randomize,
         'max_forward_command_m_s':max_speed,'hardware_validated':False}
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    for source in [Path(__file__),*sorted((ROOT/'spot_micro').glob('*.py'))]:
        dest=out/'source'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    print('[SpotMicro] Student run:',out,flush=True)

    def save(name):
        # The supervised optimizer is separate from the fresh PPO optimizer.
        ppo_optimizer=torch.optim.Adam(student.parameters(),lr=ppo['algorithm']['learning_rate'])
        infos={'curriculum_transitions':100_000_000,'style_curriculum_transitions':100_000_000,
               'teacher_checkpoint':str(teacher_path),'dagger_rounds':len(history)}
        if raw.robustness_curriculum:
            infos['robustness_curriculum_state']=raw.robustness_curriculum.state_dict()
            infos['robustness_transitions']=raw.robustness_offset+raw.common_step_counter*raw.num_envs
        if raw.terrain.terrain_origins is not None:
            infos['terrain_levels']=raw.terrain.terrain_levels.cpu().tolist()
        torch.save({'model_state_dict':student.state_dict(),'optimizer_state_dict':ppo_optimizer.state_dict(),
                    'iter':0,'infos':infos},out/name)
        (out/'distillation.json').write_text(json.dumps({'run':run,'rounds':history},indent=2)+'\n')

    try:
        for iteration in range(a.rounds):
            beta=max(0.,1.-iteration/max(.65*a.rounds,1.))
            falls=0
            xs=[];ys=[];hold_x=[];hold_y=[]
            with torch.no_grad():
                for step in range(a.steps_per_round):
                    target=teacher.act_inference(teacher_observations(observations,indices))
                    action=student.act_inference(observations)
                    xs.append(observations['policy'][:train_envs].clone());ys.append(target[:train_envs].clone())
                    if train_envs<a.num_envs:
                        hold_x.append(observations['policy'][train_envs:].clone());hold_y.append(target[train_envs:].clone())
                    teacher_mask=torch.rand(a.num_envs,1,device='cuda:0')<beta
                    observations,_,_,_=wrapper.step(torch.where(teacher_mask,target,action))
                    falls+=int(raw.reset_terminated.sum())
            x=torch.cat(xs);y=torch.cat(ys);count=len(x)
            if count>capacity:
                raise ValueError('Rollout exceeds replay capacity')
            first=min(count,capacity-cursor)
            replay_x[cursor:cursor+first]=x[:first];replay_y[cursor:cursor+first]=y[:first]
            if first<count:
                replay_x[:count-first]=x[first:];replay_y[:count-first]=y[first:]
            cursor=(cursor+count)%capacity;size=min(capacity,size+count)
            losses=[]
            for _ in range(a.updates_per_round):
                ids=torch.randint(size,(2048,),device='cuda:0')
                prediction=student.action_postprocessor(student.actor(student.actor_obs_normalizer(replay_x[ids])))
                loss=(prediction-replay_y[ids]).square().mean()
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach()))
            with torch.no_grad():
                hx=torch.cat(hold_x) if hold_x else x
                hy=torch.cat(hold_y) if hold_y else y
                prediction=student.action_postprocessor(student.actor(student.actor_obs_normalizer(hx)))
                hold_rmse=float((prediction-hy).square().mean().sqrt())
            result={'round':iteration,'teacher_probability':beta,'train_mse':float(np.mean(losses)),
                    'heldout_environment_action_rmse':hold_rmse,'fall_resets':falls,
                    'environment_steps_per_robot':(iteration+1)*a.steps_per_round}
            history.append(result)
            print('DAGGER',json.dumps(result),flush=True)
            if (iteration+1)%20==0:
                save(f'model_round_{iteration+1}.pt')
        save('model_final.pt')
        export_actor(student,robot,out/'export',run)
        print('[SpotMicro] Student done:',out/'model_final.pt',flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
