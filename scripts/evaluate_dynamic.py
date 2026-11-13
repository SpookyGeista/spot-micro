#!/usr/bin/env python3
"""Evaluate commanded speed, survival and measured gait in one simulator session."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--speeds',type=float,nargs='+',default=[0.,.4,.6,.8,1.])
parser.add_argument('--yaw',action='store_true')
parser.add_argument('--sequence',action='store_true',help='Also test forward, stop, turn and restart without manual resets')
parser.add_argument('--sequence-only',action='store_true')
parser.add_argument('--continuous',action='store_true',help='Disable time-limit resets during every test case')
parser.add_argument('--steps',type=int,default=1000)
parser.add_argument('--num-envs',type=int,default=32)
parser.add_argument('--randomize',action='store_true')
parser.add_argument('--seed',type=int,default=2027)
parser.add_argument('--environment-profile',type=Path,help='Use terrain/randomization/disturbances from this profile; keep checkpoint robot and sensor contract')
parser.add_argument('--flat',action='store_true',help='Evaluate on a plane even for a rough-ground checkpoint')
parser.add_argument('--terrain-level',type=int,default=0)
parser.add_argument('--push-strength',type=float,default=0.,help='Random force curriculum fraction; zero disables random pushes')
parser.add_argument('--kick-body-weights',type=float,nargs='*',default=[],help='Additional fixed lateral push cases at 0.4 m/s')
parser.add_argument('--kick-duration',type=float,default=.12)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.steps < 1 or args.num_envs < 1 or args.kick_duration <= 0:
    parser.error('steps, num-envs and kick-duration must be positive')
if args.kick_body_weights and args.steps <= 100:
    parser.error('Kick tests need more than 100 steps')
if any(force < 0 for force in args.kick_body_weights) or args.push_strength < 0:
    parser.error('Push strengths must be nonnegative')
args.output.mkdir(parents=True,exist_ok=True)
console=(args.output/'console.log').open('w',buffering=1)
os.dup2(console.fileno(),1);os.dup2(console.fileno(),2)
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg
from spot_micro.runner import SpotOnPolicyRunner
from spot_micro.collision_geometry import lower_leg_vertices, mesh_clearance
from spot_micro.teleop import set_live_command
from isaaclab.utils.math import quat_apply


def gait_metrics(data,dt):
    valid=data['episode_step']>50
    if not valid.any():
        return {'unavailable':'No trajectory survived beyond the one-second settling interval'}
    contact=data['foot_contact']
    swings=[[] for _ in range(4)]
    peaks=[[] for _ in range(4)]
    for env in range(contact.shape[1]):
        for leg in range(4):
            start=None
            for t in range(contact.shape[0]):
                if not valid[t,env]:
                    start=None
                elif not contact[t,env,leg] and start is None:
                    start=t
                elif contact[t,env,leg] and start is not None:
                    duration=(t-start)*dt
                    if duration>=.04:
                        swings[leg].append(duration)
                        if 'foot_clearance' in data:
                            peaks[leg].append(float(data['foot_clearance'][start:t,env,leg].max()))
                    start=None
    metrics={
        'contact_fraction_by_leg':contact[valid].mean(0).tolist(),
        'mean_completed_swing_duration_s_by_leg':[float(np.mean(x)) if x else None for x in swings],
        'mean_swing_peak_clearance_proxy_m_by_leg':[float(np.mean(x)) if x else None for x in peaks],
        'completed_swings_ge_40ms_per_leg_per_s':[len(x)/max(valid.sum()*dt,1e-9) for x in swings],
    }
    bits=(contact[valid]*np.array([1,2,4,8])).sum(-1)
    metrics['support_patterns']={name:float(np.isin(bits,patterns).mean()) for name,patterns in (
        ('diagonal',[6,9]),('pace',[5,10]),('bound',[3,12]),('tripod',[7,11,13,14]),('all_four',[15]),('flight',[0]))}
    metrics['joint_velocity_p99_rad_s']=float(np.quantile(abs(data['joint_speed'][valid]),.99))
    if 'foot_position_body' in data:
        values=data['foot_position_body'][valid,:,0]
        metrics['foot_fore_aft_span_p95_minus_p05_m_by_leg']=(np.quantile(values,.95,axis=0)-np.quantile(values,.05,axis=0)).tolist()
    if 'collision_clearance' in data:
        metrics['collision_clearance_p95_m_by_leg']=np.quantile(data['collision_clearance'][valid],.95,axis=0).tolist()
        metrics['collision_clearance_max_m_by_leg']=data['collision_clearance'][valid].max(0).tolist()
    q=data['joint_position'][valid]
    metrics['joint_span_p95_minus_p05_rad']=(np.quantile(q,.95,axis=0)-np.quantile(q,.05,axis=0)).tolist()
    return metrics


def main():
    register()
    checkpoint=args.checkpoint.resolve()
    robot=json.loads((checkpoint.parent/'robot.json').read_text())
    if args.environment_profile:
        profile=json.loads(args.environment_profile.read_text())
        for key in ('terrain','domain_randomization','disturbances','robustness_curriculum'):
            if key in profile:
                robot[key]=profile[key]
    if args.flat:
        robot.pop('terrain',None)
    evaluation_profile=args.output/'environment.json'
    evaluation_profile.write_text(json.dumps(robot,indent=2)+'\n')
    cfg=make_env_cfg(evaluation_profile,args.num_envs,'cuda:0',args.seed,randomize=args.randomize)
    env=gym.make('SpotMicro-Velocity-Flat-v0',cfg=cfg)
    raw=env.unwrapped
    raw.record_collision_geometry=True
    support_vertices=lower_leg_vertices(cfg.robot.spawn.asset_path)
    wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=SpotOnPolicyRunner(wrapper,copy.deepcopy(json.loads((checkpoint.parent/'ppo.json').read_text())),device='cuda:0')
    info=runner.load(str(checkpoint),load_optimizer=False,map_location='cuda:0')
    raw.terrain_curriculum_enabled=False
    raw.random_pushes_enabled=args.push_strength>0
    raw.push_strength_override=args.push_strength
    if raw.terrain.terrain_origins is not None:
        if not 0 <= args.terrain_level < raw.terrain.max_terrain_level:
            raise ValueError('Terrain level is outside the configured range')
        raw.terrain.terrain_levels[:]=args.terrain_level
        raw.terrain.env_origins[:]=raw.terrain.terrain_origins[raw.terrain.terrain_levels,raw.terrain.terrain_types]
    if args.kick_body_weights and raw.pushes is None:
        raise ValueError('Kick tests need a profile with disturbances configured')
    # Compare actual full reward weights, independently of the eval batch size.
    if raw.style_parameters:
        raw.style_curriculum_offset=max(raw.style_curriculum_offset,raw.style_parameters.get('ramp_transitions',1))
    policy=runner.get_inference_policy(device='cuda:0')
    summary={'checkpoint':str(checkpoint),'policy_role':raw.params.get('policy_role','sensor_actor'),
             'num_envs':args.num_envs,'steps':args.steps,'randomization':args.randomize,'seed':args.seed,
             'continuous':args.continuous,
             'terrain_level':args.terrain_level,'terrain':robot.get('terrain',{'type':'plane'}),
             'push_strength':args.push_strength,'kick_duration_s':args.kick_duration,
             'checkpoint_info':info,'reward_scoring':'full style weights','cases':{}}
    materials=raw.robot.root_physx_view.get_material_properties()
    masses=raw.robot.root_physx_view.get_masses().sum(-1)
    summary['physical_parameters_measured']={
        'total_mass_kg_min_max':[float(masses.min()),float(masses.max())],
        'static_friction_min_max':[float(materials[...,0].min()),float(materials[...,0].max())],
        'dynamic_friction_min_max':[float(materials[...,1].min()),float(materials[...,1].max())]}
    commands=[(f'forward_{v:.2f}',(v,0.,0.)) for v in args.speeds]
    if args.yaw:
        commands += [('yaw_positive',(0.,0.,.6)),('yaw_negative',(0.,0.,-.6))]
    if args.sequence_only:
        commands=[]
    if args.sequence or args.sequence_only:
        commands.append(('command_sequence',(.6,0.,0.)))
    commands += [(f'kick_{force:.2f}bw',(.4,0.,0.)) for force in args.kick_body_weights]
    original_episode_length_s = raw.cfg.episode_length_s
    try:
        for case_index,(label,command) in enumerate(commands):
            # A time-limit reset must not hide a stop/turn/restart transition.
            # Keep the usual timeout for the independent fixed-command cases.
            raw.cfg.episode_length_s = (
                max(original_episode_length_s, (args.steps + 2) * raw.step_dt)
                if label == 'command_sequence' or args.continuous else original_episode_length_s
            )
            raw.command_override=command
            recordings={}
            rewards={}
            falls=timeouts=0
            mean_logits=[]
            command_trace=[]
            fall_trace=[]
            with torch.inference_mode():
                # Model construction consumes RNG differently for teacher and
                # student dimensions. Pair initial conditions by test case.
                torch.manual_seed(args.seed+1000*case_index)
                obs,_=wrapper.reset()
                for step in range(args.steps):
                    if label.startswith('kick_') and step==100:
                        force_bw=float(label[5:-2])
                        local_force=torch.zeros(args.num_envs,3,device=raw.device)
                        direction=torch.where(torch.arange(args.num_envs,device=raw.device)%2==0,1.,-1.)
                        local_force[:,1]=direction*force_bw*raw.pushes.masses*9.81
                        raw.pushes.inject(torch.arange(args.num_envs,device=raw.device),
                            quat_apply(raw.robot.data.root_quat_w,local_force),torch.zeros_like(local_force),args.kick_duration)
                    current_command=command
                    if label=='command_sequence':
                        stages=[(.6,0.,0.),(0.,0.,0.),(0.,0.,.6),(.4,0.,0.)]
                        current_command=stages[min(3,step*4//args.steps)]
                        set_live_command(raw,obs,current_command)
                    command_trace.append(current_command)
                    action=policy(obs)
                    if step%10==0:
                        actor=runner.alg.policy
                        x=actor.actor(actor.actor_obs_normalizer(actor.get_actor_obs(obs)))
                        mean_logits.append(x.cpu().numpy())
                    obs,_,_,_=wrapper.step(action)
                    falls+=int(raw.reset_terminated.sum());timeouts+=int(raw.reset_time_outs.sum())
                    fall_trace.append(int(raw.reset_terminated.sum()))
                    for name,value in raw.step_metrics.items():
                        recordings.setdefault(name,[]).append(value.cpu().numpy())
                    for name,value in raw.reward_terms.items():
                        rewards[name]=rewards.get(name,0.)+float(value.mean())/args.steps
            data={k:np.stack(v) for k,v in recordings.items()}
            if 'foot_link_position_w' in data:
                # On rough ground this is clearance above the local toe ray hit,
                # not a whole-mesh signed distance to every terrain triangle.
                ground=data['foot_ground_height_w'] if raw.terrain.terrain_origins is not None else raw.env_origins[:,2].cpu().numpy()
                data['collision_clearance']=mesh_clearance(data['foot_link_position_w'],data['foot_link_quaternion_w'],support_vertices,ground)
            vel=data['velocity']
            data['commands']=np.asarray(command_trace)
            valid=data['episode_step']>50
            result={'command':command,'mean_velocity':vel.mean((0,1)).tolist(),
                    'reset_seed':args.seed+1000*case_index,
                    'steady_mean_velocity':vel[valid].mean(0).tolist() if valid.any() else None,
                    'velocity_rmse':np.sqrt(((vel-data['commands'][:,None,:])**2).mean((0,1))).tolist(),
                    'fall_resets':falls,'time_limit_resets':timeouts,
                    'terrain_boundary_resets':int(data['terrain_boundary_reset'].sum()),
                    'episode_time_limit_s':raw.cfg.episode_length_s,
                    'mean_base_height_m':float(data['height'][valid].mean()) if valid.any() else None,
                    'max_joint_speed_rad_s':float(np.abs(data['joint_speed']).max()),
                    'max_torque_nm':float(np.abs(data['torque']).max()),
                    'mean_reward_terms_per_s':rewards,'gait':gait_metrics(data,raw.step_dt)}
            if 'external_force_w_n' in data:
                force_norm=np.linalg.norm(data['external_force_w_n'],axis=-1)
                result['max_external_force_n']=float(force_norm.max())
                result['mean_external_force_impulse_norm_ns']=float(force_norm.sum(0).mean()*raw.step_dt)
                result['max_external_torque_nm']=float(np.linalg.norm(data['external_torque_w_nm'],axis=-1).max())
            if label=='command_sequence':
                result['segments']=[]
                for stage in range(4):
                    start=stage*args.steps//4;stop=(stage+1)*args.steps//4
                    settled=start+min(50,(stop-start)//2)
                    result['segments'].append({'command':command_trace[start],
                        'steady_mean_velocity':vel[settled:stop].mean((0,1)).tolist(),
                        'fall_resets':sum(fall_trace[start:stop])})
            logits=torch.tensor(np.concatenate(mean_logits))
            processor=getattr(runner.alg.policy,'action_postprocessor',None)
            if processor is not None:
                logits.requires_grad_()
                latent=processor.latent_mean(logits)
                derivative=torch.autograd.grad(latent.sum(),logits)[0]
                result['median_latent_mean_derivative']=float(derivative.abs().median())
                result['median_abs_raw_actor']=float(logits.detach().abs().median())
                result['latent_noise_std']=runner.alg.policy.latent_std.detach().cpu().tolist()
            np.savez_compressed(args.output/f'{label}_trajectories.npz',**data)
            summary['cases'][label]=result
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print('RESULT',label,json.dumps(result),flush=True)
    finally:
        raw.cfg.episode_length_s = original_episode_length_s
        env.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
