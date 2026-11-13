import copy
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from spot_micro.bounded_policy import DirectSquashedActorCritic
from spot_micro.contract import ROOT, load_robot, metadata
from spot_micro.control import BatchedController
from spot_micro.curriculum import PerformanceCurriculum
from spot_micro.export import ExportedActor
from spot_micro.runtime import PolicyRuntime


class DynamicTrainingTests(unittest.TestCase):
    def test_expanded_student_initialization_omits_privileged_inputs(self):
        from spot_micro.distillation import initialize_student,sensor_projection
        teacher_cfg=load_robot(ROOT / 'configs/spotmicro_teacher_dynamic.json')
        cfg={**teacher_cfg,'policy_role':'student','history_length':20,'observe_acceleration':True}
        x=torch.randn(8,520)
        indices=sensor_projection(teacher_cfg,cfg)
        teacher_obs=TensorDict({'critic':torch.cat((x[:,indices],torch.zeros(8,38)),-1)},[8])
        student_obs=TensorDict({'policy':x,'critic':torch.cat((x,torch.randn(8,38)),-1)},[8])
        kwargs={'actor_hidden_dims':[16],'critic_hidden_dims':[16],'noise_std_type':'log'}
        teacher=DirectSquashedActorCritic(teacher_obs,{'policy':['critic'],'critic':['critic']},12,**kwargs)
        student=DirectSquashedActorCritic(student_obs,{'policy':['policy'],'critic':['critic']},12,**kwargs)
        initialize_student(student,teacher,teacher_cfg,cfg)
        torch.testing.assert_close(student.act_inference(student_obs),teacher.act_inference(teacher_obs))
        before=student.act_inference(student_obs)
        student_obs['critic'][:,-38:]=1e6
        torch.testing.assert_close(student.act_inference(student_obs),before)

    def test_long_imu_history_matches_pi_and_teacher_projection(self):
        from spot_micro.distillation import sensor_projection,teacher_observations
        cfg=load_robot(ROOT / 'configs/spotmicro_teacher_dynamic.json')
        student_cfg={**cfg,'policy_role':'student','history_length':20,'observe_acceleration':True,'accel_scale':.1,'accel_clip_m_s2':39.24}
        runtime=PolicyRuntime(student_cfg)
        controller=BatchedController(student_cfg,1,'cpu')
        teacher_controller=BatchedController(cfg,1,'cpu')
        indices=sensor_projection(cfg,student_cfg)
        rng=np.random.default_rng(7)
        for step in range(100):
            gyro=rng.normal(0,.3,(1,3)).astype(np.float32)
            accel=(rng.normal(0,2,(1,3))+[0,0,9.81]).astype(np.float32)
            cmd=np.array([[.4,0,.2]],dtype=np.float32)
            obs=controller.observe(torch.from_numpy(gyro),torch.from_numpy(accel),torch.from_numpy(cmd))
            expected=runtime.observe(gyro[0],accel[0],cmd[0])
            np.testing.assert_allclose(obs.numpy()[0],expected,atol=1e-5,rtol=1e-5)
            teacher_obs=teacher_controller.observe(torch.from_numpy(gyro),torch.from_numpy(accel),torch.from_numpy(cmd))
            torch.testing.assert_close(obs[:,indices],teacher_obs)
            action=rng.normal(0,.3,(1,12)).astype(np.float32)
            target=controller.apply(torch.from_numpy(action),torch.from_numpy(cmd))
            teacher_controller.apply(torch.from_numpy(action),torch.from_numpy(cmd))
            np.testing.assert_allclose(target.numpy()[0],runtime.apply(action[0],cmd[0]),atol=1e-6)
        self.assertEqual(metadata(student_cfg)['observation_size'],520)
        oracle=torch.full((1,38),999.)
        projected=teacher_observations({'policy':obs,'critic':torch.cat((obs,oracle),-1)},indices)['critic']
        self.assertEqual(projected.shape[-1],153)
        self.assertTrue((projected[:,-38:]==999).all())
        self.assertFalse((obs==999).any())
        saturated=runtime.observe([0,0,0],[100,-100,9.81],[0,0,0])
        np.testing.assert_allclose(saturated[-3:],[3.924,-3.924,.981],rtol=1e-6)

    def test_collision_clearance_uses_rotated_geometry(self):
        from spot_micro.collision_geometry import mesh_clearance
        points=np.array([[-.2,0,-.1],[.2,0,.1]])
        positions=np.array([[[0,0,1]]*4],dtype=float)
        q=np.array([[[1,0,0,0]]*4],dtype=float)
        np.testing.assert_allclose(mesh_clearance(positions,q,[points]*4),.9)
        q[:]=[2**-.5,0,2**-.5,0]
        np.testing.assert_allclose(mesh_clearance(positions,q,[points]*4),.8)
        # Different local ground heights under the four feet on rough terrain.
        np.testing.assert_allclose(mesh_clearance(positions,q,[points]*4,np.array([[.1,.2,.3,.4]])),[[.7,.6,.5,.4]])

    def test_fresh_actor_explores_physical_targets_and_preserves_export(self):
        torch.manual_seed(4)
        cfg = load_robot(ROOT / 'configs/spotmicro_teacher_dynamic.json')
        obs = TensorDict({'critic': torch.randn(4096,153)}, [4096])
        policy = DirectSquashedActorCritic(obs, {'policy':['critic'],'critic':['critic']},12,
            actor_hidden_dims=[32],critic_hidden_dims=[32],noise_std_type='log',
            init_noise_std=.45,min_noise_std=.15,max_noise_std=.8)
        actions = policy.act(obs)
        controller = BatchedController(cfg,4096,'cpu')
        targets = controller.apply(actions,torch.zeros(4096,3))
        self.assertGreater(float(targets.std(0).min()),cfg['servo']['deadband_rad']*5)
        self.assertTrue(torch.isfinite(policy.get_actions_log_prob(actions)).all())
        mean_action = policy.act_inference(obs)
        mean_action.mean().backward()
        self.assertGreater(float(policy.actor[-1].bias.grad.abs().min()),.05)
        torch.testing.assert_close(ExportedActor(policy)(obs['critic']),mean_action)

    def test_privileged_policy_is_not_mislabelled_for_hardware(self):
        cfg = load_robot(ROOT / 'configs/spotmicro_teacher_dynamic.json')
        info = metadata(cfg)
        self.assertEqual(info['observation_size'],153)
        self.assertTrue(info['simulation_only'])
        with self.assertRaisesRegex(ValueError,'simulation-only'):
            PolicyRuntime(cfg)

    def test_speed_dependent_phase_matches_sensor_runtime(self):
        cfg = copy.deepcopy(load_robot(ROOT / 'configs/spotmicro_teacher_dynamic.json'))
        cfg.pop('policy_role')
        runtime = PolicyRuntime(cfg)
        controller = BatchedController(cfg,1,'cpu')
        for vx in (.0,.1,.6,2.):
            command=np.array([vx,0.,0.],dtype=np.float32)
            for _ in range(40):
                runtime.apply(np.zeros(12,dtype=np.float32),command)
                controller.apply(torch.zeros(1,12),torch.from_numpy(command[None]))
                phase_error=(float(controller.phase[0])-float(runtime.phase)+.5)%1.-.5
                self.assertAlmostEqual(phase_error,0.,places=5)

    def test_curriculum_waits_for_tracking_and_survival_and_restores(self):
        cfg={'start_m_s':.25,'max_m_s':.45,'min_transitions':100,'increment_m_s':.1}
        c=PerformanceCurriculum(cfg)
        self.assertFalse(c.observe(1000,0,10000,0,100))
        self.assertEqual(c.maximum,.25)
        self.assertFalse(c.observe(1000,1000,10000,10000,200))
        self.assertEqual(c.maximum,.25)
        self.assertTrue(c.observe(1000,1000,10000,500,300))
        self.assertAlmostEqual(c.maximum,.35)
        c.observe(10,10,1000,50,350)
        restored=PerformanceCurriculum(cfg)
        restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.state_dict(),c.state_dict())
        self.assertTrue(restored.observe(200,200,5000,250,400))
        self.assertAlmostEqual(restored.maximum,.45)


if __name__ == '__main__':
    unittest.main()
