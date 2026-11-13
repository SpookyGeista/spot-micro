import unittest
import torch
from spot_micro.robustness import PushPulses, RobustnessCurriculum, gait_quality_terms


class RobustnessTests(unittest.TestCase):
    def test_physical_pulse_has_finite_duration_and_reset_clears_it(self):
        pulses = PushPulses({}, torch.tensor([2.52, 3.]), .02)
        ids = torch.tensor([0])
        pulses.inject(ids, torch.tensor([[10., 0., 0.]]), torch.zeros(1, 3), .1)
        impulse = torch.zeros(2, 3)
        for _ in range(10):
            force, _ = pulses.step(random_events=False)
            impulse += force*.02
        torch.testing.assert_close(impulse, torch.tensor([[1., 0., 0.], [0., 0., 0.]]))
        pulses.inject(ids, torch.ones(1, 3), torch.ones(1, 3), .2)
        pulses.reset(ids)
        self.assertEqual(float(pulses.force.abs().sum()+pulses.torque.abs().sum()), 0.)

    def test_random_push_is_bounded_by_each_robots_weight(self):
        masses = torch.linspace(1., 4., 256)
        pulses = PushPulses({'max_force_body_weights':1.5,'max_torque_nm':.4},masses,.02)
        pulses.timer[:] = 0
        force, torque = pulses.step(.5)
        normalized = force[:,:2].norm(dim=-1)/(masses*9.81)
        self.assertTrue(bool((normalized <= .75+1e-6).all()))
        self.assertTrue(bool((normalized >= .75*.25-1e-6).all()))
        self.assertLessEqual(float(torque.abs().max()),.2)

    def test_curriculum_does_not_advance_on_bad_survival_and_restores(self):
        c = RobustnessCurriculum({'min_transitions':1})
        self.assertFalse(c.observe(200,100,6000,60.,10))
        self.assertEqual(c.strength,.2)
        self.assertTrue(c.observe(200,190,6000,600.,20))
        restored = RobustnessCurriculum({})
        restored.load_state_dict(c.state_dict())
        self.assertEqual(restored.strength,.4)

    def test_swing_reward_requires_touchdown_and_matching_diagonal_timing(self):
        times = torch.tensor([[.2,0.,0.,.2],[.2,0.,.2,0.]])
        peaks = torch.full((2,4),.03)
        moving = torch.tensor([True,False])
        landed = torch.tensor([[True,False,False,True],[True,True,True,True]])
        terms = gait_quality_terms(times,peaks,landed,moving,{'swing_peak':.1,'diagonal_timing':.1})
        self.assertGreater(float(terms['swing_peak'][0]),0.)
        self.assertEqual(float(terms['swing_peak'][1]),0.)
        self.assertEqual(float(terms['diagonal_timing'][0]),0.)
        terms = gait_quality_terms(times,peaks,torch.zeros_like(landed),torch.ones_like(moving),{'swing_peak':.1,'diagonal_timing':.1})
        self.assertEqual(float(terms['swing_peak'].sum()),0.)
        self.assertLess(float(terms['diagonal_timing'][1]),0.)


if __name__ == '__main__':
    unittest.main()
