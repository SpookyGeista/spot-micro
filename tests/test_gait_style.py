import unittest
import torch
from spot_micro.gait_style import style_terms


class GaitStyleTests(unittest.TestCase):
    def run_terms(self, *, contact=None, velocity=None, clearance=None, phase=None,
                  moving=False, previous_contact=None, previous_vz=None, dt=.02):
        contact = torch.ones(1,4,dtype=torch.bool) if contact is None else contact
        return style_terms(contact,torch.zeros(1,4) if clearance is None else clearance,
                           torch.zeros(1,4,3) if velocity is None else velocity,
                           torch.zeros(1,4) if phase is None else phase,
                           torch.tensor([moving]),torch.zeros(1,3),torch.zeros(1,3),
                           contact if previous_contact is None else previous_contact,
                           torch.zeros(1,4) if previous_vz is None else previous_vz,
                           {"foot_slip":.5,"foot_clearance":.4,"touchdown":.05,"body_wobble":.15},
                           {"clearance_m":.025},dt)

    def test_stationary_stance_has_no_style_cost(self):
        terms,_ = self.run_terms()
        self.assertTrue(all(float(v)==0 for v in terms.values()))

    def test_slip_cost_applies_only_to_stance_feet(self):
        contact = torch.tensor([[True,False,True,False]])
        velocity = torch.zeros(1,4,3); velocity[:,~contact[0],0] = 1
        terms,_ = self.run_terms(contact=contact,velocity=velocity)
        self.assertEqual(float(terms["foot_slip"]),0)
        velocity[:,contact[0],0] = .15
        terms,_ = self.run_terms(contact=contact,velocity=velocity)
        self.assertAlmostEqual(float(terms["foot_slip"]),-.25,places=5)

    def test_lifted_swing_foot_matches_mid_swing_target(self):
        phase = torch.tensor([[.8,.3,.3,.8]])
        clearance = torch.tensor([[.025,0,0,.025]])
        terms,_ = self.run_terms(phase=phase,clearance=clearance,moving=True)
        self.assertAlmostEqual(float(terms["foot_clearance"]),0,places=6)
        terms,_ = self.run_terms(phase=phase,moving=True)
        self.assertLess(float(terms["foot_clearance"]),-.1)

    def test_touchdown_is_an_event_with_dt_independent_integral(self):
        previous = torch.zeros(1,4,dtype=torch.bool)
        vz = torch.full((1,4),-.25)
        first,_ = self.run_terms(previous_contact=previous,previous_vz=vz,dt=.02)
        second,_ = self.run_terms(previous_contact=previous,previous_vz=vz,dt=.01)
        self.assertAlmostEqual(float(first["touchdown"])*.02,float(second["touchdown"])*.01)
        ongoing,_ = self.run_terms(previous_vz=vz)
        self.assertEqual(float(ongoing["touchdown"]),0)


if __name__=="__main__":
    unittest.main()
