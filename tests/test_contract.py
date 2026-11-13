import copy
import unittest

import numpy as np
import torch

from spot_micro.contract import load_robot, metadata
from spot_micro.control import BatchedController, ServoDynamics
from spot_micro.runtime import PolicyRuntime


class DeploymentParityTests(unittest.TestCase):
    def test_observation_history_and_actuator_processing_match_pi(self):
        cfg = load_robot()
        runtimes = [PolicyRuntime(cfg), PolicyRuntime(cfg)]
        batched = BatchedController(cfg, 2, "cpu")
        rng = np.random.default_rng(42)
        for step in range(250):
            if step == 130:
                # A reset in one robot must not alter the other robot's history.
                runtimes[0].reset()
                batched.reset(torch.tensor([0]))
            gyro = rng.normal(0, 0.3, (2, 3)).astype(np.float32)
            accel = (rng.normal(0, 0.4, (2, 3)) + [0, 0, 9.81]).astype(np.float32)
            if step % 11 == 0:
                accel *= 2  # Reject large translational acceleration in the tilt filter.
            command = np.tile([0.08, 0.0, 0.1], (2, 1)).astype(np.float32)
            if step % 17 == 0:
                command[:] = 0
            actual = batched.observe(torch.from_numpy(gyro), torch.from_numpy(accel), torch.from_numpy(command)).numpy()
            expected = np.stack([runtimes[i].observe(gyro[i], accel[i], command[i]) for i in range(2)])
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
            action = rng.normal(0, 2, (2, 12)).astype(np.float32)
            old = batched.target.clone()
            actual = batched.apply(torch.from_numpy(action), torch.from_numpy(command)).numpy()
            expected = np.stack([runtimes[i].apply(action[i], command[i]) for i in range(2)])
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
            self.assertLessEqual(float((batched.target - old).abs().max()), cfg["max_target_rate_rad_s"] * cfg["control_dt"] + 1e-6)

    def test_no_ground_truth_state_in_deployment_contract(self):
        info = metadata(load_robot())
        self.assertEqual(info["observation_size"], 115)
        fields = [x["name"] for x in info["observation_frame_fields"]]
        self.assertNotIn("joint_positions", fields)
        self.assertNotIn("base_velocity", fields)
        self.assertNotIn("foot_contacts", fields)
        runtime = PolicyRuntime(load_robot())
        with self.assertRaises(ValueError):
            runtime.observe([float("nan"), 0, 0], [0, 0, 9.81], [0, 0, 0])
        with self.assertRaises(ValueError):
            runtime.apply([float("inf")] * 12, [0, 0, 0])


class ServoModelTests(unittest.TestCase):
    def test_delay_reset_and_torque_speed_envelope(self):
        cfg = copy.deepcopy(load_robot()["servo"])
        cfg["position_bias_rad"] = 0
        model = ServoDynamics(cfg, 2, "cpu")
        ids = torch.arange(2)
        zero = torch.zeros(2, 12)
        model.reset(ids, zero, randomize=False)
        model.delay[:] = 2
        target = torch.ones(2, 12)
        self.assertTrue(torch.equal(model.compute(target, zero, zero), zero))
        self.assertTrue(torch.equal(model.compute(target, zero, zero), zero))
        self.assertTrue((model.compute(target, zero, zero) > 0).all())
        self.assertLessEqual(float(model.last_torque.max()), cfg["torque_cap_nm"] + 1e-6)
        velocity = torch.full_like(zero, cfg["no_load_speed_rad_s"])
        self.assertTrue((model.compute(target, zero, velocity) <= 1e-6).all())
        model.reset(ids, zero, randomize=False)
        self.assertTrue(torch.equal(model.compute(zero, zero, zero), zero))


if __name__ == "__main__":
    unittest.main()
