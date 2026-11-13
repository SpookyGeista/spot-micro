import unittest
from types import SimpleNamespace

import torch

from spot_micro.contract import FRAME_SIZE
from spot_micro.teleop import KeyboardCommand, set_live_command


class TeleopTests(unittest.TestCase):
    def test_repeat_stop_and_key_release_do_not_leave_motion(self):
        keys = KeyboardCommand()
        keys.press("W")
        keys.press("W")
        self.assertEqual(keys.command(), (.08, 0, 0))
        keys.press("SPACE")
        self.assertEqual(keys.command(), (0, 0, 0))
        keys.release("SPACE")
        keys.release("W")
        self.assertEqual(keys.command(), (0, 0, 0))
        keys.press("S")
        self.assertEqual(keys.command(), (-.03, 0, 0))
        keys.clear()
        self.assertEqual(keys.command(), (0, 0, 0))

    def test_live_command_reaches_actor_without_advancing_sensor_history(self):
        history = torch.randn(1, 5, FRAME_SIZE)
        before = history.clone()
        raw = SimpleNamespace(commands=torch.zeros(1, 3), controller=SimpleNamespace(history=history),
                              params={"command_scale": [5, 5, 2], "history_length": 5})
        obs = {"policy": history.flatten(1).clone(), "critic": torch.zeros(1, 153)}
        set_live_command(raw, obs, (.08, -.04, .3))
        torch.testing.assert_close(raw.commands, torch.tensor([[.08, -.04, .3]]))
        torch.testing.assert_close(history[:, :-1], before[:, :-1])
        torch.testing.assert_close(history[:, -1, 3:], before[:, -1, 3:])
        torch.testing.assert_close(obs["policy"][:, 92:95], torch.tensor([[.4, -.2, .6]]))
        torch.testing.assert_close(obs["critic"][:, 92:95], obs["policy"][:, 92:95])

    def test_live_command_updates_extended_imu_history_without_touching_privileged_state(self):
        history = torch.randn(2, 20, 26)
        raw = SimpleNamespace(commands=torch.zeros(2, 3), controller=SimpleNamespace(history=history),
                              params={"command_scale": [2, 2, 1], "history_length": 20,
                                      "observe_acceleration": True})
        obs = {"policy": history.flatten(1).clone(),
               "critic": torch.cat((history.flatten(1), torch.randn(2, 38)), dim=1)}
        before = {key: value.clone() for key, value in obs.items()}
        set_live_command(raw, obs, (.6, 0, .6))
        for key, value in obs.items():
            torch.testing.assert_close(value[:, 494:497], torch.tensor([[1.2, 0., .6]]).expand(2, -1))
            torch.testing.assert_close(value[:, :494], before[key][:, :494])
            torch.testing.assert_close(value[:, 497:], before[key][:, 497:])


if __name__ == "__main__":
    unittest.main()
