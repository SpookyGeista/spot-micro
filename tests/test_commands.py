import json
from pathlib import Path
import unittest

from spot_micro.commands import command_ranges
from spot_micro.teleop import KeyboardCommand


class CommandTests(unittest.TestCase):
    def test_two_mps_curriculum_retains_low_speeds_and_ramps_forward_limit(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/spotmicro_speed_2mps.json").read_text())
        self.assertEqual(command_ranges(cfg, 0)[0], (-.12,.4))
        self.assertAlmostEqual(command_ranges(cfg, .5)[0][1], 1.2)
        self.assertEqual(command_ranges(cfg, 1)[0], (-.12,2.))
        self.assertEqual(command_ranges(cfg, 1)[1:], command_ranges(cfg, 0)[1:])

    def test_legacy_checkpoints_preserve_ranges(self):
        ranges = command_ranges({}, .5)
        for actual, expected in zip(ranges, ((-.015, .11), (-.02, .02), (-.30, .30))):
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)

    def test_fast_profile_and_keyboard_match_training_envelope(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/spotmicro_fast.json').read_text())
        initial, final = command_ranges(cfg, 0), command_ranges(cfg, 1)
        self.assertGreater(final[0][1], initial[0][1])
        self.assertEqual(final[0][1], .4)
        keys = KeyboardCommand(cfg['teleop_speeds'])
        for key, axis, sign in [('W', 0, 1), ('S', 0, -1), ('A', 1, 1), ('D', 1, -1), ('Q', 2, 1), ('E', 2, -1)]:
            keys.clear(); keys.press(key)
            value = keys.command()[axis]
            self.assertGreater(sign*value, 0)
            self.assertGreaterEqual(value, final[axis][0])
            self.assertLessEqual(value, final[axis][1])
        keys.press('SPACE')
        self.assertEqual(keys.command(), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
