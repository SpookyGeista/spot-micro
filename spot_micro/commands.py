"""Versioned velocity curriculum; old checkpoints retain the original ranges."""
import math


def gait_frequency(parameters, planar_speed):
    schedule = parameters.get("gait_frequency_schedule")
    if not schedule:
        return parameters["gait_frequency_hz"]
    return (schedule["base_hz"] + schedule["gain_hz_per_m_s"]*planar_speed).clip(
        schedule["base_hz"], schedule["max_hz"])


def command_ranges(parameters, progress):
    settings = parameters.get("command_curriculum", {})
    initial = settings.get("initial_vx_vy_wz", [[0, .06], [0, 0], [-.15, .15]])
    final = settings.get("final_vx_vy_wz", [[-.03, .16], [-.04, .04], [-.45, .45]])
    if len(initial) != 3 or len(final) != 3:
        raise ValueError("Three velocity command ranges are required")
    fraction = max(0.0, min(float(progress), 1.0))
    ranges = []
    for begin, end in zip(initial, final):
        if len(begin) != 2 or len(end) != 2 or not all(math.isfinite(x) for x in (*begin, *end)):
            raise ValueError("Command ranges must be finite pairs")
        if begin[0] > begin[1] or end[0] > end[1]:
            raise ValueError("Command ranges must be ordered")
        ranges.append(tuple(a + fraction * (b-a) for a, b in zip(begin, end)))
    return ranges
