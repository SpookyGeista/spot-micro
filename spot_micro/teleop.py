"""Keyboard velocity commands for the simulator; no hardware access."""
from .contract import frame_size


class KeyboardCommand:
    def __init__(self, speeds=None):
        self.speeds = speeds or {"forward_m_s": .08, "backward_m_s": .03,
                                 "lateral_m_s": .04, "yaw_rad_s": .3}
        self.pressed = set()
        self.reset_requested = False
        self.quit_requested = False

    def press(self, key):
        if key == "SPACE":
            self.pressed.clear()
        elif key == "R":
            self.pressed.clear()
            self.reset_requested = True
        elif key == "ESCAPE":
            self.quit_requested = True
        self.pressed.add(key)

    def release(self, key):
        self.pressed.discard(key)

    def clear(self):
        self.pressed.clear()

    def command(self):
        if "SPACE" in self.pressed:
            return (0.0, 0.0, 0.0)
        def held(*keys):
            return any(key in self.pressed for key in keys)
        forward = held("W", "UP") - held("S", "DOWN")
        lateral = held("A", "LEFT") - held("D", "RIGHT")
        yaw = held("Q") - held("E")
        return (forward * self.speeds["forward_m_s" if forward >= 0 else "backward_m_s"],
                lateral * self.speeds["lateral_m_s"], yaw * self.speeds["yaw_rad_s"])


def set_live_command(raw, observations, command):
    """Update only the latest command fields, without repeating IMU/history updates."""
    raw.command_override = tuple(command)
    raw.commands[:] = raw.commands.new_tensor(command)
    scaled = raw.commands * raw.commands.new_tensor(raw.params["command_scale"])
    raw.controller.history[:, -1, :3] = scaled
    start = (raw.params["history_length"] - 1) * frame_size(raw.params)
    # Privileged observations begin with the same sensor history. A simulation
    # teacher uses this group for its actor too, so both command fields must match.
    for name in ("policy", "critic"):
        if name in observations:
            observations[name][:, start:start + 3] = scaled.clamp(-10, 10)
