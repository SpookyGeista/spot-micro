"""Optional PCA9685 backend. Configuration starts with all PWM outputs off."""
from __future__ import annotations

import math
import threading
import time


class PCA9685:
    def __init__(self, bus, cfg, *, sleep=time.sleep):
        self.bus = bus
        self.address = cfg["address"]
        self.oscillator_hz = float(cfg["oscillator_hz"])
        frequency = float(cfg["frequency_hz"])
        if not 20 <= frequency <= 100 or not 20_000_000 <= self.oscillator_hz <= 30_000_000:
            raise ValueError("Unsupported servo PWM frequency or oscillator estimate")
        self.prescale = round(self.oscillator_hz / (4096 * frequency)) - 1
        if not 3 <= self.prescale <= 255:
            raise ValueError("PWM prescale cannot be represented")
        self.actual_frequency_hz = self.oscillator_hz / (4096 * (self.prescale + 1))
        self.sleep = sleep
        self.lock = threading.RLock()
        self.initialized = False
        self.disabled = True
        self.written_channels = set()
        self.pending = {}

    def initialize(self):
        with self.lock:
            self.bus.write_byte_data(self.address, 0xFD, 0x10)  # ALL_LED_OFF
            self.bus.write_byte_data(self.address, 0x00, 0x30)  # SLEEP + auto increment
            self.bus.write_byte_data(self.address, 0x01, 0x04)  # totem pole, change on STOP
            self.bus.write_byte_data(self.address, 0xFE, self.prescale)
            self.bus.write_byte_data(self.address, 0x00, 0x20)  # wake, disable all-call
            self.sleep(0.005)
            self.bus.write_byte_data(self.address, 0x00, 0xA0)  # restart + auto increment
            # ALL_LED_* writes broadcast to individual LEDn registers; they are
            # not a separate global mask. Never clear ALL_LED_OFF to arm channels.
            for channel in range(16):
                self.bus.write_i2c_block_data(self.address, 0x06 + 4 * channel, [0, 0, 0, 0x10])
            self.written_channels.clear()
            self.pending.clear()
            self.initialized = True
            self.disabled = True

    def write_pulses(self, pulses):
        if not self.initialized:
            raise RuntimeError("PCA9685 has not been initialized")
        prepared = []
        for channel, pulse_us in pulses.items():
            if not isinstance(channel, int) or not 0 <= channel < 16:
                raise ValueError("PWM channel must be an integer in 0..15")
            if not math.isfinite(pulse_us) or not 500 <= pulse_us <= 2500:
                raise ValueError("Servo pulse must be finite and within 500..2500 us")
            # Use the programmed prescale rather than assuming an exact 50 Hz.
            ticks = round(pulse_us * self.oscillator_hz / ((self.prescale + 1) * 1_000_000))
            if not 1 <= ticks <= 4095:
                raise ValueError("PWM width cannot be represented")
            prepared.append((channel, [0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F]))
        with self.lock:
            for channel, registers in prepared:
                self.pending[channel] = registers
                if not self.disabled:
                    self.bus.write_i2c_block_data(self.address, 0x06 + 4 * channel, registers)
                self.written_channels.add(channel)

    def enable(self):
        with self.lock:
            if not self.initialized or not self.written_channels:
                raise RuntimeError("Write at least one known pulse before enabling outputs")
            # Clear FULL_OFF only by writing known pulse widths to selected
            # channels. Unused channels remain FULL_OFF.
            for channel, registers in self.pending.items():
                self.bus.write_i2c_block_data(self.address, 0x06 + 4 * channel, registers)
            self.disabled = False

    def disable_all(self):
        with self.lock:
            self.disabled = True
            self.written_channels.clear()
            self.pending.clear()
            self.bus.write_byte_data(self.address, 0xFD, 0x10)
