"""Lightweight NVAPI/NVML helper for adjusting GPU memory clocks.

Attempts to use NVIDIA's NVML (via pynvml) to adjust the application
memory clock. It falls back gracefully if NVML is unavailable so the
server can still run on machines without the library installed.
"""
from __future__ import annotations

from typing import Tuple

try:  # Optional dependency
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - best effort only
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False


class NvAPIError(RuntimeError):
    pass


class NvAPI:
    """Minimal wrapper around NVML for memory clock control."""

    def __init__(self, device_index: int = 0) -> None:
        if not _NVML_AVAILABLE:
            raise NvAPIError("pynvml not available")
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        # Cache default clocks so offsets can be applied relative to them
        self.base_mem = pynvml.nvmlDeviceGetDefaultApplicationsClock(
            self.handle, pynvml.NVML_CLOCK_MEM
        )
        self.base_graphics = pynvml.nvmlDeviceGetDefaultApplicationsClock(
            self.handle, pynvml.NVML_CLOCK_GRAPHICS
        )

    def set_mem_clock_offset(self, offset_mhz: int) -> Tuple[bool, str]:
        """Apply an offset (in MHz) to the default memory clock.

        Returns (ok, message)
        """
        target_mem = self.base_mem + offset_mhz
        try:
            pynvml.nvmlDeviceSetApplicationsClocks(
                self.handle, target_mem, self.base_graphics
            )
            return True, f"memory clock set to {target_mem} MHz"
        except pynvml.NVMLError as e:  # pragma: no cover - depends on hardware
            return False, str(e)

    def reset(self) -> bool:
        """Reset clocks back to defaults."""
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(self.handle)
            return True
        except pynvml.NVMLError:  # pragma: no cover - depends on hardware
            return False
