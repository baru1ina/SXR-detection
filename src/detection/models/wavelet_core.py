from dataclasses import dataclass
from typing import Optional

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from src.visualization.plots import plot_enhance_crash_energy


@dataclass
class WaveletTransformResult:
    signal: np.ndarray
    scales: np.ndarray
    coeffs: np.ndarray
    raw_energy: np.ndarray
    energy: np.ndarray
    signed_response: np.ndarray
    response_scale: int


class WaveletEdgeCore:
    def __init__(
        self,
        dt: float,
        crash_time_min: float = 10e-6,
        crash_time_max: float = 50e-6,
        wavelet_name: str = "gaus1",
        response_scale_seconds: Optional[float] = None,
        enhance_energy: bool = True,
    ):
        self.dt = float(dt)
        self.crash_time_min = float(crash_time_min)
        self.crash_time_max = float(crash_time_max)
        self.wavelet_name = wavelet_name
        self.response_scale_seconds = response_scale_seconds
        self.enhance_energy = bool(enhance_energy)

        self.last_result: Optional[WaveletTransformResult] = None

    def compute_scale_range(self) -> np.ndarray:
        scale_min = int(self.crash_time_min / (2 * self.dt))
        scale_max = int(self.crash_time_max / (2 * self.dt))
        scale_min = max(scale_min, 1)
        scale_max = max(scale_max, scale_min + 1)
        return np.arange(scale_min, scale_max, dtype=int)

    @staticmethod
    def enhance_crash_energy(energy: np.ndarray, signal: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy, dtype=float)
        if len(energy) < 9:
            return energy.copy()

        energy_median = medfilt(energy, kernel_size=5)
        noise = energy - energy_median
        noise_level = float(np.std(noise))

        energy_background = gaussian_filter1d(energy_median, sigma=1.0)
        mask = energy_median > 6 * noise_level
        energy_clean = np.where(mask, energy_median, energy_background)

        median_val = float(np.median(energy_clean))
        mad = float(np.median(np.abs(energy_clean - median_val)))
        denom = median_val + mad + 1e-12

        energy_norm = energy_clean / denom
        threshold = median_val + 2 * mad
        energy_norm = np.where(energy_norm > threshold, energy_norm ** 1.5, energy_norm)

        plot_enhance_crash_energy(
            energy_median,
            energy,
            noise,
            noise_level,
            energy_norm,
            channel_name="SXR",
            filename="shot",
            mode="wavelet_diagnostics",
        )

        return energy_norm

    def _response_scale_index(self, scales: np.ndarray, coeffs: np.ndarray) -> int:
        if len(scales) == 0:
            return 0

        if self.response_scale_seconds is not None and np.isfinite(self.response_scale_seconds):
            target_scale = max(1.0, float(self.response_scale_seconds) / (2 * self.dt))
            return int(np.argmin(np.abs(scales.astype(float) - target_scale)))

        scale_power = np.nanstd(coeffs, axis=1)
        if np.all(~np.isfinite(scale_power)):
            return len(scales) // 2
        return int(np.nanargmax(scale_power))

    def transform(self, signal: np.ndarray) -> WaveletTransformResult:
        x = np.asarray(signal, dtype=float)
        scales = self.compute_scale_range()
        coeffs, _ = pywt.cwt(
            x,
            scales,
            self.wavelet_name,
            sampling_period=self.dt,
        )

        raw_energy = np.sqrt(np.sum(coeffs ** 2, axis=0))
        energy = self.enhance_crash_energy(raw_energy, signal) if self.enhance_energy else raw_energy.copy()
        std = float(np.std(energy))
        if std > 1e-12:
            energy = energy / std

        response_idx = self._response_scale_index(scales, coeffs)
        signed_response = np.asarray(coeffs[response_idx], dtype=float)
        response_std = float(np.std(signed_response))
        if response_std > 1e-12:
            signed_response = signed_response / response_std

        result = WaveletTransformResult(
            signal=x,
            scales=scales,
            coeffs=coeffs,
            raw_energy=raw_energy,
            energy=energy,
            signed_response=signed_response,
            response_scale=int(scales[response_idx]),
        )
        self.last_result = result
        return result
