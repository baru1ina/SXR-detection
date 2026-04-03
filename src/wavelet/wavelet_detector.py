import numpy as np
import pywt
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from src.anomaly.smoothing import smooth_signal_gauss, smooth_signal_savgol, smooth_signal_median


class WaveletSawtoothDetector:
    def __init__(
        self,
        dt,
        period,
        crash_time_min=10e-6,
        crash_time_max=100e-6,
        smoothing_window=51,
        smoothing_poly=3,
        percentile_threshold=99,
        min_period=1e-3,
        wavelet_name="mexh",
        period_map=None
    ):
        self.dt = dt
        self.crash_time_min = crash_time_min
        self.crash_time_max = crash_time_max
        self.smoothing_window = smoothing_window
        self.smoothing_poly = smoothing_poly
        self.percentile_threshold = percentile_threshold
        self.min_period = min_period
        self.wavelet_name = wavelet_name
        self.period = period
        self.period_map = period_map

        self.scales = None
        self.coeffs = None
        self.energy = None


    @staticmethod
    def validate_crash_candidate(signal, crash_idx, dt, original_signal=None):
        if original_signal is not None:
            signal = original_signal

        window_us = 100  # мкс
        window_samples = int(window_us * 1e-6 / dt)

        left = max(0, crash_idx - window_samples)
        right = min(len(signal), crash_idx + window_samples)

        segment = signal[left:right]
        time = np.arange(len(segment)) * dt

        deriv = np.gradient(segment, dt)
        max_neg_deriv = np.min(deriv)

        pre_crash = segment[:len(segment) // 2]
        post_crash = segment[len(segment) // 2:]

        if len(pre_crash) > 0 and len(post_crash) > 0:
            pre_max = np.max(pre_crash)
            crash_value = segment[len(segment) // 2]
            drop_ratio = (pre_max - crash_value) / (pre_max + 1e-8)
        else:
            drop_ratio = 0

        pos_deriv = deriv[deriv > 0]
        neg_deriv = -deriv[deriv < 0]

        if len(pos_deriv) > 0 and len(neg_deriv) > 0:
            asymmetry = np.max(neg_deriv) / (np.mean(pos_deriv) + 1e-8)
        else:
            asymmetry = 1

        is_valid = (
                max_neg_deriv < -1e5 and
                drop_ratio > 0.2 and
                asymmetry > 2
        )

        return is_valid, {
            'max_neg_deriv': max_neg_deriv,
            'drop_ratio': drop_ratio,
            'asymmetry': asymmetry
        }

    def compute_scale_range_v0(self):
        f_c = 0.25

        f_max = 1.0 / self.crash_time_min
        f_min = 1.0 / self.crash_time_max

        scale_min = int(f_c / (f_max * self.dt))
        scale_max = int(f_c / (f_min * self.dt))

        scale_min = max(scale_min, 1)
        scale_max = max(scale_max, scale_min + 1)

        return np.arange(scale_min, scale_max)

    def compute_scale_range(self):

        # тут фигня, непонятно почему вообще это приближение работает:(
        scale_min = int(self.crash_time_min / (2 * self.dt))
        # scale_min = int(0.159 * self.crash_time_min / (self.dt))
        scale_max = int(self.crash_time_max / (2 * self.dt))
        # scale_max = int(0.159 * self.crash_time_max / (self.dt))

        scale_min = max(scale_min, 1)
        scale_max = max(scale_max, scale_min + 1)

        return np.arange(scale_min, scale_max)

    def compute_cwt(self, signal):
        self.scales = self.compute_scale_range()
        coeffs, freqs = pywt.cwt(
            signal,
            self.scales,
            self.wavelet_name,
            sampling_period=self.dt
        )

        self.coeffs = coeffs
        return coeffs

    def compute_energy(self):
        self.energy = np.sqrt(np.sum(self.coeffs**2, axis=0))
        return self.energy

    @staticmethod
    def local_threshold(energy, window=100):
        threshold = np.zeros_like(energy)
        for i in range(len(energy)):
            window_energy = energy[max(0, i - window):min(len(energy), i + window)]
            threshold[i] = np.percentile(window_energy, 98)
        return threshold

    @staticmethod
    def enhance_crash_energy(energy, dt, crash_duration=20e-6):

        energy_median = medfilt(energy, kernel_size=5)
        high_freq = energy - energy_median
        noise_level = np.std(high_freq)
        adaptive_window = np.clip(noise_level * 1000, 3, 15)
        adaptive_window = int(adaptive_window) if adaptive_window % 2 else int(adaptive_window) + 1
        energy_smoothed = savgol_filter(energy, adaptive_window, 2)
        energy_enhanced = energy_smoothed ** 1.5

        energy_enhanced = energy_enhanced / np.percentile(energy_enhanced, 95)

        return energy_enhanced

    @staticmethod
    def enhance_crash_energy_v2(energy, dt, crash_duration=20e-6):
        noise_estimate = np.std(np.diff(energy))

        kernel_size = int(np.clip(noise_estimate * 1000, 3, 11))
        if kernel_size % 2 == 0:
            kernel_size += 1

        energy_median = medfilt(energy, kernel_size=kernel_size)

        noise = energy - energy_median
        noise_level = np.std(noise)

        sigma = 0.5
        energy_background = gaussian_filter1d(energy_median, sigma=sigma)

        mask = energy_median > 2 * noise_level
        energy_clean = np.where(mask, energy_median, energy_background)

        median_val = np.median(energy_clean)
        mad = np.median(np.abs(energy_clean - median_val))
        energy_norm = energy_clean / (median_val + mad)

        threshold = median_val + 2 * mad
        enhancement_factor = 1.2
        energy_norm = np.where(energy_norm > threshold,
                               energy_norm ** enhancement_factor,
                               energy_norm)
        return energy_norm

    @staticmethod
    def enhance_crash_energy_v3(energy, dt):
        from scipy.signal import medfilt
        from scipy.ndimage import gaussian_filter1d

        energy_median = medfilt(energy, kernel_size=5)

        noise = energy - energy_median
        noise_level = np.std(noise)

        sigma = 1.0
        energy_background = gaussian_filter1d(energy_median, sigma=sigma)

        mask = energy_median > 3 * noise_level
        energy_clean = np.where(mask, energy_median, energy_background)

        median_val = np.median(energy_clean)
        mad = np.median(np.abs(energy_clean - median_val))

        energy_norm = energy_clean / (median_val + mad)

        threshold = median_val + 2 * mad
        enhancement_factor = 1.5
        energy_norm = np.where(energy_norm > threshold,
                               energy_norm ** enhancement_factor,
                               energy_norm)
        return energy_norm


    def detect(self, signal, time_plasma):

        smoothed = signal
        self.compute_cwt(smoothed)
        energy = self.compute_energy()
        energy = self.enhance_crash_energy_v3(energy, dt = self.dt)
        energy = energy / np.std(energy)

        # threshold = np.percentile(energy, self.percentile_threshold)
        # threshold = np.mean(self.local_threshold(energy))
        # threshold = np.median(energy) + 1.5 * np.std(energy)
        threshold = np.median(energy) + np.std(energy)
        # threshold = np.percentile(energy, 95) + 1.5 * (np.percentile(energy, 75) - np.percentile(energy, 25))

        # min_distance = int(self.min_period / self.dt)
        if self.period_map is None:
            min_distance = int(self.period / self.dt)
        else:
            local_period = np.median(self.period_map)
            print(f"period map= {self.period_map}")
            print(f"local_period= {local_period}")
            min_distance = int(local_period / self.dt)

        peaks, properties = find_peaks(
            energy,
            height=threshold,
            distance=min_distance
        )

        fontsize=20
        labelsize=20

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(time_plasma* 1000, signal, 'b-', label='Сигнал', alpha=0.7)
        ax1.plot(time_plasma[peaks] * 1000, signal[peaks], 'r^',
                 markersize=8, label='Обнаруженные срывы')
        ax1.set_xlabel('Время (мс)', fontsize=fontsize)
        ax1.set_ylabel('Амплитуда', fontsize=fontsize)
        ax1.set_title('Исходный сигнал с отмеченными моментами срывов', fontsize=fontsize)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=labelsize)

        ax2.plot(time_plasma * 1000, energy, 'g-', label='Энергетический профиль')
        ax2.axhline(y=threshold, color='r', linestyle='--',
                    label=f'Порог')
        ax2.plot(time_plasma[peaks] * 1000, energy[peaks], 'r^',
                 markersize=8, label='Пики')
        ax2.set_xlabel('Время (мс)',fontsize=fontsize)
        ax2.set_ylabel('Нормированная энергия',fontsize=fontsize)
        ax2.set_title('Энергетический профиль вейвлет-преобразования',fontsize=fontsize)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax2.tick_params(axis='both', labelsize=labelsize)
        plt.tight_layout()
        plt.show()

        return peaks, energy, threshold

    def diagnostics(self):
        if self.energy is None:
            return {}

        return {
            "energy_mean": float(np.mean(self.energy)),
            "energy_std": float(np.std(self.energy)),
            "energy_max": float(np.max(self.energy)),
            "energy_95": float(np.percentile(self.energy, 95)),
            "energy_99": float(np.percentile(self.energy, 99)),
            "scale_range": (
                int(self.scales[0]),
                int(self.scales[-1])
            )
        }

