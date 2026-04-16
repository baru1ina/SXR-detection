import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
from scipy.signal import medfilt

from src.anomaly.smoothing import smooth_signal_gauss, smooth_signal_savgol, smooth_signal_median

# гпт-хрючево

class CPDDetector:
    def __init__(self,
                 dt,
                 penalty=3,
                 model="rbf",
                 min_distance_factor=0.8,
                 drop_window_factor=0.1,
                 drop_threshold_factor=0.2):

        self.dt = dt
        self.penalty = penalty
        self.model = model

        self.min_distance_factor = min_distance_factor
        self.drop_window_factor = drop_window_factor
        self.drop_threshold_factor = drop_threshold_factor


    def _get_candidates(self, x):
        algo = rpt.KernelCPD(kernel="rbf").fit(x)
        bkps = algo.predict(pen=self.penalty)
        return np.array(bkps[:-1], dtype=int)

    def _cluster_candidates(self, bkps, period_map):
        clustered = []
        current = [bkps[0]]
        for p in bkps[1:]:
            window = period_map[max(0, p - 50):p + 50]
            local_period = np.median(window)
            min_distance = int(self.min_distance_factor *
                               local_period / self.dt)

            if p - current[-1] < min_distance:
                current.append(p)
            else:
                clustered.append(current)
                current = [p]
        clustered.append(current)
        candidates = []
        for cluster in clustered:
            candidates.append(cluster[len(cluster) // 2])
        return np.array(candidates, dtype=int)

    def _validate_crashes(self, x, candidates):
        crash_window = int(50e-6 / self.dt)
        valid = []
        for idx in candidates:
            if idx + crash_window >= len(x):
                continue
            segment = x[idx:idx + crash_window]
            drop = x[idx] - np.min(segment)
            if drop > self.drop_threshold:
                valid.append(idx)
        return np.array(valid, dtype=int)

    def _preprocess_signal(self, x, method="diff"):
        if method == "raw":
            return x.copy()

        elif method == "diff":
            dx = np.diff(x)
            return np.concatenate([[0], dx])

        elif method == "diff2":
            dx2 = np.diff(x, n=2)
            return np.concatenate([[0, 0], dx2])

        elif method == "absdiff":
            dx = np.diff(x)
            dx = np.abs(dx)
            return np.concatenate([[0], dx])

        elif method == "wavelet_energy":
            import pywt
            scales = np.arange(1, 32)
            coeffs, _ = pywt.cwt(x, scales, "gaus1")
            energy = np.sum(np.abs(coeffs) ** 2, axis=0)
            energy = energy - np.mean(energy)
            return energy
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def detect(self, signal, period_map, debug=False):
        if len(signal) < 100:
            return np.array([], dtype=int)

        x = medfilt(signal, kernel_size=31)

        x = x - np.mean(x)

        import matplotlib.pyplot as plt

        dx = self._preprocess_signal(x, method="diff")

        # plt.plot(np.linspace(0, dx, len(dx)), dx)
        plt.plot(dx)
        plt.show()

        bkps = self._get_candidates(dx)

        if len(bkps) == 0:
            return np.array([], dtype=int)

        if debug:
            print(f"[CPD] raw: {len(bkps)}")

        candidates = self._cluster_candidates(bkps, period_map)

        if debug:
            print(f"[CPD] clustered: {len(candidates)}")

        self.drop_threshold = self.drop_threshold_factor * np.std(x)
        valid = self._validate_crashes(x, candidates)

        if debug:
            print(f"[CPD] final: {len(valid)}")
        return valid

    def detect_old(self, signal, period=None, debug=False):
        if len(signal) < 100: return np.array([])

        from scipy.signal import medfilt
        x = medfilt(signal, kernel_size=31)

        # x = x - np.mean(x)

        x = self._preprocess_signal(x, method="wavelet_energy")
        # x = self._preprocess_signal(x, method="diff")
        # x = self._preprocess_signal(x, method="diff2")
        # x = self._preprocess_signal(x, method="absdiff")

        import matplotlib.pyplot as plt
        plt.plot(x)

        algo = rpt.KernelCPD(kernel="rbf").fit(x)
        bkps = algo.predict(pen=self.penalty)

        bkps = np.array(bkps[:-1])

        for idx, _ in enumerate(bkps):
            plt.axvline(bkps[idx], linestyle="--", color="r", alpha=0.5)
        plt.show()

        print(f"[CPD] raw breakpoints: {len(bkps)}")

        if len(bkps) == 0:
            return np.array([])

        if period is None and len(bkps) > 1:
            diffs = np.diff(bkps) * self.dt
            period = np.median(diffs)
            print(f"[CPD] estimated period: {period:.6e} s")

        if period is None:
            return np.array([])

        min_distance = int(self.min_distance_factor * period / self.dt)
        clustered = []
        current = [bkps[0]]

        for p in bkps[1:]:
            if p - current[-1] < min_distance:
                current.append(p)
            else:
                clustered.append(current)
                current = [p]

        clustered.append(current)

        candidates = [int(np.mean(c)) for c in clustered]

        print(f"[CPD] clustered candidates: {len(candidates)}")

        drop_window = int(self.drop_window_factor * period / self.dt)
        std = np.std(x)
        valid = []

        # for idx in candidates:
        #     if idx + drop_window >= len(x):
        #         continue
        #
        #     drop = x[idx] - x[idx + drop_window]
        #
        #     if drop > self.drop_threshold_factor * std:
        #         valid.append(idx)
        #
        # print(f"[CPD] valid after drop: {len(valid)}")

        valid = candidates
        reset_times = np.array(valid) * self.dt

        return np.array(valid)

