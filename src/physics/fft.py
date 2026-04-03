import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def harmonic_score(freqs, spectrum, f0):
    harmonics = [f0, 2*f0, 3*f0]
    energy = 0
    for h in harmonics:
        idx = np.argmin(np.abs(freqs - h))
        energy += spectrum[idx]
    return energy


def fft_score(signal, dt, estimated_period, min_freq=0, max_freq=5000):
    n = len(signal)

    # max_freq = 100000

    fft = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, dt)

    print("max freq:", np.max(freqs))

    mask = (freqs > min_freq) & (freqs < max_freq)

    if not np.any(mask):
        return 0

    # spectrum_full = np.abs(fft)
    spectrum = np.abs(fft[mask])
    # freqs_masked = freqs[mask]

    peak = np.max(spectrum)
    mean = np.mean(spectrum)

    if mean == 0:
        score = 0
    else:
        score = peak / mean

    n = len(signal)
    # time = np.arange(n) * dt * 1000

    harm_energy = harmonic_score(freqs, spectrum, 1 / estimated_period)

    is_noise = harm_energy < 0.2 * np.sum(spectrum)

    return score, is_noise


def fft_score_with_drift(signal, logger, dt, times, periods, scores,
                         min_freq=100, max_freq=5000,
                         window_ms=20, step_ms=5):

    window = int(window_ms * 1e-3 / dt)
    step = int(step_ms * 1e-3 / dt)

    normalized_spectra = []
    window_f0_list = []
    window_scores_list = []
    harmonic_grid = np.arange(0.5, 5.1, 0.05)

    window_ratios_2_1 = []
    window_ratios_3_1 = []
    window_harmonic_ratios = []

    logger.info(f"Дебаг fft_score_with_drift")
    logger.info(f"Параметры: окно={window_ms} мс, шаг={step_ms} мс")
    logger.info(f"Диапазон частот: {min_freq} - {max_freq} Гц")

    for i, start in enumerate(range(0, len(signal) - window, step)):
        seg = signal[start:start + window]
        local_period = periods[i] if not np.isnan(periods[i]) else None
        local_score = scores[i] if i < len(scores) else 0

        if local_period is None:
            continue

        f0 = 1.0 / local_period
        window_f0_list.append(f0)
        window_scores_list.append(local_score)

        n = len(seg)
        fft = np.fft.rfft(seg - np.mean(seg))
        freqs = np.fft.rfftfreq(n, dt)
        spectrum = np.abs(fft)

        harmonic_numbers = freqs / f0

        valid = (harmonic_numbers >= 0.5) & (harmonic_numbers <= 5)
        if np.sum(valid) < 10:
            continue

        f_interp = interp1d(harmonic_numbers[valid],
                            spectrum[valid],
                            kind='linear',
                            bounds_error=False,
                            fill_value=0)

        spectrum_norm = f_interp(harmonic_grid)
        normalized_spectra.append(spectrum_norm)

        idx_1 = np.argmin(np.abs(harmonic_grid - 1))
        idx_2 = np.argmin(np.abs(harmonic_grid - 2))
        idx_3 = np.argmin(np.abs(harmonic_grid - 3))

        A1_win = spectrum_norm[idx_1]
        A2_win = spectrum_norm[idx_2]
        A3_win = spectrum_norm[idx_3]

        ratio_2_1_win = A2_win / (A1_win + 1e-8)
        ratio_3_1_win = A3_win / (A1_win + 1e-8)

        window_ratios_2_1.append(ratio_2_1_win)
        window_ratios_3_1.append(ratio_3_1_win)

        harmonic_energy_win = A1_win ** 2 + A2_win ** 2 + A3_win ** 2
        total_energy_win = np.sum(spectrum_norm ** 2)
        harmonic_ratio_win = harmonic_energy_win / (total_energy_win + 1e-8)
        window_harmonic_ratios.append(harmonic_ratio_win)

    logger.info(f"Обработано окон: {len(normalized_spectra)}")
    logger.info(f"Средняя частота: {np.mean(window_f0_list):.2f} Гц")
    logger.info(f"Средняя оценка автокорреляции: {np.mean(window_scores_list):.3f}")

    avg_spectrum = np.mean(normalized_spectra, axis=0)

    idx_1 = np.argmin(np.abs(harmonic_grid - 1))
    idx_2 = np.argmin(np.abs(harmonic_grid - 2))
    idx_3 = np.argmin(np.abs(harmonic_grid - 3))
    idx_4 = np.argmin(np.abs(harmonic_grid - 4))

    A1 = avg_spectrum[idx_1]
    A2 = avg_spectrum[idx_2]
    A3 = avg_spectrum[idx_3]
    A4 = avg_spectrum[idx_4]

    ratio_2_1 = A2 / (A1 + 1e-8)
    ratio_3_1 = A3 / (A1 + 1e-8)
    ratio_4_1 = A4 / (A1 + 1e-8)

    harmonic_energy = A1 ** 2 + A2 ** 2 + A3 ** 2
    total_energy = np.sum(avg_spectrum ** 2)
    harmonic_ratio = harmonic_energy / (total_energy + 1e-8)

    logger.info(f"Количество окон: {len(normalized_spectra)}")
    logger.info(f"Средняя частота: {np.mean(window_f0_list):.2f} Гц (период {1000 / np.mean(window_f0_list):.2f} мс)")
    logger.info(f"Дрейф частоты: {np.std(window_f0_list):.2f} Гц")
    logger.info(f"Амплитуды: A1={A1:.2f}, A2={A2:.2f}, A3={A3:.2f}, A4={A4:.2f}")
    logger.info(f"Отношения: A2/A1={ratio_2_1:.3f}, A3/A1={ratio_3_1:.3f}, A4/A1={ratio_4_1:.3f}")
    logger.info(f"Гармоническое отношение: {harmonic_ratio:.4f} ({harmonic_ratio * 100:.2f}%)")

    return harmonic_ratio, ratio_2_1, np.array(window_f0_list)

