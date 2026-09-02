import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# from src.physics.sawtooth_filter import is_sawtooth


def plot_spectrum_with_harmonics(signal, dt, f0):
    n = len(signal)

    fft = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, dt)
    spectrum = np.abs(fft)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, spectrum, label="Спектр")

    for k in range(1, 6):
        plt.axvline(k * f0, linestyle='--', label=f"{k}f0" if k == 1 else None)

    plt.xlabel("Частота (Гц)")
    plt.xlim(0, 5000)
    plt.ylabel("Амплитуда")
    plt.title("Спектр с гармониками")
    plt.legend()
    plt.grid()
    plt.show()

def plot_normalized_spectrum(harmonic_grid, avg_spectrum):
    plt.figure(figsize=(8, 4))
    plt.plot(harmonic_grid, avg_spectrum, marker='o')

    for k in range(1, 5):
        plt.axvline(k, linestyle='--', alpha=0.5)

    plt.xlabel("Номер гармоники (f / f0)")
    plt.ylabel("Амплитуда")
    plt.title("Нормированный спектр")
    plt.grid()
    plt.show()


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


def classify_harmonic_ratio(h_ratio):
    if h_ratio > 0.3:
        return "Шум"
    elif h_ratio > 0.2:
        return "... ?"
    elif h_ratio > 0.1:
        return "... ?"
    else:
        return "... ?"


def fft_score_with_drift(signal, logger, dt, times, periods, scores,
                         min_freq=100, max_freq=5000,
                         window_ms=20, step_ms=5,
                         ac_threshold=0.3,
                         harmonic_threshold=0.4,
                         min_ac_fraction=0.25):

    #TODO: важная оговорка! - окно сейчас 20 мс. Это хорошо для устойчивого спектра, но плохо для очень короткой пачки
    # быстрых мелких колебаний. Если в конце пачка с периодом 1 мс, то в 20 мс теоретически помещается много циклов,
    # но если сама пачка длится, например, только 3-5 мс, FFT-окно смешает её с соседним режимом. Тогда спектр будет
    # “средней смесью”: часть энергии от старого периода, часть от быстрой пачки, часть от перехода.
    # Надо подумать, может, стоит адаптировать ширину окна в зависимости от значения на карте периодов.

    window = int(window_ms * 1e-3 / dt)
    step = int(step_ms * 1e-3 / dt)

    window_f0_list = []
    harmonic_ratios = []
    peakiness_list = []

    # harmonic_counts = []
    decay_scores = []

    for i, start in enumerate(range(0, len(signal) - window, step)):
        seg = signal[start:start + window]

        local_period = periods[i] if i < len(periods) and not np.isnan(periods[i]) else None

        if local_period is None:
            continue

        f0 = 1.0 / local_period
        window_f0_list.append(f0)

        n = len(seg)

        win = np.hanning(n)
        seg_win = (seg - np.mean(seg)) * win

        fft = np.fft.rfft(seg_win, n=4 * n)
        freqs = np.fft.rfftfreq(4 * n, dt)

        spectrum = np.abs(fft)

        mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(mask):
            continue

        freqs = freqs[mask]
        spectrum = spectrum[mask]

        spectrum_energy = spectrum ** 2
        total_energy = np.sum(spectrum_energy) + 1e-12

        peakiness = np.max(spectrum) / (np.mean(spectrum) + 1e-12)
        peakiness_list.append(peakiness)

        def band_energy(center, width=0.15):
            band = (freqs > center * (1 - width)) & (freqs < center * (1 + width))
            if not np.any(band):
                return 0
            return np.sum(spectrum_energy[band])

        harmonic_amps = []
        for n_h in range(1, 7):
            A = band_energy(n_h * f0)
            harmonic_amps.append(A)

        harmonic_amps = np.array(harmonic_amps)

        harmonic_energy = np.sum(harmonic_amps[:3])
        harmonic_ratio = harmonic_energy / total_energy
        harmonic_ratios.append(harmonic_ratio)

        noise_level = np.median(spectrum_energy)
        detected = harmonic_amps > 3 * noise_level
        # num_harmonics = np.sum(detected)
        # harmonic_counts.append(num_harmonics)

        A1 = harmonic_amps[0] + 1e-12
        decay_errors = []

        for k in range(1, len(harmonic_amps)):
            if harmonic_amps[k] == 0:
                continue

            expected = A1 / (k + 1)**2
            actual = harmonic_amps[k]

            decay_errors.append(abs(actual - expected) / A1)

        if len(decay_errors) > 0:
            decay_score = np.mean(decay_errors)
        else:
            decay_score = np.inf

        decay_scores.append(decay_score)

    if len(harmonic_ratios) == 0:
        return 0, 0, np.array([]), True

    #TODO: тут есть проблема с усреднением. В теории у нас "идеальная" пила с одним периодом и, следовательно
    # одним набором гармоник, а не дрейфующий по всей пиле период с разными наборами гармоник в каждом окне.
    # Хорошо, что сделан fft по окнам (хотя есть нюансы), плохо, что в итоге, усредняя по всем окнам, мы никак не учитываем
    # их неоднородность. Таким образом, итоговые физические пороги не имеют реального физического смысла, т.к. смешивают
    # разные режимы в одно число. Есть идея не говорить “весь сигнал saw / not saw” в сигнале по усредненным характеристикам,
    # а строить локальную маску. Проблема в том, что текущая реализация fft_score_with_drift() отбраковывает сигналы с
    # наводкой (типа очень большой пик в сигнале), но пропускают шумные сигналы с периодическим фоновым шумом. Поэтому
    # для отбраковки таких сигналов и добавлен критерий на соотношении дисперсий в области тока и предыстории.
    # Можно ли как-то настроить именно на этом этапе? Или надо пробрасывать в конечный детектор и смотреть итоговые значения на соответствие маске?


    # harmonic_ratio = np.mean(harmonic_ratios)
    # peakiness = np.mean(peakiness_list)
    # mean_decay = np.mean(decay_scores) if len(decay_scores) > 0 else np.inf
    # autocorr_mean = np.mean(scores) if len(scores) > 0 else 0

    harmonic_ratio = np.median(harmonic_ratios)
    peakiness = np.median(peakiness_list)
    mean_decay = np.median(decay_scores) if len(decay_scores) > 0 else np.inf
    score_array = np.asarray(scores, dtype=float)
    period_array = np.asarray(periods, dtype=float)
    reliable_ac = (
        np.isfinite(score_array)
        & np.isfinite(period_array)
        & (score_array > 0)
    )
    autocorr_mean = np.median(score_array[reliable_ac]) if np.any(reliable_ac) else 0
    ac_valid_fraction = float(np.mean(reliable_ac)) if len(reliable_ac) else 0.0

    f0_array = np.array(window_f0_list)

    if ac_valid_fraction < min_ac_fraction:
        is_saw = False
    elif autocorr_mean < ac_threshold:
        is_saw = False
    elif peakiness < 8:
        is_saw = False
    elif harmonic_ratio < harmonic_threshold:
        is_saw = False
    elif mean_decay > 0.5:
        is_saw = False
    else:
        is_saw = True

    # is_saw = (
    #         autocorr_mean > 0.6 and
    #         peakiness > 8 and
    #         mean_decay < 2.0 and
    #         mean_harmonics >= 3)
    #     is_saw = (
    #     harmonic_ratio > 0.08 and
    #     peakiness > 10 and
    #     autocorr_mean > 0.7 and
    #     mean_harmonics >= 3 and
    #     mean_decay < 3.0

    # classification = classify_harmonic_ratio(harmonic_ratio)

    # logger.warning(
    #     f"Гармоническое отношение: {harmonic_ratio:.4f} ({harmonic_ratio * 100:.2f}%) | {classification}"
    # )

    #TODO: можно добавить какую-то общую оценки типа "если хороших окон >= 50% от активной области -> saw подтверждается".
    # Тогда в статистику можно будет выводить какие-нибудь объединенные интервалы на окнах и предположение о характере
    # колебаний именно в окне (пила/синусоида/шум и т.д.).

    logger.info(f"Пиковость спектра: {peakiness:.2f}")
    logger.info(
        f"Автокорреляция: {autocorr_mean:.3f}; "
        f"доля валидных окон: {ac_valid_fraction:.3f}"
    )
    logger.info(f"Доля энергии первых трёх гармоник: {harmonic_ratio:.3f}")
    logger.info(f"Отклонение от 1/n: {mean_decay:.3f}")

    logger.info(f"has saw = {is_saw}")

    return harmonic_ratio, peakiness, f0_array, is_saw
