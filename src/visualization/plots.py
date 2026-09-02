import os

import matplotlib.pyplot as plt
from config.path import PATH_TO_RES_TEMP
import numpy as np

from config.constants import *


def plot_shot(shot):
    n = shot.n_channels
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ch in enumerate(shot.channel_names):
        axes[i].plot(shot.time, shot.signals[:, i])
        axes[i].set_title(ch)
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_with_crashes(shot, crash_times, channel_name=None, mode="wavelet"):
    if channel_name is None:
        channel_name = shot.channel_names[0]

    if channel_name not in shot.channel_names:
        raise ValueError(
            f"Channel {channel_name} not found. "
            f"Available: {shot.channel_names}"
        )

    ch_index = shot.channel_names.index(channel_name)

    time = shot.time
    signal = shot.signals[:, ch_index]

    plt.figure(figsize=(12, 5))
    plt.plot(time, signal, color='k')

    for t in crash_times:
        idx = np.argmin(np.abs(time - t))
        plt.scatter(t, signal[idx], color='r', marker='.', s=50, zorder=5)
        plt.axvline(t, linestyle="--", color="r", alpha=0.5)

    plt.title(f"{channel_name} with detected crashes", fontsize=FONTSIZE)
    plt.xlabel("Time (s)", fontsize=FONTSIZE)
    plt.ylabel("Signal", fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    # plt.legend()
    plt.grid(True)
    os.makedirs(f"{PATH_TO_RES_TEMP}/{mode}", exist_ok=True)
    plt.savefig(f"{PATH_TO_RES_TEMP}/{mode}/{channel_name}_{shot.metadata['file']}_with_crashes.png")
    plt.show()


def plot_signal_energy_period_map(
    time,
    signal,
    energy,
    period_map=None,
    period_map_acf=None,
    crash_times=None,
    active_mask=None,
    channel_name="SXR",
    filename="shot",
    mode="wavelet_diagnostics",
):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    energy = np.asarray(energy, dtype=float)

    n = min(len(time), len(signal), len(energy))
    if n == 0:
        return
    time = time[:n]
    signal = signal[:n]
    energy = energy[:n]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(time, signal, color="k", linewidth=1.0)
    axes[0].set_ylabel("signal")
    axes[0].set_title(f"{channel_name}: signal, wavelet energy and period maps")
    axes[0].grid(True)

    axes[1].plot(time, energy, linewidth=1.0)
    axes[1].set_ylabel("wavelet energy")
    axes[1].grid(True)

    has_period_map = False
    if period_map_acf is not None:
        pm_acf = np.asarray(period_map_acf, dtype=float)[:n]
        axes[2].plot(
            time[:len(pm_acf)],
            pm_acf * 1e3,
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label="ACF estimate",
        )
        has_period_map = True

    if period_map is not None:
        pm = np.asarray(period_map, dtype=float)[:n]
        axes[2].step(
            time[:len(pm)],
            pm * 1e3,
            where="post",
            color="tab:orange",
            linewidth=1.4,
            label="Refined map (events + ACF fallback)",
        )
        has_period_map = True

    if has_period_map:
        axes[2].set_ylabel("period, ms")
        axes[2].legend(loc="best")
    else:
        axes[2].text(
            0.5,
            0.5,
            "period maps are unavailable",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_ylabel("period, ms")

    axes[2].set_xlabel("time, s")
    axes[2].grid(True)

    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)[:n]
        if len(mask) == n and np.any(mask):
            ymin, ymax = axes[0].get_ylim()
            axes[0].fill_between(time, ymin, ymax, where=mask, alpha=0.08)
            ymin, ymax = axes[1].get_ylim()
            axes[1].fill_between(time, ymin, ymax, where=mask, alpha=0.08)
            ymin, ymax = axes[2].get_ylim()
            axes[2].fill_between(time, ymin, ymax, where=mask, alpha=0.08)

    if crash_times is not None:
        for t in crash_times:
            for ax in axes:
                ax.axvline(float(t), linestyle="--", color="r", alpha=0.45)

    os.makedirs(f"{PATH_TO_RES_TEMP}/{mode}", exist_ok=True)
    safe_channel = str(channel_name).replace("/", "_").replace("\\", "_")
    safe_file = str(filename).replace("/", "_").replace("\\", "_")
    plt.tight_layout()
    plt.savefig(f"{PATH_TO_RES_TEMP}/{mode}/{safe_channel}_{safe_file}_energy_period_map.png")
    plt.show()


def plot_enhance_crash_energy(
    energy_median,
    energy,
    noise,
    noise_level,
    energy_norm,
    channel_name="SXR",
    filename="shot",
    mode="wavelet_diagnostics",
):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(energy, color="g", linewidth=1.0, label="energy")
    axes[0].plot(energy_median, 'b--', linewidth=1.0, label="median energy")
    axes[0].plot(noise, 'r--', linewidth=1.0, label="noise (σ={:.4f})".format(noise_level))

    # axes[0].axhline(2.5*noise_level, color="r", linewidth=1.0)
    axes[0].axhline(3*noise_level, color="r", linewidth=1.0)

    axes[1].plot(energy, color="g", linewidth=1.0, label="energy")
    axes[1].plot(energy_norm, 'r', linewidth=1.0, label="energy enhanced")

    axes[0].grid(True, alpha=0.5)
    axes[1].grid(True, alpha=0.5)

    axes[0].legend()
    axes[1].legend()

    os.makedirs(f"{PATH_TO_RES_TEMP}/{mode}", exist_ok=True)
    safe_channel = str(channel_name).replace("/", "_").replace("\\", "_")
    safe_file = str(filename).replace("/", "_").replace("\\", "_")
    plt.tight_layout()
    plt.savefig(f"{PATH_TO_RES_TEMP}/{mode}/{safe_channel}_{safe_file}_energy_enhance_diagnostic.png")
    plt.show()






