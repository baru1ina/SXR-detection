import matplotlib.pyplot as plt
from config.path import PATH_TO_RES_TEMP
import numpy as np


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


def plot_with_crashes(shot, crash_times, channel_name=None):
    fs = 20
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

    plt.title(f"{channel_name} with detected crashes", fontsize=fs)
    plt.xlabel("Time (s)", fontsize=fs)
    plt.ylabel("Signal", fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    # plt.legend()
    plt.grid(True)

    plt.savefig(f"{PATH_TO_RES_TEMP}/{channel_name}_{shot.metadata['file']}_with_crashes.png")
    plt.show()
