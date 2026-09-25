from collections import Counter
from typing import Callable, Iterable

import numpy as np

from src.detection.utils.common import CrashCandidate


DetectorFactory = Callable[[float, float], object]


def collect_multisxr_candidates(
    prepared,
    downsample: int,
    channels: Iterable[str],
    coincidence_window_s: float,
    detector_factory: DetectorFactory,
    debug: bool = False,
) -> list[CrashCandidate]:
    """Union nearby POSR/CPD proposals; never require multi-channel support."""

    downsample = int(downsample)
    if downsample < 1:
        raise ValueError("downsample must be at least 1")
    if not np.isfinite(coincidence_window_s) or coincidence_window_s <= 0:
        raise ValueError("coincidence_window_s must be positive and finite")

    ordered_channels = list(dict.fromkeys(channels))
    if not ordered_channels:
        raise ValueError("At least one proposal channel is required")
    period = float(prepared.estimated_period)
    dt_effective = float(prepared.dt) * downsample
    period_map = (
        None if prepared.period_map is None else prepared.period_map[::downsample]
    )
    plateau_mask = getattr(prepared, "plasma_plateau_mask", None)
    plateau_mask_ds = (
        None if plateau_mask is None else np.asarray(plateau_mask, dtype=bool)[::downsample]
    )

    raw_candidates: list[CrashCandidate] = []
    source_counts = Counter()
    rejected_before_plateau = 0
    for channel_name in ordered_channels:
        signal = prepared.channel_signals.get(channel_name)
        if signal is None:
            continue
        signal_ds = np.asarray(signal, dtype=float)[::downsample]
        detector = detector_factory(dt_effective, period)
        proposals = detector.detect_candidates(
            signal_ds,
            period=period,
            period_map=period_map,
            # The mask comes from the reference SXR. Applying it to every
            # channel would hide proposals visible only in auxiliary SXR.
            active_mask=None,
            channel=channel_name,
            debug=debug,
        )
        for candidate in proposals:
            index = int(candidate.index)
            if plateau_mask_ds is not None and (
                not 0 <= index < len(plateau_mask_ds) or not plateau_mask_ds[index]
            ):
                rejected_before_plateau += 1
                continue
            if 0 <= index * downsample < len(prepared.time_plasma):
                candidate.channel = channel_name
                raw_candidates.append(candidate)
                source_counts[channel_name] += 1

    if not raw_candidates:
        return []

    max_gap = max(1, int(round(coincidence_window_s / dt_effective)))
    raw_candidates.sort(key=lambda candidate: candidate.index)
    clusters: list[list[CrashCandidate]] = [[raw_candidates[0]]]
    for candidate in raw_candidates[1:]:
        # Bound cluster width by its first member to avoid chaining across
        # several distinct nearby crashes.
        if candidate.index - clusters[-1][0].index <= max_gap:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    merged = []
    for cluster in clusters:
        reference_members = [
            candidate
            for candidate in cluster
            if candidate.channel == prepared.channel_name
        ]
        pool = reference_members or cluster
        representative = max(
            pool,
            key=lambda candidate: (
                float(candidate.score) if np.isfinite(candidate.score) else 0.0
            ),
        )
        source_channels = sorted({candidate.channel for candidate in cluster})
        source_methods = sorted({str(candidate.method or "unknown") for candidate in cluster})
        source_channel_methods = {
            channel_name: sorted(
                {str(candidate.method or "unknown") for candidate in cluster if candidate.channel == channel_name}
            )
            for channel_name in source_channels
        }
        source_scores = [
            float(candidate.score) if np.isfinite(candidate.score) else 0.0
            for candidate in cluster
        ]
        index = int(representative.index)
        merged.append(
            CrashCandidate(
                index=index,
                time=float(prepared.time_plasma[index * downsample]),
                score=float(max(source_scores)),
                direction=str(representative.direction),
                channel=prepared.channel_name,
                method="multi_sxr_union",
                meta={
                    "source_channels": source_channels,
                    "source_methods": source_methods,
                    "source_channel_methods": source_channel_methods,
                    "source_scores": source_scores,
                    "source_count": len(source_channels),
                    "proposal_count": len(cluster),
                    "time_spread_s": (
                        max(candidate.index for candidate in cluster)
                        - min(candidate.index for candidate in cluster)
                    ) * dt_effective,
                },
            )
        )

    if debug:
        print(
            f"[MultiSXR] raw={len(raw_candidates)}, merged={len(merged)}, "
            f"rejected_before_plateau={rejected_before_plateau}, "
            f"sources={dict(source_counts)}"
        )
    return sorted(merged, key=lambda candidate: candidate.index)
