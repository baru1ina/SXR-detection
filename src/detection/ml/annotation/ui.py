from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.path import (
    DEFAULT_ANNOTATION_UI_LOG_PATH,
    DEFAULT_EXPERT_LABELS_PATH,
    DEFAULT_PSEUDO_LABELS_PATH,
    IP_CHANNEL,
    SXR_CHANNELS,
)
from src.detection.ml.annotation.backend import (
    AutomaticSuggestion,
    build_shot_catalog,
    decimate_minmax,
    default_display_channels,
    load_channels,
    merge_reviewed_interval,
)
from src.detection.ml.annotation.plot_viewport import render_plot_viewport_listener
from src.detection.ml.annotation.store import (
    AnnotationConflictError,
    load_annotation_document,
    upsert_shot_annotation,
)
from src.detection.ml.train.expert_labels import (
    EXPERT_LABEL_SCHEMA_VERSION,
    parse_expert_labels,
)
from src.detection.ml.train.pseudo_labels import load_pseudo_labels
from src.logger import setup_logger
from src.preprocessing.plasma_detection import detect_plasma_interval


SOURCE_LABELS = {
    "wavelet_teacher": "Wavelet-разметка",
}
SOURCE_COLORS = {
    "wavelet_teacher": "#4c78a8",
}
SOURCE_SYMBOLS = {
    "wavelet_teacher": "circle",
}

EVENT_LABELS = {
    "sawtooth": "Срыв",
    "uncertain": "Возможный срыв",
    "false_positive": "Ложный срыв",
}
SHOT_LABELS = {
    "sawtooth": "Пилообразные колебания присутствуют",
    "no_sawtooth": "Пилообразных колебаний нет",
    "uncertain": "Не уверен(а)",
}

ACTION_LABELS = {
    "add_sawtooth": "Добавить срыв",
    "reject": "Пометить как ложный срыв",
    "select_interval": "Выделить интервал",
    "inspect": "Выбрать / редактировать",
}
ACTION_ICON_FILES = {
    "inspect": "redact.png",
    "add_sawtooth": "accept.png",
    "reject": "cancel.png",
    "select_interval": "interval.png",
}
ANNOTATION_MEDIA_DIR = Path(__file__).resolve().with_name("media")
EVENT_TOLERANCE_S = 0.3e-3


def _canonical_shot_id(value: str) -> str:
    value = str(value).strip().casefold()
    return f"sht{value}" if value.isdigit() else value


@st.cache_data(show_spinner=False)
def _action_icon_label(action: str, fallback: str) -> str:
    """Return an embedded PNG label suitable for a Streamlit button."""

    icon_path = ANNOTATION_MEDIA_DIR / ACTION_ICON_FILES[action]
    try:
        encoded = b64encode(icon_path.read_bytes()).decode("ascii")
    except OSError:
        return fallback
    return f"![{fallback}](data:image/png;base64,{encoded})"


def _render_action_icon_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button:has(img) {
            min-height: 3.25rem;
            padding: 0.35rem;
        }
        div[data-testid="stButton"] button img {
            width: 1.5rem !important;
            height: 1.5rem !important;
            max-height: none !important;
            object-fit: contain;
            margin: 0 !important;
        }
        div[data-testid="stButton"] button:has(img) p {
            margin: 0;
            line-height: 1;
        }
        .annotation-shot-meta {
            color: var(--text-color);
            font-size: 1.05rem;
            line-height: 1.5;
            margin: -0.45rem 0 0.8rem;
            opacity: 0.78;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_view_revision(
    source_dir: str,
    channels: tuple[str, ...],
    signal_height: int,
    display_start_s: float,
    display_end_s: float,
    show_grid: bool,
    y_ranges: dict[str, tuple[float, float]],
) -> str:
    """Return a compact widget revision for settings that rebuild the view."""

    payload = "\0".join(
        (
            str(Path(source_dir)),
            str(signal_height),
            f"{display_start_s:.12g}",
            f"{display_end_s:.12g}",
            str(show_grid),
            *(
                f"{name}\0{bounds[0]:.12g}\0{bounds[1]:.12g}"
                for name, bounds in y_ranges.items()
            ),
            *channels,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _plot_viewport_token(state_key: str) -> str:
    return sha256(state_key.encode("utf-8")).hexdigest()[:16]


def _plot_viewport_listener_key(state_key: str) -> str:
    return f"plot_viewport_listener::{_plot_viewport_token(state_key)}"


def _valid_axis_range(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    return start, end


def _consume_plot_viewport(state_key: str, channels: tuple[str, ...]) -> None:
    """Copy the latest browser relayout ranges into persistent input fields."""

    listener_state = st.session_state.get(_plot_viewport_listener_key(state_key))
    if not hasattr(listener_state, "get"):
        return
    viewport = listener_state.get("viewport")
    if not hasattr(viewport, "get"):
        return
    sequence = viewport.get("sequence")
    applied_key = f"plot_viewport_sequence::{state_key}"
    if sequence is None or st.session_state.get(applied_key) == sequence:
        return

    x_range = _valid_axis_range(viewport.get("x_range"))
    if x_range is not None:
        st.session_state[f"display_start::{state_key}"] = x_range[0]
        st.session_state[f"display_end::{state_key}"] = x_range[1]

    raw_y_ranges = viewport.get("y_ranges")
    if hasattr(raw_y_ranges, "items"):
        for channel, raw_range in raw_y_ranges.items():
            if channel not in channels:
                continue
            y_range = _valid_axis_range(raw_range)
            if y_range is None:
                continue
            st.session_state[f"display_y_min::{state_key}::{channel}"] = y_range[0]
            st.session_state[f"display_y_max::{state_key}::{channel}"] = y_range[1]

    active_y_channel = viewport.get("active_y_channel")
    if active_y_channel in channels:
        st.session_state[f"display_y_channel::{state_key}"] = active_y_channel
    st.session_state[applied_key] = sequence


def _annotation_logger():
    log_path = Path(DEFAULT_ANNOTATION_UI_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return setup_logger(log_to_file=True, log_file_path=log_path.name)


@st.cache_data(show_spinner=False)
def _available_channels(source_dir: str, relative_path: str) -> list[str]:
    logger = _annotation_logger()
    from src.io.loader import SHTLoader

    return SHTLoader(source_dir, logger).list_available_channels(relative_path)


@st.cache_data(show_spinner="Загрузка каналов SHT…")
def _plot_channel_data(
    source_dir: str,
    relative_path: str,
    channels: tuple[str, ...],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, float]:
    """Load and decimate signals once instead of on every UI rerun."""

    logger = _annotation_logger()
    shot = load_channels(source_dir, relative_path, channels, logger)
    plot_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    starts: list[float] = []
    ends: list[float] = []
    for name in shot.channel_names:
        channel = shot.get_channel(name)
        time = np.asarray(channel.time, dtype=float)
        values = np.asarray(channel.values, dtype=float)
        starts.append(float(time[0]))
        ends.append(float(time[-1]))
        plot_data[name] = decimate_minmax(time, values)
    return plot_data, min(starts), max(ends)


def _natural_y_ranges(
    plot_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[float, float]]:
    """Return finite per-channel Y bounds with a small visual padding."""

    result: dict[str, tuple[float, float]] = {}
    for name, (_, values) in plot_data.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            result[name] = (-1.0, 1.0)
            continue
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        span = upper - lower
        padding = span * 0.05 if span > 0 else max(abs(lower) * 0.05, 1.0)
        result[name] = (lower - padding, upper + padding)
    return result


@st.cache_data(show_spinner=False)
def _plasma_display_interval(
    source_dir: str,
    relative_path: str,
) -> tuple[float, float] | None:
    """Reuse the detector's 20%-of-maximum current interval for the UI."""

    logger = _annotation_logger()
    shot = load_channels(source_dir, relative_path, (IP_CHANNEL,), logger)
    current = shot.get_channel(IP_CHANNEL)
    interval = detect_plasma_interval(
        np.asarray(current.values, dtype=float),
        np.asarray(current.time, dtype=float),
        threshold_ratio=0.2,
    )
    if interval is None:
        return None
    return float(interval[0]), float(interval[1])


def _render_graph_view_controls(
    source_dir: str,
    relative_path: str,
    state_key: str,
    full_start: float,
    full_end: float,
    natural_y_ranges: dict[str, tuple[float, float]],
) -> tuple[
    int,
    float,
    float,
    bool,
    tuple[float, float],
    dict[str, tuple[float, float]],
    str,
]:
    """Render persistent graph bounds and return the effective view."""

    with st.expander("Вид графика", expanded=False):
        signal_height = st.slider(
            "Высота одного канала, px",
            min_value=180,
            max_value=420,
            value=260,
            step=20,
            key=f"signal_height::{state_key}",
        )
        show_grid = st.checkbox(
            "Сетка",
            value=True,
            key=f"show_grid::{state_key}",
        )
        crop_key = f"crop_by_current::{state_key}"
        crop_by_current = st.toggle(
            "Обрезать сигнал по току (20%)",
            key=crop_key,
            help=(
                f"Используется {IP_CHANNEL}: остаётся интервал, в котором ток "
                "выше 20% своего максимума."
            ),
        )

        allowed_start, allowed_end = float(full_start), float(full_end)
        if crop_by_current:
            try:
                plasma_interval = _plasma_display_interval(source_dir, relative_path)
            except Exception as exc:
                plasma_interval = None
                st.warning(f"Не удалось определить интервал по току: {exc}")
            if plasma_interval is None:
                st.warning("По току не найден плазменный интервал; показан весь сигнал.")
            else:
                allowed_start = max(allowed_start, float(plasma_interval[0]))
                allowed_end = min(allowed_end, float(plasma_interval[1]))

        start_key = f"display_start::{state_key}"
        end_key = f"display_end::{state_key}"
        crop_revision_key = f"display_crop_revision::{state_key}"
        previous_crop = st.session_state.get(crop_revision_key)
        if start_key not in st.session_state or previous_crop != crop_by_current:
            st.session_state[start_key] = allowed_start
        if end_key not in st.session_state or previous_crop != crop_by_current:
            st.session_state[end_key] = allowed_end
        st.session_state[crop_revision_key] = crop_by_current

        st.session_state[start_key] = min(
            max(float(st.session_state[start_key]), allowed_start), allowed_end
        )
        st.session_state[end_key] = min(
            max(float(st.session_state[end_key]), allowed_start), allowed_end
        )

        channels = tuple(natural_y_ranges)
        y_channel_key = f"display_y_channel::{state_key}"
        if st.session_state.get(y_channel_key) not in channels:
            st.session_state[y_channel_key] = channels[0]
        y_channel = st.selectbox(
            "Канал для оси Y",
            channels,
            key=y_channel_key,
        )
        for channel, natural_range in natural_y_ranges.items():
            y_min_key = f"display_y_min::{state_key}::{channel}"
            y_max_key = f"display_y_max::{state_key}::{channel}"
            if y_min_key not in st.session_state:
                st.session_state[y_min_key] = float(natural_range[0])
            if y_max_key not in st.session_state:
                st.session_state[y_max_key] = float(natural_range[1])

        y_min_key = f"display_y_min::{state_key}::{y_channel}"
        y_max_key = f"display_y_max::{state_key}::{y_channel}"
        first_bounds_row = st.columns(2)
        display_start = first_bounds_row[0].number_input(
            "Левая граница X, с",
            format="%.6f",
            step=0.001,
            min_value=allowed_start,
            max_value=allowed_end,
            key=start_key,
        )
        display_y_min = first_bounds_row[1].number_input(
            "Нижняя граница Y",
            format="%.6g",
            key=y_min_key,
        )
        second_bounds_row = st.columns(2)
        display_end = second_bounds_row[0].number_input(
            "Правая граница X, с",
            format="%.6f",
            step=0.001,
            min_value=allowed_start,
            max_value=allowed_end,
            key=end_key,
        )
        display_y_max = second_bounds_row[1].number_input(
            "Верхняя граница Y",
            format="%.6g",
            key=y_max_key,
        )

        def reset_display_bounds() -> None:
            st.session_state[start_key] = allowed_start
            st.session_state[end_key] = allowed_end
            for channel, natural_range in natural_y_ranges.items():
                st.session_state[f"display_y_min::{state_key}::{channel}"] = float(
                    natural_range[0]
                )
                st.session_state[f"display_y_max::{state_key}::{channel}"] = float(
                    natural_range[1]
                )

        st.button(
            "Сбросить границы",
            width="stretch",
            key=f"reset_display_bounds::{state_key}",
            on_click=reset_display_bounds,
        )

        effective_start = max(float(display_start), allowed_start)
        effective_end = min(float(display_end), allowed_end)
        if effective_end <= effective_start:
            st.error("Правая граница должна быть больше левой.")
            st.stop()
        if (
            abs(effective_start - float(display_start)) > 1e-12
            or abs(effective_end - float(display_end)) > 1e-12
        ):
            st.caption(
                f"Эффективный диапазон: {effective_start:.6f}–"
                f"{effective_end:.6f} с"
            )
        if float(display_y_max) <= float(display_y_min):
            st.error("Верхняя граница Y должна быть больше нижней.")
            st.stop()

        y_ranges = {
            channel: (
                float(st.session_state[f"display_y_min::{state_key}::{channel}"]),
                float(st.session_state[f"display_y_max::{state_key}::{channel}"]),
            )
            for channel in channels
        }
    return (
        signal_height,
        effective_start,
        effective_end,
        show_grid,
        (allowed_start, allowed_end),
        y_ranges,
        y_channel,
    )


def _find_raw_annotation(document, shot_id: str):
    canonical = _canonical_shot_id(shot_id)
    for item in document.payload["shots"]:
        if _canonical_shot_id(item.get("shot_id", "")) == canonical:
            return deepcopy(item)
    return None


def _new_draft(shot) -> dict:
    return {
        "shot_id": shot.shot_id,
        "relative_path": shot.relative_path,
        "review_status": "partial",
        "shot_label": "uncertain",
        "reviewed_intervals_s": [],
        "events": [],
        "note": "",
    }


def _ensure_local_review(draft: dict, time_s: float, tolerance_s: float) -> None:
    radius = max(float(tolerance_s), 1e-6)
    draft["reviewed_intervals_s"] = merge_reviewed_interval(
        draft["reviewed_intervals_s"],
        max(0.0, float(time_s) - radius),
        float(time_s) + radius,
    )


def _add_manual_event(
    draft: dict,
    time_s: float,
    label: str,
) -> tuple[int, bool]:
    """Single entry point for graph, contextual, and exact-time additions."""

    if label not in {"sawtooth", "uncertain"}:
        raise ValueError(f"Unsupported manual event label: {label}")
    events = draft["events"]
    duplicate_index = next(
        (
            index
            for index, event in enumerate(events)
            if event["label"] == label
            and abs(float(event["time_s"]) - float(time_s)) < 1e-12
        ),
        None,
    )
    if duplicate_index is not None:
        return duplicate_index, False

    event = {
        "time_s": float(time_s),
        "label": label,
    }
    events.append(event)
    events.sort(key=lambda item: float(item["time_s"]))
    event_index = events.index(event)
    _ensure_local_review(draft, event["time_s"], EVENT_TOLERANCE_S)
    if label == "sawtooth":
        draft["shot_label"] = "sawtooth"
    return event_index, True


def _reject_suggestion(
    draft: dict,
    suggestion: AutomaticSuggestion,
) -> tuple[int, bool]:
    """Single entry point for all automatic-proposal rejections."""

    events = draft["events"]
    duplicate_index = next(
        (
            index
            for index, event in enumerate(events)
            if event["label"] == "false_positive"
            and abs(float(event["time_s"]) - suggestion.time_s) < 1e-12
        ),
        None,
    )
    if duplicate_index is not None:
        return duplicate_index, False

    event = {
        "time_s": suggestion.time_s,
        "label": "false_positive",
    }
    events.append(event)
    events.sort(key=lambda item: float(item["time_s"]))
    event_index = events.index(event)
    _ensure_local_review(draft, suggestion.time_s, EVENT_TOLERANCE_S)
    return event_index, True


def _teacher_suggestions(teacher_shot) -> tuple[AutomaticSuggestion, ...]:
    return tuple(
        AutomaticSuggestion(
            source="wavelet_teacher",
            time_s=float(event.time_s),
            score=float(event.score),
        )
        for event in teacher_shot.events
    )


def _event_locator(event: dict) -> tuple[str, float]:
    return event["label"], round(float(event["time_s"]), 12)


def _find_event_by_locator(events: list[dict], locator) -> int | None:
    if locator is None:
        return None
    return next(
        (index for index, event in enumerate(events) if _event_locator(event) == tuple(locator)),
        None,
    )


def _suggestion_locator(suggestion: AutomaticSuggestion) -> tuple[str, float]:
    return suggestion.source, round(float(suggestion.time_s), 12)


def _find_suggestion_by_locator(suggestions, locator) -> int | None:
    if locator is None:
        return None
    return next(
        (
            index
            for index, suggestion in enumerate(suggestions)
            if _suggestion_locator(suggestion) == tuple(locator)
        ),
        None,
    )


def _is_uncertain_event(event: dict) -> bool:
    return event.get("label") == "uncertain"


def _selection_points(plot_state) -> list[dict]:
    if plot_state is None:
        return []
    selection = (
        plot_state.get("selection", {})
        if isinstance(plot_state, dict)
        else getattr(plot_state, "selection", {})
    )
    if isinstance(selection, dict):
        return list(selection.get("points", []))
    return list(getattr(selection, "points", []))


def _point_number(point: dict) -> int:
    return int(point.get("point_number", point.get("pointNumber", 0)))


def _curve_number(point: dict) -> int:
    return int(point.get("curve_number", point.get("curveNumber", 0)))


def _add_signal_marker_line(
    figure,
    time_s: float,
    *,
    color: str,
    signal_y0: float,
    signal_y1: float,
    dash: str = "dot",
    width: float = 1.0,
    opacity: float = 0.35,
) -> None:
    figure.add_shape(
        type="line",
        x0=float(time_s),
        x1=float(time_s),
        y0=signal_y0,
        y1=signal_y1,
        xref="x",
        yref="paper",
        line={"color": color, "width": width, "dash": dash},
        opacity=opacity,
        layer="above",
    )


def _build_figure(
    plot_data,
    draft,
    suggestions,
    work_queue=(),
    signal_height: int = 240,
    action_mode: str = "inspect",
    display_range: tuple[float, float] | None = None,
    show_grid: bool = True,
    allowed_x_range: tuple[float, float] | None = None,
    y_ranges: dict[str, tuple[float, float]] | None = None,
    plot_token: str | None = None,
):
    allow_free_time_selection = action_mode in {"add_sawtooth", "select_interval"}
    names = list(plot_data)
    rail_row = len(names) + 1
    subplot_gap_px = 48
    figure_height = (
        int(signal_height) * len(names)
        + 260
        + subplot_gap_px * len(names)
    )
    figure = make_subplots(
        rows=len(names) + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=subplot_gap_px / figure_height,
        row_heights=[*([1.0] * len(names)), 0.65],
        subplot_titles=[*names, "Метки"],
    )
    bottom_signal_axis = "yaxis" if len(names) == 1 else f"yaxis{len(names)}"
    signal_y0 = float(figure.layout[bottom_signal_axis].domain[0])
    signal_y1 = float(figure.layout.yaxis.domain[1])
    selected_event_keys = {
        tuple(item["locator"])
        for item in work_queue
        if item["kind"] == "event"
    }
    selected_suggestion_keys = {
        tuple(item["locator"])
        for item in work_queue
        if item["kind"] == "suggestion"
    }
    trace_targets: list[dict] = []
    for row, name in enumerate(names, 1):
        plot_time, plot_values = plot_data[name]
        figure.add_trace(
            go.Scattergl(
                x=plot_time,
                y=plot_values,
                mode="lines+markers" if allow_free_time_selection else "lines",
                name=name,
                showlegend=False,
                line={"width": 1},
                marker=(
                    {"size": 10, "opacity": 0.01}
                    if allow_free_time_selection
                    else None
                ),
                hovertemplate=f"{name}<br>t=%{{x:.6f}} s<extra></extra>",
            ),
            row=row,
            col=1,
        )
        trace_targets.append({"kind": "signal", "channel": name})

    manual_indices = [
        index
        for index, event in enumerate(draft["events"])
        if event["label"] != "false_positive"
    ]
    figure.add_trace(
        go.Scattergl(
            x=[draft["events"][index]["time_s"] for index in manual_indices],
            y=[2] * len(manual_indices),
            mode="markers",
            name="Ручная разметка",
            marker={
                "size": [
                    16
                    if _event_locator(draft["events"][index]) in selected_event_keys
                    else 11
                    for index in manual_indices
                ],
                "color": [
                    "#e6a700"
                    if _is_uncertain_event(draft["events"][index])
                    else "#1b9e77"
                    for index in manual_indices
                ],
                "symbol": [
                    "circle-open"
                    if _is_uncertain_event(draft["events"][index])
                    else "circle"
                    for index in manual_indices
                ],
                "line": {
                    "width": [
                        3
                        if _event_locator(draft["events"][index])
                        in selected_event_keys
                        else 1
                        for index in manual_indices
                    ],
                    "color": [
                        "#00c8ff"
                        if _event_locator(draft["events"][index])
                        in selected_event_keys
                        else "#ffffff"
                        for index in manual_indices
                    ],
                },
            },
            text=[
                EVENT_LABELS[
                    "uncertain"
                    if _is_uncertain_event(draft["events"][index])
                    else draft["events"][index]["label"]
                ]
                for index in manual_indices
            ],
            hovertemplate="%{text}<br>t=%{x:.6f} s<extra></extra>",
        ),
        row=rail_row,
        col=1,
    )
    trace_targets.append({"kind": "manual", "indices": manual_indices})
    for index in manual_indices:
        event = draft["events"][index]
        is_uncertain = _is_uncertain_event(event)
        _add_signal_marker_line(
            figure,
            event["time_s"],
            color="#e6a700" if is_uncertain else "#1b9e77",
            signal_y0=signal_y0,
            signal_y1=signal_y1,
            dash="dash" if is_uncertain else "solid",
            width=1.5,
            opacity=0.65,
        )

    rejected_times = {
        round(float(event["time_s"]), 12)
        for event in draft["events"]
        if event["label"] == "false_positive"
    }
    source_levels = {
        "wavelet_teacher": 1,
    }
    for source in SOURCE_LABELS:
        indices = [
            index
            for index, suggestion in enumerate(suggestions)
            if suggestion.source == source
        ]
        if not indices:
            continue
        rejected = [
            round(float(suggestions[index].time_s), 12) in rejected_times
            for index in indices
        ]
        selected = [
            _suggestion_locator(suggestions[index]) in selected_suggestion_keys
            for index in indices
        ]
        figure.add_trace(
            go.Scattergl(
                x=[suggestions[index].time_s for index in indices],
                y=[source_levels[source]] * len(indices),
                mode="markers",
                name=SOURCE_LABELS[source],
                marker={
                    "size": [15 if is_selected else 10 for is_selected in selected],
                    "color": [
                        "#b7b7b7" if is_rejected else SOURCE_COLORS[source]
                        for is_rejected in rejected
                    ],
                    "symbol": [
                        "x" if is_rejected else SOURCE_SYMBOLS[source]
                        for is_rejected in rejected
                    ],
                    "line": {
                        "width": [3 if is_selected else 0 for is_selected in selected],
                        "color": "#00c8ff",
                    },
                },
                text=[
                    f"{SOURCE_LABELS[source]} · score={suggestions[index].score:.3g}"
                    + (" · отклонена" if is_rejected else "")
                    for index, is_rejected in zip(indices, rejected)
                ],
                hovertemplate="%{text}<br>t=%{x:.6f} s<extra></extra>",
            ),
            row=rail_row,
            col=1,
        )
        trace_targets.append({"kind": "suggestion", "indices": indices})
        for index, is_rejected in zip(indices, rejected):
            if is_rejected:
                continue
            _add_signal_marker_line(
                figure,
                suggestions[index].time_s,
                color=SOURCE_COLORS[source],
                signal_y0=signal_y0,
                signal_y1=signal_y1,
                dash="dot",
                width=1.0,
                opacity=0.3,
            )

    rejected_indices = [
        index
        for index, event in enumerate(draft["events"])
        if event["label"] == "false_positive"
    ]
    figure.add_trace(
        go.Scattergl(
            x=[draft["events"][index]["time_s"] for index in rejected_indices],
            y=[0] * len(rejected_indices),
            mode="markers",
            name="Ложные срабатывания",
            marker={
                "size": [
                    15
                    if _event_locator(draft["events"][index]) in selected_event_keys
                    else 11
                    for index in rejected_indices
                ],
                "color": "#d62728",
                "symbol": "x",
                "line": {
                    "width": [
                        3
                        if _event_locator(draft["events"][index])
                        in selected_event_keys
                        else 0
                        for index in rejected_indices
                    ],
                    "color": "#00c8ff",
                },
            },
            hovertemplate="Ложный срыв<br>t=%{x:.6f} s<extra></extra>",
        ),
        row=rail_row,
        col=1,
    )
    trace_targets.append({"kind": "rejected", "indices": rejected_indices})
    for index in rejected_indices:
        _add_signal_marker_line(
            figure,
            draft["events"][index]["time_s"],
            color="#d62728",
            signal_y0=signal_y0,
            signal_y1=signal_y1,
            dash="dash",
            width=1.2,
            opacity=0.45,
        )

    for item in _selection_overlay_items(list(work_queue), draft):
        time_s = float(item["time_s"])
        tolerance_s = float(item["tolerance_s"])
        figure.add_shape(
            type="rect",
            x0=time_s - tolerance_s,
            x1=time_s + tolerance_s,
            y0=signal_y0,
            y1=signal_y1,
            xref="x",
            yref="paper",
            fillcolor="rgba(0, 168, 214, 0.14)",
            line={"width": 0},
            layer="above",
        )
        if item["kind"] != "boundary":
            _add_signal_marker_line(
                figure,
                time_s,
                color="#00a8d6",
                signal_y0=signal_y0,
                signal_y1=signal_y1,
                dash="dash",
                width=2.0,
                opacity=0.95,
            )

    boundaries = sorted(
        float(item["time_s"])
        for item in work_queue
        if item["kind"] == "boundary"
    )
    for boundary_s in boundaries:
        figure.add_shape(
            type="line",
            x0=boundary_s,
            x1=boundary_s,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"color": "#00a8d6", "width": 2, "dash": "dash"},
            opacity=0.9,
            layer="above",
        )
    if len(boundaries) == 2:
        figure.add_shape(
            type="rect",
            x0=boundaries[0],
            x1=boundaries[1],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor="rgba(0, 168, 214, 0.08)",
            line={"width": 0},
            layer="below",
        )

    for start_s, end_s in draft["reviewed_intervals_s"]:
        figure.add_shape(
            type="rect",
            x0=start_s,
            x1=end_s,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor="rgba(70, 130, 180, 0.07)",
            line={"width": 0},
            layer="below",
        )
    figure.update_layout(
        height=figure_height,
        margin={"l": 10, "r": 30, "t": 90, "b": 55},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "x": 0,
            "font": {"size": 14},
            "itemsizing": "constant",
        },
        hovermode="closest",
        clickmode="event+select",
        dragmode="pan",
        uirevision=f"{draft['shot_id']}:{','.join(names)}",
        meta={"sxr_viewport_token": plot_token} if plot_token else None,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=[
            "Отклонённые<br>метки",
            "Wavelet<br>разметка",
            "Ручная<br>разметка",
        ],
        range=[-0.5, 2.5],
        tickfont={"size": 12},
        fixedrange=True,
        row=rail_row,
        col=1,
    )
    for row in range(1, rail_row):
        figure.update_xaxes(
            showline=True,
            showgrid=show_grid,
            mirror=True,
            linewidth=1,
            linecolor="rgba(105, 125, 145, 0.55)",
            row=row,
            col=1,
        )
        figure.update_yaxes(
            showline=True,
            showgrid=show_grid,
            mirror=True,
            linewidth=1,
            linecolor="rgba(105, 125, 145, 0.55)",
            row=row,
            col=1,
        )
    figure.update_xaxes(
        showline=True,
        showgrid=False,
        mirror=False,
        linewidth=1,
        linecolor="rgba(105, 125, 145, 0.55)",
        row=rail_row,
        col=1,
    )
    figure.update_yaxes(
        showline=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(105, 125, 145, 0.38)",
        mirror=False,
        linewidth=1,
        linecolor="rgba(105, 125, 145, 0.55)",
        row=rail_row,
        col=1,
    )
    if display_range is not None:
        figure.update_xaxes(range=list(display_range))
    if allowed_x_range is not None:
        figure.update_xaxes(
            minallowed=float(allowed_x_range[0]),
            maxallowed=float(allowed_x_range[1]),
        )
    if y_ranges is not None:
        for row, name in enumerate(names, 1):
            bounds = y_ranges.get(name)
            if bounds is not None:
                figure.update_yaxes(range=list(bounds), row=row, col=1)
    figure.update_xaxes(title_text="Время, с", row=rail_row, col=1)
    figure.update_annotations(font={"size": 16})

    return figure, trace_targets


def _set_flash(level: str, message: str) -> None:
    st.session_state.annotation_flash = (level, message)


def _reject_from_ui(draft: dict, suggestion: AutomaticSuggestion) -> None:
    _, created = _reject_suggestion(draft, suggestion)
    _set_flash(
        "success" if created else "info",
        (
            "Помечено как ложный срыв"
            if created
            else "Метка уже помечена как ложный срыв"
        ),
    )


def _queue_item_key(item: dict) -> tuple:
    if item["kind"] == "event":
        return ("event", *tuple(item["locator"]))
    if item["kind"] == "suggestion":
        return ("suggestion", *tuple(item["locator"]))
    return (
        item["kind"],
        round(float(item["time_s"]), 12),
        item.get("reference_sxr"),
    )


def _enqueue_work_item(item: dict) -> bool:
    queue = st.session_state.work_queue
    key = _queue_item_key(item)
    if any(_queue_item_key(existing) == key for existing in queue):
        return False
    queue.append(item)
    return True


def _work_item_label(item: dict) -> str:
    time_s = float(item["time_s"])
    if item["kind"] == "boundary":
        return f"Граница интервала · {time_s:.6f} s"
    if item["kind"] == "suggestion":
        return f"{SOURCE_LABELS[item['source']]} · {time_s:.6f} s"
    if item["kind"] == "event":
        return f"{EVENT_LABELS[item['label']]} · {time_s:.6f} s"
    channel = item.get("reference_sxr")
    return f"Точка · {time_s:.6f} s" + (f" · {channel}" if channel else "")


def _clear_work_queue() -> None:
    st.session_state.work_queue = []
    st.session_state.selected_event_locator = None
    st.session_state.selected_suggestion_locator = None
    st.session_state.selected_time_s = None
    st.session_state.selected_channel = None


def _selection_overlay_items(work_queue: list[dict], draft: dict) -> list[dict]:
    """Resolve the visible time tolerance for every queued selection."""

    items: list[dict] = []
    for item in work_queue:
        items.append(
            {
                "kind": item["kind"],
                "time_s": float(item["time_s"]),
                "tolerance_s": EVENT_TOLERANCE_S,
            }
        )
    return items


def _render_action_buttons(state_key: str) -> str:
    st.subheader("Область редактирования ")
    _render_action_icon_css()
    current = st.session_state.action_mode
    columns = st.columns(len(ACTION_LABELS))
    for column, (action, label) in zip(columns, ACTION_LABELS.items()):
        clicked = column.button(
            _action_icon_label(action, label),
            help=label,
            type="primary" if action == current else "secondary",
            width="stretch",
            key=f"action::{action}::{state_key}",
        )
        if clicked and action != current:
            st.session_state.action_mode = action
            _clear_work_queue()
            st.session_state.last_plot_selection = None
            st.rerun()
    return current


def _queue_plot_selection(
    points: list[dict],
    trace_targets: list[dict],
    suggestions,
    draft: dict,
    action_mode: str,
) -> bool:
    signatures = tuple(
        sorted(
            (
                _curve_number(point),
                _point_number(point),
                round(float(point["x"]), 12),
            )
            for point in points
        )
    )
    if not signatures or signatures == st.session_state.last_plot_selection:
        return False
    st.session_state.last_plot_selection = signatures
    if action_mode == "select_interval":
        selected_times = sorted({float(point["x"]) for point in points})
        boundaries = [
            item for item in st.session_state.work_queue if item["kind"] == "boundary"
        ]
        if len(selected_times) >= 2:
            boundaries = [
                {"kind": "boundary", "time_s": selected_times[0]},
                {"kind": "boundary", "time_s": selected_times[-1]},
            ]
        else:
            boundary = {"kind": "boundary", "time_s": selected_times[0]}
            if len(boundaries) >= 2:
                boundaries = [boundary]
            elif not any(
                abs(float(item["time_s"]) - selected_times[0]) < 1e-12
                for item in boundaries
            ):
                boundaries.append(boundary)
        st.session_state.work_queue = boundaries
        if len(boundaries) == 2:
            _select_interval_items(draft, suggestions)
        st.session_state.plot_selection_warning = None
        return True

    original_queue = list(st.session_state.work_queue)
    if action_mode == "inspect":
        st.session_state.work_queue = []
    queue_changed = False
    for point in points:
        curve = _curve_number(point)
        point_number = _point_number(point)
        if not 0 <= curve < len(trace_targets):
            continue
        target = trace_targets[curve]
        time_s = float(point["x"])
        reference_sxr = None
        suggestion = None
        event = None
        if target["kind"] == "signal":
            reference_sxr = (
                target["channel"] if target["channel"] in SXR_CHANNELS else None
            )
        elif point_number < len(target.get("indices", [])):
            target_index = target["indices"][point_number]
            if target["kind"] == "suggestion":
                suggestion = suggestions[target_index]
            else:
                event = draft["events"][target_index]

        if action_mode == "add_sawtooth":
            queue_changed |= _enqueue_work_item(
                {
                    "kind": "point",
                    "time_s": time_s,
                    "reference_sxr": reference_sxr,
                }
            )
        elif action_mode == "reject":
            if suggestion is None:
                continue
            queue_changed |= _enqueue_work_item(
                {
                    "kind": "suggestion",
                    "time_s": suggestion.time_s,
                    "source": suggestion.source,
                    "locator": list(_suggestion_locator(suggestion)),
                }
            )
        elif suggestion is not None:
            queue_changed |= _enqueue_work_item(
                {
                    "kind": "suggestion",
                    "time_s": suggestion.time_s,
                    "source": suggestion.source,
                    "locator": list(_suggestion_locator(suggestion)),
                }
            )
            st.session_state.selected_suggestion_locator = _suggestion_locator(suggestion)
            st.session_state.selected_event_locator = None
        elif event is not None:
            queue_changed |= _enqueue_work_item(
                {
                    "kind": "event",
                    "time_s": float(event["time_s"]),
                    "label": event["label"],
                    "locator": list(_event_locator(event)),
                }
            )
            st.session_state.selected_event_locator = _event_locator(event)
            st.session_state.selected_suggestion_locator = None
            st.session_state.sync_editor_selection = True
    st.session_state.plot_selection_warning = None
    if action_mode == "inspect" and queue_changed:
        candidates = list(st.session_state.work_queue)
        original_keys = {_queue_item_key(item) for item in original_queue}
        newly_selected = [
            item for item in candidates if _queue_item_key(item) not in original_keys
        ]
        chosen = (newly_selected or candidates)[-1]
        st.session_state.work_queue = [chosen]
        if chosen["kind"] == "suggestion":
            st.session_state.selected_suggestion_locator = tuple(chosen["locator"])
            st.session_state.selected_event_locator = None
        else:
            st.session_state.selected_event_locator = tuple(chosen["locator"])
            st.session_state.selected_suggestion_locator = None
        queue_changed = [
            _queue_item_key(item) for item in st.session_state.work_queue
        ] != [_queue_item_key(item) for item in original_queue]
    elif action_mode == "inspect":
        st.session_state.work_queue = original_queue
    return queue_changed


def _select_interval_items(draft: dict, suggestions) -> tuple[int, int]:
    boundaries = sorted(
        float(item["time_s"])
        for item in st.session_state.work_queue
        if item["kind"] == "boundary"
    )
    if len(boundaries) != 2:
        return 0, 0
    start_s, end_s = boundaries
    queue = [
        {"kind": "boundary", "time_s": start_s},
        {"kind": "boundary", "time_s": end_s},
    ]
    events = [
        {
            "kind": "event",
            "time_s": float(event["time_s"]),
            "label": event["label"],
            "locator": list(_event_locator(event)),
        }
        for event in draft["events"]
        if start_s <= float(event["time_s"]) <= end_s
    ]
    automatic = [
        {
            "kind": "suggestion",
            "time_s": suggestion.time_s,
            "source": suggestion.source,
            "locator": list(_suggestion_locator(suggestion)),
        }
        for suggestion in suggestions
        if start_s <= suggestion.time_s <= end_s
    ]
    st.session_state.work_queue = queue + events + automatic
    return len(events), len(automatic)


def _apply_add_queue(draft: dict, label: str) -> tuple[int, int]:
    created_count = 0
    duplicate_count = 0
    for item in st.session_state.work_queue:
        if item["kind"] not in {"point", "suggestion"}:
            continue
        _, created = _add_manual_event(
            draft,
            float(item["time_s"]),
            label,
        )
        created_count += int(created)
        duplicate_count += int(not created)
    if label == "sawtooth" and created_count:
        st.session_state.pending_shot_label = "sawtooth"
    return created_count, duplicate_count


def _apply_reject_queue(draft: dict, suggestions) -> tuple[int, int]:
    created_count = 0
    missing_count = 0
    for item in st.session_state.work_queue:
        if item["kind"] != "suggestion":
            continue
        index = _find_suggestion_by_locator(suggestions, item["locator"])
        if index is None:
            missing_count += 1
            continue
        _, created = _reject_suggestion(draft, suggestions[index])
        created_count += int(created)
    return created_count, missing_count


def _apply_accept_suggestion_queue(
    draft: dict,
    suggestions,
    label: str,
) -> tuple[int, int]:
    created_count = 0
    missing_count = 0
    for item in st.session_state.work_queue:
        if item["kind"] != "suggestion":
            continue
        index = _find_suggestion_by_locator(suggestions, item["locator"])
        if index is None:
            missing_count += 1
            continue
        suggestion = suggestions[index]
        _, created = _add_manual_event(
            draft,
            suggestion.time_s,
            label,
        )
        created_count += int(created)
    if label == "sawtooth" and created_count:
        st.session_state.pending_shot_label = "sawtooth"
    return created_count, missing_count


def _delete_queued_events(draft: dict) -> int:
    indices = {
        index
        for item in st.session_state.work_queue
        if item["kind"] == "event"
        for index in [_find_event_by_locator(draft["events"], item["locator"])]
        if index is not None
    }
    for index in sorted(indices, reverse=True):
        draft["events"].pop(index)
    return len(indices)


def _render_decision_sidebar(
    draft: dict,
    state_key: str,
    full_start: float,
    full_end: float,
    annotation_path: str,
    shot_id: str,
) -> None:
    pending_label = st.session_state.pop("pending_shot_label", None)
    review_status_key = f"review_status::{state_key}"
    shot_label_key = f"shot_label::{state_key}"
    if pending_label is not None:
        draft["shot_label"] = pending_label
        st.session_state[shot_label_key] = pending_label

    st.divider()
    st.subheader("Решение по файлу")
    review_status = st.selectbox(
        "Статус просмотра",
        ["partial", "complete"],
        index=["partial", "complete"].index(draft["review_status"]),
        format_func={"partial": "Частично", "complete": "Полностью"}.get,
        key=review_status_key,
    )
    shot_label = st.selectbox(
        "Содержимое файла",
        list(SHOT_LABELS),
        index=list(SHOT_LABELS).index(draft["shot_label"]),
        format_func=SHOT_LABELS.get,
        key=shot_label_key,
    )
    note = st.text_input(
        "Комментарий",
        draft.get("note", ""),
        key=f"note::{state_key}",
    )
    draft.update(review_status=review_status, shot_label=shot_label, note=note)
    if review_status == "complete":
        draft["reviewed_intervals_s"] = [
            [max(0.0, float(full_start)), float(full_end)]
        ]

    if st.button("Сохранить", type="primary", width="stretch"):
        try:
            parse_expert_labels(
                {"schema_version": EXPERT_LABEL_SCHEMA_VERSION, "shots": [draft]}
            )
            saved = upsert_shot_annotation(
                annotation_path,
                deepcopy(draft),
                expected_revision=st.session_state.annotation_revision,
            )
            st.session_state.annotation_revision = saved.revision
            st.success(f"Разметка {shot_id} сохранена")
        except AnnotationConflictError as exc:
            st.error(f"Конфликт сохранения: {exc}")
        except Exception as exc:
            st.error(f"Разметка не сохранена: {exc}")

    if st.button(
        "Обновить",
        width="stretch",
        key=f"refresh_after_save::{state_key}",
        help=(
            "Перечитать экспертную разметку и список файлов. При фильтре "
            "«Только неразмеченные» откроется следующий доступный файл."
        ),
    ):
        st.session_state.active_annotation = None
        st.rerun()


def _render_event_editor(
    draft: dict,
    event_index: int,
    state_key: str,
) -> None:
    existing = draft["events"][event_index]
    st.markdown(
        f"**Редактирование метки около {float(existing['time_s']):.6f} s**"
    )
    revision = _event_locator(existing)
    with st.form(f"workspace_editor::{state_key}::{revision}"):
        event_time = st.number_input(
            "Время, s", value=float(existing["time_s"]), format="%.6f"
        )
        mark_sawtooth = st.form_submit_button(
            "Пометить как срыв", type="primary", width="stretch"
        )
        mark_uncertain = st.form_submit_button(
            "Пометить как возможный срыв", width="stretch"
        )
        st.form_submit_button(
            "Пометить как ложный срыв",
            disabled=True,
            help=(
                "Метка уже помечена как ложный срыв."
                if existing["label"] == "false_positive"
                else "Это действие доступно для исходной автоматической метки."
            ),
            width="stretch",
        )
        delete_manual = st.form_submit_button(
            "Удалить ручные метки",
            width="stretch",
        )
        if delete_manual:
            draft["events"].pop(event_index)
            _clear_work_queue()
            _set_flash("success", "Ручная метка удалена")
            st.rerun()
        if mark_sawtooth or mark_uncertain:
            event_label = "uncertain" if mark_uncertain else "sawtooth"
            updated = {
                "time_s": float(event_time),
                "label": event_label,
            }
            draft["events"][event_index] = updated
            draft["events"].sort(key=lambda item: float(item["time_s"]))
            _ensure_local_review(draft, event_time, EVENT_TOLERANCE_S)
            if event_label == "sawtooth":
                st.session_state.pending_shot_label = "sawtooth"
            _clear_work_queue()
            _set_flash("success", "Метка обновлена")
            st.rerun()


def _render_new_event_editor(
    draft: dict,
    time_s: float,
    state_key: str,
    key_suffix: str,
    suggestion: AutomaticSuggestion | None = None,
) -> None:
    st.markdown(f"**Редактирование метки около {float(time_s):.6f} s**")
    with st.form(f"new_event_editor::{state_key}::{key_suffix}"):
        event_time = st.number_input("Время, s", value=float(time_s), format="%.6f")
        mark_sawtooth = st.form_submit_button(
            "Пометить как срыв", type="primary", width="stretch"
        )
        mark_uncertain = st.form_submit_button(
            "Пометить как возможный срыв", width="stretch"
        )
        mark_false = st.form_submit_button(
            "Пометить как ложный срыв",
            disabled=suggestion is None,
            help=(
                None
                if suggestion is not None
                else "Это действие доступно только для автоматической метки."
            ),
            width="stretch",
        )
        st.form_submit_button(
            "Удалить ручные метки",
            disabled=True,
            help="Удаление доступно только для уже сохранённой ручной метки.",
            width="stretch",
        )
        if mark_false and suggestion is not None:
            _reject_from_ui(draft, suggestion)
            _clear_work_queue()
            st.rerun()
        if mark_sawtooth or mark_uncertain:
            event_label = "uncertain" if mark_uncertain else "sawtooth"
            index, created = _add_manual_event(
                draft,
                float(event_time),
                event_label,
            )
            if event_label == "sawtooth" and created:
                st.session_state.pending_shot_label = "sawtooth"
            _clear_work_queue()
            _set_flash(
                "success" if created else "info",
                "Метка добавлена" if created else "Такая метка уже существует",
            )
            del index
            st.rerun()


def _render_workspace(
    draft: dict,
    suggestions,
    available_sxr: list[str],
    state_key: str,
    full_start: float,
    action_mode: str,
) -> None:
    queue = st.session_state.work_queue
    if action_mode == "inspect" and len(queue) > 1:
        queue = [queue[-1]]
        st.session_state.work_queue = queue
    warning = st.session_state.pop("plot_selection_warning", None)
    if warning:
        st.warning(warning)
    selected_count = sum(item["kind"] != "boundary" for item in queue)

    remove_indices = st.multiselect(
        "Убрать из выбора",
        range(len(queue)),
        format_func=lambda index: _work_item_label(queue[index]),
        key=f"queue_remove::{state_key}::{len(queue)}",
        disabled=not queue,
    )
    queue_controls = st.columns(2)
    if queue_controls[0].button(
        "Убрать отмеченные",
        disabled=not remove_indices,
        width="stretch",
        key=f"queue_remove_button::{state_key}",
    ):
        remove_set = set(remove_indices)
        st.session_state.work_queue = [
            item for index, item in enumerate(queue) if index not in remove_set
        ]
        st.session_state.last_plot_selection = None
        st.rerun()
    if queue_controls[1].button(
        "Очистить",
        disabled=not queue,
        width="stretch",
        key=f"queue_clear::{state_key}",
    ):
        _clear_work_queue()
        st.session_state.last_plot_selection = None
        st.rerun()

    if action_mode == "select_interval":
        boundaries = sorted(
            float(item["time_s"])
            for item in queue
            if item["kind"] == "boundary"
        )
        if len(boundaries) < 2:
            pass
        else:
            interval_start_key = f"selection_interval_start::{state_key}"
            interval_end_key = f"selection_interval_end::{state_key}"
            interval_revision_key = f"selection_interval_revision::{state_key}"
            interval_error_key = f"selection_interval_error::{state_key}"
            boundary_revision = tuple(round(value, 12) for value in boundaries)
            if st.session_state.get(interval_revision_key) != boundary_revision:
                st.session_state[interval_start_key] = boundaries[0]
                st.session_state[interval_end_key] = boundaries[1]
                st.session_state[interval_revision_key] = boundary_revision

            def update_interval_from_fields() -> None:
                start_s = float(st.session_state[interval_start_key])
                end_s = float(st.session_state[interval_end_key])
                if end_s <= start_s:
                    st.session_state[interval_error_key] = (
                        "Правая граница интервала должна быть больше левой."
                    )
                    return
                st.session_state.pop(interval_error_key, None)
                st.session_state.work_queue = [
                    {"kind": "boundary", "time_s": start_s},
                    {"kind": "boundary", "time_s": end_s},
                ]
                _select_interval_items(draft, suggestions)
                st.session_state.last_plot_selection = None

            interval_columns = st.columns(2)
            interval_columns[0].number_input(
                "Левая граница интервала, с",
                format="%.6f",
                step=0.001,
                key=interval_start_key,
                on_change=update_interval_from_fields,
            )
            interval_columns[1].number_input(
                "Правая граница интервала, с",
                format="%.6f",
                step=0.001,
                key=interval_end_key,
                on_change=update_interval_from_fields,
            )
            interval_error = st.session_state.get(interval_error_key)
            if interval_error:
                st.error(interval_error)

            event_count = sum(item["kind"] == "event" for item in queue)
            suggestion_count = sum(
                item["kind"] == "suggestion" for item in queue
            )
            if st.button(
                f"Пометить как срыв ({suggestion_count})",
                disabled=not suggestion_count,
                width="stretch",
                key=f"interval_accept::{state_key}",
            ):
                created, missing = _apply_accept_suggestion_queue(
                    draft, suggestions, "sawtooth"
                )
                _clear_work_queue()
                _set_flash(
                    "success",
                    f"Добавлено срывов: {created}"
                    + (f"; не найдено: {missing}" if missing else ""),
                )
                st.rerun()
            if st.button(
                f"Пометить как возможный срыв ({suggestion_count})",
                disabled=not suggestion_count,
                width="stretch",
                key=f"interval_uncertain::{state_key}",
            ):
                created, missing = _apply_accept_suggestion_queue(
                    draft, suggestions, "uncertain"
                )
                _clear_work_queue()
                _set_flash(
                    "success",
                    f"Добавлено неуверенных: {created}"
                    + (f"; не найдено: {missing}" if missing else ""),
                )
                st.rerun()
            if st.button(
                f"Пометить как ложный срыв ({suggestion_count})",
                disabled=not suggestion_count,
                width="stretch",
                key=f"interval_reject::{state_key}",
            ):
                created, missing = _apply_reject_queue(draft, suggestions)
                _clear_work_queue()
                _set_flash(
                    "success",
                    f"Помечено как ложные срывы: {created}"
                    + (f"; не найдено: {missing}" if missing else ""),
                )
                st.rerun()
            if st.button(
                f"Удалить ручные метки ({event_count})",
                disabled=not event_count,
                width="stretch",
                key=f"interval_delete::{state_key}",
            ):
                deleted = _delete_queued_events(draft)
                _clear_work_queue()
                _set_flash("success", f"Удалено меток: {deleted}")
                st.rerun()
    elif action_mode == "add_sawtooth":
        if st.button(
            f"Пометить как срыв ({selected_count})",
            type="primary",
            disabled=not selected_count,
            width="stretch",
            key=f"batch_add_sawtooth::{state_key}",
        ):
            created, duplicates = _apply_add_queue(draft, "sawtooth")
            _clear_work_queue()
            _set_flash(
                "success",
                f"Добавлено меток: {created}"
                + (f"; уже существовало: {duplicates}" if duplicates else ""),
            )
            st.rerun()
        if st.button(
            f"Пометить как возможный срыв ({selected_count})",
            disabled=not selected_count,
            width="stretch",
            key=f"batch_add_uncertain::{state_key}",
        ):
            created, duplicates = _apply_add_queue(draft, "uncertain")
            _clear_work_queue()
            _set_flash(
                "success",
                f"Добавлено возможных срывов: {created}"
                + (f"; уже существовало: {duplicates}" if duplicates else ""),
            )
            st.rerun()
        with st.expander("Добавить по точному времени", expanded=False):
            exact_time = st.number_input(
                "Время, s",
                value=float(st.session_state.selected_time_s or full_start),
                format="%.6f",
                key=f"exact_time::{state_key}",
            )
            if st.button(
                "Добавить в выбранные",
                width="stretch",
                key=f"exact_add::{state_key}",
            ):
                reference_sxr = (
                    st.session_state.selected_channel
                    if st.session_state.selected_channel in available_sxr
                    else (available_sxr[0] if available_sxr else None)
                )
                _enqueue_work_item(
                    {
                        "kind": "point",
                        "time_s": float(exact_time),
                        "reference_sxr": reference_sxr,
                    }
                )
                st.rerun()
    elif action_mode == "reject":
        suggestion_count = sum(item["kind"] == "suggestion" for item in queue)
        if st.button(
            f"Пометить как ложный срыв ({suggestion_count})",
            type="primary",
            disabled=not suggestion_count,
            width="stretch",
            key=f"batch_reject::{state_key}",
        ):
            created, missing = _apply_reject_queue(draft, suggestions)
            _clear_work_queue()
            _set_flash(
                "success",
                f"Помечено как ложные срывы: {created}"
                + (f"; не найдено: {missing}" if missing else ""),
            )
            st.rerun()
    if action_mode == "inspect" and len(queue) == 1:
        st.divider()
        item = queue[0]
        if item["kind"] == "suggestion":
            suggestion_index = _find_suggestion_by_locator(
                suggestions, item["locator"]
            )
            if suggestion_index is not None:
                suggestion = suggestions[suggestion_index]
                _render_new_event_editor(
                    draft,
                    suggestion.time_s,
                    state_key,
                    f"suggestion::{suggestion.source}::{suggestion.time_s:.12f}",
                    suggestion=suggestion,
                )
        elif item["kind"] == "event":
            event_index = _find_event_by_locator(draft["events"], item["locator"])
            if event_index is not None:
                _render_event_editor(draft, event_index, state_key)
        elif item["kind"] == "point":
            _render_new_event_editor(
                draft,
                float(item["time_s"]),
                state_key,
                f"point::{float(item['time_s']):.12f}",
            )

def main() -> None:
    st.set_page_config(page_title="SXR expert annotation", layout="wide")

    with st.sidebar:
        st.header("Файлы")
        with st.expander("Источники данных", expanded=False):
            pseudo_path = st.text_input("Wavelet-разметка", DEFAULT_PSEUDO_LABELS_PATH)
            annotation_path = st.text_input("Экспертная разметка", DEFAULT_EXPERT_LABELS_PATH)
            source_dir_override = st.text_input(
                "Каталог SHT",
                "",
                placeholder="Пусто — использовать source_dir из wavelet JSON",
            ).strip()

        try:
            pseudo = load_pseudo_labels(pseudo_path)
            document = load_annotation_document(annotation_path)
            expert = parse_expert_labels(document.payload)
        except Exception as exc:
            st.error(f"Не удалось загрузить данные разметки: {exc}")
            st.stop()

        catalog = build_shot_catalog(pseudo, expert)
        with st.expander("Поиск и фильтры", expanded=True):
            search_query = st.text_input(
                "Поиск по имени файла",
                placeholder="sht41114 или easy/",
            ).strip().casefold()
            categories = sorted({item.category for item in catalog}, key=str.casefold)
            selected_categories = st.multiselect(
                "Категории", categories, default=categories
            )
            status_filter = st.selectbox(
                "Статус",
                ["Все", "Только неразмеченные", "Только размеченные"],
            )
            filtered = [item for item in catalog if item.category in selected_categories]
            if search_query:
                filtered = [
                    item
                    for item in filtered
                    if search_query in item.shot_id.casefold()
                    or search_query in item.relative_path.casefold()
                    or search_query in Path(item.relative_path).name.casefold()
                ]
            if status_filter == "Только неразмеченные":
                filtered = [item for item in filtered if not item.is_annotated]
            elif status_filter == "Только размеченные":
                filtered = [item for item in filtered if item.is_annotated]
            if not filtered:
                st.warning("Нет файлов с выбранными фильтрами")
                st.stop()
            labels = {
                f"{'✓' if item.is_annotated else '○'} {Path(item.relative_path).name} · "
                f"{item.category} · {item.split} · teacher={item.teacher_event_count}": item
                for item in filtered
            }
            selected_label = st.selectbox("Файл", list(labels))
            shot = labels[selected_label]

    teacher_by_id = {item.shot_id: item for item in pseudo.train + pseudo.validation}
    teacher_shot = teacher_by_id[shot.shot_id]
    state_key = f"{Path(annotation_path).resolve()}::{shot.shot_id}"
    if st.session_state.get("active_annotation") != state_key:
        st.session_state.active_annotation = state_key
        st.session_state.annotation_revision = document.revision
        st.session_state.annotation_draft = (
            _find_raw_annotation(document, shot.shot_id) or _new_draft(shot)
        )
        st.session_state.selected_time_s = None
        st.session_state.selected_channel = None
        st.session_state.selected_event_locator = None
        st.session_state.selected_suggestion_locator = None
        st.session_state.last_plot_selection = None
        st.session_state.work_queue = []
        st.session_state.action_mode = "inspect"
        st.session_state.plot_selection_warning = None
    if "work_queue" not in st.session_state:
        st.session_state.work_queue = []
    if "action_mode" not in st.session_state:
        st.session_state.action_mode = "inspect"
    if "plot_selection_warning" not in st.session_state:
        st.session_state.plot_selection_warning = None
    draft = st.session_state.annotation_draft

    source_dir = source_dir_override or pseudo.source_dir
    try:
        available = _available_channels(source_dir, shot.relative_path)
    except Exception as exc:
        st.error(f"Не удалось прочитать список каналов: {exc}")
        st.stop()

    with st.sidebar:
        with st.expander("Каналы", expanded=True):
            channel_state_key = f"channels::{state_key}"
            if channel_state_key not in st.session_state:
                st.session_state[channel_state_key] = default_display_channels(available)

            selected_channel_set = {
                name for name in st.session_state[channel_state_key] if name in available
            }

            channel_search_key = f"channel_search::{state_key}"
            active_search = str(st.session_state.get(channel_search_key, ""))
            active_query = active_search.strip().casefold()
            active_matches = sorted(
                (name for name in available if active_query in name.casefold()),
                key=str.casefold,
            )
            pills_key_prefix = f"channel_pills::{state_key}::"
            pills_query_key = f"channel_pills_query::{state_key}"
            if st.session_state.get(pills_query_key) != active_query:
                for session_key in list(st.session_state):
                    if str(session_key).startswith(pills_key_prefix):
                        del st.session_state[session_key]
                st.session_state[pills_query_key] = active_query
            active_pills_key = (
                f"{pills_key_prefix}"
                f"{sha256(active_query.encode('utf-8')).hexdigest()[:12]}"
            )
            # Pill values are updated by Streamlit before the rerun starts.
            # Reconcile the currently visible subset before drawing the summary.
            if active_pills_key in st.session_state:
                selected_channel_set.difference_update(active_matches)
                selected_channel_set.update(st.session_state[active_pills_key] or [])

            selected_channels = [
                name for name in available if name in selected_channel_set
            ]
            st.session_state[channel_state_key] = selected_channels

            channel_search = st.text_input(
                "Поиск по названию канала",
                placeholder="SXR, D-alfa, МГД...",
                key=channel_search_key,
            ).strip().casefold()
            matching_channels = sorted(
                (name for name in available if channel_search in name.casefold()),
                key=str.casefold,
            )
            st.caption(f"Доступные каналы: {len(matching_channels)} из {len(available)}")
            with st.container(height=300, border=True):
                if not matching_channels:
                    st.info("Каналы с таким названием не найдены")
                else:
                    pills_key = (
                        f"{pills_key_prefix}"
                        f"{sha256(channel_search.encode('utf-8')).hexdigest()[:12]}"
                    )
                    visible_selection = st.pills(
                        "Доступные каналы",
                        matching_channels,
                        selection_mode="multi",
                        default=[
                            name for name in matching_channels if name in selected_channel_set
                        ],
                        key=pills_key,
                        label_visibility="collapsed",
                    )
                    selected_channel_set.difference_update(matching_channels)
                    selected_channel_set.update(visible_selection or [])

            selected_channels = [
                name for name in available if name in selected_channel_set
            ]
            st.session_state[channel_state_key] = selected_channels
    if not selected_channels:
        st.warning("Выберите хотя бы один канал")
        st.stop()

    plot_data, full_start, full_end = _plot_channel_data(
        source_dir, shot.relative_path, tuple(selected_channels)
    )
    natural_y_ranges = _natural_y_ranges(plot_data)
    plotted_channels = tuple(plot_data)
    _consume_plot_viewport(state_key, plotted_channels)
    plot_token = _plot_viewport_token(state_key)
    with st.sidebar:
        (
            signal_height,
            display_start,
            display_end,
            show_grid,
            allowed_x_range,
            y_ranges,
            active_y_channel,
        ) = (
            _render_graph_view_controls(
                source_dir,
                shot.relative_path,
                state_key,
                full_start,
                full_end,
                natural_y_ranges,
            )
        )
    suggestions = _teacher_suggestions(teacher_shot)

    st.title("Экспертная разметка SXR")
    st.markdown(
        '<div class="annotation-shot-meta">'
        f"{escape(shot.relative_path)} · {escape(shot.category)} · "
        f"{escape(shot.split)}"
        "</div>",
        unsafe_allow_html=True,
    )
    flash = st.session_state.pop("annotation_flash", None)
    if flash:
        level, message = flash
        getattr(st, level)(message)

    action_mode = st.session_state.action_mode
    graph_column, workspace_column = st.columns([2.1, 1.0], gap="medium")

    with graph_column:
        figure, trace_targets = _build_figure(
            plot_data,
            draft,
            suggestions,
            work_queue=st.session_state.work_queue,
            signal_height=signal_height,
            action_mode=action_mode,
            display_range=(display_start, display_end),
            show_grid=show_grid,
            allowed_x_range=allowed_x_range,
            y_ranges=y_ranges,
            plot_token=plot_token,
        )
        view_revision = _plot_view_revision(
            source_dir,
            plotted_channels,
            signal_height,
            display_start,
            display_end,
            show_grid,
            y_ranges,
        )
        plot_key = (
            f"annotation_plot::{shot.shot_id}::{action_mode}::{view_revision}"
        )

        def handle_plot_selection() -> None:
            _queue_plot_selection(
                _selection_points(st.session_state.get(plot_key)),
                trace_targets,
                suggestions,
                draft,
                action_mode,
            )

        st.plotly_chart(
            figure,
            width="stretch",
            on_select=handle_plot_selection,
            selection_mode="points",
            key=plot_key,
            config={"scrollZoom": True, "displaylogo": False},
        )
        render_plot_viewport_listener(
            plot_token=plot_token,
            y_axes={
                ("yaxis" if row == 1 else f"yaxis{row}"): name
                for row, name in enumerate(plotted_channels, 1)
            },
            active_y_channel=active_y_channel,
            key=_plot_viewport_listener_key(state_key),
        )

    with workspace_column:
        with st.container(border=True):
            action_mode = _render_action_buttons(state_key)
            _render_workspace(
                draft,
                suggestions,
                [name for name in SXR_CHANNELS if name in available],
                state_key,
                full_start,
                action_mode,
            )
            _render_decision_sidebar(
                draft,
                state_key,
                full_start,
                full_end,
                annotation_path,
                shot.shot_id,
            )
