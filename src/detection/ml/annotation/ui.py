from copy import deepcopy
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.path import (
    DEFAULT_EXPERT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PSEUDO_LABELS_PATH,
    SXR_CHANNELS,
)
from src.detection.ml.annotation.backend import (
    AutomaticSuggestion,
    build_shot_catalog,
    compute_automatic_suggestions,
    decimate_minmax,
    default_display_channels,
    load_channels,
    merge_reviewed_interval,
)
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


SOURCE_LABELS = {
    "wavelet_teacher": "Wavelet teacher",
    "posr": "Raw POSR",
    "cpd": "Raw CPD",
    "feature_ml": "Feature-ML",
}
SOURCE_COLORS = {
    "wavelet_teacher": "#4c78a8",
    "posr": "#8f63b8",
    "cpd": "#e45756",
    "feature_ml": "#111111",
}
SOURCE_SYMBOLS = {
    "wavelet_teacher": "triangle-down",
    "posr": "diamond",
    "cpd": "square",
    "feature_ml": "circle",
}
EVENT_LABELS = {
    "sawtooth": "Срыв",
    "uncertain": "Не уверена",
    "false_positive": "Ложная автоматическая метка",
}
SHOT_LABELS = {
    "sawtooth": "Пила присутствует",
    "no_sawtooth": "Пилы нет",
    "uncertain": "Не уверена",
}
ACTION_LABELS = {
    "inspect": "Выбрать / редактировать",
    "add_sawtooth": "Добавить срыв",
    "add_uncertain": "Добавить неуверенную",
    "reject": "Отклонить автоматическую",
}


def _canonical_shot_id(value: str) -> str:
    value = str(value).strip().casefold()
    return f"sht{value}" if value.isdigit() else value


@st.cache_data(show_spinner=False)
def _available_channels(source_dir: str, relative_path: str) -> list[str]:
    logger = setup_logger(log_to_file=False)
    from src.io.loader import SHTLoader

    return SHTLoader(source_dir, logger).list_available_channels(relative_path)


@st.cache_data(show_spinner="Загрузка каналов SHT…")
def _plot_channel_data(
    source_dir: str,
    relative_path: str,
    channels: tuple[str, ...],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, float]:
    """Load and decimate signals once instead of on every UI rerun."""

    logger = setup_logger(log_to_file=False)
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


@st.cache_data(show_spinner="Расчёт POSR, CPD и ML…")
def _computed_suggestions(
    source_dir: str,
    relative_path: str,
    include_posr: bool,
    include_cpd: bool,
    model_path: str,
) -> tuple[AutomaticSuggestion, ...]:
    logger = setup_logger(log_to_file=False)
    effective_model = model_path if model_path and Path(model_path).is_file() else None
    return compute_automatic_suggestions(
        source_dir,
        relative_path,
        logger,
        include_posr=include_posr,
        include_cpd=include_cpd,
        model_path=effective_model,
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
    reference_sxr: str | None,
    *,
    tolerance_s: float = 0.3e-3,
    visible_sxr: list[str] | None = None,
    confidence: str | None = None,
    note: str = "",
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
            and event.get("source") is None
            and abs(float(event["time_s"]) - float(time_s)) < 1e-12
        ),
        None,
    )
    if duplicate_index is not None:
        return duplicate_index, False

    visible = list(
        dict.fromkeys(name for name in [reference_sxr, *(visible_sxr or [])] if name)
    )
    event = {
        "time_s": float(time_s),
        "label": label,
        "tolerance_s": float(tolerance_s),
        "confidence": "uncertain" if label == "uncertain" else (confidence or "certain"),
        "reference_sxr": reference_sxr,
        "visible_sxr": visible,
        "note": note,
    }
    events.append(event)
    events.sort(key=lambda item: float(item["time_s"]))
    event_index = events.index(event)
    _ensure_local_review(draft, event["time_s"], event["tolerance_s"])
    if label == "sawtooth":
        draft["shot_label"] = "sawtooth"
    return event_index, True


def _reject_suggestion(
    draft: dict,
    suggestion: AutomaticSuggestion,
    note: str = "",
) -> tuple[int, bool]:
    """Single entry point for all automatic-proposal rejections."""

    events = draft["events"]
    duplicate_index = next(
        (
            index
            for index, event in enumerate(events)
            if event["label"] == "false_positive"
            and event.get("source") == suggestion.source
            and abs(float(event["time_s"]) - suggestion.time_s) < 1e-12
        ),
        None,
    )
    if duplicate_index is not None:
        return duplicate_index, False

    tolerance_s = 0.3e-3
    event = {
        "time_s": suggestion.time_s,
        "label": "false_positive",
        "tolerance_s": tolerance_s,
        "confidence": "certain",
        "reference_sxr": suggestion.reference_sxr,
        "visible_sxr": list(suggestion.source_channels),
        "source": suggestion.source,
        "note": note,
    }
    events.append(event)
    events.sort(key=lambda item: float(item["time_s"]))
    event_index = events.index(event)
    _ensure_local_review(draft, suggestion.time_s, tolerance_s)
    return event_index, True


def _teacher_suggestions(teacher_shot) -> tuple[AutomaticSuggestion, ...]:
    return tuple(
        AutomaticSuggestion(
            source="wavelet_teacher",
            time_s=float(event.time_s),
            score=float(event.score),
            reference_sxr=teacher_shot.channel,
            source_channels=tuple(event.metadata.get("source_channels", ())),
        )
        for event in teacher_shot.events + teacher_shot.uncertain_events
    )


def _event_locator(event: dict) -> tuple[str, str | None, float]:
    return event["label"], event.get("source"), round(float(event["time_s"]), 12)


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


def _event_option(index: int, event: dict) -> str:
    return f"{index + 1}. {event['time_s']:.6f} s · {EVENT_LABELS[event['label']]}"


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


def _build_figure(plot_data, draft, suggestions):
    """Build WebGL signals plus a compact clickable annotation rail."""

    names = list(plot_data)
    figure = make_subplots(
        rows=len(names) + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.025, 0.12 / max(len(names), 1)),
        row_heights=[0.18, *([1.0] * len(names))],
        subplot_titles=["Метки", *names],
    )
    trace_targets: list[dict] = []
    for row, name in enumerate(names, 2):
        plot_time, plot_values = plot_data[name]
        figure.add_trace(
            go.Scattergl(
                x=plot_time,
                y=plot_values,
                mode="lines",
                name=name,
                showlegend=False,
                line={"width": 1},
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
            y=[5] * len(manual_indices),
            mode="markers",
            name="Ручная разметка",
            marker={
                "size": 12,
                "color": [
                    "#1b9e77"
                    if draft["events"][index]["label"] == "sawtooth"
                    else "#e6a700"
                    for index in manual_indices
                ],
                "symbol": [
                    "triangle-up"
                    if draft["events"][index]["label"] == "sawtooth"
                    else "diamond"
                    for index in manual_indices
                ],
                "line": {"width": 1, "color": "#ffffff"},
            },
            text=[EVENT_LABELS[draft["events"][index]["label"]] for index in manual_indices],
            hovertemplate="%{text}<br>t=%{x:.6f} s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    trace_targets.append({"kind": "manual", "indices": manual_indices})

    rejected_keys = {
        (event.get("source"), round(float(event["time_s"]), 12))
        for event in draft["events"]
        if event["label"] == "false_positive"
    }
    source_levels = {
        "wavelet_teacher": 4,
        "posr": 3,
        "cpd": 2,
        "feature_ml": 1,
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
            (source, round(float(suggestions[index].time_s), 12)) in rejected_keys
            for index in indices
        ]
        figure.add_trace(
            go.Scattergl(
                x=[suggestions[index].time_s for index in indices],
                y=[source_levels[source]] * len(indices),
                mode="markers",
                name=SOURCE_LABELS[source],
                marker={
                    "size": 10,
                    "color": [
                        "#b7b7b7" if is_rejected else SOURCE_COLORS[source]
                        for is_rejected in rejected
                    ],
                    "symbol": [
                        "x" if is_rejected else SOURCE_SYMBOLS[source]
                        for is_rejected in rejected
                    ],
                },
                text=[
                    f"{SOURCE_LABELS[source]} · score={suggestions[index].score:.3g}"
                    + (" · отклонена" if is_rejected else "")
                    for index, is_rejected in zip(indices, rejected)
                ],
                hovertemplate="%{text}<br>t=%{x:.6f} s<extra></extra>",
            ),
            row=1,
            col=1,
        )
        trace_targets.append({"kind": "suggestion", "indices": indices})

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
            marker={"size": 11, "color": "#d62728", "symbol": "x"},
            text=[
                SOURCE_LABELS.get(
                    draft["events"][index].get("source"),
                    draft["events"][index].get("source", ""),
                )
                for index in rejected_indices
            ],
            hovertemplate="Ложное срабатывание · %{text}<br>t=%{x:.6f} s<extra></extra>",
        ),
        row=1,
        col=1,
    )
    trace_targets.append({"kind": "rejected", "indices": rejected_indices})

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
        height=max(560, 175 * len(names) + 150),
        margin={"l": 70, "r": 20, "t": 50, "b": 45},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="closest",
        clickmode="event+select",
        dragmode="pan",
        uirevision=f"{draft['shot_id']}:{','.join(names)}",
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2, 3, 4, 5],
        ticktext=["Отклонено", "ML", "CPD", "POSR", "Teacher", "Вручную"],
        fixedrange=True,
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="Время, с", row=len(names) + 1, col=1)
    return figure, trace_targets


def _set_flash(level: str, message: str) -> None:
    st.session_state.annotation_flash = (level, message)


def _select_added_event(events: list[dict], index: int) -> None:
    st.session_state.selected_event_locator = _event_locator(events[index])
    st.session_state.selected_suggestion_locator = None
    st.session_state.sync_editor_selection = True


def _add_from_ui(draft: dict, time_s: float, label: str, reference_sxr: str | None) -> None:
    index, created = _add_manual_event(draft, time_s, label, reference_sxr)
    _select_added_event(draft["events"], index)
    if label == "sawtooth":
        st.session_state.pending_shot_label = "sawtooth"
    _set_flash(
        "success" if created else "info",
        f"Метка добавлена: {time_s:.6f} s" if created else "Такая метка уже есть",
    )


def _reject_from_ui(draft: dict, suggestion: AutomaticSuggestion) -> None:
    _, created = _reject_suggestion(draft, suggestion)
    _set_flash(
        "success" if created else "info",
        "Автометка отклонена" if created else "Автометка уже отклонена",
    )


def _render_interval_controls(draft: dict, state_key: str, full_start: float, full_end: float) -> None:
    with st.expander("Просмотренные интервалы", expanded=False):
        for index, (start_s, end_s) in enumerate(draft["reviewed_intervals_s"], 1):
            st.caption(f"{index}. {start_s:.6f} – {end_s:.6f} s")
        interval_start = st.number_input(
            "Начало, s",
            value=full_start,
            format="%.6f",
            key=f"interval_start::{state_key}",
        )
        interval_end = st.number_input(
            "Конец, s",
            value=full_end,
            format="%.6f",
            key=f"interval_end::{state_key}",
        )
        if st.button("Добавить интервал", key=f"add_interval::{state_key}"):
            try:
                draft["reviewed_intervals_s"] = merge_reviewed_interval(
                    draft["reviewed_intervals_s"], interval_start, interval_end
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if draft["reviewed_intervals_s"]:
            remove_interval = st.selectbox(
                "Удалить интервал",
                range(len(draft["reviewed_intervals_s"])),
                format_func=lambda index: (
                    f"{draft['reviewed_intervals_s'][index][0]:.6f} – "
                    f"{draft['reviewed_intervals_s'][index][1]:.6f} s"
                ),
                key=f"remove_interval::{state_key}",
            )
            if st.button("Удалить", key=f"remove_interval_button::{state_key}"):
                draft["reviewed_intervals_s"].pop(remove_interval)
                st.rerun()


def _render_decision_sidebar(
    draft: dict,
    state_key: str,
    full_start: float,
    full_end: float,
    annotation_path: str,
    shot_id: str,
) -> None:
    pending_label = st.session_state.pop("pending_shot_label", None)
    pending_status = st.session_state.pop("pending_review_status", None)
    review_status_key = f"review_status::{state_key}"
    shot_label_key = f"shot_label::{state_key}"
    if pending_label is not None:
        draft["shot_label"] = pending_label
        st.session_state[shot_label_key] = pending_label
    if pending_status is not None:
        draft["review_status"] = pending_status
        st.session_state[review_status_key] = pending_status

    st.divider()
    st.header("Решение по файлу")
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

    _render_interval_controls(draft, state_key, full_start, full_end)
    with st.expander("В файле нет пилы", expanded=False):
        confirmed = st.checkbox(
            "Файл просмотрен целиком",
            key=f"no_saw_confirm::{state_key}",
        )
        if st.button(
            "Подтвердить отсутствие пилы",
            disabled=not confirmed,
            key=f"no_saw_button::{state_key}",
        ):
            draft["review_status"] = "complete"
            draft["shot_label"] = "no_sawtooth"
            draft["reviewed_intervals_s"] = [[max(0.0, full_start), full_end]]
            draft["events"] = [
                event for event in draft["events"] if event["label"] == "false_positive"
            ]
            st.session_state.pending_review_status = "complete"
            st.session_state.pending_shot_label = "no_sawtooth"
            st.rerun()

    if st.button("Сохранить файл", type="primary", width="stretch"):
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


def main() -> None:
    st.set_page_config(page_title="SXR expert annotation", layout="wide")

    with st.sidebar:
        st.header("Файлы")
        with st.expander("Пути к данным", expanded=False):
            pseudo_path = st.text_input("Wavelet-разметка", DEFAULT_PSEUDO_LABELS_PATH)
            annotation_path = st.text_input("Экспертная разметка", DEFAULT_EXPERT_LABELS_PATH)
            model_path = st.text_input("ML-модель (необязательно)", DEFAULT_MODEL_PATH)

    try:
        pseudo = load_pseudo_labels(pseudo_path)
        document = load_annotation_document(annotation_path)
        expert = parse_expert_labels(document.payload)
    except Exception as exc:
        st.error(f"Не удалось загрузить данные разметки: {exc}")
        st.stop()

    catalog = build_shot_catalog(pseudo, expert)
    with st.sidebar:
        search_query = st.text_input(
            "Поиск по имени файла",
            placeholder="sht41114 или easy/",
        ).strip().casefold()
        categories = sorted({item.category for item in catalog}, key=str.casefold)
        with st.expander("Фильтры", expanded=False):
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
        st.session_state.computed_suggestions = ()
        st.session_state.selected_event_locator = None
        st.session_state.selected_suggestion_locator = None
        st.session_state.last_plot_selection = None
    draft = st.session_state.annotation_draft

    source_dir = pseudo.source_dir
    try:
        available = _available_channels(source_dir, shot.relative_path)
    except Exception as exc:
        st.error(f"Не удалось прочитать список каналов: {exc}")
        st.stop()

    with st.sidebar:
        with st.expander("График и автометки", expanded=True):
            selected_channels = st.multiselect(
                "Отображаемые каналы",
                available,
                default=default_display_channels(available),
                help="Можно добавить D-alpha, МГД и другие каналы этого SHT.",
                key=f"channels::{state_key}",
            )
            include_posr = st.checkbox("Raw POSR", True, key=f"posr::{state_key}")
            include_cpd = st.checkbox("Raw CPD", True, key=f"cpd::{state_key}")
            include_ml = st.checkbox(
                "Feature-ML",
                Path(model_path).is_file(),
                key=f"ml::{state_key}",
            )
            calculate = st.button("Рассчитать автометки", width="stretch")
    if not selected_channels:
        st.warning("Выберите хотя бы один канал")
        st.stop()

    plot_data, full_start, full_end = _plot_channel_data(
        source_dir, shot.relative_path, tuple(selected_channels)
    )
    if calculate:
        try:
            st.session_state.computed_suggestions = _computed_suggestions(
                source_dir,
                shot.relative_path,
                include_posr,
                include_cpd,
                model_path if include_ml else "",
            )
        except Exception as exc:
            st.error(f"Ошибка расчёта автометок: {exc}")
    suggestions = (
        *_teacher_suggestions(teacher_shot),
        *st.session_state.computed_suggestions,
    )

    with st.sidebar:
        _render_decision_sidebar(
            draft,
            state_key,
            full_start,
            full_end,
            annotation_path,
            shot.shot_id,
        )

    st.title("Экспертная разметка SXR")
    st.caption(
        f"{shot.relative_path} · {shot.category} · {shot.split}. "
        "Кликните по сигналу или маркеру; колесо мыши масштабирует, перетаскивание сдвигает."
    )
    flash = st.session_state.pop("annotation_flash", None)
    if flash:
        level, message = flash
        getattr(st, level)(message)

    action_mode = st.radio(
        "Действие при клике",
        list(ACTION_LABELS),
        format_func=ACTION_LABELS.get,
        horizontal=True,
        key=f"plot_action::{state_key}",
    )
    action_help = {
        "inspect": "Выберите точку или маркер — ниже появятся контекстные действия.",
        "add_sawtooth": "Клик по сигналу или автометке сразу добавит срыв.",
        "add_uncertain": "Клик по сигналу или автометке сразу добавит неуверенную метку.",
        "reject": "Кликните по маркеру Teacher, POSR, CPD или ML, чтобы отклонить его.",
    }
    st.caption(action_help[action_mode])

    figure, trace_targets = _build_figure(plot_data, draft, suggestions)
    plot_state = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"annotation_plot::{shot.shot_id}",
        config={"scrollZoom": True, "displaylogo": False},
    )
    points = _selection_points(plot_state)
    if points:
        point = points[-1]
        curve = _curve_number(point)
        point_number = _point_number(point)
        if 0 <= curve < len(trace_targets):
            target = trace_targets[curve]
            time_s = float(point["x"])
            signature = (curve, point_number, round(time_s, 12))
            if signature != st.session_state.last_plot_selection:
                st.session_state.last_plot_selection = signature
                st.session_state.selected_time_s = time_s
                reference_sxr = None
                suggestion = None
                if target["kind"] == "signal":
                    reference_sxr = target["channel"]
                    st.session_state.selected_channel = reference_sxr
                    st.session_state.selected_event_locator = None
                    st.session_state.selected_suggestion_locator = None
                elif point_number < len(target.get("indices", [])):
                    target_index = target["indices"][point_number]
                    if target["kind"] == "suggestion":
                        suggestion = suggestions[target_index]
                        reference_sxr = suggestion.reference_sxr
                        st.session_state.selected_suggestion_locator = _suggestion_locator(
                            suggestion
                        )
                        st.session_state.selected_event_locator = None
                    else:
                        event = draft["events"][target_index]
                        reference_sxr = event.get("reference_sxr")
                        st.session_state.selected_event_locator = _event_locator(event)
                        st.session_state.selected_suggestion_locator = None
                        st.session_state.sync_editor_selection = True

                if action_mode in {"add_sawtooth", "add_uncertain"}:
                    label = (
                        "sawtooth" if action_mode == "add_sawtooth" else "uncertain"
                    )
                    _add_from_ui(draft, time_s, label, reference_sxr)
                    st.rerun()
                if action_mode == "reject":
                    if suggestion is None:
                        _set_flash(
                            "warning",
                            "Для отклонения выберите маркер Teacher, POSR, CPD или ML.",
                        )
                    else:
                        _reject_from_ui(draft, suggestion)
                    st.rerun()

    events = draft["events"]
    selected_suggestion_index = _find_suggestion_by_locator(
        suggestions, st.session_state.selected_suggestion_locator
    )
    selected_event_index = _find_event_by_locator(
        events, st.session_state.selected_event_locator
    )

    with st.container(border=True):
        if selected_suggestion_index is not None:
            suggestion = suggestions[selected_suggestion_index]
            st.markdown(
                f"**{SOURCE_LABELS[suggestion.source]}** · {suggestion.time_s:.6f} s · "
                f"score={suggestion.score:.3g}"
            )
            accept_col, uncertain_col, reject_col = st.columns(3)
            if accept_col.button("Принять как срыв", width="stretch"):
                _add_from_ui(
                    draft, suggestion.time_s, "sawtooth", suggestion.reference_sxr
                )
                st.rerun()
            if uncertain_col.button("Принять как неуверенную", width="stretch"):
                _add_from_ui(
                    draft, suggestion.time_s, "uncertain", suggestion.reference_sxr
                )
                st.rerun()
            if reject_col.button("Ложное срабатывание", width="stretch"):
                _reject_from_ui(draft, suggestion)
                st.rerun()
        elif (
            selected_event_index is not None
            and events[selected_event_index]["label"] == "false_positive"
        ):
            rejected = events[selected_event_index]
            st.markdown(
                f"**Ложное срабатывание** · {float(rejected['time_s']):.6f} s · "
                f"{SOURCE_LABELS.get(rejected.get('source'), rejected.get('source', ''))}"
            )
            if st.button("Отменить отклонение"):
                events.pop(selected_event_index)
                st.session_state.selected_event_locator = None
                _set_flash("success", "Отклонение отменено")
                st.rerun()
        elif st.session_state.selected_time_s is not None:
            selected_time = float(st.session_state.selected_time_s)
            selected_channel = st.session_state.selected_channel
            st.markdown(
                f"**Выбранная точка:** {selected_time:.6f} s"
                + (f" · {selected_channel}" if selected_channel else "")
            )
            add_col, uncertain_col = st.columns(2)
            if add_col.button("Добавить срыв", width="stretch"):
                _add_from_ui(draft, selected_time, "sawtooth", selected_channel)
                st.rerun()
            if uncertain_col.button("Добавить неуверенную", width="stretch"):
                _add_from_ui(draft, selected_time, "uncertain", selected_channel)
                st.rerun()
        else:
            st.caption(
                "Выберите точку на сигнале или маркер на дорожке «Метки»."
            )

    st.subheader("Точная правка ручной метки")
    editable_indices = [
        index for index, event in enumerate(events) if event["label"] != "false_positive"
    ]
    available_sxr = [name for name in SXR_CHANNELS if name in available]
    if editable_indices:
        editor_key = f"event_editor::{state_key}"
        if st.session_state.pop("sync_editor_selection", False):
            selected_event_index = _find_event_by_locator(
                events, st.session_state.selected_event_locator
            )
            if selected_event_index in editable_indices:
                st.session_state[editor_key] = selected_event_index
        if st.session_state.get(editor_key) not in editable_indices:
            st.session_state[editor_key] = editable_indices[0]
        edit_index = st.selectbox(
            "Метка",
            editable_indices,
            format_func=lambda index: _event_option(index, events[index]),
            key=editor_key,
        )
        existing = events[edit_index]
        editor_revision = _event_locator(existing)
        with st.form(f"event_editor_form::{state_key}::{editor_revision}"):
            row = st.columns([1.2, 1.2, 1, 1.8, 1.2])
            event_time = row[0].number_input(
                "Время, s", value=float(existing["time_s"]), format="%.6f"
            )
            event_label = row[1].selectbox(
                "Тип",
                ["sawtooth", "uncertain"],
                index=["sawtooth", "uncertain"].index(existing["label"]),
                format_func=EVENT_LABELS.get,
            )
            tolerance_ms = row[2].number_input(
                "Допуск, ms",
                min_value=0.001,
                value=float(existing.get("tolerance_s", 0.0003)) * 1e3,
                format="%.3f",
            )
            initial_reference = existing.get("reference_sxr")
            reference_sxr = (
                row[3].selectbox(
                    "Опорный SXR",
                    available_sxr,
                    index=(
                        available_sxr.index(initial_reference)
                        if initial_reference in available_sxr
                        else 0
                    ),
                )
                if available_sxr
                else None
            )
            confidence = row[4].selectbox(
                "Уверенность",
                ["certain", "uncertain"],
                index=1 if existing.get("confidence") == "uncertain" else 0,
                format_func={
                    "certain": "Уверена",
                    "uncertain": "Не уверена",
                }.get,
            )
            details = st.columns([2, 2])
            initial_visible = [
                name for name in existing.get("visible_sxr", []) if name in available_sxr
            ]
            visible_sxr = details[0].multiselect(
                "Где видна метка",
                available_sxr,
                default=initial_visible,
            )
            event_note = details[1].text_input(
                "Комментарий", existing.get("note", "")
            )
            if st.form_submit_button("Обновить метку", type="primary"):
                visible = list(
                    dict.fromkeys(
                        name for name in [reference_sxr, *visible_sxr] if name
                    )
                )
                updated = {
                    "time_s": float(event_time),
                    "label": event_label,
                    "tolerance_s": float(tolerance_ms) * 1e-3,
                    "confidence": (
                        "uncertain" if event_label == "uncertain" else confidence
                    ),
                    "reference_sxr": reference_sxr,
                    "visible_sxr": visible,
                    "note": event_note,
                }
                events[edit_index] = updated
                events.sort(key=lambda item: float(item["time_s"]))
                _ensure_local_review(draft, event_time, updated["tolerance_s"])
                st.session_state.selected_event_locator = _event_locator(updated)
                st.session_state.sync_editor_selection = True
                if event_label == "sawtooth":
                    st.session_state.pending_shot_label = "sawtooth"
                _set_flash("success", "Метка обновлена")
                st.rerun()
        if st.button("Удалить эту метку", key=f"delete_event::{state_key}"):
            events.pop(edit_index)
            st.session_state.selected_event_locator = None
            _set_flash("success", "Метка удалена")
            st.rerun()
    else:
        st.caption(
            "Ручных меток пока нет. Добавьте первую кликом по графику."
        )

    with st.expander("Добавить по точному времени", expanded=False):
        exact_columns = st.columns([2, 2, 3])
        exact_time = exact_columns[0].number_input(
            "Время, s",
            value=float(st.session_state.selected_time_s or full_start),
            format="%.6f",
            key=f"exact_time::{state_key}",
        )
        exact_label = exact_columns[1].selectbox(
            "Тип",
            ["sawtooth", "uncertain"],
            format_func=EVENT_LABELS.get,
            key=f"exact_label::{state_key}",
        )
        if exact_columns[2].button(
            "Добавить",
            width="stretch",
            key=f"exact_add::{state_key}",
        ):
            reference_sxr = (
                st.session_state.selected_channel
                if st.session_state.selected_channel in available_sxr
                else (available_sxr[0] if available_sxr else None)
            )
            _add_from_ui(draft, exact_time, exact_label, reference_sxr)
            st.rerun()
