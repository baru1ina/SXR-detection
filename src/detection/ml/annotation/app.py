from copy import deepcopy
from pathlib import Path
import sys


# ``streamlit run path/to/app.py`` adds the script directory, not the repository
# root, to sys.path.  The project intentionally uses root-level imports such as
# ``config.path`` throughout, so make the direct browser entry point equivalent
# to running the other commands from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _canonical_shot_id(value: str) -> str:
    value = str(value).strip().casefold()
    return f"sht{value}" if value.isdigit() else value


@st.cache_data(show_spinner=False)
def _available_channels(source_dir: str, relative_path: str) -> list[str]:
    logger = setup_logger(log_to_file=False)
    from src.io.loader import SHTLoader

    return SHTLoader(source_dir, logger).list_available_channels(relative_path)


@st.cache_data(show_spinner="Загрузка каналов SHT…")
def _channel_data(
    source_dir: str,
    relative_path: str,
    channels: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    logger = setup_logger(log_to_file=False)
    shot = load_channels(source_dir, relative_path, channels, logger)
    return {
        name: (shot.get_channel(name).time, shot.get_channel(name).values)
        for name in shot.channel_names
    }


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


def _build_figure(channel_data, draft, teacher, suggestions):
    names = list(channel_data)
    figure = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.025, 0.14 / max(len(names), 1)),
        subplot_titles=names,
    )
    for row, name in enumerate(names, 1):
        time, values = channel_data[name]
        plot_time, plot_values = decimate_minmax(time, values)
        figure.add_trace(
            go.Scattergl(
                x=plot_time,
                y=plot_values,
                mode="lines",
                name=name,
                line={"width": 1},
                hovertemplate=f"{name}<br>t=%{{x:.6f}} s<extra></extra>",
            ),
            row=row,
            col=1,
        )
    for start_s, end_s in draft["reviewed_intervals_s"]:
        figure.add_vrect(
            x0=start_s,
            x1=end_s,
            fillcolor="rgba(70, 130, 180, 0.08)",
            line_width=0,
            layer="below",
        )
    for event in teacher.events:
        figure.add_vline(
            x=float(event.time_s), line_color="#4c78a8", line_width=1, opacity=0.5
        )
    source_colors = {"posr": "#8f63b8", "cpd": "#e45756", "feature_ml": "#111111"}
    for suggestion in suggestions:
        if suggestion.source == "wavelet_teacher":
            continue
        figure.add_vline(
            x=suggestion.time_s,
            line_color=source_colors.get(suggestion.source, "#999999"),
            line_width=1,
            line_dash="dot",
            opacity=0.35,
        )
    event_colors = {
        "sawtooth": "#1b9e77",
        "uncertain": "#e6a700",
        "false_positive": "#d62728",
    }
    for event in draft["events"]:
        figure.add_vline(
            x=float(event["time_s"]),
            line_color=event_colors[event["label"]],
            line_width=3,
            line_dash="dash" if event["label"] != "sawtooth" else "solid",
        )
    figure.update_layout(
        height=max(520, 210 * len(names)),
        margin={"l": 55, "r": 20, "t": 50, "b": 45},
        showlegend=False,
        hovermode="x unified",
        clickmode="event+select",
        dragmode="zoom",
        uirevision=f"{draft['shot_id']}:{','.join(names)}",
    )
    figure.update_xaxes(title_text="Время, с", row=len(names), col=1)
    return figure


def _selection_points(plot_state) -> list[dict]:
    if plot_state is None:
        return []
    if isinstance(plot_state, dict):
        selection = plot_state.get("selection", {})
    else:
        selection = getattr(plot_state, "selection", {})
    if isinstance(selection, dict):
        return list(selection.get("points", []))
    return list(getattr(selection, "points", []))


def _event_option(index: int, event: dict) -> str:
    source = f" · {SOURCE_LABELS.get(event.get('source'), event.get('source'))}" if event.get("source") else ""
    return f"{index + 1}. {event['time_s']:.6f} s · {EVENT_LABELS[event['label']]}{source}"


def main() -> None:
    st.set_page_config(page_title="SXR expert annotation", layout="wide")
    st.title("Экспертная разметка срывов SXR")
    st.caption(
        "Клик по линии задаёт время и опорный канал. Синяя линия — wavelet teacher; "
        "пунктиром показаны рассчитанные proposals; толстые линии — ручная разметка."
    )

    with st.sidebar:
        st.header("Данные")
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
        categories = sorted({item.category for item in catalog}, key=str.casefold)
        selected_categories = st.multiselect("Категории", categories, default=categories)
        status_filter = st.selectbox(
            "Статус", ["Все", "Только неразмеченные", "Только размеченные"]
        )
        filtered = [item for item in catalog if item.category in selected_categories]
        if status_filter == "Только неразмеченные":
            filtered = [item for item in filtered if not item.is_annotated]
        elif status_filter == "Только размеченные":
            filtered = [item for item in filtered if item.is_annotated]
        if not filtered:
            st.warning("Нет разрядов с выбранными фильтрами")
            st.stop()
        labels = {
            f"{'✓' if item.is_annotated else '○'} {item.shot_id} · {item.category} · "
            f"{item.split} · teacher={item.teacher_event_count}": item
            for item in filtered
        }
        selected_label = st.selectbox("Разряд", list(labels))
        shot = labels[selected_label]

    teacher_by_id = {
        item.shot_id: item for item in pseudo.train + pseudo.validation
    }
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
    draft = st.session_state.annotation_draft

    source_dir = pseudo.source_dir
    try:
        available = _available_channels(source_dir, shot.relative_path)
    except Exception as exc:
        st.error(f"Не удалось прочитать список каналов: {exc}")
        st.stop()

    default_channels = default_display_channels(available)
    selected_channels = st.multiselect(
        "Отображаемые каналы",
        available,
        default=default_channels,
        help="Можно добавить D-alpha, МГД и другие каналы этого SHT.",
    )
    if not selected_channels:
        st.warning("Выберите хотя бы один канал")
        st.stop()
    channel_data = _channel_data(source_dir, shot.relative_path, tuple(selected_channels))
    full_start = min(float(time[0]) for time, _ in channel_data.values())
    full_end = max(float(time[-1]) for time, _ in channel_data.values())

    suggestion_controls = st.columns([1, 1, 1, 3])
    include_posr = suggestion_controls[0].checkbox("Raw POSR", True)
    include_cpd = suggestion_controls[1].checkbox("Raw CPD", True)
    include_ml = suggestion_controls[2].checkbox("Feature-ML", Path(model_path).is_file())
    if suggestion_controls[3].button("Рассчитать автоматические метки"):
        try:
            st.session_state.computed_suggestions = _computed_suggestions(
                source_dir,
                shot.relative_path,
                include_posr,
                include_cpd,
                model_path if include_ml else "",
            )
        except Exception as exc:
            st.error(f"Ошибка расчёта автоматических меток: {exc}")
    suggestions = (
        *_teacher_suggestions(teacher_shot),
        *st.session_state.computed_suggestions,
    )

    figure = _build_figure(channel_data, draft, teacher_shot, suggestions)
    plot_state = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"annotation_plot::{shot.shot_id}",
    )
    points = _selection_points(plot_state)
    if points:
        point = points[-1]
        st.session_state.selected_time_s = float(point["x"])
        curve = int(point.get("curve_number", point.get("curveNumber", 0)))
        if 0 <= curve < len(selected_channels):
            st.session_state.selected_channel = selected_channels[curve]

    st.subheader("Решение по разряду")
    pending_label = st.session_state.pop("pending_shot_label", None)
    pending_status = st.session_state.pop("pending_review_status", None)
    if pending_label is not None:
        draft["shot_label"] = pending_label
        st.session_state[f"shot_label::{state_key}"] = pending_label
    if pending_status is not None:
        draft["review_status"] = pending_status
        st.session_state[f"review_status::{state_key}"] = pending_status
    decision_columns = st.columns([1, 1, 2])
    review_status = decision_columns[0].selectbox(
        "Статус просмотра",
        ["partial", "complete"],
        index=["partial", "complete"].index(draft["review_status"]),
        format_func={"partial": "Частично", "complete": "Полностью"}.get,
        key=f"review_status::{state_key}",
    )
    shot_label = decision_columns[1].selectbox(
        "Содержимое файла",
        list(SHOT_LABELS),
        index=list(SHOT_LABELS).index(draft["shot_label"]),
        format_func=SHOT_LABELS.get,
        key=f"shot_label::{state_key}",
    )
    note = decision_columns[2].text_input(
        "Комментарий к разряду", draft.get("note", ""), key=f"note::{state_key}"
    )
    draft.update(review_status=review_status, shot_label=shot_label, note=note)

    no_saw_confirm = st.checkbox(
        "Я просмотрела весь файл и подтверждаю, что пилообразных срывов нет",
        key=f"no_saw_confirm::{state_key}",
    )
    if st.button("Пометить файл как не содержащий пилу", disabled=not no_saw_confirm):
        draft["review_status"] = "complete"
        draft["shot_label"] = "no_sawtooth"
        draft["reviewed_intervals_s"] = [[max(0.0, full_start), full_end]]
        draft["events"] = [
            event
            for event in draft["events"]
            if event["label"] == "false_positive"
        ]
        st.session_state.pending_review_status = "complete"
        st.session_state.pending_shot_label = "no_sawtooth"
        st.rerun()

    interval_col, event_col = st.columns([1, 2])
    with interval_col:
        st.subheader("Просмотренные интервалы")
        if draft["reviewed_intervals_s"]:
            st.dataframe(
                [
                    {"№": i + 1, "начало, с": start, "конец, с": end}
                    for i, (start, end) in enumerate(draft["reviewed_intervals_s"])
                ],
                hide_index=True,
                width="stretch",
            )
        with st.form(f"interval_form::{state_key}"):
            interval_start = st.number_input(
                "Начало, с", value=full_start, format="%.6f"
            )
            interval_end = st.number_input("Конец, с", value=full_end, format="%.6f")
            if st.form_submit_button("Добавить просмотренный интервал"):
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
                format_func=lambda i: (
                    f"{draft['reviewed_intervals_s'][i][0]:.6f} – "
                    f"{draft['reviewed_intervals_s'][i][1]:.6f} s"
                ),
                key=f"remove_interval::{state_key}",
            )
            if st.button("Удалить выбранный интервал"):
                draft["reviewed_intervals_s"].pop(remove_interval)
                st.rerun()

    with event_col:
        st.subheader("Ручные метки")
        events = draft["events"]
        editable_indices = [
            i for i, event in enumerate(events) if event["label"] != "false_positive"
        ]
        options = ["Новая метка"] + [
            _event_option(i, events[i]) for i in editable_indices
        ]
        chosen = st.selectbox("Создать или изменить", options, key=f"event_choice::{state_key}")
        edit_index = (
            None
            if chosen == "Новая метка"
            else editable_indices[options.index(chosen) - 1]
        )
        existing = events[edit_index] if edit_index is not None else None
        selected_time = st.session_state.selected_time_s
        initial_time = float(existing["time_s"]) if existing else (
            float(selected_time) if selected_time is not None else full_start
        )
        event_key_suffix = str(edit_index) if edit_index is not None else "new"
        event_time_key = f"event_time::{state_key}::{event_key_suffix}"
        selected_time_revision_key = f"selected_time_applied::{state_key}"
        if event_time_key not in st.session_state:
            st.session_state[event_time_key] = initial_time
        if (
            edit_index is None
            and selected_time is not None
            and st.session_state.get(selected_time_revision_key) != selected_time
        ):
            st.session_state[event_time_key] = float(selected_time)
            st.session_state[selected_time_revision_key] = float(selected_time)
        available_sxr = [name for name in SXR_CHANNELS if name in available]
        clicked_channel = st.session_state.selected_channel
        initial_reference = existing.get("reference_sxr") if existing else (
            clicked_channel if clicked_channel in available_sxr else (available_sxr[0] if available_sxr else None)
        )
        editable_labels = ["sawtooth", "uncertain"]
        initial_label = existing["label"] if existing and existing["label"] in editable_labels else "sawtooth"
        with st.form(f"event_form::{state_key}::{edit_index}"):
            event_time = st.number_input("Время, с", format="%.6f", key=event_time_key)
            event_label = st.selectbox(
                "Тип", editable_labels,
                index=editable_labels.index(initial_label),
                format_func=EVENT_LABELS.get,
            )
            tolerance_ms = st.number_input(
                "Допуск, ms",
                min_value=0.001,
                value=float(existing.get("tolerance_s", 0.0003) * 1e3) if existing else 0.3,
                format="%.3f",
            )
            reference_sxr = st.selectbox(
                "Опорный SXR",
                available_sxr,
                index=available_sxr.index(initial_reference) if initial_reference in available_sxr else 0,
            ) if available_sxr else None
            initial_visible = existing.get("visible_sxr", []) if existing else (
                [reference_sxr] if reference_sxr else []
            )
            visible_sxr = st.multiselect(
                "На каких SXR видна метка", available_sxr, default=initial_visible
            )
            confidence = st.selectbox(
                "Уверенность", ["certain", "uncertain"],
                index=1 if event_label == "uncertain" else 0,
                format_func={"certain": "Уверена", "uncertain": "Не уверена"}.get,
            )
            event_note = st.text_input("Комментарий", existing.get("note", "") if existing else "")
            if st.form_submit_button("Добавить" if edit_index is None else "Обновить"):
                visible = list(dict.fromkeys([name for name in [reference_sxr, *visible_sxr] if name]))
                event = {
                    "time_s": float(event_time),
                    "label": event_label,
                    "tolerance_s": float(tolerance_ms) * 1e-3,
                    "confidence": "uncertain" if event_label == "uncertain" else confidence,
                    "reference_sxr": reference_sxr,
                    "visible_sxr": visible,
                    "note": event_note,
                }
                if edit_index is None:
                    events.append(event)
                else:
                    events[edit_index] = event
                events.sort(key=lambda item: float(item["time_s"]))
                _ensure_local_review(draft, event_time, event["tolerance_s"])
                if event_label == "sawtooth":
                    st.session_state.pending_shot_label = "sawtooth"
                st.rerun()
        if edit_index is not None and st.button("Удалить выбранную ручную метку"):
            events.pop(edit_index)
            st.rerun()

    st.subheader("Отклонить неправильную автоматическую метку")
    if suggestions:
        suggestion_options = {
            f"{item.time_s:.6f} s · {SOURCE_LABELS[item.source]} · score={item.score:.3g} · "
            f"SXR={', '.join(item.source_channels) or item.reference_sxr}": item
            for item in suggestions
        }
        selected_suggestion_label = st.selectbox(
            "Автоматическая метка", list(suggestion_options), key=f"reject::{state_key}"
        )
        selected_suggestion = suggestion_options[selected_suggestion_label]
        rejection_note = st.text_input(
            "Причина отклонения", key=f"rejection_note::{state_key}"
        )
        if st.button("Пометить как ложное срабатывание"):
            tolerance_s = 0.3e-3
            duplicate = any(
                event["label"] == "false_positive"
                and event.get("source") == selected_suggestion.source
                and abs(float(event["time_s"]) - selected_suggestion.time_s) < 1e-9
                for event in events
            )
            if not duplicate:
                events.append({
                    "time_s": selected_suggestion.time_s,
                    "label": "false_positive",
                    "tolerance_s": tolerance_s,
                    "confidence": "certain",
                    "reference_sxr": selected_suggestion.reference_sxr,
                    "visible_sxr": list(selected_suggestion.source_channels),
                    "source": selected_suggestion.source,
                    "note": rejection_note,
                })
                events.sort(key=lambda item: float(item["time_s"]))
                _ensure_local_review(draft, selected_suggestion.time_s, tolerance_s)
            st.rerun()
    else:
        st.info("В wavelet teacher нет меток. При необходимости рассчитайте POSR/CPD/ML.")

    rejected_indices = [
        i for i, event in enumerate(events) if event["label"] == "false_positive"
    ]
    if rejected_indices:
        rejected_choice = st.selectbox(
            "Сохранённые ложные срабатывания",
            rejected_indices,
            format_func=lambda i: _event_option(i, events[i]),
            key=f"remove_rejected::{state_key}",
        )
        if st.button("Удалить отметку о ложном срабатывании"):
            events.pop(rejected_choice)
            st.rerun()

    st.divider()
    if st.button("Сохранить разряд", type="primary", width="stretch"):
        try:
            parse_expert_labels({
                "schema_version": EXPERT_LABEL_SCHEMA_VERSION,
                "shots": [draft],
            })
            saved = upsert_shot_annotation(
                annotation_path,
                deepcopy(draft),
                expected_revision=st.session_state.annotation_revision,
            )
            st.session_state.annotation_revision = saved.revision
            st.success(f"Разметка {shot.shot_id} сохранена в {annotation_path}")
        except AnnotationConflictError as exc:
            st.error(f"Конфликт сохранения: {exc}")
        except Exception as exc:
            st.error(f"Разметка не сохранена: {exc}")


if __name__ == "__main__":
    main()
