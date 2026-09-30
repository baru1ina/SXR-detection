"""Browser bridge that reports Plotly zoom/pan ranges back to Streamlit."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import streamlit.components.v2 as components_v2


_VIEWPORT_LISTENER = components_v2.component(
    "sxr_plot_viewport_listener",
    html='<span class="sxr-plot-viewport-listener" aria-hidden="true"></span>',
    css=".sxr-plot-viewport-listener { display: none; }",
    js=r"""
        export default function(component) {
            const {data, setStateValue} = component;
            let plot = null;
            let handler = null;
            let observer = null;
            let retryTimer = null;

            function finiteRange(value) {
                if (!Array.isArray(value) || value.length !== 2) return null;
                const result = value.map(Number);
                return result.every(Number.isFinite) ? result : null;
            }

            function findPlot() {
                const plots = Array.from(document.querySelectorAll('.js-plotly-plot'));
                const matching = plots.filter((candidate) => {
                    const meta = candidate?._fullLayout?.meta;
                    return meta?.sxr_viewport_token === data.plotToken;
                });
                return matching.at(-1) ?? plots.at(-1) ?? null;
            }

            function collectViewport(eventData) {
                const layout = plot?._fullLayout;
                if (!layout) return null;
                const xRange = finiteRange(layout.xaxis?.range);
                if (!xRange) return null;

                const yRanges = {};
                for (const [axisName, channel] of Object.entries(data.yAxes ?? {})) {
                    const range = finiteRange(layout[axisName]?.range);
                    if (range) yRanges[channel] = range;
                }

                let activeYChannel = data.activeYChannel ?? null;
                for (const [axisName, channel] of Object.entries(data.yAxes ?? {})) {
                    if (Object.keys(eventData ?? {}).some((key) => key.startsWith(`${axisName}.`))) {
                        activeYChannel = channel;
                        break;
                    }
                }
                return {
                    x_range: xRange,
                    y_ranges: yRanges,
                    active_y_channel: activeYChannel,
                    sequence: Date.now(),
                };
            }

            function detach() {
                if (plot && handler && typeof plot.removeListener === 'function') {
                    plot.removeListener('plotly_relayout', handler);
                }
                plot = null;
                handler = null;
            }

            function attachLatest() {
                const candidate = findPlot();
                if (!candidate || typeof candidate.on !== 'function') return false;
                const stillAttached = (
                    candidate === plot
                    && handler !== null
                    && typeof candidate.listeners === 'function'
                    && candidate.listeners('plotly_relayout').includes(handler)
                );
                if (stillAttached) return true;
                detach();
                plot = candidate;
                handler = (eventData) => {
                    window.requestAnimationFrame(() => {
                        const viewport = collectViewport(eventData);
                        if (viewport) setStateValue('viewport', viewport);
                    });
                };
                plot.on('plotly_relayout', handler);
                return true;
            }

            attachLatest();
            observer = new MutationObserver(() => attachLatest());
            observer.observe(document.body, {childList: true, subtree: true});
            retryTimer = window.setInterval(attachLatest, 250);

            return () => {
                if (observer !== null) observer.disconnect();
                if (retryTimer !== null) window.clearInterval(retryTimer);
                detach();
            };
        }
    """,
)


def render_plot_viewport_listener(
    *,
    plot_token: str,
    y_axes: Mapping[str, str],
    active_y_channel: str,
    key: str,
    on_viewport_change: Callable[[], None] | None = None,
) -> None:
    """Mount an invisible relayout listener immediately after a Plotly chart."""

    _VIEWPORT_LISTENER(
        key=key,
        data={
            "plotToken": plot_token,
            "yAxes": dict(y_axes),
            "activeYChannel": active_y_channel,
        },
        default={"viewport": None},
        height=1,
        on_viewport_change=on_viewport_change or (lambda: None),
    )
