from typing import Dict, List, Any

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _format_topic_list(topic_ids: List[int], df, max_len: int = 4) -> str:
    """Return comma‑separated topic labels, truncated when long."""
    if not topic_ids:
        return "no topics"

    labels: List[str] = []
    for tid in topic_ids:
        row = df.loc[df["id"] == tid, "topic_label"]
        labels.append(row.values[0] if not row.empty else f"Topic {tid}")

    if len(labels) > max_len:
        return ", ".join(labels[:max_len]) + f" … (+{len(labels)-max_len})"
    return ", ".join(labels)

# ----------------------------------------------------------------------
# Title generation
# ----------------------------------------------------------------------

class ChartTitleGenerator:
    """Consistent, mathematics‑aligned titles & subtitles."""

    # ------------------ Radar‑chart heading ------------------
    @staticmethod
    def get_portfolio_analysis_title(view_mode: str, is_market_mode: bool) -> str:
        if view_mode == "focus":  # composition
            return "Tech Composition (Corpus‑wide)" if is_market_mode else "Tech Composition (Portfolio)"
        else:                       # share
            return "Tech Share (Corpus‑wide)" if is_market_mode else "Tech Share (Portfolio)"

    # ------------------ Time‑series ------------------
    @staticmethod
    def get_time_series_title(
        view_mode: str,
        time_granularity: str,           # reserved – not used yet
        view_mode_series: str,           # 'groups' | 'topics'
        is_market_mode: bool,
        selected_count: int
    ) -> Dict[str, str]:
        """Return `{title, subtitle, subtitle_short}`."""

        if is_market_mode and view_mode_series == "groups":
            raise ValueError("Group lines are not allowed in corpus‑wide mode.")

        pct = "100"
        eq  = lambda num, den: f"Formula: ({num} ÷ {den}) {pct}"
        strip = lambda s: s.replace("Formula: ", "")

        # ---------- Matrix ----------
        if view_mode == "focus":
            if view_mode_series == "groups":
                title   = "Group tech composition (%)"
                num, den = "Group patents in selected tech", "Group total patents"
            else:  # topics
                if is_market_mode:
                    title   = "Corpus tech composition (%)"
                    num, den = "Patents in tech", "Patents in selected tech"
                else:
                    title   = "Portfolio tech composition (%)"
                    num, den = "Group patents in tech", "Group patents in selected tech"
        else:  # share
            if view_mode_series == "groups":
                if selected_count == 1:
                    title = "Group tech share (%)"
                else:
                    title = "Group combined tech share (%)"
                num, den = "Group patents in tech", "All patents in tech"
            else:  # topics
                if is_market_mode:
                    title = "Percent of patents in dataset mentioning technology (%)"
                    num, den = "Patents in tech", "Total patents in corpus"
                else:
                    title = "Portfolio tech share (%)"
                    num, den = "Group patents in tech", "All patents in tech"

        subtitle       = eq(num, den)
        subtitle_short = strip(subtitle)
        return {"title": title, "subtitle": subtitle, "subtitle_short": subtitle_short}

    # Compatibility shim – keep existing Dash callbacks working
    @staticmethod
    def get_time_series_subtitle(view_mode: str, view_mode_series: str,
                                 is_market_mode: bool, selected_count: int) -> str:
        info = ChartTitleGenerator.get_time_series_title(
            view_mode, "yearly", view_mode_series, is_market_mode, selected_count
        )
        return info["subtitle_short"]

# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------

class ChartLabelsGenerator:
    """Axis & hover labels plus UI indicators."""

    @staticmethod
    def get_time_series_labels(view_mode: str, view_mode_series: str,
                               is_market_mode: bool, time_granularity: str) -> Dict[str, str]:
        # Y‑axis + hover unit
        if view_mode == "focus":
            y_label = "Tech Composition (%)"
            hover   = "Composition"
        else:
            y_label = "Tech Share (%)"
            hover   = "Tech Share"

        if view_mode_series == "groups" and is_market_mode:
            hover += " (avg)"

        # Indicators for summary string
        mode_indicator   = "🎯 Composition" if view_mode == "focus" else "🏆 Share"
        gran_indicator   = "📅 Yearly" if time_granularity == "yearly" else "📆 Monthly"
        view_indicator   = "👥 Groups" if view_mode_series == "groups" else "🏷 Topics"

        return {
            "y_axis_label": y_label,
            "hover_label": hover,
            "hover_unit": "%",
            "mode_indicator": mode_indicator,
            "granularity_indicator": gran_indicator,
            "view_indicator": view_indicator,
        }

# ----------------------------------------------------------------------
# Captions – kept brutally short
# ----------------------------------------------------------------------

class RadarChartCaptions:
    @staticmethod
    def generate_caption(groups: Dict[str, Any], topic_ids: List[int], df,
                         view_mode: str, is_market_mode: bool) -> str:
        topics_txt = _format_topic_list(topic_ids, df)
        if view_mode == "focus":
            first = "share each tech takes within the selected basket"
        else:
            first = "share each entity holds of its tech market"
        scope = "across the corpus" if is_market_mode else "within your portfolio"
        return f"Radar shows the {first} {scope} for: {topics_txt}."

class TimeSeriesCaptions:
    @staticmethod
    def generate_caption(topic_ids: List[int], df, view_mode: str,
                          view_mode_series: str, is_market_mode: bool,
                          granularity: str) -> str:
        topics_txt = _format_topic_list(topic_ids, df)
        period = "years" if granularity == "yearly" else "months"
        if view_mode == "focus":
            noun = "composition"
        else:
            noun = "share"
        if view_mode_series == "groups":
            subject = "each group"
        else:
            subject = "each technology"
        scope = "corpus‑wide" if is_market_mode else "your portfolio"
        return (f"Lines track the {noun} of {subject} across {period} in the {scope} dataset "
                f"for: {topics_txt}.")

# Convenience wrappers for legacy imports -------------------------------------------------

def generate_radar_caption(entities: Dict[str, Any], selected_topics: List[int],
                          view_mode: str, market_mode: bool, df) -> str:
    return RadarChartCaptions.generate_caption(entities, selected_topics, df, view_mode, market_mode)


def generate_time_series_caption(series_data: Dict[str, Any], selected_topics: List[int],
                                view_mode: str, granularity: str, view_mode_series: str,
                                market_mode: bool, *_, df) -> str:
    return TimeSeriesCaptions.generate_caption(selected_topics, df, view_mode,
                                              view_mode_series, market_mode, granularity)

