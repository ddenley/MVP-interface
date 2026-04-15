import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update, MATCH, ALL
import json
import pandas as pd
import plotly.graph_objects as go
from vizcore.data_loader import load_bubble_map_data, load_topic_info_data, pair_topic_label_data, pair_topic_summary_data
from vizcore.svg_builder import render_bubble_svg
import gc

# Char Descriptions
from chart_descriptions import (
    ChartTitleGenerator,
    ChartLabelsGenerator,
    generate_radar_caption,
    generate_time_series_caption
)


LOADING_TIME = 500

# Load data once at startup (unchanged)
print("Loading data...")
df = load_bubble_map_data("data/bubble_layout.jsonl")
topic_df = load_topic_info_data(
    "data/topic_info_spad-dataset-jsonl_sentence_clusters_junk_sentences_removed_semantic_chunked_w10_p95_embedded_chunks.parquet")
df = pair_topic_label_data(df, topic_df)
df = pair_topic_summary_data(df, topic_df)
print(f"✅ Loaded {len(df)} patent topics")

# Load available applicants (unchanged)
print("Loading applicant data...")
try:
    applicant_dist = pd.read_parquet("Distributions/topic_by_applicant.parquet")
    applicant_counts = applicant_dist.groupby('value')['count'].sum().sort_values(ascending=False)
    available_applicants = [
        {'label': f"{name} ({count} patents)", 'value': name}
        for name, count in applicant_counts.items()
    ]
    print(f"✅ Loaded {len(available_applicants)} applicants")
except Exception as e:
    available_applicants = []
    print(f"⚠️ Could not load applicant distributions: {e}")

#Load time series data
print("Loading time series data...")
try:
    monthly_dist = pd.read_parquet("Distributions/topic_by_applicant_month.parquet")
    yearly_dist = pd.read_parquet("Distributions/topic_by_applicant_year.parquet")
    print(f"✅ Loaded monthly data: {len(monthly_dist)} records")
    print(f"✅ Loaded yearly data: {len(yearly_dist)} records")

    # Debug: Print first few rows and column names
    print(f"📋 Yearly data columns: {list(yearly_dist.columns)}")
    print(f"📋 Monthly data columns: {list(monthly_dist.columns)}")
    print(f"📋 Sample yearly data:")
    print(yearly_dist.head(3))
except Exception as e:
    monthly_dist = pd.DataFrame()
    yearly_dist = pd.DataFrame()
    print(f"⚠️ Could not load time series data: {e}")

# Load document distribution data for document explorer
print("Loading document distribution data...")
try:
    document_dist_df = pd.read_parquet("Distributions/document_distribution.parquet")
    print(f"✅ Loaded document distribution: {len(document_dist_df)} patents")

    # Get list of available topic count columns for quick reference
    topic_count_cols = [col for col in document_dist_df.columns if col.endswith('_count')]
    topic_chunks_cols = [col for col in document_dist_df.columns if col.endswith('_chunks')]
    print(f"📋 Available topic columns: {len(topic_count_cols)} count columns, {len(topic_chunks_cols)} chunk columns")
except Exception as e:
    document_dist_df = pd.DataFrame()
    topic_count_cols = []
    topic_chunks_cols = []
    print(f"⚠️ Could not load document distribution: {e}")

gc.collect()


GROUP_COLORS = [
    '#2E86AB',  # Professional Blue
    '#A23B72',  # Deep Rose
    '#F18F01',  # Vibrant Orange
    '#C73E1D',  # Corporate Red
    '#592E83'  # Royal Purple
]

TOPIC_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

GROUP_LINE_STYLES = ['solid', 'dash', 'dot']  # Up to 3 groups


# Helper functions
def get_next_available_color(groups_data):
    used_colors = {group_info['color'] for group_info in groups_data.values()}
    for color in GROUP_COLORS:
        if color not in used_colors:
            return color
    return GROUP_COLORS[0]


def get_next_group_name(groups_data):
    existing_names = {group_info['name'] for group_info in groups_data.values()}
    if "Default Group" not in existing_names:
        return "Default Group"
    counter = 2
    while f"Default Group {counter}" in existing_names:
        counter += 1
    return f"Default Group {counter}"


def get_all_assigned_applicants(groups_data, exclude_group=None):
    """Get set of all applicants assigned to any group (except excluded group)."""
    assigned = set()
    for group_id, group_info in groups_data.items():
        if group_id != exclude_group:
            assigned.update(group_info.get('applicants', []))
    return assigned


# Check if we should use market-wide analysis mode
def should_use_market_mode(groups_data, active_groups):
    """Determine if we should show market-wide analysis instead of user groups."""
    if not groups_data or not active_groups:
        return True

    # Check if any active groups have applicants
    for group_id, is_active in active_groups.items():
        if is_active and group_id in groups_data:
            if groups_data[group_id].get('applicants', []):
                return False  # At least one active group has applicants

    return True  # No active groups with applicants


# Calculate market-wide focus data
def calculate_market_focus_data(selected_topics, applicant_dist, df, view_mode='focus'):
    """Calculate market-wide topic analysis when no groups are active."""
    print(f"\n🌍 CALCULATING MARKET-WIDE DATA (MODE: {view_mode.upper()})...")

    if not selected_topics:
        return {}

    selected_topic_ids = [int(tid) for tid in selected_topics if int(tid) != -1]
    focus_data = {}

    # Calculate total patents across all topics
    total_all_patents = applicant_dist[applicant_dist['topic_id'] != -1]['count'].sum()
    print(f"   Total market patents (all topics): {total_all_patents}")

    # For radar chart, we create a single "Total Market" entity
    market_entity_id = 'total_market'

    # Process all selected topics
    for topic_id in selected_topic_ids:
        topic_patents = applicant_dist[applicant_dist['topic_id'] == topic_id]['count'].sum()

        # Initialize the topic in focus_data regardless of patent count
        if topic_id not in focus_data:
            focus_data[topic_id] = []

        if view_mode == 'focus':
            # FIXED: Portfolio Focus = Composition within selected topics (sums to 100%)
            combined_selected_patents = sum([
                applicant_dist[applicant_dist['topic_id'] == tid]['count'].sum()
                for tid in selected_topic_ids
            ])
            percentage = (topic_patents / combined_selected_patents) * 100 if combined_selected_patents > 0 else 0
            total_base = combined_selected_patents
            mode_label = "selected area composition"
        else:  # share
            # FIXED: Market Share = Global innovation focus (% of all patents)
            percentage = (topic_patents / total_all_patents) * 100 if total_all_patents > 0 else 0
            total_base = total_all_patents
            mode_label = "global innovation focus"

        focus_data[topic_id].append({
            'group_id': market_entity_id,
            'name': 'Total Market',
            'color': '#2E86AB',  # Use consistent color
            'percentage': float(percentage),
            'patent_count': int(topic_patents),
            'total_patents': int(total_base),
            'mode': view_mode,
            'is_market_mode': True
        })

        print(f"Topic {topic_id}: {percentage:.1f}% {mode_label} ({topic_patents} patents)")

    print(f"Market-wide analysis complete for {len(focus_data)} topics!")
    return focus_data


# Calculate market-wide time series data with FIXED intuitive logic
def calculate_market_time_series_data(selected_topics, view_mode='focus', time_granularity='yearly',
                                      view_mode_series='topics'):
    """Calculate market-wide time series data when no groups are active."""
    print(
        f"\n🌍 CALCULATING MARKET TIME SERIES (MODE: {view_mode.upper()}, GRANULARITY: {time_granularity.upper()}, VIEW: {view_mode_series.upper()})...")

    # Choose the appropriate dataset
    time_dist = yearly_dist if time_granularity == 'yearly' else monthly_dist
    time_col = 'year_bin' if time_granularity == 'yearly' else 'month_bin'

    if time_dist.empty or not selected_topics:
        return {}

    selected_topic_ids = [int(tid) for tid in selected_topics if int(tid) != -1]

    # Get all unique time periods and sort them
    all_periods = sorted(time_dist[time_col].unique())
    print(f"   Time periods: {len(all_periods)} from {all_periods[0]} to {all_periods[-1]}")

    # Initialize result structure
    time_series_data = {}

    if view_mode_series == 'groups':
        # GROUPS VIEW: Single aggregated line combining all selected topics
        print(f"   GROUPS VIEW: Aggregating {len(selected_topic_ids)} topics into combined market activity")

        time_series_data['market_combined'] = {
            'name': f'Combined Market Activity ({len(selected_topic_ids)} topics)',
            'color': '#2E86AB',  # Use primary blue for market analysis
            'periods': [],
            'values': [],
            'raw_counts': [],
            'is_market_mode': True
        }

        for period in all_periods:
            # Calculate combined patents across ALL selected topics for this period
            combined_period_count = 0

            for topic_id in selected_topic_ids:
                topic_period_count = time_dist[
                    (time_dist['topic_id'] == topic_id) &
                    (time_dist[time_col] == period)
                    ]['count'].sum()
                combined_period_count += int(topic_period_count)

            # FIXED: Calculate the value based on view mode with corrected logic
            if view_mode == 'focus':
                # FIXED: Portfolio Focus = For market-wide groups view, show composition trends
                # This doesn't make much sense as a single line, so let's show average composition
                value = 100.0 / len(selected_topic_ids)  # Average composition per topic
            else:  # share
                # FIXED: Market Share = Global market positioning
                total_period_patents = time_dist[
                    (time_dist[time_col] == period) &
                    (time_dist['topic_id'] != -1)
                    ]['count'].sum()

                if total_period_patents > 0:
                    # Calculate average focus percentage across selected topics
                    total_focus_percentage = (combined_period_count / total_period_patents * 100)
                    value = total_focus_percentage / len(selected_topic_ids)  # Average focus per topic
                else:
                    value = 0

            time_series_data['market_combined']['periods'].append(period)
            time_series_data['market_combined']['values'].append(value)
            time_series_data['market_combined']['raw_counts'].append(combined_period_count)

        print(f"   Created combined market trend: {len(all_periods)} periods")

    else:
        # TOPICS VIEW: Individual topic lines
        print(f"   TOPICS VIEW: Creating {len(selected_topic_ids)} individual topic trends")

        for topic_id in selected_topic_ids:
            # Get topic label for display
            topic_row = df[df['id'] == topic_id]
            if not topic_row.empty:
                topic_label = topic_row.iloc[0].get('topic_label', f'Topic {topic_id}')
                display_label = topic_label[:40] + "..." if len(topic_label) > 40 else topic_label
            else:
                display_label = f'Topic {topic_id}'

            time_series_data[f'topic_{topic_id}'] = {
                'name': display_label,
                'color': GROUP_COLORS[len(time_series_data) % len(GROUP_COLORS)],
                'periods': [],
                'values': [],
                'raw_counts': [],
                'is_market_mode': True
            }

            for period in all_periods:
                # Calculate total patents for this topic in this period
                topic_period_count = time_dist[
                    (time_dist['topic_id'] == topic_id) &
                    (time_dist[time_col] == period)
                    ]['count'].sum()

                # FIXED: Calculate the value based on view mode with corrected logic
                if view_mode == 'focus':
                    # FIXED: Portfolio Focus = Composition within selected topics
                    combined_selected_period_count = sum([
                        time_dist[
                            (time_dist['topic_id'] == tid) &
                            (time_dist[time_col] == period)
                            ]['count'].sum()
                        for tid in selected_topic_ids
                    ])

                    value = (
                                topic_period_count / combined_selected_period_count * 100) if combined_selected_period_count > 0 else 0
                else:  # share
                    # FIXED: Market Share = Global market positioning
                    total_period_patents = time_dist[
                        (time_dist[time_col] == period) &
                        (time_dist['topic_id'] != -1)
                        ]['count'].sum()

                    value = (topic_period_count / total_period_patents * 100) if total_period_patents > 0 else 0

                time_series_data[f'topic_{topic_id}']['periods'].append(period)
                time_series_data[f'topic_{topic_id}']['values'].append(value)
                time_series_data[f'topic_{topic_id}']['raw_counts'].append(int(topic_period_count))

    entity_type = "combined market activity" if view_mode_series == 'groups' else "individual topics"
    print(f"Market time series calculation complete for {entity_type}!")
    gc.collect()
    return time_series_data

def calculate_group_focus_data(groups_data, active_groups, applicant_dist, df, view_mode='focus', selected_topics=None):
    """Optimized: Calculate focus % for each active group on each topic using precomputed matrix."""

    use_market_mode = should_use_market_mode(groups_data, active_groups)
    if use_market_mode and selected_topics:
        print(f"\n🌍 USING MARKET-WIDE MODE (VIEW: {view_mode.upper()})...")
        return calculate_market_focus_data(selected_topics, applicant_dist, df, view_mode)

    print(f"\n📊 CALCULATING FOCUS DATA (MODE: {view_mode.upper()})...")

    valid_topic_ids = [int(tid) for tid in df['id'].unique() if tid != -1]
    focus_data = {tid: [] for tid in valid_topic_ids}

    # --- Step 1: Build matrix of (applicant, topic_id) → count
    print("   Creating pivot table for (applicant, topic_id) → count...")
    pivot = (
        applicant_dist[applicant_dist['topic_id'] != -1]
        .pivot_table(index='value', columns='topic_id', values='count', aggfunc='sum', fill_value=0)
    )

    # --- Step 2: Calculate total patents in market per topic
    print("   Calculating market topic totals...")
    market_topic_totals = pivot.sum(axis=0).to_dict()

    # --- Step 3: Process each group
    for group_id, is_active in active_groups.items():
        if not is_active or group_id not in groups_data:
            continue

        group_info = groups_data[group_id]
        applicants = group_info.get('applicants', [])
        if not applicants:
            continue

        print(f"\n  Processing {group_info['name']} ({group_id}):")
        print(f"    Applicants: {applicants}")

        # Only keep applicants present in matrix
        group_applicants = [a for a in applicants if a in pivot.index]
        if not group_applicants:
            print("    ⚠️ No data for applicants in pivot table.")
            continue

        # Slice once
        group_matrix = pivot.loc[group_applicants]
        group_topic_sums = group_matrix.sum(axis=0).to_dict()
        group_total_patents = sum(group_topic_sums.values())

        print(f"    Total valid patents in group: {group_total_patents}")
        if group_total_patents == 0:
            continue

        for topic_id in valid_topic_ids:
            topic_patent_count = group_topic_sums.get(topic_id, 0)
            if topic_patent_count == 0:
                continue

            if view_mode == 'focus':
                percentage = (topic_patent_count / group_total_patents) * 100
                total_base = group_total_patents
                mode_label = "portfolio focus"
            else:
                market_total = market_topic_totals.get(topic_id, 0)
                if market_total == 0:
                    continue
                percentage = (topic_patent_count / market_total) * 100
                total_base = market_total
                mode_label = "market share"

            focus_data[topic_id].append({
                'group_id': group_id,
                'name': group_info['name'],
                'color': group_info['color'],
                'percentage': float(percentage),
                'patent_count': int(topic_patent_count),
                'total_patents': int(total_base),
                'mode': view_mode,
                'is_market_mode': False
            })

            if percentage > 5:
                print(f"      Topic {topic_id}: {percentage:.1f}% {mode_label} ({topic_patent_count} of {total_base})")

    # Sort group entries per topic
    for topic_id in focus_data:
        focus_data[topic_id].sort(key=lambda x: x['group_id'])

    print(f"✅ Focus data calculation complete!")
    gc.collect()
    return focus_data

def df_topic_lookup(df, topic_id):
    row = df[df["topic_id"] == topic_id]
    if not row.empty:
        return row.iloc[0]
    return {}

def calculate_time_series_focus_data(groups_data, active_groups, selected_topics, view_mode='focus',
                                     time_granularity='yearly', view_mode_series='groups'):
    """Optimized: Time series focus data for selected topics and active groups with market mode support."""

    use_market_mode = should_use_market_mode(groups_data, active_groups)
    if use_market_mode and selected_topics:
        print(f"\n🌍 USING MARKET-WIDE MODE FOR TIME SERIES...")
        return calculate_market_time_series_data(selected_topics, view_mode, time_granularity, view_mode_series)

    print(f"\n📈 CALCULATING TIME SERIES DATA (MODE: {view_mode.upper()}, GRANULARITY: {time_granularity.upper()}, VIEW: {view_mode_series.upper()})...")

    time_dist = yearly_dist if time_granularity == 'yearly' else monthly_dist
    time_col = 'year_bin' if time_granularity == 'yearly' else 'month_bin'

    if time_dist.empty or not selected_topics:
        return {}

    selected_topic_ids = [int(tid) for tid in selected_topics if int(tid) != -1]
    all_periods = sorted(time_dist[time_col].dropna().unique())
    time_series_data = {}

    # Save reference to global topic dataframe (for topic label lookup)
    topic_lookup_df = df

    # --- Precompute: Filter valid topic_ids and group into (applicant, topic_id, period) → count
    print("   Precomputing applicant-topic-period matrix...")
    time_series_df = (
        time_dist[
            (time_dist['topic_id'].isin(selected_topic_ids)) &
            (time_dist['topic_id'] != -1)
        ]
        .groupby(['applicants', 'topic_id', time_col])['count']
        .sum()
        .reset_index()
    )

    # Create fast lookup: group → DataFrame slice
    for group_id, is_active in active_groups.items():
        if not is_active or group_id not in groups_data:
            continue

        group_info = groups_data[group_id]
        applicants = group_info.get('applicants', [])
        if not applicants:
            continue

        print(f"\n  Processing {group_info['name']} ({group_id}) with {len(applicants)} applicants")

        group_df = time_series_df[time_series_df['applicants'].isin(applicants)]

        # Create full pivot: (topic_id, period) → patent count
        pivot = (
            group_df.groupby(['topic_id', time_col])['count']
            .sum()
            .unstack(fill_value=0)
        )

        if view_mode_series == 'groups':
            # Aggregate all selected topics → one line per group
            group_line = {
                'name': group_info['name'],
                'color': group_info['color'],
                'periods': [],
                'values': [],
                'raw_counts': [],
                'is_market_mode': False
            }

            for period in all_periods:
                # Total patents across all topics in this period
                period_sum = pivot.get(period, pd.Series()).sum()
                raw_count = int(period_sum)

                if view_mode == 'focus':
                    # Total R&D output of group that year/month
                    group_total_all_topics = time_dist[
                        (time_dist['applicants'].isin(applicants)) &
                        (time_dist[time_col] == period) &
                        (time_dist['topic_id'] != -1)
                    ]['count'].sum()
                    value = (raw_count / group_total_all_topics * 100) if group_total_all_topics else 0
                else:  # share
                    market_total = time_dist[
                        (time_dist['topic_id'].isin(selected_topic_ids)) &
                        (time_dist[time_col] == period)
                    ]['count'].sum()
                    value = (raw_count / market_total * 100) if market_total else 0

                group_line['periods'].append(period)
                group_line['values'].append(value)
                group_line['raw_counts'].append(raw_count)

            time_series_data[group_id] = group_line

        else:  # topics view – multiple lines per group-topic combo
            for topic_id in selected_topic_ids:
                # Get topic label from global topic dataframe (not filtered df)
                topic_row = topic_lookup_df[topic_lookup_df['id'] == topic_id]
                if not topic_row.empty:
                    topic_label = topic_row.iloc[0].get('topic_label', f'Topic {topic_id}')
                else:
                    topic_label = f'Topic {topic_id}'
                short_label = topic_label[:25] + "…" if len(topic_label) > 25 else topic_label
                line_name = f"{group_info['name']}: {short_label}"
                line_id = f"group_{group_id}_topic_{topic_id}"
                group_idx = list(groups_data.keys()).index(group_id)

                topic_line = {
                    'name': line_name,
                    'color': TOPIC_COLORS[topic_id % len(TOPIC_COLORS)],
                    'line_style': GROUP_LINE_STYLES[group_idx % len(GROUP_LINE_STYLES)],
                    'periods': [],
                    'values': [],
                    'raw_counts': [],
                    'is_market_mode': False
                }

                for period in all_periods:
                    count = pivot.at[topic_id, period] if (topic_id in pivot.index and period in pivot.columns) else 0
                    raw_count = int(count)

                    if view_mode == 'focus':
                        topic_total = pivot.loc[topic_id].sum() if topic_id in pivot.index else 0
                        value = (raw_count / topic_total * 100) if topic_total else 0
                    else:  # share
                        market_total = time_dist[
                            (time_dist['topic_id'] == topic_id) &
                            (time_dist[time_col] == period)
                        ]['count'].sum()
                        value = (raw_count / market_total * 100) if market_total else 0

                    topic_line['periods'].append(period)
                    topic_line['values'].append(value)
                    topic_line['raw_counts'].append(raw_count)

                time_series_data[line_id] = topic_line

    print(f"✅ Time series calculation complete for {len(time_series_data)} lines.")
    return time_series_data


# Portfolio management functions (ENHANCED with duplicate checking) - UNCHANGED
def generate_groups_display(groups_data, selected_group, active_groups, editing_group):
    """Generate the visual display of all groups with click functionality."""
    group_elements = []

    for group_id, group_info in groups_data.items():
        group_color = group_info['color']
        group_name = group_info['name']
        applicants = group_info['applicants']

        is_selected = group_id == selected_group
        is_active = active_groups.get(group_id, False)
        is_editing = editing_group == group_id

        background_color = '#E3F2FD' if is_selected else '#F8F9FA'
        border_color = '#2196F3' if is_selected else '#E6E8EB'

        if applicants:
            applicant_items = []
            for applicant in applicants:
                applicant_items.append(
                    html.Div([
                        html.Span(applicant, style={'fontSize': '12px'}),
                        html.Button("×",
                                    id={'type': 'remove-applicant', 'group': group_id, 'applicant': applicant},
                                    style={'backgroundColor': 'transparent', 'border': 'none',
                                           'color': '#dc3545', 'fontSize': '14px', 'cursor': 'pointer',
                                           'marginLeft': '8px', 'padding': '0px 4px'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between',
                              'alignItems': 'center', 'marginBottom': '4px'})
                )
            applicant_display = html.Div(applicant_items, style={'marginLeft': '28px', 'marginBottom': '8px'})
        else:
            applicant_display = html.Div([
                html.Span("No applicants added",
                          style={'color': '#687589', 'fontSize': '12px', 'fontStyle': 'italic'})
            ], style={'marginLeft': '28px', 'marginBottom': '8px'})

        group_header_left = [
            html.Span("●", style={'color': group_color, 'fontSize': '18px', 'marginRight': '10px'}),
        ]

        if is_editing:
            group_header_left.extend([
                dcc.Input(
                    id={'type': 'group-name-input', 'group': group_id},
                    value=group_name,
                    type='text',
                    style={
                        'flex': '1',
                        'marginRight': '8px',
                        'padding': '2px 8px',
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'color': group_color,
                        'border': f'2px solid {group_color}',
                        'borderRadius': '4px',
                        'backgroundColor': 'white'
                    },
                    autoFocus=True,
                    maxLength=30
                ),
                html.Button("✓",
                            id={'type': 'save-group-name', 'group': group_id},
                            style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                                   'padding': '2px 8px', 'borderRadius': '3px', 'fontSize': '12px',
                                   'cursor': 'pointer', 'marginRight': '4px'}),
                html.Button("✕",
                            id={'type': 'cancel-group-edit', 'group': group_id},
                            style={'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none',
                                   'padding': '2px 8px', 'borderRadius': '3px', 'fontSize': '12px',
                                   'cursor': 'pointer', 'marginRight': '4px'})
            ])
        else:
            group_header_left.extend([
                html.Strong(group_name, style={'color': group_color, 'flex': '1'}),
                html.Button("✏️",
                            id={'type': 'edit-group-name', 'group': group_id},
                            style={'backgroundColor': 'transparent', 'border': 'none',
                                   'padding': '0px 4px', 'fontSize': '12px', 'cursor': 'pointer',
                                   'color': '#687589', 'marginLeft': '4px'},
                            title='Edit group name'),
                html.Span(f" ({len(applicants)} applicants)",
                          style={'color': '#687589', 'fontSize': '12px', 'marginLeft': '8px'})
            ])

        if is_selected and not is_editing:
            group_header_left.append(
                html.Span("✓ Selected", style={'color': '#2196F3', 'fontSize': '10px', 'fontWeight': 'bold',
                                               'marginLeft': '8px', 'padding': '2px 6px', 'backgroundColor': '#E3F2FD',
                                               'borderRadius': '10px', 'border': '1px solid #2196F3'})
            )

        if is_editing:
            header_left = html.Div(
                group_header_left,
                style={'display': 'flex', 'alignItems': 'center', 'flex': '1', 'cursor': 'default'}
            )
        else:
            header_left = html.Div(
                group_header_left,
                id={'type': 'group-box', 'group': group_id},
                style={'display': 'flex', 'alignItems': 'center', 'flex': '1', 'cursor': 'pointer'}
            )

        header_right = []

        toggle_style = {
            'position': 'relative',
            'width': '40px',
            'height': '20px',
            'backgroundColor': '#28a745' if is_active else '#ccc',
            'borderRadius': '20px',
            'cursor': 'pointer',
            'transition': 'background-color 0.3s ease',
            'marginRight': '8px',
            'display': 'inline-block'
        }

        toggle_circle_style = {
            'position': 'absolute',
            'top': '2px',
            'left': '20px' if is_active else '2px',
            'width': '16px',
            'height': '16px',
            'backgroundColor': 'white',
            'borderRadius': '50%',
            'transition': 'left 0.3s ease',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
        }

        active_toggle = html.Div(
            [html.Div(style=toggle_circle_style)],
            id={'type': 'active-toggle', 'group': group_id},
            n_clicks=0,
            style=toggle_style,
            title='Toggle active visualization',
        )

        header_right.append(active_toggle)

        if len(groups_data) > 1:
            header_right.append(
                html.Button("🗑️",
                            id={'type': 'remove-group', 'group': group_id},
                            style={'backgroundColor': 'transparent', 'border': 'none',
                                   'color': '#dc3545', 'fontSize': '12px', 'cursor': 'pointer',
                                   'marginLeft': '4px', 'padding': '2px 4px',
                                   'borderRadius': '3px'})
            )

        group_header = html.Div([
            header_left,
            html.Div(header_right, style={'display': 'flex', 'alignItems': 'center'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'})

        group_element = html.Div([
            group_header,
            applicant_display
        ], style={
            'padding': '12px',
            'backgroundColor': background_color,
            'borderRadius': '6px',
            'border': f'2px solid {border_color}',
            'marginBottom': '10px',
            'transition': 'all 0.2s ease'
        })

        group_elements.append(group_element)

    return group_elements


# Initialize the Dash app
app = dash.Dash(__name__, title="SPAD Patent Analysis Explorer", suppress_callback_exceptions=True)

# Loading visual
# Add CSS for loading animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .analysis-spinner {
                border: 3px solid #E6E8EB;
                border-top: 3px solid #2E86AB;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .loading-pulse {
                background: linear-gradient(90deg, #E6E8EB 25%, #F8F9FA 50%, #E6E8EB 75%);
                background-size: 200% 100%;
                animation: loading-pulse 1.5s infinite;
                border-radius: 6px;
            }

            @keyframes loading-pulse {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def create_loading_spinner(component_type, message="Analyzing..."):
    """Create loading state that shows until analysis is complete"""
    messages = {
        'radar': 'Calculating portfolio focus...',
        'timeseries': 'Processing time series data...',
        'documents': 'Searching patent database...',
        'general': 'Analyzing data...'
    }

    return html.Div([
        html.Div([
            html.Div(className="analysis-spinner"),
            html.H5(message, style={'color': '#687589', 'marginTop': '15px', 'marginBottom': '8px'}),
            html.P(messages.get(component_type, 'Processing...'),
                   style={'color': '#687589', 'fontSize': '12px', 'fontStyle': 'italic'}),
            html.P("⏳ Loading will complete when fresh data is ready...",
                   style={'color': '#2E86AB', 'fontSize': '11px', 'fontStyle': 'italic', 'marginTop': '8px'})
        ], style={'textAlign': 'center', 'padding': '40px'})
    ], style={
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
        'minHeight': '300px', 'backgroundColor': '#F8F9FA',
        'border': '2px solid #E6E8EB', 'borderRadius': '10px'
    })

# Create initial SVG with no selections
initial_svg = render_bubble_svg(df)

app.layout = html.Div([
    # Floating Portfolio Manager Toggle (unchanged)
    html.Div([
        html.Button([
            html.Span("📊", style={'marginRight': '8px'}),
            "Portfolio Manager"
        ], id='portfolio-toggle-btn', n_clicks=0,
            style={
                'position': 'fixed',
                'top': '20px',
                'left': '20px',
                'zIndex': '1000',
                'backgroundColor': '#2E86AB',
                'color': 'white',
                'border': 'none',
                'padding': '12px 20px',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'fontSize': '14px',
                'fontWeight': '600',
                'boxShadow': '0 4px 12px rgba(46, 134, 171, 0.3)',
                'transition': 'all 0.3s ease'
            })
    ]),

    # UPDATED: View Mode Toggle Button
    html.Div([
        html.Button(
            [
                html.Span("🎯", style={'marginRight': '8px'}),
                html.Span(id='view-mode-text', children='Portfolio Focus')
            ],
            id='view-mode-toggle-btn',
            n_clicks=0,
            style={
                'position': 'fixed', 'top': '20px', 'right': '20px',
                'zIndex': 1000,
                'backgroundColor': '#6A1B9A', 'color': 'white',
                'border': 'none', 'padding': '12px 20px',
                'borderRadius': '8px', 'cursor': 'pointer',
                'fontSize': '14px', 'fontWeight': '600',
                'boxShadow': '0 4px 12px rgba(106,27,154,.3)'
            }
        )
    ]),

    # Portfolio Manager Overlay (unchanged structure)
    html.Div([
        html.Div(id='portfolio-backdrop', n_clicks=0,
                 style={
                     'position': 'fixed',
                     'top': '0',
                     'left': '0',
                     'width': '100vw',
                     'height': '100vh',
                     'backgroundColor': 'rgba(0,0,0,0.5)',
                     'zIndex': '1000',
                     'display': 'none',
                     'transition': 'opacity 0.3s ease'
                 }),

        html.Div([
            html.Div([
                html.Div([
                    html.H3("📊 Portfolio Manager",
                            style={'color': '#04193A', 'margin': '0', 'flex': '1'}),
                    html.Button("✕", id='portfolio-close-btn', n_clicks=0,
                                style={'backgroundColor': 'transparent', 'border': 'none',
                                       'fontSize': '20px', 'cursor': 'pointer', 'color': '#687589'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px',
                          'paddingBottom': '15px', 'borderBottom': '2px solid #E6E8EB'}),

                html.Div([
                    html.Div([
                        html.H4("Portfolio Groups", style={'color': '#04193A', 'marginBottom': '15px'}),
                        html.Div(id='groups-container', children=[]),

                        html.Button("+ Create New Group", id='add-group-btn', n_clicks=0,
                                    style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                                           'padding': '8px 16px', 'borderRadius': '5px', 'fontSize': '12px',
                                           'width': '100%', 'marginBottom': '20px', 'cursor': 'pointer'})
                    ]),

                    html.Div([
                        html.H4("Add Applicants", style={'color': '#04193A', 'marginBottom': '15px'}),

                        html.P("Click a group above to select it, then search for applicants to add:",
                               style={'fontSize': '12px', 'color': '#687589', 'marginBottom': '10px'}),

                        # Warning message for duplicate applicants
                        html.Div(id='duplicate-warning', style={'marginBottom': '10px'}),

                        dcc.Dropdown(
                            id='applicant-search',
                            placeholder="Type to search applicants (e.g., Sony, Canon...)",
                            options=[],
                            value=None,
                            searchable=True,
                            clearable=True,
                            style={'marginBottom': '10px', 'fontSize': '12px'},
                            optionHeight=35,
                            maxHeight=200,
                        ),

                        html.Button("Add to Selected Group", id='add-applicant-btn', n_clicks=0,
                                    style={'backgroundColor': '#0066CC', 'color': 'white', 'border': 'none',
                                           'padding': '8px 16px', 'borderRadius': '5px', 'fontSize': '12px',
                                           'width': '100%', 'cursor': 'pointer'})
                    ]),

                    html.Div(style={'height': '200px'})
                ])
            ], style={
                'height': '100vh',
                'overflowY': 'auto',
                'padding': '30px',
                'paddingBottom': '30px',
                'boxSizing': 'border-box'
            })
        ], id='portfolio-panel',
            style={
                'position': 'fixed',
                'top': '0',
                'left': '-470px',
                'width': '420px',
                'height': '100vh',
                'backgroundColor': 'white',
                'zIndex': '1001',
                'padding': '0',
                'boxShadow': '4px 0 20px rgba(0,0,0,0.15)',
                'transition': 'left 0.3s ease-in-out',
                'overflowY': 'hidden',
                'visibility': 'hidden'
            })
    ]),

    # Main Content
    html.Div([
        html.Div([
            html.H1("SPAD Patent Technology Explorer",
                    style={'color': '#04193A', 'textAlign': 'center', 'marginBottom': '20px',
                           'marginTop': '60px'}),
            html.P(f"Interactive analysis of {len(df)} patent topics with semantic clustering",
                   style={'textAlign': 'center', 'color': '#687589'}),
            html.Div([
                html.Strong("Active groups:", style={"display": "block", "marginBottom": "6px"}),
                html.Div(
                    id="active-groups-display",
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "flexWrap": "wrap",
                        "gap": "6px",
                    },
                ),
            ], style={"marginTop": "10px", "textAlign": "center"}),

            # Selected Topics Display
            html.Div([
                html.Strong("Selected topics:", style={"display": "block", "marginBottom": "6px", "marginTop": "15px"}),
                html.Div(
                    id="selected-topics-display",
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "flexWrap": "wrap",
                        "gap": "6px",
                        "minHeight": "24px",
                    },
                ),
                html.Div(
                    id="selection-limit-info",
                    style={
                        "fontSize": "11px",
                        "color": "#687589",
                        "textAlign": "center",
                        "marginTop": "4px",
                        "fontStyle": "italic"
                    }
                )
            ], style={"textAlign": "center"}),
        ], style={'padding': '20px', 'backgroundColor': '#F8F9FA', 'marginBottom': '20px'}),

        # UPDATED: SVG Display Container with responsive height
        html.Div([
            html.H3("Patent Topic Landscape", style={'color': '#04193A', 'marginBottom': '15px'}),
            html.Div([
                html.Iframe(
                    id='svg-frame',
                    srcDoc=initial_svg,
                    style={
                        'width': '100%',
                        'height': '700px',  # Initial height, will be auto-adjusted
                        'border': 'none',
                        'borderRadius': '10px',
                        'transition': 'height 0.3s ease',  # Smooth height transitions
                        'overflow': 'hidden'  # Disable scroll bars
                    }
                )
            ], style={
                'border': '2px solid #364761',
                'borderRadius': '10px',
                'padding': '10px',
                'backgroundColor': 'white',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            })
        ], style={'padding': '20px'}),

        # Analysis Dashboard - Two side-by-side containers
        html.Div([
            # Left Analysis Box
            html.Div([
                html.H4(id="portfolio-analysis-title",
                        style={'color': '#04193A', 'marginBottom': '15px', 'textAlign': 'center'}),
                html.Div(
                    id='portfolio-radar-content',
                    style={'minHeight': '400px'}
                )
            ], style={
                'flex': '1',
                'backgroundColor': 'white',
                'border': '2px solid #364761',
                'borderRadius': '10px',
                'padding': '20px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            }),

            # Right Analysis Box - ENHANCED with time series
            html.Div([
                # Header with view toggle and time dropdown
                html.Div([
                    html.Div([
                        html.H4("Topic Insights", style={'color': '#04193A', 'margin': '0', 'marginRight': '15px'}),
                        # View toggle switch
                        html.Div([
                            html.Span("Groups", style={'fontSize': '10px', 'color': '#687589', 'marginRight': '6px'}),
                            html.Div([
                                html.Div(
                                    id='view-toggle-circle',
                                    style={
                                        'position': 'absolute',
                                        'top': '2px',
                                        'left': '2px',  # Will be updated by callback
                                        'width': '16px',
                                        'height': '16px',
                                        'backgroundColor': 'white',
                                        'borderRadius': '50%',
                                        'transition': 'left 0.3s ease',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.3)'
                                    }
                                )
                            ],
                                id='view-toggle-switch',
                                n_clicks=0,
                                title="", # Updated via callback
                                style={
                                    'position': 'relative',
                                    'width': '40px',
                                    'height': '20px',
                                    'backgroundColor': '#2E86AB',  # Will be updated by callback
                                    'borderRadius': '20px',
                                    'cursor': 'pointer',
                                    'transition': 'background-color 0.3s ease',
                                    'display': 'inline-block'
                                }),
                            html.Span("Topics", style={'fontSize': '10px', 'color': '#687589', 'marginLeft': '6px'})
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1'}),

                    html.Div([
                        dcc.Dropdown(
                            id='time-granularity-dropdown',
                            options=[
                                {'label': '📅 Yearly', 'value': 'yearly'},
                                {'label': '📆 Monthly', 'value': 'monthly'}
                            ],
                            value='yearly',
                            clearable=False,
                            style={'minWidth': '120px', 'fontSize': '12px'}
                        )
                    ])
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
                          'marginBottom': '15px'}),

                html.Div(
                    id='time-series-content',
                    style={'minHeight': '350px'}
                )
            ], style={
                'flex': '1',
                'backgroundColor': 'white',
                'border': '2px solid #364761',
                'borderRadius': '10px',
                'padding': '20px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            })
        ], style={
            'display': 'flex',
            'gap': '20px',
            'padding': '0 20px 20px 20px'
        }),

        # Document Explorer Container
        html.Div([
            html.H3("Document Explorer", style={'color': '#04193A', 'marginBottom': '15px'}),
            html.Div(
                id='document-explorer-content',
                style={'minHeight': '300px'}
            )
        ], style={'padding': '20px'}),

        # Selection Info Panel
        html.Div([
            html.H3("Selected Patent Topics", style={'color': '#04193A'}),
            html.Div(id='selection-info',
                     children="Click up to 5 topic bubbles to explore patent clusters and analyze technological themes")
        ], style={'padding': '20px', 'backgroundColor': '#E6E8EB', 'margin': '20px', 'borderRadius': '10px'}),

        # Analysis Summary Panel
        html.Div([
            html.H3("Analysis Overview", style={'color': '#04193A'}),
            html.Div([
                html.P(f"📊 Total Topics: {len(df)}", style={'margin': '5px 0'}),
                html.P(f"🔗 Clusters: {len(df['cluster'].unique()) if 'cluster' in df.columns else 'N/A'}",
                       style={'margin': '5px 0'}),
            ])
        ], style={'padding': '20px', 'backgroundColor': '#F8F9FA', 'margin': '20px', 'borderRadius': '10px'})
    ]),

    # CLEANED UP: Stores for State Management (NO DUPLICATES)
    dcc.Store(id='selected-bubbles', data=[]),
    dcc.Store(id='portfolio-open', data=False),
    dcc.Store(id='applicant-groups', data={
        'group_1': {
            'name': 'Default Group',
            'color': GROUP_COLORS[0],
            'applicants': []
        }
    }),
    dcc.Store(id='group-counter', data=1),
    dcc.Store(id='available-applicants-store', data=available_applicants),
    dcc.Store(id='selected-group', data='group_1'),
    dcc.Store(id='active-groups', data={'group_1': True}),
    dcc.Store(id='group-focus-data', data={}),
    dcc.Store(id='editing-group', data=None),
    dcc.Store(id='view-mode', data='focus'),
    dcc.Store(id='time-series-data', data={}),
    dcc.Store(id='view-mode-series', data='topics'),  # SINGLE DEFINITION: Default to topics mode

    # Chunk modal data
    dcc.Store(id='chunk-modal-data', data={}),
    dcc.Store(id='chunk-modal-open', data=False),

    # Client-side interaction stores
    dcc.Store(id='click-data', data=None),
    dcc.Interval(id='click-detector', interval=10, n_intervals=0),

    # FIXED: Loading state management stores
    dcc.Store(id='selection-pending', data=False),
    dcc.Store(id='debounce-trigger', data=0),
    dcc.Store(id='analysis-complete', data=0),
    dcc.Interval(id='debounce-interval', interval=500, n_intervals=0, disabled=True),
])


def _get_patent_applicants_display(patent_row):
    """Generate applicant display for a patent row."""
    # Get applicants from the 'applicants' column
    applicants_raw = patent_row.get('applicants')

    if applicants_raw is None or (isinstance(applicants_raw, float) and pd.isna(applicants_raw)):
        return [
            html.Span("No applicants listed", style={'color': '#687589', 'fontSize': '11px', 'fontStyle': 'italic'})]

    # Handle different data types
    try:
        if isinstance(applicants_raw, list):
            applicants_list = applicants_raw
        elif isinstance(applicants_raw, str):
            # Try to parse as JSON first, then fall back to comma-separated
            try:
                applicants_list = json.loads(applicants_raw)
            except:
                applicants_list = [app.strip() for app in applicants_raw.split(',') if app.strip()]
        else:
            applicants_list = [str(applicants_raw)]
    except:
        return [html.Span("Error parsing applicants", style={'color': '#C73E1D', 'fontSize': '11px'})]

    if not applicants_list:
        return [
            html.Span("No applicants listed", style={'color': '#687589', 'fontSize': '11px', 'fontStyle': 'italic'})]

    # Show top 3 applicants with overflow indicator
    display_applicants = applicants_list[:3]
    applicant_elements = []

    for i, applicant in enumerate(display_applicants):
        # Truncate long applicant names
        display_name = applicant[:25] + "..." if len(applicant) > 25 else applicant

        applicant_elements.append(
            html.Span(
                display_name,
                title=applicant,  # Full name on hover
                style={
                    'fontSize': '11px', 'color': '#364761', 'display': 'block',
                    'backgroundColor': '#E6E8EB', 'padding': '2px 6px', 'borderRadius': '10px',
                    'textAlign': 'center', 'marginBottom': '2px'
                }
            )
        )

    # Add overflow indicator if there are more applicants
    if len(applicants_list) > 3:
        applicant_elements.append(
            html.Span(
                f"+ {len(applicants_list) - 3} more",
                style={
                    'fontSize': '10px', 'color': '#687589', 'fontStyle': 'italic',
                    'textAlign': 'center', 'display': 'block'
                }
            )
        )

    return applicant_elements


def _render_market_wide_patents(matching_patents, selected_topics, groups_data, active_groups):
    """Render patents in market-wide mode (existing logic)."""
    # Sort by total signal count (descending)
    matching_patents = matching_patents.sort_values('total_signal_count', ascending=False)

    # Limit to top 50 patents for performance
    top_patents = matching_patents.head(50)

    print(f"   Found {len(matching_patents)} patents mentioning ALL topics, showing top {len(top_patents)}")

    # Create patent rows using existing logic
    patent_rows = []
    for patent_idx, row in top_patents.iterrows():
        patent_row = _create_patent_row(patent_idx, row, selected_topics)
        patent_rows.append(patent_row)

    # Create header and content
    topics_selected = len(selected_topics)
    header_text = f"Patents Mentioning ALL {topics_selected} Selected Topic{'s' if topics_selected > 1 else ''}"

    header = html.Div([
        html.H4(header_text,
                style={'color': '#04193A', 'margin': '0', 'marginBottom': '10px'}),
        html.P(
            f"Found {len(matching_patents)} patents that mention all selected topics, showing top {len(top_patents)} by total mentions",
            style={'color': '#687589', 'fontSize': '12px', 'margin': '0', 'marginBottom': '10px'}),
        html.P("💡 Click on topic buttons to view the actual patent chunks",
               style={'color': '#2E86AB', 'fontSize': '11px', 'fontStyle': 'italic', 'margin': '0',
                      'marginBottom': '15px'})
    ])

    content = html.Div([
        header,
        html.Div(patent_rows, style={'maxHeight': '500px', 'overflowY': 'auto'}),

        # Chunk Modal
        html.Div(
            id='chunks-modal',
            style={'display': 'none'},
            children=[]
        )
    ], style={
        'border': '2px solid #364761', 'borderRadius': '10px', 'padding': '20px',
        'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    })

    return content


def _render_group_comparative_patents(matching_patents, selected_topics, groups_data, active_groups):
    """Render patents grouped by active groups."""
    print(f"   🏢 GROUP COMPARATIVE MODE: Organizing patents by active groups")

    # Get active groups with applicants
    active_group_ids = [gid for gid, is_active in active_groups.items()
                        if is_active and gid in groups_data and groups_data[gid].get('applicants', [])]

    if not active_group_ids:
        return html.Div([
            html.Div("👥", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Active Groups with Applicants",
                    style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Activate groups with applicants in the Portfolio Manager to see group comparisons.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={
            'border': '2px solid #364761', 'borderRadius': '10px', 'padding': '20px',
            'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'minHeight': '200px'
        })

    # Group patents by matching groups
    grouped_patents = {}
    unmatched_patents = []

    for patent_idx, patent_row in matching_patents.iterrows():
        # FIXED: Better handling of patent applicants
        patent_applicants_raw = patent_row.get('applicants')
        patent_applicants = []

        # Handle different data types safely
        try:
            if patent_applicants_raw is None:
                patent_applicants = []
            elif isinstance(patent_applicants_raw, float) and pd.isna(patent_applicants_raw):
                patent_applicants = []
            elif isinstance(patent_applicants_raw, list):
                patent_applicants = patent_applicants_raw
            elif isinstance(patent_applicants_raw, str):
                # Try to parse as JSON, fallback to comma-separated
                try:
                    patent_applicants = json.loads(patent_applicants_raw)
                except:
                    patent_applicants = [app.strip() for app in patent_applicants_raw.split(',') if app.strip()]
            elif hasattr(patent_applicants_raw, '__iter__'):
                # Handle numpy arrays or other iterables
                patent_applicants = list(patent_applicants_raw)
            else:
                patent_applicants = [str(patent_applicants_raw)]
        except Exception as e:
            print(f"   ⚠️ Error parsing applicants for {patent_row.get('lens_id', 'unknown')}: {e}")
            patent_applicants = []

        patent_matched = False

        for group_id in active_group_ids:
            group_applicants = groups_data[group_id].get('applicants', [])

            # Check if any patent applicant matches any group applicant (case-insensitive partial matching)
            if patent_applicants and group_applicants:
                for patent_app in patent_applicants:
                    for group_app in group_applicants:
                        # Use case-insensitive partial matching
                        if (isinstance(patent_app, str) and isinstance(group_app, str) and
                                (group_app.lower() in patent_app.lower() or patent_app.lower() in group_app.lower())):
                            if group_id not in grouped_patents:
                                grouped_patents[group_id] = []
                            grouped_patents[group_id].append((patent_idx, patent_row))
                            patent_matched = True
                            break
                    if patent_matched:
                        break
                if patent_matched:
                    break

        # Track unmatched patents
        if not patent_matched:
            unmatched_patents.append((patent_idx, patent_row))

    # Sort patents within each group by signal count
    for group_id in grouped_patents:
        grouped_patents[group_id].sort(key=lambda x: x[1]['total_signal_count'], reverse=True)

    # Create group sections
    group_sections = []
    total_displayed_patents = 0

    for group_id in active_group_ids:
        if group_id not in grouped_patents:
            # Show empty group with message
            group_info = groups_data[group_id]
            group_header = html.Div([
                html.Div([
                    html.Strong(f"📊 {group_info['name']}",
                                style={'color': 'white', 'fontSize': '14px', 'fontWeight': '600'}),
                    html.Span("0 patents",
                              style={'color': 'white', 'fontSize': '12px', 'marginLeft': '15px', 'opacity': '0.9'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
            ], style={
                'backgroundColor': group_info['color'],
                'padding': '12px 20px',
                'borderRadius': '8px',
                'marginBottom': '15px',
                'textAlign': 'center',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'opacity': '0.6'  # Dim empty groups
            })

            empty_message = html.Div([
                html.P("No patents found for this group's applicants in the selected topics.",
                       style={'color': '#687589', 'fontSize': '12px', 'fontStyle': 'italic',
                              'textAlign': 'center', 'margin': '10px 0'})
            ])

            group_section = html.Div([
                group_header,
                empty_message
            ], style={'marginBottom': '25px'})

            group_sections.append(group_section)
            continue

        group_info = groups_data[group_id]
        group_patents = grouped_patents[group_id][:20]  # Limit per group for performance
        total_displayed_patents += len(group_patents)

        # Create group header bar
        group_header = html.Div([
            html.Div([
                html.Strong(f"📊 {group_info['name']}",
                            style={'color': 'white', 'fontSize': '14px', 'fontWeight': '600'}),
                html.Span(
                    f"{len(grouped_patents[group_id])} patent{'s' if len(grouped_patents[group_id]) != 1 else ''}",
                    style={'color': 'white', 'fontSize': '12px', 'marginLeft': '15px', 'opacity': '0.9'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
        ], style={
            'backgroundColor': group_info['color'],
            'padding': '12px 20px',
            'borderRadius': '8px',
            'marginBottom': '15px',
            'textAlign': 'center',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })

        # Create patent rows for this group
        group_patent_rows = []
        for patent_idx, patent_row in group_patents:
            patent_row_element = _create_patent_row(patent_idx, patent_row, selected_topics)
            group_patent_rows.append(patent_row_element)

        # Add group section
        group_section = html.Div([
            group_header,
            html.Div(group_patent_rows, style={'marginBottom': '25px'})
        ])

        group_sections.append(group_section)

    # Add unmatched patents section if any exist
    if unmatched_patents:
        unmatched_patents.sort(key=lambda x: x[1]['total_signal_count'], reverse=True)
        unmatched_subset = unmatched_patents[:10]  # Show top 10 unmatched

        unmatched_header = html.Div([
            html.Div([
                html.Strong("📄 Other Patents",
                            style={'color': 'white', 'fontSize': '14px', 'fontWeight': '600'}),
                html.Span(
                    f"{len(unmatched_patents)} patent{'s' if len(unmatched_patents) != 1 else ''} (not matching active groups)",
                    style={'color': 'white', 'fontSize': '12px', 'marginLeft': '15px', 'opacity': '0.9'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
        ], style={
            'backgroundColor': '#687589',  # Gray for unmatched
            'padding': '12px 20px',
            'borderRadius': '8px',
            'marginBottom': '15px',
            'textAlign': 'center',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })

        unmatched_patent_rows = []
        for patent_idx, patent_row in unmatched_subset:
            patent_row_element = _create_patent_row(patent_idx, patent_row, selected_topics)
            unmatched_patent_rows.append(patent_row_element)

        unmatched_section = html.Div([
            unmatched_header,
            html.Div(unmatched_patent_rows)
        ])

        group_sections.append(unmatched_section)
        total_displayed_patents += len(unmatched_subset)

    # Create overall header
    topics_selected = len(selected_topics)
    header = html.Div([
        html.H4(f"Group Patent Comparison - ALL {topics_selected} Selected Topic{'s' if topics_selected > 1 else ''}",
                style={'color': '#04193A', 'margin': '0', 'marginBottom': '10px'}),
        html.P(
            f"Found {len(matching_patents)} total patents, showing {total_displayed_patents} organized by active groups",
            style={'color': '#687589', 'fontSize': '12px', 'margin': '0', 'marginBottom': '10px'}),
        html.P("💡 Click on topic buttons to view patent chunks • Patents grouped by applicant matching (partial names)",
               style={'color': '#2E86AB', 'fontSize': '11px', 'fontStyle': 'italic', 'margin': '0',
                      'marginBottom': '15px'})
    ])

    content = html.Div([
        header,
        html.Div(group_sections, style={'maxHeight': '600px', 'overflowY': 'auto'}),

        # Chunk Modal
        html.Div(
            id='chunks-modal',
            style={'display': 'none'},
            children=[]
        )
    ], style={
        'border': '2px solid #364761', 'borderRadius': '10px', 'padding': '20px',
        'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    })

    return content

def _create_patent_row(patent_idx, row, selected_topics):
    """Create a patent row element (extracted from existing logic)."""
    # Calculate topic-specific data
    topic_buttons = []
    total_chunks = 0

    for topic_id in selected_topics:
        count_col = f'topic_{topic_id}_count'
        chunks_col = f'topic_{topic_id}_chunks'

        if count_col in row and row[count_col] > 0:
            count = int(row[count_col])
            chunks = row.get(chunks_col, [])

            # Handle chunks data structure
            try:
                if chunks is not None and hasattr(chunks, '__len__'):
                    chunk_count = len(chunks)
                    chunks_list = list(chunks) if hasattr(chunks, '__iter__') else []
                else:
                    chunk_count = 0
                    chunks_list = []
            except:
                chunk_count = 0
                chunks_list = []

            total_chunks += chunk_count

            # Get topic label
            topic_row = df[df['id'] == topic_id]
            topic_label = topic_row.iloc[0].get('topic_label',
                                                f'Topic {topic_id}') if not topic_row.empty else f'Topic {topic_id}'
            short_label = topic_label[:25] + "..." if len(topic_label) > 25 else topic_label

            # Create clickable topic button
            topic_button = html.Button(
                f"{short_label} ({count})",
                id={'type': 'topic-chunk-btn', 'patent_idx': patent_idx, 'topic_id': topic_id},
                n_clicks=0,
                style={
                    'backgroundColor': '#2E86AB', 'color': 'white', 'border': 'none',
                    'padding': '4px 8px', 'borderRadius': '12px', 'fontSize': '11px',
                    'cursor': 'pointer', 'marginRight': '6px', 'marginBottom': '4px',
                    'transition': 'all 0.2s ease', 'fontWeight': '600'
                },
                title=f"Click to view {chunk_count} chunks for {topic_label}"
            )
            topic_buttons.append(topic_button)

    # Create patent row
    patent_row = html.Div([
        # Patent Title and Lens ID
        html.Div([
            html.A(
                row.get('patent_title', 'No Title Available')[:100] + (
                    "..." if len(str(row.get('patent_title', ''))) > 100 else ''),
                href=f"https://www.lens.org/lens/patent/{row['lens_id']}",
                target="_blank",
                style={
                    'color': '#04193A', 'fontSize': '14px', 'display': 'block', 'marginBottom': '6px',
                    'textDecoration': 'none', 'fontWeight': '600',
                    'transition': 'color 0.2s ease'
                },
                className='patent-title-link'
            ),
            html.Div([
                html.Span(f"Lens ID: {row['lens_id']}",
                          style={'color': '#687589', 'fontSize': '12px', 'marginRight': '15px'}),
                html.Span(f"Published: {row.get('date_published', 'N/A')}",
                          style={'color': '#687589', 'fontSize': '12px', 'marginRight': '15px'}),
                html.Span(f"Jurisdiction: {row.get('jurisdiction', 'N/A')}",
                          style={'color': '#687589', 'fontSize': '12px'})
            ])
        ], style={'flex': '1', 'marginRight': '15px'}),

        # Signal Count
        html.Div([
            html.Div(
                str(int(row['total_signal_count'])),
                style={
                    'backgroundColor': '#F18F01', 'color': 'white', 'padding': '8px 12px',
                    'borderRadius': '20px', 'fontSize': '14px', 'fontWeight': '600',
                    'textAlign': 'center', 'minWidth': '40px'
                }
            ),
            html.Small("mentions",
                       style={'color': '#687589', 'fontSize': '10px', 'textAlign': 'center', 'display': 'block',
                              'marginTop': '2px'})
        ], style={'marginRight': '15px', 'textAlign': 'center'}),

        # Clickable Topic Chunks
        html.Div([
            html.Div(f"{total_chunks} chunks",
                     style={'fontSize': '12px', 'fontWeight': '600', 'color': '#04193A', 'marginBottom': '6px'}),
            html.Div(topic_buttons, style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style={'flex': '1', 'marginRight': '15px'}),

        # Applicants Section
        html.Div([
            html.Div("Applicants", style={'fontSize': '12px', 'fontWeight': '600', 'color': '#04193A', 'marginBottom': '6px'}),
            html.Div(
                children=_get_patent_applicants_display(row),
                style={'display': 'flex', 'flexDirection': 'column', 'gap': '3px'}
            )
        ], style={'flex': '0 0 200px'})

    ], style={
        'display': 'flex', 'alignItems': 'flex-start', 'padding': '15px',
        'border': '1px solid #E6E8EB', 'borderRadius': '8px', 'marginBottom': '10px',
        'backgroundColor': 'white', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
        'transition': 'all 0.2s ease'
    })

    return patent_row



# FIXED: Document Explorer callback with proper loading logic
@app.callback(
    [Output('document-explorer-content', 'children'),
     Output('analysis-complete', 'data')],
    [Input('debounce-trigger', 'data'),
     Input('active-groups', 'data'),
     Input('applicant-groups', 'data')],
    [State('selected-bubbles', 'data'),
     State('selection-pending', 'data'),
     State('analysis-complete', 'data')],
    prevent_initial_call=False
)
def update_document_explorer_with_completion(debounce_trigger, active_groups, groups_data,
                                             selected_topics, is_pending, current_complete):
    """Document explorer that signals when analysis is complete."""

    # FIXED: Show loading if pending OR if we're waiting for initial debounce trigger
    should_show_loading = is_pending or (selected_topics and debounce_trigger == 0)

    if should_show_loading:
        return create_loading_spinner('documents', 'Searching Patent Database...'), no_update

    # Your existing logic here...
    if not selected_topics or document_dist_df.empty:
        # Return ready state + increment completion counter
        ready_state = html.Div([
            html.Div([
                html.Div("📄",
                         style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
                html.H5("Document Explorer Ready",
                        style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
                html.P("Select topics from the visualization above to explore related patent documents.",
                       style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '200px'})
        ], style={
            'border': '2px solid #364761', 'borderRadius': '10px', 'padding': '20px',
            'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'minHeight': '200px'
        })
        return ready_state, current_complete + 1

    print(f"\n📄 Document Explorer: Analysis starting for topics: {selected_topics}")

    # Get relevant topic count columns for selected topics
    selected_count_cols = [f'topic_{topic_id}_count' for topic_id in selected_topics
                           if f'topic_{topic_id}_count' in topic_count_cols]

    if not selected_count_cols:
        no_data_content = html.Div([
            html.Div("⚠️", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Data Available", style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("No document data available for the selected topics.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={
            'border': '2px solid #364761', 'borderRadius': '10px', 'padding': '20px',
            'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'minHeight': '200px'
        })
        return no_data_content, current_complete + 1  # Signal completion

    # Find patents that mention ALL of the selected topics (AND logic)
    mask = pd.Series([True] * len(document_dist_df))
    for col in selected_count_cols:
        mask = mask & (document_dist_df[col] > 0)

    matching_patents = document_dist_df[mask].copy()

    if matching_patents.empty:
        topic_labels = []
        for topic_id in selected_topics:
            topic_row = df[df['id'] == topic_id]
            if not topic_row.empty:
                label = topic_row.iloc[0].get('topic_label', f'Topic {topic_id}')
                topic_labels.append(label[:30] + "..." if len(label) > 30 else label)

        topics_text = ", ".join(topic_labels)

        no_patents_content = html.Div([
            html.Div("🔍", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Patents Found", style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P(f"No patents found that mention ALL of these topics together:",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4',
                          'marginBottom': '8px'}),
            html.P(topics_text,
                   style={'color': '#04193A', 'textAlign': 'center', 'fontWeight': '600', 'fontSize': '12px',
                          'marginBottom': '8px'}),
            html.P("Try selecting fewer topics or different topic combinations.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'fontSize': '11px'})
        ], style={
            'border': '2px solid #364761', 'borderRadius': '10px', 'padding': '20px',
            'backgroundColor': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'minHeight': '200px'
        })
        return no_patents_content, current_complete + 1  # Signal completion

    # Calculate total signal count for each patent across all selected topics
    matching_patents['total_signal_count'] = matching_patents[selected_count_cols].sum(axis=1)

    # Check if we should use group comparative mode
    use_market_mode = should_use_market_mode(groups_data, active_groups)

    gc.collect()

    if use_market_mode:
        # MARKET-WIDE MODE: Show all patents (existing logic)
        content = _render_market_wide_patents(matching_patents, selected_topics, groups_data, active_groups)
        return content, current_complete + 1  # Signal completion
    else:
        # GROUP COMPARATIVE MODE: Group patents by active groups
        content = _render_group_comparative_patents(matching_patents, selected_topics, groups_data, active_groups)
        return content, current_complete + 1  # Signal completion


# NEW: Chunk Modal Callback - FIXED
@app.callback(
    [Output('chunks-modal', 'children'),
     Output('chunks-modal', 'style')],
    [Input({'type': 'topic-chunk-btn', 'patent_idx': ALL, 'topic_id': ALL}, 'n_clicks'),
     Input({'type': 'close-chunks-modal', 'modal_id': ALL}, 'n_clicks')],  # FIXED: Use pattern-matching
    [State('chunks-modal', 'style')],
    prevent_initial_call=True
)
def handle_chunk_modal(topic_btn_clicks, close_clicks, current_style):
    """Handle opening and closing the chunks modal."""

    ctx = callback_context
    if not ctx.triggered:
        return [], {'display': 'none'}

    trigger_id = ctx.triggered[0]['prop_id']

    # Close modal - FIXED: Check for pattern-matching close button
    if 'close-chunks-modal' in trigger_id and any(close_clicks):
        return [], {'display': 'none'}

    # Open modal for topic chunks
    if 'topic-chunk-btn' in trigger_id and any(topic_btn_clicks):
        # Find which button was clicked
        button_data = None
        for i, clicks in enumerate(topic_btn_clicks):
            if clicks and clicks > 0:
                # Parse the trigger to get the button data
                trigger_data = json.loads(trigger_id.split('.')[0])
                patent_idx = trigger_data['patent_idx']
                topic_id = trigger_data['topic_id']
                button_data = (patent_idx, topic_id)
                break

        if button_data is None:
            return [], {'display': 'none'}

        patent_idx, topic_id = button_data

        # Get the patent data
        try:
            patent_row = document_dist_df.iloc[patent_idx]
            chunks_col = f'topic_{topic_id}_chunks'
            chunks = patent_row.get(chunks_col, [])

            # Handle chunks data structure
            try:
                if chunks is not None and hasattr(chunks, '__len__'):
                    chunks_list = list(chunks)
                else:
                    chunks_list = []
            except:
                chunks_list = []

            # Get topic info
            topic_row = df[df['id'] == topic_id]
            topic_label = topic_row.iloc[0].get('topic_label', f'Topic {topic_id}') if not topic_row.empty else f'Topic {topic_id}'

            # Create modal content
            if not chunks_list:
                modal_content = [
                    html.Div([
                        html.H4(f"No Chunks Found", style={'color': '#04193A', 'marginBottom': '10px'}),
                        html.P("No text chunks found for this topic in this patent.")
                    ])
                ]
            else:
                # Create chunk displays
                chunk_displays = []
                for i, chunk_data in enumerate(chunks_list):
                    if isinstance(chunk_data, dict):
                        chunk_text = chunk_data.get('text', 'No text available')
                        section = chunk_data.get('section', 'Unknown')
                    else:
                        chunk_text = str(chunk_data)
                        section = 'Unknown'

                    # Truncate very long chunks
                    display_text = chunk_text[:1000] + "..." if len(chunk_text) > 1000 else chunk_text

                    chunk_display = html.Div([
                        html.Div([
                            html.Span(f"Chunk {i+1}", style={'fontWeight': '600', 'color': '#04193A'}),
                            html.Span(f" • {section.title()}", style={'color': '#687589', 'fontSize': '12px', 'marginLeft': '8px'})
                        ], style={'marginBottom': '8px'}),
                        html.Div(
                            display_text,
                            style={
                                'backgroundColor': '#F8F9FA', 'padding': '12px', 'borderRadius': '6px',
                                'border': '1px solid #E6E8EB', 'lineHeight': '1.5', 'fontSize': '13px',
                                'maxHeight': '200px', 'overflowY': 'auto'
                            }
                        )
                    ], style={'marginBottom': '15px'})

                    chunk_displays.append(chunk_display)

                modal_content = [
                    html.Div([
                        html.H4(f"{topic_label}", style={'color': '#04193A', 'marginBottom': '5px'}),
                        html.P(f"Patent: {patent_row.get('patent_title', 'No Title')[:80]}...",
                               style={'color': '#687589', 'fontSize': '12px', 'marginBottom': '15px'}),
                        html.P(f"Found {len(chunks_list)} text chunk{'s' if len(chunks_list) != 1 else ''} mentioning this topic:",
                               style={'color': '#364761', 'fontSize': '13px', 'marginBottom': '20px'}),
                        html.Div(chunk_displays, style={'maxHeight': '400px', 'overflowY': 'auto'})
                    ])
                ]

            # Modal wrapper - FIXED: Use pattern-matching for close button
            modal = html.Div([
                # Modal backdrop
                html.Div(
                    style={
                        'position': 'fixed', 'top': '0', 'left': '0', 'width': '100vw', 'height': '100vh',
                        'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '2000'
                    }
                ),
                # Modal content
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("📄 Patent Chunks", style={'color': '#04193A', 'margin': '0', 'flex': '1'}),
                            html.Button("✕",
                                       id={'type': 'close-chunks-modal', 'modal_id': f'modal_{patent_idx}_{topic_id}'},
                                       n_clicks=0,
                                       style={'backgroundColor': 'transparent', 'border': 'none',
                                              'fontSize': '20px', 'cursor': 'pointer', 'color': '#687589'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px',
                                  'paddingBottom': '15px', 'borderBottom': '2px solid #E6E8EB'}),

                        *modal_content

                    ], style={'padding': '30px'})
                ], style={
                    'position': 'fixed', 'top': '50%', 'left': '50%', 'transform': 'translate(-50%, -50%)',
                    'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 10px 30px rgba(0,0,0,0.3)',
                    'maxWidth': '80vw', 'maxHeight': '80vh', 'overflowY': 'auto', 'zIndex': '2001'
                })
            ])

            return modal, {'display': 'block'}

        except Exception as e:
            print(f"Error creating chunk modal: {e}")
            return [], {'display': 'none'}

    return [], {'display': 'none'}

@app.callback(
    [Output('view-mode', 'data'),
     Output('view-mode-text', 'children')],
    Input('view-mode-toggle-btn', 'n_clicks'),
    State('view-mode', 'data'),
    prevent_initial_call=True
)
def toggle_view_mode(n_clicks, current_mode):
    """Toggle between portfolio focus and market share views."""
    new_mode = 'share' if current_mode == 'focus' else 'focus'

    if new_mode == 'focus':
        button_text = 'Portfolio Focus'
    else:
        button_text = 'Patent Share'

    print(f"🔄 View mode toggled to: {new_mode}")
    return new_mode, button_text

# ============================================================================
# SINGLE DEFINITION: VIEW MODE TOGGLE CALLBACK (NO DUPLICATES)
# ============================================================================

@app.callback(
    [Output('view-mode-series', 'data'),
     Output('view-toggle-switch', 'style'),
     Output('view-toggle-circle', 'style'),
     Output('view-toggle-switch', 'title')],
    [Input('view-toggle-switch', 'n_clicks'),
     Input('active-groups', 'data'),
     Input('applicant-groups', 'data')],
    [State('view-mode-series', 'data')],
    prevent_initial_call=False
)
def toggle_view_mode_series(n_clicks, active_groups, groups_data, current_mode):
    """Toggle between groups and topics view for time series with conditional disable."""

    # Get the trigger context to see what caused the callback
    ctx = callback_context

    print(f"🔍 toggle_view_mode_series FIRED - trigger: {ctx.triggered}")

    # Check if we have active groups with applicants
    has_active_groups_with_applicants = False

    if groups_data and active_groups:
        for group_id, is_active in active_groups.items():
            if is_active and group_id in groups_data:
                if groups_data[group_id].get('applicants', []):
                    has_active_groups_with_applicants = True
                    break

    # If no active groups with applicants, force topics mode and disable toggle
    if not has_active_groups_with_applicants:
        print("🚫 No active groups with applicants - forcing Topics mode and disabling toggle")

        # Force topics mode
        forced_mode = 'topics'

        # Disabled switch appearance (greyed out)
        switch_style = {
            'position': 'relative', 'width': '40px', 'height': '20px',
            'backgroundColor': '#CDD1D8',  # Grey background when disabled
            'borderRadius': '20px', 'cursor': 'not-allowed',  # Change cursor
            'transition': 'background-color 0.3s ease', 'display': 'inline-block',
            'opacity': '0.6'  # Make it look disabled
        }

        # Circle positioned for topics (right side) and greyed
        circle_style = {
            'position': 'absolute', 'top': '2px', 'left': '22px',  # Right position for topics
            'width': '16px', 'height': '16px', 'backgroundColor': '#9BA3B0',  # Grey circle
            'borderRadius': '50%', 'transition': 'left 0.3s ease',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'
        }

        # Helpful tooltip explanation
        tooltip_text = "Add organizations to active groups in the portfolio manager to compare groups"

        # FIXED: Return early - don't allow any toggling when disabled
        return forced_mode, switch_style, circle_style, tooltip_text

    # ONLY EXECUTE TOGGLE LOGIC WHEN GROUPS ARE AVAILABLE
    # Only allow toggling if the click came from the switch (not from groups/applicants changing)
    if ctx.triggered and 'view-toggle-switch' in ctx.triggered[0]['prop_id']:
        # Normal toggle operation when active groups with applicants exist
        new_mode = 'topics' if current_mode == 'groups' else 'groups'
        print(f"🔄 Time series view mode toggled to: {new_mode}")
    else:
        # Groups/applicants changed but switch wasn't clicked - keep current mode
        new_mode = current_mode or 'groups'  # Default to groups if current_mode is None
        print(f"🔄 Groups/applicants changed - keeping current mode: {new_mode}")

    # Update switch appearance based on mode (normal enabled state)
    if new_mode == 'groups':
        switch_style = {
            'position': 'relative', 'width': '40px', 'height': '20px',
            'backgroundColor': '#2E86AB', 'borderRadius': '20px', 'cursor': 'pointer',
            'transition': 'background-color 0.3s ease', 'display': 'inline-block'
        }
        circle_style = {
            'position': 'absolute', 'top': '2px', 'left': '2px',
            'width': '16px', 'height': '16px', 'backgroundColor': 'white',
            'borderRadius': '50%', 'transition': 'left 0.3s ease',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.3)'
        }
        tooltip_text = "Switch to view individual topics over time"
    else:  # topics
        switch_style = {
            'position': 'relative', 'width': '40px', 'height': '20px',
            'backgroundColor': '#F18F01', 'borderRadius': '20px', 'cursor': 'pointer',
            'transition': 'background-color 0.3s ease', 'display': 'inline-block'
        }
        circle_style = {
            'position': 'absolute', 'top': '2px', 'left': '22px',
            'width': '16px', 'height': '16px', 'backgroundColor': 'white',
            'borderRadius': '50%', 'transition': 'left 0.3s ease',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.3)'
        }
        tooltip_text = "Switch to view portfolio groups over time"

    return new_mode, switch_style, circle_style, tooltip_text


# UPDATED: Portfolio analysis title based on mode using professional generator
@app.callback(
    Output('portfolio-analysis-title', 'children'),
    [Input('view-mode', 'data'),
     Input('active-groups', 'data'),
     Input('applicant-groups', 'data')]
)
def update_portfolio_title(view_mode, active_groups, groups_data):
    """Update the portfolio analysis box title based on current mode."""
    use_market_mode = should_use_market_mode(groups_data, active_groups)
    return ChartTitleGenerator.get_portfolio_analysis_title(view_mode, use_market_mode)




# UPDATED: Time series data calculation callback
@app.callback(
    Output('time-series-data', 'data'),
    [Input('active-groups', 'data'),
     Input('debounce-trigger', 'data'),  # CHANGED: Use debounce trigger
     Input('view-mode', 'data'),
     Input('time-granularity-dropdown', 'value'),
     Input('view-mode-series', 'data')],
    [State('selected-bubbles', 'data'),  # CHANGED: Get from state
     State('applicant-groups', 'data')],
    prevent_initial_call=False
)
def calculate_time_series_data(active_groups, debounce_trigger, view_mode, time_granularity, view_mode_series,
                               selected_topics, groups_data):
    """Calculate and store time series data with forced topics mode when no active groups."""
    if not selected_topics or not groups_data:
        return {}

    # Check if we should force topics mode due to lack of active groups with applicants
    has_active_groups_with_applicants = False
    if groups_data and active_groups:
        for group_id, is_active in active_groups.items():
            if is_active and group_id in groups_data:
                if groups_data[group_id].get('applicants', []):
                    has_active_groups_with_applicants = True
                    break

    # Force topics mode if no active groups with applicants
    effective_view_mode_series = view_mode_series
    if not has_active_groups_with_applicants and view_mode_series == 'groups':
        effective_view_mode_series = 'topics'
        print("🔄 Forcing topics mode for time series calculation - no active groups with applicants")

    # Calculate time series focus data with effective view mode
    time_series_data = calculate_time_series_focus_data(
        groups_data, active_groups, selected_topics, view_mode, time_granularity, effective_view_mode_series
    )
    gc.collect()

    return time_series_data


# FIXED: Time series chart callback with proper loading logic
@app.callback(
    [Output('time-series-content', 'children'),
     Output('analysis-complete', 'data', allow_duplicate=True)],
    [Input('time-series-data', 'data'),
     Input('view-mode', 'data'),
     Input('time-granularity-dropdown', 'value'),
     Input('view-mode-series', 'data')],
    [State('selected-bubbles', 'data'),
     State('active-groups', 'data'),
     State('applicant-groups', 'data'),
     State('selection-pending', 'data'),
     State('analysis-complete', 'data')],
    prevent_initial_call=True
)
def update_time_series_chart_debounced(time_series_data, view_mode, time_granularity, view_mode_series,
                                       selected_topics, active_groups, groups_data, is_pending, current_complete):
    """Generate time series line chart with debounced loading."""

    # FIXED: Show loading if pending OR if waiting for initial data
    if is_pending:
        return create_loading_spinner('timeseries', 'Processing Time Series Data...'), no_update

    # Check if we should use market mode
    use_market_mode = should_use_market_mode(groups_data, active_groups)

    # Validate requirements
    active_group_ids = [gid for gid, is_active in (active_groups or {}).items() if is_active]
    selected_count = len(selected_topics or [])

    # FIXED: Generate professional labels using the new system
    labels = ChartLabelsGenerator.get_time_series_labels(
        view_mode, view_mode_series, use_market_mode, time_granularity
    )

    # FIXED: Generate professional chart title and subtitle
    title_info = ChartTitleGenerator.get_time_series_title(
        view_mode, time_granularity, view_mode_series, use_market_mode, selected_count
    )

    chart_subtitle = ChartTitleGenerator.get_time_series_subtitle(
        view_mode, view_mode_series, use_market_mode, selected_count
    )

    # Check if we have topics
    if selected_count == 0:
        html_content = html.Div([
            html.Div("🎯", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Topics Selected", style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Select topics from the visualization above to see time trends.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '300px'})
        return html_content, current_complete + 1  # Signal completion

    if view_mode == 'focus' and selected_count < 1:
        remaining = 2 - selected_count
        html_content = html.Div([
            html.Div("📊", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("Need More Topics for Composition Analysis",
                    style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P(
                f"Select {remaining} more topic{'s' if remaining > 1 else ''} to analyze relative importance over time. "
                f"({selected_count}/2 minimum required for composition trends)",
                style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'}),
            html.Hr(style={'width': '60%', 'margin': '15px auto', 'border': '1px solid #E6E8EB'}),
            html.P("💡 Tip: Switch to Market Share mode to analyze single topics over time.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontSize': '12px', 'fontStyle': 'italic'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '300px'})
        return html_content, current_complete + 1  # Signal completion

    # Check if we have groups when not in market mode
    if not use_market_mode and not active_group_ids:
        html_content = html.Div([
            html.Div("📈", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Active Portfolio Groups",
                    style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Activate at least one portfolio group to see time series analysis.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '300px'})
        return html_content, current_complete + 1  # Signal completion

    # Check if we have time series data
    if not time_series_data:
        html_content = html.Div([
            html.Div("⏳", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Time Series Data", style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("No patent data available for the selected analysis.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '300px'})
        return html_content, current_complete + 1  # Signal completion

    # Create the line chart
    try:
        fig = go.Figure()

        max_value = 0
        total_traces = 0

        # Add traces for each item (group or topic)
        for item_id, item_data in time_series_data.items():
            periods = item_data['periods']
            values = item_data['values']
            raw_counts = item_data['raw_counts']
            item_name = item_data['name']
            item_color = item_data['color']
            is_market_mode_item = item_data.get('is_market_mode', False)

            # TODO: INCLUDE EMPTY VALUES DECIDE YES/NO
            # # Skip if no data
            # if not periods or max(values) == 0:
            #     continue

            # NEW: Build line style with both color and dash pattern
            line_style = dict(color=item_color, width=3)

            if is_market_mode_item:
                line_style['dash'] = 'solid'  # Solid line for market mode
            elif 'line_style' in item_data:
                # Topics view: Use group line style
                line_style['dash'] = item_data['line_style']
            else:
                # Groups view: Keep solid lines
                line_style['dash'] = 'solid'

            # FIXED: Build hover template with correct units using professional labels
            if labels['hover_unit'] == "":
                # Raw counts - no percentage sign
                hover_format = f'{labels["hover_label"]}: %{{y}}'
            else:
                # Percentages
                hover_format = f'{labels["hover_label"]}: %{{y:.1f}}{labels["hover_unit"]}'

            fig.add_trace(go.Scatter(
                x=periods,
                y=values,
                mode='lines+markers',
                name=item_name,
                line=line_style,
                marker=dict(color=item_color, size=6),
                hovertemplate=f'<b>{item_name}</b><br>' +
                              f'Period: %{{x}}<br>' +
                              # f'{hover_format}<br>' +
                              f'Patents: %{{customdata}}<br>' +
                              # ('<i>(Market Analysis)</i><br>' if is_market_mode_item else '') +
                              '<extra></extra>',
                customdata=raw_counts
            ))

            max_value = max(max_value, max(values))
            total_traces += 1

        if total_traces == 0:
            html_content = html.Div([
                html.Div("📊",
                         style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
                html.H5("No Data for Selected Period",
                        style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
                html.P("No patents found for the selected analysis during this time period.",
                       style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '300px'})
            return html_content, current_complete + 1  # Signal completion

        # Configure layout with professional title
        fig.update_layout(
            title=dict(
                text=f"{title_info['title']}<br><span style='font-size:12px; color:#687589'>{title_info['subtitle']}</span>",
                x=0.5,
                font=dict(size=14, color='#04193A')
            ),
            xaxis=dict(
                title=f"{time_granularity.capitalize()} Period",
                showgrid=True,
                gridcolor='rgba(104, 117, 137, 0.2)',
                tickfont=dict(size=10, color='#687589')
            ),
            yaxis=dict(
                title=labels['y_axis_label'],  # FIXED: Now uses professional label generator
                showgrid=True,
                gridcolor='rgba(104, 117, 137, 0.2)',
                tickfont=dict(size=10, color='#687589'),
                range=[0, max_value * 1.1]
            ),
            showlegend=True,  # Always show legend, even for single traces
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=11, color='#04193A')
            ),
            margin=dict(l=50, r=120, t=50, b=50),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        # FIXED: Generate professional summary using label generator
        if labels['hover_unit'] == "":
            max_value_text = f"{max_value:.0f} patents max"
        else:
            max_value_text = f"{max_value:.1f}% max"

        summary_text = (
            f"{labels['mode_indicator']} • {labels['granularity_indicator']} {time_granularity.capitalize()} • "
            f"{labels['view_indicator']} • {chart_subtitle} • "
            f"{total_traces} trend line{'s' if total_traces > 1 else ''} • "
            f"{max_value_text}")

        # FIXED: Generate professional caption using new system
        try:
            caption_text = generate_time_series_caption(
                time_series_data, selected_topics, view_mode, time_granularity,
                view_mode_series, use_market_mode, active_groups, groups_data, df
            )

            caption_div = html.Div([
                html.Div([
                    html.Strong("Chart Explanation: ", style={'color': '#04193A', 'fontSize': '12px'}),
                    html.Span(
                        caption_text,
                        style={'color': '#364761', 'fontSize': '12px', 'lineHeight': '1.4'}
                    )
                ], style={
                    'backgroundColor': '#F8F9FA', 'padding': '12px', 'borderRadius': '6px',
                    'border': '1px solid #E6E8EB', 'marginTop': '15px'
                })
            ])
        except Exception as e:
            print(f"❌ Time series caption generation failed: {e}")
            # Fallback
            caption_div = html.Div([
                html.Div([
                    html.Strong("Chart Explanation: ", style={'color': '#04193A', 'fontSize': '12px'}),
                    html.Span(
                        "This time series chart shows trends over time for selected topics.",
                        style={'color': '#364761', 'fontSize': '12px', 'lineHeight': '1.4'}
                    )
                ], style={
                    'backgroundColor': '#F8F9FA', 'padding': '12px', 'borderRadius': '6px',
                    'border': '1px solid #E6E8EB', 'marginTop': '15px'
                })
            ])

        html_content =  html.Div([
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False}
            ),
            html.Div([
                html.Small(summary_text,
                           style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic'})
            ], style={'marginTop': '10px'}),
            caption_div
        ])
        return html_content, current_complete + 1  # Signal completion

    except Exception as e:
        print(f"Error generating time series chart: {e}")
        html_content =  html.Div([
            html.Div("⚠️", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("Chart Generation Error",
                    style={'color': '#C73E1D', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Unable to generate time series chart. Please check your data and try again.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '300px'})
        return html_content, current_complete + 1  # Signal completion


# Active groups display (unchanged)
@app.callback(
    Output("active-groups-display", "children"),
    Input("active-groups", "data"),
    State("applicant-groups", "data"),
)
def render_active_chips(active_groups, groups_data):
    if not active_groups:
        return no_update

    chips = []
    for gid, is_on in active_groups.items():
        if is_on and gid in groups_data:
            info = groups_data[gid]
            chips.append(
                html.Span(
                    info["name"],
                    style={
                        "display": "inline-block",
                        "backgroundColor": info["color"],
                        "color": "white",
                        "padding": "4px 12px",
                        "borderRadius": "16px",
                        "fontSize": "12px",
                        "fontWeight": "600",
                    },
                )
            )

    if chips:
        return chips

    return html.Span("No active groups", style={"color": "#687589", "fontStyle": "italic"})


# Selected topics display (unchanged)
@app.callback(
    [Output("selected-topics-display", "children"),
     Output("selection-limit-info", "children")],
    Input("selected-bubbles", "data"),
)
def render_selected_topics(selected_ids):
    """Render chips for selected topics and show selection limit info."""
    if not selected_ids:
        return (
            html.Span("No topics selected", style={"color": "#687589", "fontStyle": "italic"}),
            "Click up to 5 topic bubbles to analyze"
        )

    selected_data = df[df['id'].isin(selected_ids)]

    chips = []
    for _, row in selected_data.iterrows():
        topic_id = row['id']
        topic_label = row.get('topic_label', f"Topic {topic_id}")
        display_label = topic_label[:25] + "..." if len(topic_label) > 25 else topic_label

        chips.append(
            html.Span(
                display_label,
                style={
                    "display": "inline-block",
                    "backgroundColor": "#364761",
                    "color": "white",
                    "padding": "4px 10px",
                    "borderRadius": "12px",
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "margin": "2px",
                    "border": "1px solid #2E86AB"
                },
                title=f"Topic {topic_id}: {topic_label}"
            )
        )

    limit_text = f"{len(selected_ids)}/5 topics selected"
    if len(selected_ids) >= 5:
        limit_text += " (limit reached)"
        limit_style = {"color": "#C73E1D", "fontWeight": "600"}
    else:
        limit_style = {"color": "#687589"}

    limit_info = html.Span(limit_text, style=limit_style)
    return chips, limit_info


# FIXED: Portfolio radar chart with proper loading logic
@app.callback(
    [Output('portfolio-radar-content', 'children'),
     Output('analysis-complete', 'data', allow_duplicate=True)],
    [Input('debounce-trigger', 'data'),
     Input('active-groups', 'data'),
     Input('group-focus-data', 'data'),
     Input('applicant-groups', 'data'),
     Input('view-mode', 'data')],
    [State('selected-bubbles', 'data'),
     State('selection-pending', 'data'),
     State('analysis-complete', 'data')],
    prevent_initial_call=True
)
def update_portfolio_radar_debounced(debounce_trigger, active_groups, focus_data, groups_data, view_mode,
                                     selected_topics, is_pending, current_complete):
    """Generate radar chart with debounced loading states."""

    # FIXED: Show loading if pending OR if waiting for initial data
    if is_pending or (selected_topics and debounce_trigger == 0):
        return create_loading_spinner('radar', 'Calculating Portfolio Focus...'), no_update

    # Check if we should use market mode
    use_market_mode = should_use_market_mode(groups_data, active_groups)

    # Initialize variables for entities to show
    entities_to_show = {}
    active_entity_ids = []

    # If using market mode, use market focus data
    if use_market_mode and selected_topics:
        print("🌍 Radar chart using market-wide mode")
        # Recalculate focus data for market mode
        focus_data = calculate_market_focus_data(selected_topics, applicant_dist, df, view_mode)
        # Create a single market entity
        entities_to_show = {
            'total_market': {
                'name': 'Total Market',
                'color': '#2E86AB',
                'is_market_mode': True
            }
        }
        active_entity_ids = ['total_market']
    else:
        # Use regular portfolio groups
        active_entity_ids = [gid for gid, is_active in (active_groups or {}).items() if is_active]
        entities_to_show = {gid: groups_data[gid] for gid in active_entity_ids if gid in groups_data}

    selected_count = len(selected_topics or [])

    # Determine chart title and labels based on view mode
    if view_mode == 'focus':
        if use_market_mode:
            hover_label = "Global Focus"
            base_title = "Focus Analysis"
        else:
            hover_label = "Focus"
            base_title = "Focus Analysis"
    else:  # share
        if use_market_mode:
            hover_label = "Composition"
            base_title = "Composition Analysis"
        else:
            hover_label = "Market Share"
            base_title = "Market Share Analysis"

    # Update title for market mode
    if use_market_mode:
        if view_mode == 'focus':
            chart_title = f"Global Innovation Focus"
        else:
            chart_title = f"Technology Area Composition"
    else:
        chart_title = f"Portfolio {base_title}"

    # Check if we have entities to show (groups or market mode)
    if not active_entity_ids:
        no_groups_content = html.Div([
            html.Div("📊",
                     style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("No Active Portfolio Groups",
                    style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Activate at least one portfolio group from the Portfolio Manager to see analysis.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '350px'})
        return no_groups_content, current_complete + 1

    # Check if we have enough topics
    if selected_count < 3:
        remaining = 3 - selected_count
        if use_market_mode:
            message = f"Select {remaining} more topic{'s' if remaining > 1 else ''} to see market analysis. ({selected_count}/3 minimum required)"
        else:
            message = f"Select {remaining} more topic{'s' if remaining > 1 else ''} to generate radar analysis. ({selected_count}/3 minimum required)"

        need_topics_content = html.Div([
            html.Div("🎯", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("Need More Topics", style={'color': '#687589', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P(message,
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic', 'lineHeight': '1.4'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '350px'})
        return need_topics_content, current_complete + 1

    # Generate radar chart
    try:
        # Get topic labels for selected topics
        topic_labels = []
        for topic_id in selected_topics:
            topic_row = df[df['id'] == topic_id]
            if not topic_row.empty:
                label = topic_row.iloc[0].get('topic_label', f'Topic {topic_id}')
                display_label = label[:35] + "..." if len(label) > 35 else label
                topic_labels.append(display_label)

        # Calculate angles for even distribution
        import math
        n_topics = len(topic_labels)
        angles = [i * 360 / n_topics for i in range(n_topics)]

        # Create radar chart traces
        traces = []

        for entity_id, entity_info in entities_to_show.items():
            entity_name = entity_info['name']
            entity_color = entity_info['color']

            # Get focus percentages for this entity on selected topics
            focus_values = []

            for topic_id in selected_topics:
                topic_focus = 0.0
                # Handle both string and integer keys in focus_data
                topic_key = topic_id if topic_id in focus_data else str(topic_id)

                if focus_data and topic_key in focus_data:
                    for ring in focus_data[topic_key]:
                        if ring['group_id'] == entity_id:
                            topic_focus = ring['percentage']
                            break

                focus_values.append(topic_focus)

            # Close the polygon by adding first value at end
            radar_values = focus_values + [focus_values[0]]
            radar_labels = topic_labels + [topic_labels[0]]
            radar_angles = angles + [angles[0]]

            # Create trace for this entity
            trace = go.Scatterpolar(
                r=radar_values,
                theta=radar_angles,
                mode='lines+markers',
                name=entity_name,
                line=dict(color=entity_color, width=3),
                marker=dict(color=entity_color, size=6),
                hovertemplate=f'<b>{entity_name}</b><br>' +
                              '%{theta}<br>' +
                              f'{hover_label}: %{{r:.1f}}%<br>' +
                              '<extra></extra>',
                fill=None
            )
            traces.append(trace)

        # Create layout
        max_value = max([max(trace.r[:-1]) for trace in traces] + [10]) if traces else 10

        # FIXED: Better max value scaling
        if view_mode == 'share':
            # For market share, ensure we don't compress too much empty space
            max_value = min(max_value * 1.1 + 5, 100)  # Always leave ~5% headroom
        else:
            # For focus mode, standard scaling
            max_value = max_value * 1.2

        layout = go.Layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value],
                    tickfont=dict(size=10, color='#687589'),
                    gridcolor='rgba(104, 117, 137, 0.2)',
                    linecolor='rgba(104, 117, 137, 0.3)'
                ),
                angularaxis=dict(
                    tickvals=angles,
                    ticktext=topic_labels,
                    tickfont=dict(size=11, color='#04193A'),
                    linecolor='rgba(104, 117, 137, 0.3)',
                    gridcolor='rgba(104, 117, 137, 0.2)'
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color='#04193A')
            ),
            margin=dict(l=20, r=20, t=20, b=80),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)

        gc.collect()

        # Return chart with summary
        max_focus = max([max(trace.r[:-1]) for trace in traces]) if traces else 0

        if use_market_mode:
            if view_mode == 'focus':
                mode_indicator = "🎯 Global Innovation Focus"
                entity_indicator = "🌍 Market Analysis"
            else:
                mode_indicator = "🏆 Technology Area Composition"
                entity_indicator = "🌍 Selected Topics"
            entity_count = 1
        else:
            mode_indicator = "🎯 Portfolio Focus" if view_mode == 'focus' else "🏆 Market Share"
            entity_indicator = "👥 Portfolio Groups"
            entity_count = len(active_entity_ids)

        # FIXED: Max value with correct units
        if view_mode == 'share' and not use_market_mode:
            max_value_text = f"{max_focus:.1f}%"
        else:
            max_value_text = f"{max_focus:.1f}%"

        # Generate caption
        caption_text = generate_radar_caption(entities_to_show, selected_topics, view_mode, use_market_mode, df)

        caption_div = html.Div([
            html.Div([
                html.Strong("Chart Explanation: ", style={'color': '#04193A', 'fontSize': '12px'}),
                html.Span(
                    caption_text,
                    style={'color': '#364761', 'fontSize': '12px', 'lineHeight': '1.4'}
                )
            ], style={
                'backgroundColor': '#F8F9FA', 'padding': '12px', 'borderRadius': '6px',
                'border': '1px solid #E6E8EB', 'marginTop': '15px'
            })
        ])

        final_content = html.Div([
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False}
            ),
            html.Div([
                html.Small(
                    f"{mode_indicator} • {entity_indicator} • Comparing {entity_count} entit{'ies' if entity_count > 1 else 'y'} "
                    f"across {selected_count} selected topics • "
                    f"Max value: {max_value_text}",
                    style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic'})
            ], style={'marginTop': '10px'}),
            caption_div
        ])
        return final_content, current_complete + 1

    except Exception as e:
        print(f"Error generating radar chart: {e}")
        error_content = html.Div([
            html.Div("⚠️", style={'fontSize': '48px', 'textAlign': 'center', 'marginBottom': '20px', 'opacity': '0.3'}),
            html.H5("Chart Generation Error",
                    style={'color': '#C73E1D', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Unable to generate radar chart. Please check your data and try again.",
                   style={'color': '#687589', 'textAlign': 'center', 'fontStyle': 'italic'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '350px'})
        return error_content, current_complete + 1


# Portfolio panel toggle callbacks (unchanged)
@app.callback(
    [Output('portfolio-panel', 'style'),
     Output('portfolio-backdrop', 'style'),
     Output('portfolio-open', 'data')],
    [Input('portfolio-toggle-btn', 'n_clicks'),
     Input('portfolio-close-btn', 'n_clicks'),
     Input('portfolio-backdrop', 'n_clicks')],
    State('portfolio-open', 'data')
)
def toggle_portfolio_panel(toggle_clicks, close_clicks, backdrop_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        panel_style = {
            'position': 'fixed',
            'top': '0',
            'left': '-470px',
            'width': '420px',
            'height': '100vh',
            'backgroundColor': 'white',
            'zIndex': '1001',
            'padding': '0',
            'boxShadow': '4px 0 20px rgba(0,0,0,0.15)',
            'transition': 'left 0.3s ease-in-out',
            'visibility': 'hidden',
            'overflowY': 'hidden'
        }
        backdrop_style = {'display': 'none'}
        return panel_style, backdrop_style, False

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    should_open = not is_open if triggered_id == 'portfolio-toggle-btn' else False

    if should_open:
        panel_style = {
            'position': 'fixed',
            'top': '0',
            'left': '0px',
            'width': '420px',
            'height': '100vh',
            'backgroundColor': 'white',
            'zIndex': '1001',
            'padding': '0',
            'boxShadow': '4px 0 20px rgba(0,0,0,0.15)',
            'transition': 'left 0.3s ease-in-out',
            'visibility': 'visible',
            'overflowY': 'hidden'
        }
        backdrop_style = {
            'position': 'fixed', 'top': '0', 'left': '0', 'width': '100vw', 'height': '100vh',
            'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '1000', 'display': 'block'
        }
    else:
        panel_style = {
            'position': 'fixed',
            'top': '0',
            'left': '-470px',
            'width': '420px',
            'height': '100vh',
            'backgroundColor': 'white',
            'zIndex': '1001',
            'padding': '0',
            'boxShadow': '4px 0 20px rgba(0,0,0,0.15)',
            'transition': 'left 0.3s ease-in-out',
            'visibility': 'hidden',
            'overflowY': 'hidden'
        }
        backdrop_style = {'display': 'none'}

    return panel_style, backdrop_style, should_open


# Body scroll lock (unchanged)
app.clientside_callback(
    """
    function(isOpen) {
        if (isOpen) {
            const y = window.scrollY || window.pageYOffset;
            document.body.dataset.scrollY = y;
            document.body.style.position = "fixed";
            document.body.style.top = `-${y}px`;
            document.body.style.left = "0";
            document.body.style.right = "0";
        } else {
            const y = document.body.dataset.scrollY || "0";
            document.body.style.position = "";
            document.body.style.top = "";
            document.body.style.left = "";
            document.body.style.right = "";
            document.documentElement.style.overflow = "";
            window.scrollTo(0, parseInt(y, 10));
        }
        return "";
    }
    """,
    Output("portfolio-backdrop", "children"),
    Input("portfolio-open", "data"),
    prevent_initial_call=True,
)

# Keyboard handling for group editing (unchanged)
app.clientside_callback(
    """
    function(editing_group) {
        if (!editing_group) return window.dash_clientside.no_update;

        setTimeout(() => {
            const inputs = document.querySelectorAll('input[id*="group-name-input"]');
            inputs.forEach(input => {
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        const saveBtn = this.parentElement.querySelector('button[id*="save-group-name"]');
                        if (saveBtn) saveBtn.click();
                    } else if (e.key === 'Escape') {
                        const cancelBtn = this.parentElement.querySelector('button[id*="cancel-group-edit"]');
                        if (cancelBtn) cancelBtn.click();
                    }
                });
            });
        }, 100);

        return window.dash_clientside.no_update;
    }
    """,
    Output('editing-group', 'data', allow_duplicate=True),
    Input('editing-group', 'data'),
    prevent_initial_call=True
)


# Applicant search with duplicate checking (unchanged)
@app.callback(
    [Output('applicant-search', 'options'),
     Output('duplicate-warning', 'children')],
    [Input('available-applicants-store', 'data'),
     Input('selected-group', 'data'),
     Input('applicant-groups', 'data'),
     Input('applicant-search', 'search_value')],
    prevent_initial_call=False
)
def populate_applicant_options_with_duplicate_check(all_options, selected_group, groups_data, search):
    if not all_options or not selected_group:
        return [], html.Div()

    # Get all assigned applicants excluding current group
    already_assigned_elsewhere = get_all_assigned_applicants(groups_data, exclude_group=selected_group)
    already_in_current_group = set(groups_data.get(selected_group, {}).get('applicants', []))

    # Create warning message if search term matches assigned applicant
    warning_div = html.Div()
    if search and len(search) >= 2:
        # FIXED: Use exact matching instead of partial matching
        conflicting_applicants = [name for name in already_assigned_elsewhere
                                  if name.lower() == search.lower()]  # Exact match only

        if conflicting_applicants:
            warning_div = html.Div([
                html.Span("⚠️ ", style={'color': '#FF6B35', 'fontWeight': 'bold'}),
                html.Span(f"Applicant(s) already assigned to other groups: {', '.join(conflicting_applicants[:3])}"
                          + ("..." if len(conflicting_applicants) > 3 else ""),
                          style={'color': '#FF6B35', 'fontSize': '11px'})
            ], style={'backgroundColor': '#FFF3E0', 'padding': '6px 8px', 'borderRadius': '4px',
                      'border': '1px solid #FFB74D', 'marginBottom': '8px'})

    # Filter options
    if search and len(search) >= 2:
        search_lower = search.lower()
        filtered = [
            opt for opt in all_options
            if (search_lower in opt['value'].lower() or search_lower in opt['label'].lower())
               and opt['value'] not in already_in_current_group
               and opt['value'] not in already_assigned_elsewhere  # This should only exclude exact matches now
        ]
    else:
        filtered = [opt for opt in all_options
                    if opt['value'] not in already_in_current_group
                    and opt['value'] not in already_assigned_elsewhere]

    return filtered, warning_div


# UPDATED: Group management callback with proper focus data recalculation
@app.callback(
    [
        Output('applicant-groups', 'data'),
        Output('applicant-search', 'value'),
        Output('groups-container', 'children'),
        Output('group-counter', 'data'),
        Output('selected-group', 'data'),
        Output('add-group-btn', 'children'),
        Output('add-group-btn', 'disabled'),
        Output('add-group-btn', 'style'),
        Output('active-groups', 'data'),
        Output('group-focus-data', 'data'),
        Output('editing-group', 'data'),
    ],
    [
        Input('add-applicant-btn', 'n_clicks'),
        Input({'type': 'remove-applicant', 'group': ALL, 'applicant': ALL}, 'n_clicks'),
        Input('add-group-btn', 'n_clicks'),
        Input({'type': 'remove-group', 'group': ALL}, 'n_clicks'),
        Input({'type': 'group-box', 'group': ALL}, 'n_clicks'),
        Input({'type': 'active-toggle', 'group': ALL}, 'n_clicks'),
        Input({'type': 'edit-group-name', 'group': ALL}, 'n_clicks'),
        Input({'type': 'save-group-name', 'group': ALL}, 'n_clicks'),
        Input({'type': 'cancel-group-edit', 'group': ALL}, 'n_clicks'),
        Input('view-mode', 'data'),
    ],
    [
        State('applicant-search', 'value'),
        State('selected-group', 'data'),
        State('applicant-groups', 'data'),
        State('group-counter', 'data'),
        State('active-groups', 'data'),
        State('editing-group', 'data'),
        State({'type': 'group-name-input', 'group': ALL}, 'value'),
        State('selected-bubbles', 'data'),
    ],
)
def manage_everything(add_app_clicks, rem_app_clicks, add_grp_clicks,
                      rem_grp_clicks, grp_box_clicks, toggle_clicks,
                      edit_clicks, save_clicks, cancel_clicks, view_mode,
                      selected_applicant, selected_group,
                      groups_data, group_counter, active_groups, editing_group,
                      group_name_inputs, selected_topics):
    """Enhanced group management with market mode support."""
    if groups_data is None:
        groups_data = {}
    if active_groups is None:
        active_groups = {}

    max_groups = 3
    ctx = callback_context
    trig = ctx.triggered[0]['prop_id'] if ctx.triggered else None

    if trig is None:
        pass
    elif 'group-box' in trig:
        bid = json.loads(trig.split('.')[0])
        selected_group = bid['group']
    elif 'add-applicant-btn' in trig and selected_applicant and selected_group:
        if selected_group in groups_data:
            # Check for duplicates across all groups
            already_assigned = get_all_assigned_applicants(groups_data)
            if selected_applicant not in already_assigned:
                lst = groups_data[selected_group]['applicants']
                if selected_applicant not in lst:
                    lst.append(selected_applicant)
                    print(f"✅ Added {selected_applicant} to {groups_data[selected_group]['name']}")
            else:
                print(f"⚠️ Cannot add {selected_applicant}: already assigned to another group")
        selected_applicant = None
    elif 'remove-applicant' in trig:
        bid = json.loads(trig.split('.')[0])
        gid, who = bid['group'], bid['applicant']
        if gid in groups_data and who in groups_data[gid]['applicants']:
            groups_data[gid]['applicants'].remove(who)
            print(f"✅ Removed {who} from {groups_data[gid]['name']}")
    elif 'add-group-btn' in trig and len(groups_data) < max_groups:
        group_counter += 1
        gid = f'group_{group_counter}'
        groups_data[gid] = {
            'name': get_next_group_name(groups_data),
            'color': get_next_available_color(groups_data),
            'applicants': [],
        }
        active_groups[gid] = False
        print(f"✅ Created new group: {groups_data[gid]['name']}")
    elif 'remove-group' in trig:
        bid = json.loads(trig.split('.')[0])
        gid = bid['group']
        if gid in groups_data and len(groups_data) > 1:
            removed_name = groups_data[gid]['name']
            groups_data.pop(gid, None)
            active_groups.pop(gid, None)
            if selected_group == gid:
                selected_group = next(iter(groups_data))
            if editing_group == gid:
                editing_group = None
            print(f"🗑️ Removed group: {removed_name}")
    elif 'active-toggle' in trig:
        bid = json.loads(trig.split('.')[0])
        gid = bid['group']
        active_groups[gid] = not active_groups.get(gid, False)
        status = "activated" if active_groups[gid] else "deactivated"
        print(f"🔄 {groups_data[gid]['name']} {status}")
    elif 'edit-group-name' in trig:
        bid = json.loads(trig.split('.')[0])
        editing_group = bid['group']
    elif 'save-group-name' in trig:
        bid = json.loads(trig.split('.')[0])
        gid = bid['group']
        if group_name_inputs and gid in groups_data:
            all_groups_with_inputs = [g for g in groups_data.keys() if g == editing_group]
            if all_groups_with_inputs and gid in all_groups_with_inputs:
                idx = all_groups_with_inputs.index(gid)
                if idx < len(group_name_inputs) and group_name_inputs[idx]:
                    new_name = group_name_inputs[idx].strip()
                    if new_name and len(new_name) > 0:
                        existing_names = [g['name'].lower() for gid2, g in groups_data.items() if gid2 != gid]
                        if new_name.lower() not in existing_names:
                            old_name = groups_data[gid]['name']
                            groups_data[gid]['name'] = new_name
                            print(f"✏️ Renamed group from '{old_name}' to '{new_name}'")
        editing_group = None
    elif 'cancel-group-edit' in trig:
        editing_group = None

    # Calculate focus data with current view mode and selected topics
    print(f"\n🔄 Group state changed, recalculating focus data in {view_mode} mode...")
    focus_data = calculate_group_focus_data(groups_data, active_groups, applicant_dist, df, view_mode,
                                            selected_topics=selected_topics)

    # Rebuild UI
    btn_count = len(groups_data)
    btn_text = f"+ Create New Group ({btn_count}/{max_groups})" \
        if btn_count < max_groups else \
        f"Max Groups Reached ({btn_count}/{max_groups})"
    btn_disable = btn_count >= max_groups
    btn_style = {
        'backgroundColor': '#6c757d' if btn_disable else '#28a745',
        'color': 'white', 'border': 'none',
        'padding': '8px 16px', 'borderRadius': '5px',
        'fontSize': '12px', 'width': '100%', 'marginBottom': '20px',
        'cursor': 'not-allowed' if btn_disable else 'pointer'
    }

    groups_display = generate_groups_display(
        groups_data, selected_group, active_groups, editing_group
    )

    gc.collect()

    return (
        groups_data,
        selected_applicant,
        groups_display,
        group_counter,
        selected_group,
        btn_text,
        btn_disable,
        btn_style,
        active_groups,
        focus_data,
        editing_group,
    )


# UPDATED: SVG UPDATE CALLBACK
@app.callback(
    Output('svg-frame', 'srcDoc'),
    [Input('group-focus-data', 'data'),
     Input('view-mode', 'data')],  # UPDATED: Changed from 'analysis-mode'
    [State('active-groups', 'data'),
     State('applicant-groups', 'data')]
)
def update_svg(focus_data, view_mode, active_groups, groups_data):
    """
    Update the SVG when group focus changes or view mode changes.
    """
    num_active_groups = sum(1 for is_active in active_groups.values() if is_active) if active_groups else 0

    # Check if we're in market-wide mode
    use_market_mode = should_use_market_mode(groups_data, active_groups)

    print(f"\n🎨 Updating SVG with {num_active_groups} active groups in {view_mode} mode")
    print(f"   Market mode: {use_market_mode}")
    print(f"   🚀 Selection handled pure client-side - no server regeneration for clicks!")

    if focus_data and not use_market_mode:
        topics_with_rings = sum(1 for rings in focus_data.values() if rings)
        print(f"   Topics with ring data: {topics_with_rings}")

    # FIXED: Translate new mode names to what SVG builder expects
    # Portfolio Focus -> relative (rings), Market Share -> absolute (pie charts)
    # SVG still expects 'absolute' for market-share pies – do NOT change
    analysis_mode_for_svg = 'relative' if view_mode == 'focus' else 'absolute'
    print(f" Translating {view_mode} -> {analysis_mode_for_svg} for SVG builder")

    # Check if render_bubble_svg accepts focus_data and analysis_mode parameters
    import inspect
    sig = inspect.signature(render_bubble_svg)

    # Prepare parameters
    render_params = {
        'df': df,
    }

    # Only pass focus_data when NOT in market mode
    if 'focus_data' in sig.parameters:
        if use_market_mode:
            render_params['focus_data'] = None  # Clean visualization in market mode
            print("   🌍 Market mode: Rendering clean bubbles without portfolio overlays")
        else:
            render_params['focus_data'] = focus_data if num_active_groups > 0 else None
            print("   👥 Portfolio mode: Rendering bubbles with portfolio overlays")

    # FIXED: Always use analysis_mode parameter (SVG builder hasn't been updated)
    if 'analysis_mode' in sig.parameters:
        render_params['analysis_mode'] = analysis_mode_for_svg
        print(f" Passing analysis_mode='{analysis_mode_for_svg}' to SVG builder")

    # Generate SVG with translated mode
    svg_string = render_bubble_svg(**render_params)

    print(" SVG generated - should show rings (focus) or pie charts (share)")
    gc.collect()
    return svg_string


# UNIFIED: Single iframe handler combining auto-height + click interaction.
# Previously three separate callbacks all wrote to iframe.onload as a plain
# property, so each one silently overwrote the last. The resize interval then
# called whichever function happened to win that race every 100 ms, meaning
# click handlers were never reliably attached and the overflow:hidden applied
# to the iframe document body was suppressing pointer-events (killing hover).
# Fix: one shared setup function stored on window, called via addEventListener
# so nothing can clobber it, and overflow applied only to <body> margin/padding
# rather than the document root (which was blocking mouse events).

app.clientside_callback(
    """
    function(srcDoc, selectedBubbles) {

        // Shared function that does BOTH resize and click-handler attachment.
        // Stored on window so the resize-interval callback can call it safely
        // without needing its own copy of iframe.onload.
        window._setupIframe = function(iframe, selectedBubbles) {
            try {
                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                if (!iframeDoc) return;

                const svg = iframeDoc.querySelector('svg');
                const bubbleSvg = iframeDoc.querySelector('#bubble-visualization') || svg;

                // ── 1. Auto-height ────────────────────────────────────────────
                if (svg) {
                    // Fix: only zero out margins/padding, NOT overflow on the
                    // document root – setting overflow:hidden there suppresses
                    // pointer-events in modern Chrome/Firefox and breaks hover.
                    iframeDoc.body.style.margin = '0';
                    iframeDoc.body.style.padding = '0';

                    const viewBox = svg.getAttribute('viewBox');
                    if (viewBox) {
                        const parts = viewBox.split(' ').map(Number);
                        const svgWidth  = parts[2];
                        const svgHeight = parts[3];
                        if (svgWidth && svgHeight) {
                            const aspectRatio   = svgWidth / svgHeight;
                            const iframeWidth   = iframe.offsetWidth;
                            const requiredHeight = iframeWidth / aspectRatio;
                            iframe.style.height  = requiredHeight + 'px';
                            console.log('SVG Auto-Height:', svgWidth + 'x' + svgHeight,
                                        '→ iframe', iframeWidth + 'x' + Math.round(requiredHeight));
                        }
                    }
                }

                // ── 2 & 3. Per-bubble listeners + selection sync ──────────────
                // Chrome 143 (Mar 2026) broke two things for SVG inside srcdoc iframes:
                //   a) CSS :hover no longer applies transforms on <g> elements
                //   b) Event delegation from the SVG root stopped catching child clicks
                // Fix: attach click, mouseenter, mouseleave directly to each bubble.
                // Guard flag (dataset.listenersAttached) prevents duplicate listeners
                // accumulating when the resize interval re-calls this function.
                //
                // CSS class cascade (.client-selected .bubble-circle fill) is also
                // unreliable in Chrome 143+ srcdoc SVG, so we apply colours as
                // direct inline styles which always win regardless of CSS quirks.
                if (!bubbleSvg) return;

                // Helper: set circle fill/stroke directly as inline styles
                function applyBubbleStyle(bubble, isSelected) {
                    const circle = bubble.querySelector('.bubble-circle');
                    if (!circle) return;
                    if (isSelected) {
                        circle.style.fill        = '#87CEEB';
                        circle.style.stroke      = '#2196F3';
                        circle.style.strokeWidth = '3px';
                    } else {
                        circle.style.fill        = '#9BA3B0';
                        circle.style.stroke      = '#FFFFFF';
                        circle.style.strokeWidth = '1.5px';
                    }
                }

                const currentSelection = selectedBubbles || [];
                const allBubbles = bubbleSvg.querySelectorAll('.bubble-wrapper');

                allBubbles.forEach(function(bubble) {
                    const bubbleId = parseInt(bubble.getAttribute('data-bubble-id'));

                    // ── Sync selection state (runs every call) ─────────────────
                    const isSelected = currentSelection.includes(bubbleId);
                    if (isSelected) {
                        bubble.classList.add('client-selected');
                        bubble.classList.remove('client-unselected');
                    } else {
                        bubble.classList.add('client-unselected');
                        bubble.classList.remove('client-selected');
                    }
                    applyBubbleStyle(bubble, isSelected);

                    // ── Skip listener attachment if already done ───────────────
                    if (bubble.dataset.listenersAttached) return;
                    bubble.dataset.listenersAttached = 'true';

                    // ── Hover: JS scale + bring-to-front ──────────────────────
                    // Chrome 143+ fires mouseleave→mouseenter when appendChild
                    // moves an SVG element, creating an infinite loop that leaves
                    // the transform permanently set. Fix: only call appendChild when
                    // the bubble is NOT already the last child. Once it's at the end
                    // we stop moving it, the loop can't restart, and mouseleave
                    // correctly clears the transform when the mouse actually leaves.
                    bubble.addEventListener('mouseenter', function() {
                        const scale = bubble.style.getPropertyValue('--scale') || '1.8';
                        bubble.style.transform = 'scale(' + scale + ')';
                        if (bubble.parentNode && bubble.parentNode.lastElementChild !== bubble) {
                            bubble.parentNode.appendChild(bubble);
                        }
                    });
                    bubble.addEventListener('mouseleave', function() {
                        bubble.style.transform = '';
                    });

                    // ── Click ─────────────────────────────────────────────────
                    bubble.addEventListener('click', function(e) {
                        if (isNaN(bubbleId)) return;

                        const sel = window.clientSelection || [];
                        const idx = sel.indexOf(bubbleId);

                        if (idx > -1) {
                            sel.splice(idx, 1);
                            bubble.classList.remove('client-selected');
                            bubble.classList.add('client-unselected');
                            applyBubbleStyle(bubble, false);
                        } else {
                            if (sel.length >= 5) {
                                // Flash to indicate limit reached
                                bubble.style.filter = 'brightness(1.4)';
                                setTimeout(function() { bubble.style.filter = ''; }, 200);
                                return;
                            }
                            sel.push(bubbleId);
                            bubble.classList.remove('client-unselected');
                            bubble.classList.add('client-selected');
                            applyBubbleStyle(bubble, true);
                        }

                        window.clientSelection = sel;
                        window.pendingClickData = {
                            bubbleId: bubbleId,
                            timestamp: Date.now(),
                            newSelection: [...sel]
                        };

                        e.preventDefault();
                        e.stopPropagation();
                    });
                });

                window.clientSelection = [...currentSelection];

            } catch (error) {
                console.log('Error setting up iframe:', error);
            }
        };

        setTimeout(function() {
            const iframe = document.getElementById('svg-frame');
            if (!iframe) return;

            // Use addEventListener so this is never overwritten by another callback
            const sel = selectedBubbles;
            function onLoadOnce() {
                window._setupIframe(iframe, sel);
            }

            // Remove any lingering load listener then add a fresh one
            if (window._iframeLoadHandler) {
                iframe.removeEventListener('load', window._iframeLoadHandler);
            }
            window._iframeLoadHandler = onLoadOnce;
            iframe.addEventListener('load', window._iframeLoadHandler);

            // If already loaded, run immediately
            if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                window._setupIframe(iframe, sel);
            }

        }, 150);

        return window.dash_clientside.no_update;
    }
    """,
    Output('svg-frame', 'title'),
    [Input('svg-frame', 'srcDoc'),
     Input('selected-bubbles', 'data')],
    prevent_initial_call=False
)

# Resize-on-interval: just calls the shared setup function – no iframe.onload touch
app.clientside_callback(
    """
    function(n_intervals) {
        if (n_intervals % 10 !== 0) return window.dash_clientside.no_update;

        const iframe = document.getElementById('svg-frame');
        if (!iframe) return window.dash_clientside.no_update;

        if (window._setupIframe) {
            window._setupIframe(iframe, window.clientSelection || []);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('svg-frame', 'data-resize-checked'),
    Input('click-detector', 'n_intervals'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_intervals) {
        if (window.pendingClickData) {
            const clickData = window.pendingClickData;
            window.pendingClickData = null;
            return clickData;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('click-data', 'data'),
    Input('click-detector', 'n_intervals'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function() {
        // Add hover effects to patent title links
        const style = document.createElement('style');
        style.textContent = `
            .patent-title-link:hover {
                color: #2E86AB !important;
                text-decoration: underline !important;
            }
            .patent-title-link:visited {
                color: #6A1B9A !important;
            }
        `;

        // Only add if not already added
        if (!document.querySelector('#patent-link-styles')) {
            style.id = 'patent-link-styles';
            document.head.appendChild(style);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('document-explorer-content', 'data-styled'),  # Dummy output
    Input('document-explorer-content', 'children'),
    prevent_initial_call=False
)


# FIXED: Selection sync callback with proper debounce management
@app.callback(
    [Output('selected-bubbles', 'data'),
     Output('selection-pending', 'data'),
     Output('debounce-interval', 'disabled')],
    Input('click-data', 'data'),
    [State('selected-bubbles', 'data')],
    prevent_initial_call=True
)
def sync_selection_with_debounce(click_data, current_selection):
    """Update selection and trigger debounce timer."""
    if not click_data or 'newSelection' not in click_data:
        return no_update, no_update, no_update

    new_selection = click_data['newSelection']

    # Check if selection actually changed
    if new_selection == current_selection:
        return no_update, no_update, no_update

    print(f" Selection changed: {new_selection}, resetting debounce timer...")

    # Return: new selection, pending=True, disable timer (will be re-enabled by timer manager)
    return new_selection, True, True

@app.callback(
    [Output('debounce-interval', 'disabled', allow_duplicate=True),
     Output('debounce-interval', 'n_intervals', allow_duplicate=True)],
    Input('selection-pending', 'data'),
    prevent_initial_call=True
)
def manage_debounce_timer(is_pending):
    """Manage debounce timer state - reset when new selection pending."""
    if is_pending:
        print(" Starting fresh debounce timer...")
        # Small delay then enable timer with reset counter
        return False, 0
    else:
        # Keep timer disabled when not pending
        return True, 0


@app.callback(
    [Output('debounce-trigger', 'data'),
     Output('selection-pending', 'data', allow_duplicate=True),
     Output('debounce-interval', 'disabled', allow_duplicate=True)],
    Input('debounce-interval', 'n_intervals'),
    [State('selection-pending', 'data'),
     State('debounce-trigger', 'data')],
    prevent_initial_call=True
)
def complete_debounce_timer(n_intervals, is_pending, current_trigger):
    """Complete debounce and trigger analysis after 500ms."""
    if is_pending and n_intervals >= 1:  # 500ms has passed
        print(f" Debounce complete after {n_intervals} intervals, triggering analysis...")
        # Trigger analysis with incremented value, stop pending, disable timer
        return current_trigger + 1, False, True
    return no_update, no_update, no_update


# Selection info callback (unchanged)
@app.callback(
    Output('selection-info', 'children'),
    Input('selected-bubbles', 'data')
)
def update_selection_info(selected_ids):
    """Update the selection information panel."""
    if not selected_ids:
        return "Click up to 5 topic bubbles to explore patent clusters and analyze technological themes"

    selected_data = df[df['id'].isin(selected_ids)]
    if selected_data.empty:
        return "No valid topic selections"

    info_items = []
    for _, row in selected_data.iterrows():
        topic_label = row.get('topic_label', f"Topic {row['id']}")
        cluster = row.get('cluster', 'Unclustered')
        info_items.append(
            html.Div([
                html.Strong(f"{topic_label}"),
                html.Br(),
                html.Small(f"Patent Cluster: {cluster} • Topic ID: {row['id']}", style={'color': '#687589'})
            ], style={'marginBottom': '10px', 'padding': '8px', 'backgroundColor': 'white', 'borderRadius': '5px'})
        )

    limit_warning = ""
    if len(selected_ids) >= 5:
        limit_warning = html.Div([
            html.Strong("Selection limit reached:", style={'color': '#C73E1D'}),
            html.Span(" You can select up to 5 topics for analysis.", style={'color': '#687589'})
        ], style={'marginBottom': '10px', 'padding': '6px', 'backgroundColor': '#FFF3E0', 'borderRadius': '5px',
                  'border': '1px solid #FFB74D'})

    return [
        html.H4(f"Analyzing {len(selected_ids)} patent topic{'s' if len(selected_ids) != 1 else ''}",
                style={'color': '#04193A'}),
        limit_warning,
        html.Div(info_items),
        html.Hr(),
        html.P(f"Selected topics span {len(set(selected_data['cluster']))} distinct patent clusters",
               style={'fontStyle': 'italic', 'color': '#687589'})
    ]

server = app.server

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)