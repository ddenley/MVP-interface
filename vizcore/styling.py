def get_color_palette() -> dict:
    """
    Returns a dictionary of color codes used in the Vizcore application.
    :return: Dictionary with color names as keys and their hex codes as values.
    """
    return {
        'dark_blue': '#04193A',
        'navy_blue': '#03234B',
        'medium_blue': '#1D304E',
        'steel_blue': '#364761',
        'slate_blue': '#4F5E75',
        'charcoal': '#687589',
        'gray_blue': '#828C9C',
        'light_gray': '#9BA3B0',
        'silver': '#B4BAC4',
        'light_silver': '#CDD1D8',
        'pearl': '#E6E8EB',
        'white': '#FFFFFF',
        'warm_white': '#F8F9FA',
        'success_blue': '#0066CC',
        'info_blue': '#2196F3',
        'subtle_blue': '#E3F2FD',
        'selected_blue': '#87CEEB',  # NEW: Light blue for selected bubbles
    }


def assign_bubble_colors(df, selected_ids=None):
    """
    Assign colors to bubbles, with special handling for selected bubbles.
    :param df: DataFrame containing bubble data
    :param selected_ids: Set of selected bubble IDs that should be highlighted
    :return: DataFrame with 'color' column added/updated
    """
    palette = get_color_palette()
    uniform_bubble_color = palette['light_gray']
    selected_color = palette['selected_blue']

    # Default color for all bubbles
    df['color'] = uniform_bubble_color

    # Highlight selected bubbles
    if selected_ids:
        selected_mask = df['id'].isin(selected_ids)
        df.loc[selected_mask, 'color'] = selected_color

    return df


def get_hull_colors():
    palette = get_color_palette()
    hull_style = {
        'fill_color': '#B3D9FF',
        'border_color': palette['dark_blue'],
        'fill_opacity': 0.6,
        'border_opacity': 0.9,
    }
    return hull_style


def assign_bubble_colors_uniform(df):
    """
    Assign uniform colors to all bubbles - no selection-based coloring.
    Selection styling will be handled purely client-side.
    """
    palette = get_color_palette()
    uniform_bubble_color = palette['light_gray']

    # All bubbles get the same color
    df['color'] = uniform_bubble_color
    return df