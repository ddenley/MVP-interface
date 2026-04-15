import pandas as pd
import numpy as np
from .styling import get_color_palette, assign_bubble_colors, get_hull_colors, assign_bubble_colors_uniform
import shapely.geometry as sg
import shapely.ops as so
from shapely import concave_hull
import alphashape
from scipy.interpolate import splprep, splev
import math


# --------- Geometry and Visualization Functions ---------

def calculate_cluster_centroids(df):
    """Calculate centroids for each cluster of bubbles."""
    centroids = {}
    for cluster_id in df['cluster'].unique():
        cluster_bubbles = df[df['cluster'] == cluster_id]
        total_weight = cluster_bubbles['radius'].sum()
        centroid_x = (cluster_bubbles['x'] * cluster_bubbles['radius']).sum() / total_weight
        centroid_y = (cluster_bubbles['y'] * cluster_bubbles['radius']).sum() / total_weight
        centroids[cluster_id] = (centroid_x, centroid_y)
    return centroids


def generate_smooth_hull(cluster_bubbles, centroid, padding=8, num_rays=48, smoothing=0.2):
    # Minimum padding of 8 trying to prevent cutting
    padding = max(padding, 8)
    geometries = []
    for _, bubble in cluster_bubbles.iterrows():
        x, y, r = bubble['x'], bubble['y'], bubble['radius']
        # Generous buffer
        total_buffer = r + padding
        circle = sg.Point(x, y).buffer(total_buffer, cap_style=1, join_style=1)
        geometries.append(circle)

    if not geometries:
        return None

    # Union all of the buffered circles
    union_geom = so.unary_union(geometries)

    # Handle MultiPolygons
    if hasattr(union_geom, 'geoms'):
        largest_area = 0
        largest_geom = None
        for geom in union_geom.geoms:
            if hasattr(geom, 'area') and geom.area > largest_area:
                largest_area = geom.area
                largest_geom = geom
        union_geom = largest_geom if largest_geom else union_geom.geoms[0]

    # Concave hull
    try:
        hull = concave_hull(union_geom, ratio=0.25)
    except Exception as e:
        if hasattr(union_geom, 'exterior'):
            print(f"Concave hull basic gen failed: {e}")
            print(f"Falling back to alphashape")
            try:
                boundary_coords = np.array(union_geom.exterior.coords)
                alpha_value = max(cluster_bubbles['radius'].mean() * 6, 30)
                hull = alphashape.alphashape(boundary_coords, alpha_value)
            except Exception as e:
                print(f"Alphashape fallback failed: {e}")
                hull = union_geom.convex_hull
        else:
            print(f"Concave hull failed: {e}")
            hull = union_geom.convex_hull

    max_radius = cluster_bubbles['radius'].max()
    gentle_tolerance = max_radius * 0.02
    hull = hull.simplify(gentle_tolerance, preserve_topology=True)

    # Safety validation of the hull generation
    hull = validate_hull_safety(hull, cluster_bubbles, extra_margin=3)

    # Extract co-ordinates
    if hasattr(hull, 'exterior'):
        coords = np.array(hull.exterior.coords)
    else:
        raise ValueError("Hull is not a polygon, cannot extract coordinates")

    # Additional spline smoothing for hull curves
    if len(coords) > 4:
        coords = apply_smooth_spline(coords, samples=200, smoothing_factor=0.3)

    return coords


def validate_hull_safety(hull, cluster_bubbles, extra_margin=3):
    """
    Here to try and ensure no cutting of hulls
    """
    try:
        unsafe = True
        expansion_factor = 1.0
        original_hull = hull
        while unsafe and expansion_factor < 5.0:
            unsafe = False

            for _, bubble in cluster_bubbles.iterrows():
                x, y, r = bubble['x'], bubble['y'], bubble['radius']
                safety_bubble = sg.Point(x, y).buffer(r + extra_margin, cap_style=1, join_style=1)

                if hull.intersects(safety_bubble) and not hull.contains(safety_bubble):
                    unsafe = True
                    break

            if unsafe:
                expansion_factor += 0.5
                hull = hull.buffer(expansion_factor, cap_style=1, join_style=1)

        return hull

    except Exception as e:
        print(f"Hull validation failed: {e}")
        return hull


def apply_smooth_spline(coords, samples=200, smoothing_factor=0.3):
    try:
        # Close the curve properly
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        # Remove duplicate closing point for spline fitting
        coords_for_spline = coords[:-1]

        if len(coords_for_spline) < 3:
            return coords

        tck, u = splprep([coords_for_spline[:, 0], coords_for_spline[:, 1]],
                         s=smoothing_factor * len(coords_for_spline),
                         per=True,
                         k=3)
        u_smooth = np.linspace(0, 1, samples)
        x_smooth, y_smooth = splev(u_smooth, tck)
        return np.column_stack([x_smooth, y_smooth])

    except Exception as e:
        print(f"Spline smoothing failed: {e}")
        print("Simple smoothing fallback")
        try:
            tck, u = splprep([coords_for_spline[:, 0], coords_for_spline[:, 1]],
                             s=0.1 * len(coords_for_spline), per=True)
            u_smooth = np.linspace(0, 1, 150)
            x_smooth, y_smooth = splev(u_smooth, tck)
            return np.column_stack([x_smooth, y_smooth])
        except Exception as e:
            print("Simple spline smoothing failed - returning original coordinates")
            return coords


def should_draw_hull(cluster_id, df, min_bubbles=3, min_total_radius=50):
    """
    Simple function to determine if a hull should be drawn for a cluster.
    For now lets say always
    """
    return True


def should_draw_hull_deprecated(cluster_id, df, min_bubbles=3, min_total_radius=50):
    """Determine if a hull should be drawn for this cluster."""
    cluster_bubbles = df[df['cluster'] == cluster_id]

    # Skip small clusters
    if len(cluster_bubbles) < min_bubbles:
        return False

    # Skip clusters with very small total area
    total_radius = cluster_bubbles['radius'].sum()
    if total_radius < min_total_radius:
        return False

    return True


# NEW: Pie chart segment generation
def create_pie_segment_path(cx, cy, radius, start_angle, end_angle):
    """
    Create SVG path for a pie chart segment.

    :param cx, cy: Center coordinates
    :param radius: Radius of the circle
    :param start_angle, end_angle: Angles in degrees (0 = top, clockwise)
    :return: SVG path string
    """
    # Handle full circle case
    if abs(end_angle - start_angle) >= 360:
        return f"M{cx},{cy} m-{radius},0 a{radius},{radius} 0 1,0 {radius * 2},0 a{radius},{radius} 0 1,0 -{radius * 2},0 Z"

    # Convert to radians (0 degrees = top of circle, clockwise)
    start_rad = math.radians(start_angle - 90)
    end_rad = math.radians(end_angle - 90)

    # Calculate arc endpoints
    x1 = cx + radius * math.cos(start_rad)
    y1 = cy + radius * math.sin(start_rad)
    x2 = cx + radius * math.cos(end_rad)
    y2 = cy + radius * math.sin(end_rad)

    # Determine if arc should be large (> 180 degrees)
    large_arc = 1 if (end_angle - start_angle) > 180 else 0

    # Create path: Move to center, line to start, arc to end, line back to center, close
    path = f"M{cx:.2f},{cy:.2f} L{x1:.2f},{y1:.2f} A{radius:.2f},{radius:.2f} 0 {large_arc},1 {x2:.2f},{y2:.2f} L{cx:.2f},{cy:.2f} Z"

    return path


# --------- Main SVG Generation Function ---------

def render_bubble_svg(df, width=1800, height=800, canvas_boundary_buffer=50, focus_data=None,
                      analysis_mode='relative'):
    """
    Render the SVG for the bubble visualization with PURE client-side selection handling.

    COMPLETELY REMOVED: selected_ids parameter - all selection styling now handled client-side

    :param df: DataFrame containing bubble data
    :param width: Canvas width
    :param height: Canvas height
    :param canvas_boundary_buffer: Padding around the canvas
    :param focus_data: Focus data for visualization
    :param analysis_mode: 'relative' for rings, 'absolute' for pie charts
    :return: SVG string
    """
    if focus_data:
        print(
            f"🎨 SVG Builder received focus data for {len([k for k, v in focus_data.items() if v])} topics with visualization")
        print(f"🎨 Rendering in {analysis_mode.upper()} mode")

    # FIXED: Always assign uniform colors - no server-side selection styling
    df = assign_bubble_colors_uniform(df)

    # Canvas setup
    canvas_width = width + (2 * canvas_boundary_buffer)
    canvas_height = height + (2 * canvas_boundary_buffer)
    palette = get_color_palette()

    # Start building the SVG
    svg_content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg id="bubble-visualization" viewBox="0 0 {canvas_width} {canvas_height}" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: auto; max-width: 100%;">',
        f'<rect x="0" y="0" width="{canvas_width}" height="{canvas_height}" fill="{palette["warm_white"]}" stroke="{palette["silver"]}" stroke-width="2"/>',
        '<title>Bubble Visualization</title>',
        '<style>',
        # Existing styles
        '  .bubble-text { font-family: "Segoe UI", Arial, sans-serif; font-weight: 600; dominant-baseline: central; }',
        '  .cluster-hull { stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }',
        '  .st-professional { filter: drop-shadow(0px 2px 4px rgba(4, 25, 58, 0.1)); }',
        '  .bubble-label { display: flex; align-items: center; justify-content: center; text-align: center; width: 100%; height: 100%; overflow: hidden; clip-path: circle(50% at 50% 50%); pointer-events: none; }',
        '  .bubble-label-text { font-family:"Segoe UI", Arial, sans-serif; font-weight:600; color:#04193A; line-height:1.1; padding:3%; box-sizing:border-box; display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:4; overflow:hidden; text-align:center;}',
        '  .bubble-wrapper { transition: transform .25s ease; transform-box: fill-box; transform-origin: center; }',
        '  .bubble-wrapper:hover { transform: scale(var(--scale, 1.8)); }',

        # Base styles for interactive bubbles
        '  .bubble-clickable { cursor: pointer; }',

        # ENHANCED: Client-side selection styles with highest priority
        '  .bubble-wrapper.client-selected .bubble-circle { stroke: #2196F3 !important; stroke-width: 3px !important; fill: #87CEEB !important; filter: brightness(1.1) !important; }',
        '  .bubble-wrapper.client-selected .bubble-label-text { color: #03234B !important; font-weight: 700 !important; text-shadow: 0 1px 2px rgba(255,255,255,0.8) !important; }',
        '  .bubble-wrapper.client-unselected .bubble-circle { stroke: #FFFFFF !important; stroke-width: 1.5px !important; fill: #9BA3B0 !important; filter: none !important; }',
        '  .bubble-wrapper.client-unselected .bubble-label-text { color: #04193A !important; font-weight: 600 !important; text-shadow: none !important; }',

        # Smooth transitions for all state changes
        '  .bubble-circle { transition: all 0.3s ease !important; }',
        '  .bubble-label-text { transition: all 0.3s ease !important; }',

        # Pie chart styles
        '  .pie-segment { transition: opacity 0.2s ease; }',
        '  .pie-segment:hover { opacity: 1.0 !important; stroke-width: 1 !important; }',

        '</style>',
        '<defs>',
        '  <filter id="st-glow" x="-50%" y="-50%" width="200%" height="200%">',
        '    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>',
        '    <feMerge>',
        '      <feMergeNode in="coloredBlur"/>',
        '      <feMergeNode in="SourceGraphic"/>',
        '    </feMerge>',
        '  </filter>',
        '  <linearGradient id="st-gradient" x1="0%" y1="0%" x2="100%" y2="100%">',
        f'    <stop offset="0%" style="stop-color:{palette["dark_blue"]};stop-opacity:0.1"/>',
        f'    <stop offset="100%" style="stop-color:{palette["pearl"]};stop-opacity:0.05"/>',
        '  </linearGradient>',
        '</defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="url(#st-gradient)"/>'
    ]

    hull_style = get_hull_colors()

    # Generate cluster hulls (unchanged)
    if 'cluster' in df.columns:
        print("Generating the cluster hulls...")
        centroids = calculate_cluster_centroids(df)

        for cluster_idx, cluster_id in enumerate(df['cluster'].unique()):
            if not should_draw_hull(cluster_id, df):
                continue

            cluster_bubbles = df[df['cluster'] == cluster_id]
            centroid = centroids[cluster_id]
            hull_points = generate_smooth_hull(cluster_bubbles, centroid, padding=8, smoothing=0.25)

            if len(hull_points) >= 3:
                svg_points = []
                for x, y in hull_points:
                    svg_x = x + canvas_boundary_buffer
                    svg_y = canvas_height - (y + canvas_boundary_buffer)
                    svg_points.append(f"{svg_x:.2f},{svg_y:.2f}")

                path_data = f"M{svg_points[0]} " + " ".join([f"L{point}" for point in svg_points[1:]]) + " Z"

                svg_content.append(
                    f'<path d="{path_data}" '
                    f'fill="{hull_style["fill_color"]}" fill-opacity="{hull_style["fill_opacity"]}" '
                    f'stroke="{hull_style["border_color"]}" stroke-opacity="{hull_style["border_opacity"]}" '
                    f'class="cluster-hull st-professional" filter="url(#st-glow)">'
                    f'<title>Cluster {cluster_id} ({len(cluster_bubbles)} components)</title>'
                    f'</path>'
                )

    # Generate bubbles - COMPLETELY REMOVED all server-side selection logic
    for _, row in df.iterrows():
        x_orig, y_orig, r_orig = row['x'], row['y'], row['radius']
        bubble_id = row['id']
        cluster = row.get('cluster', 0)

        # Coordinate system correction with buffers
        x = x_orig + canvas_boundary_buffer
        y = canvas_height - (y_orig + canvas_boundary_buffer)
        r = r_orig + 1.5

        color = row['color']  # This is now always the uniform color

        # UPDATED: Set display text and tooltip with summary
        if 'display_label' in row and pd.notna(row['display_label']):
            display_text = row['display_label']
        else:
            display_text = f"T{bubble_id}"

        # NEW: Build enhanced tooltip with focus information
        tooltip_text = build_enhanced_tooltip(row, bubble_id, focus_data, analysis_mode)

        scale_factor = max(1.3, 2.5 - 0.015 * r)

        # FIXED: All bubbles start as "client-unselected" - client JS will update them
        base_classes = "bubble-wrapper bubble-clickable client-unselected"

        svg_content.append(
            f'<g class="{base_classes}" '
            f'data-bubble-id="{bubble_id}" '
            f'style="--scale:{scale_factor};" '
            f'onmouseenter="this.parentNode.appendChild(this)">')

        # Check if this bubble has focus data
        bubble_id_str = str(bubble_id)
        has_focus_data = focus_data and bubble_id_str in focus_data and focus_data[bubble_id_str]

        if has_focus_data and analysis_mode == 'absolute':
            # ABSOLUTE MODE: Pie chart visualization
            groups_data = focus_data[bubble_id_str]
            total_group_percentage = sum(group['percentage'] for group in groups_data)

            if total_group_percentage > 0:
                print(
                    f"🥧 Generating pie chart for bubble {bubble_id} with {len(groups_data)} segments (covering {total_group_percentage:.1f}% of market)")

                # Draw background circle (represents total market)
                svg_content.append(
                    f'<circle cx="{x}" cy="{y}" r="{r}" '
                    f'fill="#F0F0F0" '
                    f'stroke="none" '
                    f'opacity="0.3">'
                    f'<title>Total Market: 100% of patents in this topic</title>'
                    f'</circle>'
                )

                # Generate pie segments for active groups
                current_angle = 0

                for group_info in groups_data:
                    segment_percentage = group_info['percentage']
                    segment_angle = (segment_percentage / 100) * 360

                    print(
                        f"   Segment: {group_info['name']} - {segment_percentage:.1f}% market share ({segment_angle:.1f}°) - Color: {group_info['color']}")

                    if segment_angle > 0.5:  # Only draw segments larger than 0.5 degrees
                        path_data = create_pie_segment_path(
                            cx=x, cy=y, radius=r,
                            start_angle=current_angle,
                            end_angle=current_angle + segment_angle
                        )

                        svg_content.append(
                            f'<path d="{path_data}" '
                            f'fill="{group_info["color"]}" '
                            f'stroke="rgba(255,255,255,0.3)" '
                            f'stroke-width="0.5" '
                            f'opacity="0.85" '
                            f'class="pie-segment">'
                            f'<title>{group_info["name"]}: {group_info["percentage"]:.1f}% market share '
                            f'({group_info["patent_count"]} of {group_info["total_patents"]} total patents)</title>'
                            f'</path>'
                        )

                    current_angle += segment_angle

                # Add ungrouped market segment if groups don't cover 100%
                ungrouped_percentage = 100 - total_group_percentage
                if ungrouped_percentage > 0.1:
                    ungrouped_angle = (ungrouped_percentage / 100) * 360
                    path_data = create_pie_segment_path(
                        cx=x, cy=y, radius=r,
                        start_angle=current_angle,
                        end_angle=current_angle + ungrouped_angle
                    )

                    svg_content.append(
                        f'<path d="{path_data}" '
                        f'fill="rgba(150,150,150,0.4)" '
                        f'stroke="rgba(255,255,255,0.3)" '
                        f'stroke-width="0.5" '
                        f'opacity="0.6" '
                        f'class="pie-segment">'
                        f'<title>Other companies: {ungrouped_percentage:.1f}% market share (ungrouped)</title>'
                        f'</path>'
                    )

                    print(f"   Ungrouped market: {ungrouped_percentage:.1f}% ({ungrouped_angle:.1f}°)")

                # FIXED: Add border circle with uniform styling (client CSS will override) + enhanced tooltip
                svg_content.append(
                    f'<circle class="bubble-circle" cx="{x}" cy="{y}" r="{r}" '
                    f'fill="none" '
                    f'stroke="white" stroke-width="1.5" '
                    f'opacity="0.7">'
                    f'<title>{tooltip_text}</title>'
                    f'</circle>'
                )
            else:
                # No meaningful pie data, draw normal bubble
                svg_content.append(
                    f'<circle class="bubble-circle" cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                    f'stroke="white" stroke-width="1.5" opacity="0.95" '
                    f'class="st-professional bubble-circle">'
                    f'<title>{tooltip_text}</title>'
                    f'</circle>'
                )

        elif has_focus_data and analysis_mode == 'relative':
            # RELATIVE MODE: Ring visualization

            # Draw base circle first with enhanced tooltip
            svg_content.append(
                f'<circle class="bubble-circle" cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                f'stroke="white" stroke-width="1.5" opacity="0.95" '
                f'class="st-professional bubble-circle">'
                f'<title>{tooltip_text}</title>'
                f'</circle>'
            )

            # Add focus rings
            ring_width = min(r * 0.15, 8)
            ring_gap = 2

            for ring_index, ring_info in enumerate(focus_data[bubble_id_str]):
                ring_radius = r + 3 + (ring_index * (ring_width + ring_gap))
                circumference = 2 * math.pi * ring_radius
                dash_length = (ring_info['percentage'] / 100) * circumference

                # Background ring (subtle)
                svg_content.append(
                    f'<circle cx="{x}" cy="{y}" r="{ring_radius}" '
                    f'fill="none" stroke="{ring_info["color"]}" '
                    f'stroke-width="{ring_width}" '
                    f'opacity="0.1" />'
                )

                # Focus ring (prominent)
                svg_content.append(
                    f'<circle cx="{x}" cy="{y}" r="{ring_radius}" '
                    f'fill="none" stroke="{ring_info["color"]}" '
                    f'stroke-width="{ring_width}" '
                    f'stroke-dasharray="{dash_length:.2f} {circumference:.2f}" '
                    f'opacity="0.8" stroke-linecap="round" '
                    f'transform="rotate(-90 {x} {y})">'
                    f'<title>{ring_info["name"]}: {ring_info["percentage"]:.1f}% focus '
                    f'({ring_info["patent_count"]} of {ring_info["total_patents"]} patents)</title>'
                    f'</circle>'
                )

        else:
            # NO FOCUS DATA: Draw normal bubble with uniform styling + enhanced tooltip
            svg_content.append(
                f'<circle class="bubble-circle" cx="{x}" cy="{y}" r="{r}" fill="{color}" '
                f'stroke="white" stroke-width="1.5" opacity="0.95" '
                f'class="st-professional bubble-circle">'
                f'<title>{tooltip_text}</title>'
                f'</circle>'
            )

        # FIXED: Text label with uniform styling (client CSS will override for selection)
        text_color = palette["dark_blue"]
        font_weight = "600"

        foreign_width = r * 1.8
        foreign_height = r * 1.8
        font_size = max(6, r * 0.22)
        foreign_x = x - foreign_width / 2
        foreign_y = y - foreign_height / 2

        svg_content.append(
            f'<foreignObject x="{foreign_x}" y="{foreign_y}" width="{foreign_width}" height="{foreign_height}" pointer-events="none">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="bubble-label">'
            f'<div class="bubble-label-text" style="font-size: {font_size}px; width: 100%; max-width: {foreign_width * 0.9}px; color: {text_color}; font-weight: {font_weight};">{display_text}</div>'
            f'</div>'
            f'</foreignObject>'
        )
        svg_content.append('</g>')

    svg_content.append('</svg>')
    return '\n'.join(svg_content)


def build_enhanced_tooltip(row, bubble_id, focus_data, analysis_mode):
    """
    Build enhanced tooltip with focus information for each bubble.

    :param row: DataFrame row for this bubble
    :param bubble_id: Topic ID
    :param focus_data: Focus data from portfolio analysis
    :param analysis_mode: 'relative' for portfolio focus, 'absolute' for market share
    :return: Enhanced tooltip string
    """
    # Start with the base summary
    summary = row.get('summary', f"No summary available for topic {bubble_id}")
    if pd.notna(summary) and summary.strip():
        tooltip_parts = [summary.strip()]
    else:
        tooltip_parts = [f"No summary available for topic {bubble_id}"]

    # Check if this bubble has focus data
    bubble_id_str = str(bubble_id)
    has_focus_data = focus_data and bubble_id_str in focus_data and focus_data[bubble_id_str]

    if has_focus_data:
        groups_data = focus_data[bubble_id_str]

        # Add separator
        tooltip_parts.append("---")

        # Add focus information based on mode
        for group_info in groups_data:
            group_name = group_info['name']
            percentage = group_info['percentage']
            is_market_mode = group_info.get('is_market_mode', False)
            mode = group_info['mode']

            if is_market_mode:
                # Market-wide mode
                if mode == 'focus':
                    tooltip_parts.append(f"{group_name}: Topic contribution {percentage:.1f}%")
                else:  # share mode
                    tooltip_parts.append(f"{group_name}: Market composition {percentage:.1f}%")
            else:
                # Portfolio mode
                if mode == 'focus':
                    tooltip_parts.append(f"{group_name}: Portfolio focus {percentage:.1f}%")
                else:  # share mode
                    tooltip_parts.append(f"{group_name}: Market share {percentage:.1f}%")

    # Join all parts with line breaks (using &#10; for SVG compatibility)
    return "&#10;".join(tooltip_parts)