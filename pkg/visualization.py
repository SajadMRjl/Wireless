"""
Visualization module for wireless repeater detection.
Creates interactive maps, heatmaps, and analysis plots.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from geopy.distance import geodesic
from . import config


def create_base_map(center_lat: Optional[float] = None, center_lon: Optional[float] = None, zoom: Optional[int] = None) -> folium.Map:
    """
    Create base Folium map centered on Tehran.

    Args:
        center_lat, center_lon: Map center coordinates
        zoom: Initial zoom level

    Returns:
        Folium Map object
    """
    if center_lat is None:
        center_lat = config.VIZ_CONFIG['map_center_lat']
    if center_lon is None:
        center_lon = config.VIZ_CONFIG['map_center_lon']
    if zoom is None:
        zoom = config.VIZ_CONFIG['map_zoom']

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )

    return m


def add_bts_markers(map_obj: folium.Map, bts_list: List[Dict[str, Any]]) -> None:
    """
    Add BTS markers to map.

    Args:
        map_obj: Folium Map object
        bts_list: List of BTS dictionaries
    """
    for bts in bts_list:
        popup_text = f"""
        <b>{bts['id']}</b><br>
        Frequency: {bts['frequency_mhz']} MHz<br>
        Power: {bts['tx_power_dbm']} dBm<br>
        Gain: {bts['antenna_gain_dbi']} dBi<br>
        Location: ({bts['lat']:.6f}, {bts['lon']:.6f})
        """

        folium.CircleMarker(
            location=[bts['lat'], bts['lon']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=250),
            color='blue',
            fill=True,
            fillColor='lightblue',
            fillOpacity=0.7,
            weight=2
        ).add_to(map_obj)

        # Add BTS label
        folium.Marker(
            location=[bts['lat'], bts['lon']],
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 10px; color: blue; font-weight: bold;">
                    {bts['id']}
                </div>
            """)
        ).add_to(map_obj)


def add_repeater_markers(map_obj: folium.Map, repeater_list: List[Dict[str, Any]], color: str = 'red', label: str = 'Actual') -> None:
    """
    Add repeater markers to map.

    Args:
        map_obj: Folium Map object
        repeater_list: List of repeater dictionaries
        color: Marker color
        label: Label prefix for markers
    """
    for repeater in repeater_list:
        popup_text = f"""
        <b>{label} Repeater: {repeater.get('id', repeater.get('detection_id', 'Unknown'))}</b><br>
        """

        if 'gain_db' in repeater:
            popup_text += f"Gain: {repeater['gain_db']} dB<br>"

        if 'serving_bts_id' in repeater:
            popup_text += f"Serving BTS: {repeater['serving_bts_id']}<br>"

        if 'confidence_mean_z' in repeater:
            popup_text += f"Confidence (z-score): {repeater['confidence_mean_z']:.2f}<br>"
            popup_text += f"Cluster size: {repeater['cluster_size']}<br>"

        popup_text += f"Location: ({repeater['lat']:.6f}, {repeater['lon']:.6f})"

        # Choose icon based on type
        if label == 'Actual':
            icon = folium.Icon(color='red', icon='star', prefix='fa')
        else:  # Detected
            icon = folium.Icon(color='orange', icon='crosshairs', prefix='fa')

        folium.Marker(
            location=[repeater['lat'], repeater['lon']],
            popup=folium.Popup(popup_text, max_width=250),
            icon=icon
        ).add_to(map_obj)


def add_detection_error_line(map_obj: folium.Map, detected: Dict[str, Any], actual: Dict[str, Any]) -> None:
    """
    Add line showing detection error between detected and actual repeater.

    Args:
        map_obj: Folium Map object
        detected: Detected repeater dictionary
        actual: Actual repeater dictionary
    """
    error_km = geodesic(
        (detected['lat'], detected['lon']),
        (actual['lat'], actual['lon'])
    ).km

    folium.PolyLine(
        locations=[
            [detected['lat'], detected['lon']],
            [actual['lat'], actual['lon']]
        ],
        color='purple',
        weight=2,
        opacity=0.7,
        popup=f"Detection Error: {error_km*1000:.1f} meters"
    ).add_to(map_obj)


def add_signal_heatmap(map_obj: folium.Map, measurements: List[Dict[str, Any]]) -> None:
    """
    Add signal strength heatmap to map.

    Args:
        map_obj: Folium Map object
        measurements: List of measurement dictionaries
    """
    # Extract coordinates and RSSI values
    heat_data = []

    for meas in measurements:
        lat = meas['lat']
        lon = meas['lon']
        rssi = meas.get('serving_rssi', -110)

        # Normalize RSSI to 0-1 range for heatmap intensity
        # Typical range: -110 dBm (weak) to -50 dBm (strong)
        normalized_intensity = (rssi + 110) / 60  # 0 to 1
        normalized_intensity = max(0, min(1, normalized_intensity))

        heat_data.append([lat, lon, normalized_intensity])

    # Add heatmap layer
    HeatMap(
        heat_data,
        radius=config.VIZ_CONFIG['heatmap_radius'],
        blur=config.VIZ_CONFIG['heatmap_blur'],
        min_opacity=0.3,
        max_zoom=13
    ).add_to(map_obj)


def add_anomaly_overlay(map_obj: folium.Map, anomalies: List[Dict[str, Any]]) -> None:
    """
    Add anomaly points overlay to map.

    Args:
        map_obj: Folium Map object
        anomalies: List of anomaly dictionaries
    """
    for anomaly in anomalies:
        folium.CircleMarker(
            location=[anomaly['lat'], anomaly['lon']],
            radius=3,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6,
            weight=1,
            popup=f"Z-score: {anomaly['z_score']:.2f}<br>BTS: {anomaly['bts_id']}"
        ).add_to(map_obj)


def create_main_detection_map(
    bts_list: List[Dict[str, Any]],
    actual_repeaters: List[Dict[str, Any]],
    detected_repeaters: List[Dict[str, Any]],
    measurements: List[Dict[str, Any]],
    anomalies: Optional[List[Dict[str, Any]]] = None,
    validation_metrics: Optional[Dict[str, Any]] = None,
    filepath: Optional[str] = None
) -> folium.Map:
    """
    Create main detection visualization map.

    Args:
        bts_list: List of BTS dictionaries
        actual_repeaters: List of actual repeater dictionaries
        detected_repeaters: List of detected repeater dictionaries
        measurements: List of measurement dictionaries
        anomalies: List of anomaly dictionaries (optional)
        validation_metrics: Validation metrics dictionary (optional)
        filepath: Output HTML file path

    Returns:
        Folium Map object
    """
    print("Creating main detection map...")

    # Create base map
    m = create_base_map()

    # Add layers
    add_bts_markers(m, bts_list)
    add_signal_heatmap(m, measurements)

    if anomalies:
        add_anomaly_overlay(m, anomalies)

    add_repeater_markers(m, actual_repeaters, color='red', label='Actual')
    add_repeater_markers(m, detected_repeaters, color='orange', label='Detected')

    # Add detection error lines
    if validation_metrics and 'matches' in validation_metrics:
        for match in validation_metrics['matches']:
            add_detection_error_line(m, match['detected'], match['actual'])

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin-bottom: 5px;"><b>Legend</b></p>
    <p style="margin: 3px;"><span style="color: blue;">●</span> BTS Station</p>
    <p style="margin: 3px;"><span style="color: red;">★</span> Actual Repeater</p>
    <p style="margin: 3px;"><span style="color: orange;">⊕</span> Detected Repeater</p>
    <p style="margin: 3px;"><span style="color: red;">●</span> Anomaly Point</p>
    '''

    if validation_metrics:
        legend_html += f'''
        <hr style="margin: 5px 0;">
        <p style="margin: 3px;"><b>Detection Results:</b></p>
        <p style="margin: 3px; font-size: 12px;">
        Detected: {validation_metrics['num_detected']}/{validation_metrics['num_actual']}<br>
        Mean Error: {validation_metrics['mean_error_m']:.1f}m
        </p>
        '''

    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save to file
    import os

    if filepath is None:
        filepath = config.OUTPUT_PATHS['main_map_html']

    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    m.save(filepath)
    print(f"Saved main detection map to {filepath}")

    return m


def create_comparison_maps(bts_list: List[Dict[str, Any]], measurements: List[Dict[str, Any]], predictions: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
    """
    Create side-by-side comparison of actual vs predicted coverage.

    Args:
        bts_list: List of BTS dictionaries
        measurements: List of actual measurements
        predictions: List of predicted measurements
        filepath: Output HTML file path

    Returns:
        HTML string with side-by-side maps
    """
    print("Creating comparison maps...")

    # Map 1: Actual measurements (with repeater)
    m1 = create_base_map()
    add_bts_markers(m1, bts_list)
    add_signal_heatmap(m1, measurements)
    m1_html = m1._repr_html_()

    # Map 2: Predicted coverage (no repeater)
    m2 = create_base_map()
    add_bts_markers(m2, bts_list)

    # Convert predictions to measurement format for heatmap
    pred_as_meas = []
    for pred in predictions:
        # Find max predicted RSSI across all BTS
        rssi_values = [v for k, v in pred.items() if k.startswith('predicted_rssi_')]
        max_rssi = max(rssi_values) if rssi_values else -110

        pred_as_meas.append({
            'lat': pred['lat'],
            'lon': pred['lon'],
            'serving_rssi': max_rssi
        })

    add_signal_heatmap(m2, pred_as_meas)
    m2_html = m2._repr_html_()

    # Combine into side-by-side HTML
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Coverage Comparison</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
            }}
            .container {{
                display: flex;
                gap: 20px;
            }}
            .map-container {{
                flex: 1;
            }}
            h3 {{
                text-align: center;
                margin-bottom: 10px;
            }}
            iframe {{
                width: 100%;
                height: 600px;
                border: 1px solid #ccc;
            }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Signal Coverage Comparison</h1>
        <div class="container">
            <div class="map-container">
                <h3>Actual Measurements (with Repeater)</h3>
                <iframe srcdoc='{m1_html.replace("'", "&#39;")}'></iframe>
            </div>
            <div class="map-container">
                <h3>Predicted Coverage (without Repeater)</h3>
                <iframe srcdoc='{m2_html.replace("'", "&#39;")}'></iframe>
            </div>
        </div>
    </body>
    </html>
    """

    # Save to file
    import os

    if filepath is None:
        filepath = config.OUTPUT_PATHS['comparison_map_html']

    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(combined_html)

    print(f"Saved comparison maps to {filepath}")

    return combined_html


def plot_residual_histogram(residuals: List[Dict[str, Any]], bts_list: List[Dict[str, Any]], z_threshold: Optional[float] = None) -> Figure:
    """
    Plot histogram of residuals with z-score threshold.

    Args:
        residuals: List of residual dictionaries
        bts_list: List of BTS dictionaries
        z_threshold: Z-score threshold line to draw
    """
    if z_threshold is None:
        z_threshold = config.DETECTION_CONFIG['z_score_threshold']

    # Collect all residuals
    all_residuals = []
    for res_dict in residuals:
        for bts in bts_list:
            bts_id = bts['id']
            residual_key = f'residual_{bts_id}'
            if residual_key in res_dict:
                all_residuals.append(res_dict[residual_key])

    # Calculate statistics
    mean_res = np.mean(all_residuals)
    std_res = np.std(all_residuals)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')

    # Add vertical lines for mean and threshold
    plt.axvline(mean_res, color='green', linestyle='--', linewidth=2,
                label=f'Mean: {mean_res:.2f} dB')

    threshold_value = mean_res + z_threshold * std_res
    plt.axvline(threshold_value, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (z={z_threshold}): {threshold_value:.2f} dB')

    plt.xlabel('Residual (Actual - Predicted RSSI) [dB]')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution with Anomaly Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_detection_metrics(validation_metrics: Dict[str, Any]) -> Figure:
    """
    Plot detection performance metrics.

    Args:
        validation_metrics: Validation metrics dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Detection rate
    ax1 = axes[0]
    categories = ['Detected', 'Missed']
    values = [
        validation_metrics['num_detected'],
        validation_metrics['num_actual'] - validation_metrics['num_detected']
    ]
    colors = ['green', 'red']

    ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count')
    ax1.set_title(f"Detection Rate: {validation_metrics['detection_rate']*100:.1f}%")
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Error distribution
    ax2 = axes[1]
    errors = [m['error_m'] for m in validation_metrics['matches']]

    ax2.bar(range(len(errors)), errors, color='orange', alpha=0.7, edgecolor='black')
    ax2.axhline(validation_metrics['mean_error_m'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {validation_metrics['mean_error_m']:.1f}m")
    ax2.set_xlabel('Repeater Index')
    ax2.set_ylabel('Detection Error (meters)')
    ax2.set_title('Detection Error per Repeater')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


if __name__ == '__main__':
    from bts_generator import load_bts_from_csv, load_repeaters_from_csv
    from drive_test_simulator import load_measurements_from_csv
    from detection import detect_repeaters, validate_detection, build_expected_coverage_map

    # Load data
    print("Loading data...")
    bts_stations = load_bts_from_csv()
    actual_repeaters = load_repeaters_from_csv()
    measurements = load_measurements_from_csv()

    # Run detection
    detected_repeaters = detect_repeaters(measurements, bts_stations)

    # Validate
    metrics = validate_detection(detected_repeaters, actual_repeaters)

    # Create visualizations
    create_main_detection_map(
        bts_list=bts_stations,
        actual_repeaters=actual_repeaters,
        detected_repeaters=detected_repeaters,
        measurements=measurements,
        validation_metrics=metrics
    )

    print("\nVisualization complete!")
