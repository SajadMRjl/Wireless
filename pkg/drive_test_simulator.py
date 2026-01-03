"""
Drive test simulator for mobile network measurements.
Generates measurement points and calculates RSSI using dual-path propagation model.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from . import config
from .propagation import calculate_rssi_at_point


def generate_measurement_grid(bounds: Optional[Dict[str, float]] = None, grid_spacing_m: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Generate a grid of measurement points across the geographic area.

    Args:
        bounds: Dictionary with lat_min, lat_max, lon_min, lon_max
        grid_spacing_m: Spacing between measurement points in meters

    Returns:
        List of (lat, lon) tuples
    """
    if bounds is None:
        bounds = config.TEHRAN_BOUNDS
    if grid_spacing_m is None:
        grid_spacing_m = config.DRIVE_TEST_CONFIG['grid_spacing_m']

    # Convert grid spacing from meters to degrees (approximate)
    # At Tehran's latitude: 1 degree lat ≈ 111 km, 1 degree lon ≈ 92 km
    lat_spacing = (grid_spacing_m / 1000) / 111
    lon_spacing = (grid_spacing_m / 1000) / 92

    # Generate grid
    lat_points = np.arange(
        bounds['lat_min'],
        bounds['lat_max'],
        lat_spacing
    )

    lon_points = np.arange(
        bounds['lon_min'],
        bounds['lon_max'],
        lon_spacing
    )

    # Create mesh grid
    measurement_points = []
    for lat in lat_points:
        for lon in lon_points:
            measurement_points.append((lat, lon))

    print(f"Generated {len(measurement_points)} measurement points " +
          f"(grid spacing: {grid_spacing_m}m)")

    return measurement_points


def simulate_drive_test(
    bts_list: List[Dict[str, Any]],
    repeater_list: List[Dict[str, Any]],
    measurement_points: Optional[List[Tuple[float, float]]] = None,
    add_noise: Optional[bool] = None,
    save_to_csv: bool = True
) -> List[Dict[str, Any]]:
    """
    Simulate drive test measurements across measurement points.

    Implements dual-path propagation model:
    - Direct path from BTS
    - Indirect paths via repeaters
    - Logarithmic combination of all paths
    - Log-normal shadowing noise

    Args:
        bts_list: List of BTS dictionaries
        repeater_list: List of repeater dictionaries
        measurement_points: List of (lat, lon) tuples. If None, generates grid
        add_noise: Whether to add noise (default: from config)
        save_to_csv: Whether to save results to CSV

    Returns:
        List of measurement dictionaries
    """
    # Generate measurement grid if not provided
    if measurement_points is None:
        measurement_points = generate_measurement_grid()

    # Get configuration
    if add_noise is None:
        add_noise = config.NOISE_CONFIG['use_noise']

    frequency_mhz = config.BTS_CONFIG['frequency_mhz']
    rx_gain_dbi = config.DRIVE_TEST_CONFIG['rx_antenna_gain_dbi']
    sensitivity_floor = config.DRIVE_TEST_CONFIG['sensitivity_floor_dbm']
    sigma_noise = config.NOISE_CONFIG['log_normal_sigma_db']

    measurements = []
    start_time = datetime.now()

    print(f"Simulating drive test with {len(measurement_points)} measurement points...")
    print(f"BTS: {len(bts_list)}, Repeaters: {len(repeater_list)}")
    print(f"Noise: {'enabled' if add_noise else 'disabled'} " +
          f"(sigma={sigma_noise} dB)" if add_noise else "")

    # Process each measurement point
    for idx, (lat, lon) in enumerate(measurement_points):
        # Calculate RSSI from all BTS (with dual-path model)
        rssi_per_bts = calculate_rssi_at_point(
            point_lat=lat,
            point_lon=lon,
            bts_list=bts_list,
            repeater_list=repeater_list,
            frequency_mhz=frequency_mhz,
            rx_gain_dbi=rx_gain_dbi,
            add_noise=add_noise,
            sigma_noise=sigma_noise
        )

        # Apply sensitivity floor (minimum detectable signal)
        rssi_per_bts = {
            bts_id: max(rssi, sensitivity_floor)
            for bts_id, rssi in rssi_per_bts.items()
        }

        # Determine serving cell (strongest signal)
        if rssi_per_bts:
            serving_cell_id = max(rssi_per_bts, key=rssi_per_bts.get)
            serving_rssi = rssi_per_bts[serving_cell_id]
        else:
            serving_cell_id = None
            serving_rssi = sensitivity_floor

        # Create measurement record
        # CRITICAL: Store RSSI from ALL BTS, not just serving cell
        measurement = {
            'measurement_id': idx + 1,
            'timestamp': start_time + timedelta(seconds=idx),
            'lat': lat,
            'lon': lon,
            'serving_cell_id': serving_cell_id,
            'serving_rssi': serving_rssi,
            **{f'rssi_{bts_id}': rssi for bts_id, rssi in rssi_per_bts.items()}
        }

        measurements.append(measurement)

        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(measurement_points)} points...")

    print(f"Drive test simulation complete! Total measurements: {len(measurements)}")

    # Save to CSV
    if save_to_csv:
        save_measurements_to_csv(measurements)

    return measurements


def save_measurements_to_csv(measurements: List[Dict[str, Any]], filepath: Optional[str] = None) -> None:
    """
    Save measurements to CSV file.

    Args:
        measurements: List of measurement dictionaries
        filepath: Output file path (default: from config)
    """
    import os

    if filepath is None:
        filepath = config.OUTPUT_PATHS['measurements_csv']

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = pd.DataFrame(measurements)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(measurements)} measurements to {filepath}")


def load_measurements_from_csv(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load measurements from CSV file.

    Args:
        filepath: Input file path (default: from config)

    Returns:
        List of measurement dictionaries
    """
    if filepath is None:
        filepath = config.OUTPUT_PATHS['measurements_csv']

    df = pd.read_csv(filepath)
    return df.to_dict('records')


def get_rssi_vector_from_measurement(measurement: Dict[str, Any], bts_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract RSSI vector for all BTS from a measurement record.

    Args:
        measurement: Measurement dictionary
        bts_list: List of BTS dictionaries

    Returns:
        Dictionary {bts_id: rssi_dbm}
    """
    rssi_vector = {}

    for bts in bts_list:
        bts_id = bts['id']
        rssi_key = f'rssi_{bts_id}'

        if rssi_key in measurement:
            rssi_vector[bts_id] = measurement[rssi_key]
        else:
            # If not in measurement (shouldn't happen), use sensitivity floor
            rssi_vector[bts_id] = config.DRIVE_TEST_CONFIG['sensitivity_floor_dbm']

    return rssi_vector
