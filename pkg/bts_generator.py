"""
Generate synthetic BTS (Base Transceiver Stations) and repeater locations
for Tehran mobile network simulation.
"""

from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from . import config


def generate_bts_stations(
    bounds: Optional[Dict[str, float]] = None,
    separation_km: Optional[float] = None,
    frequency_mhz: Optional[float] = None,
    tx_power_dbm: Optional[float] = None,
    antenna_gain_dbi: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic BTS stations within geographic bounds.

    Places BTS stations with minimum separation constraint to avoid
    overlapping cell coverage.

    Args:
        bounds: Dictionary with lat_min, lat_max, lon_min, lon_max
        separation_km: distance between BTS (km)
        frequency_mhz: Operating frequency
        tx_power_dbm: Transmit power
        antenna_gain_dbi: Antenna gain

    Returns:
        List of BTS dictionaries
    """
    # Use config defaults if not specified
    if bounds is None:
        bounds = config.TEHRAN_BOUNDS
    if separation_km is None:
        separation_km = config.BTS_CONFIG['separation_km']
    if frequency_mhz is None:
        frequency_mhz = config.BTS_CONFIG['frequency_mhz']
    if tx_power_dbm is None:
        tx_power_dbm = config.BTS_CONFIG['tx_power_dbm']
    if antenna_gain_dbi is None:
        antenna_gain_dbi = config.BTS_CONFIG['antenna_gain_dbi']


    lat_step_deg = separation_km / 111.0  # Vertical spacing in degrees
    lon_step_deg = (separation_km * np.sin(np.pi / 3)) / 92.0  # Horizontal spacing: √3/2 * R
    

    bts_list = []
    col_index = 0

    lon_pivot = bounds['lon_min']
    while lon_pivot < bounds['lon_max']:
        lat_pivot = bounds['lat_min'] + (lat_step_deg / 2 if col_index % 2 == 1 else 0)
        
        while lat_pivot < bounds['lat_max']:
            power_variation = np.random.uniform(-2, 2)  # ±2 dBm variation
            gain_variation = np.random.uniform(-1, 1)  # ±1 dBi variation
            height_variation = np.random.uniform(-5, 10)  # Height variation: -5m to +10m
            
            bts = {
                'id': f'BTS_{len(bts_list)+1:03d}',
                'lat': lat_pivot,
                'lon': lon_pivot,
                'frequency_mhz': frequency_mhz,
                'tx_power_dbm': tx_power_dbm + power_variation,  # Realistic variation
                'antenna_gain_dbi': antenna_gain_dbi + gain_variation,  # Realistic variation
                'antenna_height_m': max(15, config.BTS_CONFIG['antenna_height_m'] + height_variation)  # Min 15m
            }
            bts_list.append(bts)

            lat_pivot += lat_step_deg

        col_index += 1
        lon_pivot += lon_step_deg

    print(f"Generated {len(bts_list)} BTS stations")
    return bts_list


def generate_repeaters(
    bts_list: List[Dict[str, Any]],
    num_repeaters: Optional[int] = None,
    bounds: Optional[Dict[str, float]] = None,
    gain_tx_db: Optional[float] = None,
    gain_rx_db: Optional[float] = None,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate unauthorized repeater locations.

    Each repeater is assigned to a serving BTS and placed within
    specified distance range from that BTS.

    Args:
        bts_list: List of BTS dictionaries
        num_repeaters: Number of repeaters to generate
        bounds: Geographic bounds
        gain_db: Repeater amplifier gain
        random_seed: Random seed

    Returns:
        List of repeater dictionaries
    """
    # Use config defaults if not specified
    if num_repeaters is None:
        num_repeaters = config.REPEATER_CONFIG['count']
    if bounds is None:
        bounds = config.TEHRAN_BOUNDS
    if gain_tx_db is None:
        gain_tx_db = config.REPEATER_CONFIG['gain_tx_db']
    if gain_rx_db is None:
        gain_rx_db = config.REPEATER_CONFIG['gain_rx_db']
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    np.random.seed(random_seed)

    if not bts_list:
        print("Error: No BTS available for repeater placement")
        return []

    repeater_list = []
    max_attempts = 1000

    for i in range(num_repeaters):
        # Randomly select a BTS to serve as a location reference (placement only)
        # Repeaters are no longer tied to a specific serving BTS
        reference_bts = np.random.choice(bts_list)

        attempts = 0
        while attempts < max_attempts:
            # Generate random angle and distance
            angle = np.random.uniform(0, 2 * np.pi)
            distance_km = np.random.uniform(0.33, 0.66)

            # Calculate approximate lat/lon offset
            # At Tehran's latitude, 1 degree lat ≈ 111 km, 1 degree lon ≈ 92 km
            lat_offset = (distance_km * np.cos(angle)) / 111
            lon_offset = (distance_km * np.sin(angle)) / 92

            lat = reference_bts['lat'] + lat_offset
            lon = reference_bts['lon'] + lon_offset

            # Check if within bounds
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lon_min'] <= lon <= bounds['lon_max']):

                # Verify actual distance
                actual_dist = geodesic(
                    (lat, lon),
                    (reference_bts['lat'], reference_bts['lon'])
                ).km

                if actual_dist <= 1:
                    repeater = {
                        'id': f'REP_{i+1:03d}',
                        'lat': lat,
                        'lon': lon,
                        'gain_tx_db': gain_tx_db,
                        'gain_rx_db': gain_rx_db,
                    }
                    repeater_list.append(repeater)
                    break

            attempts += 1

        if attempts >= max_attempts:
            print(f"Warning: Could not place repeater {i+1} after {max_attempts} attempts")

    print(f"Generated {len(repeater_list)} unauthorized repeaters")
    return repeater_list


def save_bts_to_csv(bts_list: List[Dict[str, Any]], filepath: Optional[str] = None) -> None:
    """Save BTS list to CSV file."""
    import os

    if filepath is None:
        filepath = config.OUTPUT_PATHS['bts_csv']

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = pd.DataFrame(bts_list)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(bts_list)} BTS to {filepath}")


def save_repeaters_to_csv(repeater_list: List[Dict[str, Any]], filepath: Optional[str] = None) -> None:
    """Save repeater list to CSV file."""
    import os

    if filepath is None:
        filepath = config.OUTPUT_PATHS['repeaters_csv']

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = pd.DataFrame(repeater_list)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(repeater_list)} repeaters to {filepath}")


def load_bts_from_csv(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load BTS list from CSV file."""
    if filepath is None:
        filepath = config.OUTPUT_PATHS['bts_csv']

    df = pd.read_csv(filepath)
    return df.to_dict('records')


def load_repeaters_from_csv(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load repeater list from CSV file."""
    if filepath is None:
        filepath = config.OUTPUT_PATHS['repeaters_csv']

    df = pd.read_csv(filepath)
    return df.to_dict('records')


if __name__ == '__main__':
    # Generate and save BTS stations
    bts_stations = generate_bts_stations()
    save_bts_to_csv(bts_stations)

    # Generate and save repeaters
    repeaters = generate_repeaters(bts_stations)
    save_repeaters_to_csv(repeaters)

    print("\nBTS and repeater generation complete!")
    print(f"BTS count: {len(bts_stations)}")
    print(f"Repeater count: {len(repeaters)}")
