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
    num_bts: Optional[int] = None,
    bounds: Optional[Dict[str, float]] = None,
    min_separation_km: Optional[float] = None,
    frequency_mhz: Optional[float] = None,
    tx_power_dbm: Optional[float] = None,
    antenna_gain_dbi: Optional[float] = None,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic BTS stations within geographic bounds.

    Places BTS stations with minimum separation constraint to avoid
    overlapping cell coverage.

    Args:
        num_bts: Number of BTS to generate
        bounds: Dictionary with lat_min, lat_max, lon_min, lon_max
        min_separation_km: Minimum distance between BTS (km)
        frequency_mhz: Operating frequency
        tx_power_dbm: Transmit power
        antenna_gain_dbi: Antenna gain
        random_seed: Random seed for reproducibility

    Returns:
        List of BTS dictionaries
    """
    # Use config defaults if not specified
    if num_bts is None:
        num_bts = config.BTS_CONFIG['count']
    if bounds is None:
        bounds = config.TEHRAN_BOUNDS
    if min_separation_km is None:
        min_separation_km = config.BTS_CONFIG['min_separation_km']
    if frequency_mhz is None:
        frequency_mhz = config.BTS_CONFIG['frequency_mhz']
    if tx_power_dbm is None:
        tx_power_dbm = config.BTS_CONFIG['tx_power_dbm']
    if antenna_gain_dbi is None:
        antenna_gain_dbi = config.BTS_CONFIG['antenna_gain_dbi']
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    np.random.seed(random_seed)

    bts_list = []
    max_attempts = 1000

    for i in range(num_bts):
        attempts = 0
        while attempts < max_attempts:
            # Generate random location within bounds
            lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
            lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])

            # Check minimum separation from existing BTS
            valid_location = True
            for existing_bts in bts_list:
                distance = geodesic(
                    (lat, lon),
                    (existing_bts['lat'], existing_bts['lon'])
                ).km

                if distance < min_separation_km:
                    valid_location = False
                    break

            if valid_location:
                bts = {
                    'id': f'BTS_{i+1:03d}',
                    'lat': lat,
                    'lon': lon,
                    'frequency_mhz': frequency_mhz,
                    'tx_power_dbm': tx_power_dbm,
                    'antenna_gain_dbi': antenna_gain_dbi,
                    'antenna_height_m': config.BTS_CONFIG['antenna_height_m']
                }
                bts_list.append(bts)
                break

            attempts += 1

        if attempts >= max_attempts:
            print(f"Warning: Could not place BTS {i+1} after {max_attempts} attempts")

    print(f"Generated {len(bts_list)} BTS stations")
    return bts_list


def generate_repeaters(
    bts_list: List[Dict[str, Any]],
    num_repeaters: Optional[int] = None,
    bounds: Optional[Dict[str, float]] = None,
    gain_db: Optional[float] = None,
    min_dist_km: Optional[float] = None,
    max_dist_km: Optional[float] = None,
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
        min_dist_km: Minimum distance from serving BTS
        max_dist_km: Maximum distance from serving BTS
        random_seed: Random seed

    Returns:
        List of repeater dictionaries
    """
    # Use config defaults if not specified
    if num_repeaters is None:
        num_repeaters = config.REPEATER_CONFIG['count']
    if bounds is None:
        bounds = config.TEHRAN_BOUNDS
    if gain_db is None:
        gain_db = config.REPEATER_CONFIG['gain_db']
    if min_dist_km is None:
        min_dist_km = config.REPEATER_CONFIG['min_distance_from_bts_km']
    if max_dist_km is None:
        max_dist_km = config.REPEATER_CONFIG['max_distance_from_bts_km']
    if random_seed is None:
        random_seed = config.RANDOM_SEED + 1  # Different seed from BTS

    np.random.seed(random_seed)

    if not bts_list:
        print("Error: No BTS available to assign repeaters")
        return []

    repeater_list = []
    max_attempts = 1000

    for i in range(num_repeaters):
        # Randomly select a BTS to serve
        serving_bts = np.random.choice(bts_list)

        attempts = 0
        while attempts < max_attempts:
            # Generate random angle and distance
            angle = np.random.uniform(0, 2 * np.pi)
            distance_km = np.random.uniform(min_dist_km, max_dist_km)

            # Calculate approximate lat/lon offset
            # At Tehran's latitude, 1 degree lat ≈ 111 km, 1 degree lon ≈ 92 km
            lat_offset = (distance_km * np.cos(angle)) / 111
            lon_offset = (distance_km * np.sin(angle)) / 92

            lat = serving_bts['lat'] + lat_offset
            lon = serving_bts['lon'] + lon_offset

            # Check if within bounds
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lon_min'] <= lon <= bounds['lon_max']):

                # Verify actual distance
                actual_dist = geodesic(
                    (lat, lon),
                    (serving_bts['lat'], serving_bts['lon'])
                ).km

                if min_dist_km <= actual_dist <= max_dist_km:
                    repeater = {
                        'id': f'REP_{i+1:03d}',
                        'lat': lat,
                        'lon': lon,
                        'gain_db': gain_db,
                        'serving_bts_id': serving_bts['id'],
                        'serving_bts_lat': serving_bts['lat'],
                        'serving_bts_lon': serving_bts['lon']
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
