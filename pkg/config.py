"""
Configuration file for wireless repeater detection simulation.
Contains all simulation parameters for Tehran mobile network.
"""

# Geographic Bounds (Tehran)
TEHRAN_BOUNDS = {
    'lat_min': 35.720266,
    'lat_max': 35.758200,
    'lon_min': 51.500422,
    'lon_max': 51.546223
}

# BTS (Base Transceiver Station) Parameters
BTS_CONFIG = {
    'frequency_mhz': 1800,  # LTE Band 3 (1800 MHz) - realistic frequency
    'tx_power_dbm': 43,  # Typical macro cell transmit power (40-46 dBm)
    'antenna_gain_dbi': 15,  # BTS antenna gain (reduced for urban model)
    'antenna_height_m': 30,  # Typical cell tower height (25-50m)
    'separation_km': 1,  # distance between BTS (1-2 km urban)
}

# Repeater Parameters
REPEATER_CONFIG = {
    'count': 3,  # Number of unauthorized repeaters
    'gain_db': 25,  # Amplifier gain (typical consumer repeater: 20-30 dB)
}

# Drive Test Parameters
DRIVE_TEST_CONFIG = {
    'grid_spacing_m': 100,  # Spacing between measurement points
    'rx_antenna_gain_dbi': 0,  # Mobile device antenna gain
    'sensitivity_floor_dbm': -110,  # Minimum detectable signal
}

# Noise Model Parameters
NOISE_CONFIG = {
    'log_normal_sigma_db': 8,  # Standard deviation for log-normal shadowing
    'use_noise': True,  # Enable/disable noise for testing
}

# Detection Algorithm Parameters
DETECTION_CONFIG = {
    'z_score_threshold': 2.5,  # Statistical significance threshold
    'dbscan_eps_m': 500,  # DBSCAN maximum distance (meters)
    'dbscan_min_samples': 10,  # DBSCAN minimum cluster size
}

# Visualization Parameters
VIZ_CONFIG = {
    'map_center_lat': (TEHRAN_BOUNDS['lat_min'] + TEHRAN_BOUNDS['lat_max']) / 2,
    'map_center_lon': (TEHRAN_BOUNDS['lon_min'] + TEHRAN_BOUNDS['lon_max']) / 2,
    'map_zoom': 11,
    'heatmap_radius': 15,
    'heatmap_blur': 20,
}

# Output Paths
OUTPUT_PATHS = {
    'bts_csv': './data/bts_stations.csv',
    'repeaters_csv': './data/repeaters.csv',
    'measurements_csv': './data/drive_test_measurements.csv',
}

# Random seed for reproducibility
RANDOM_SEED = 42
