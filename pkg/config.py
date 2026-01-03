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
    'frequency_mhz': 1800,  # LTE Band 3 (1800 MHz)
    'tx_power_dbm': 45,  # Transmit power (reduced for realistic urban range)
    'antenna_gain_dbi': 15,  # BTS antenna gain (typical sector antenna)
    'antenna_height_m': 30,  # Typical cell tower height
    'separation_km': 1,  # Distance between BTS
}

# Repeater Parameters
REPEATER_CONFIG = {
    'count': 3,  # Number of unauthorized repeaters
    'gain_tx_db': 60,  # Repeater output antenna gain
    'gain_rx_db': 60,  # Repeater input antenna gain
}

# Drive Test Parameters
DRIVE_TEST_CONFIG = {
    'grid_spacing_m': 100,  # Spacing between measurement points
    'rx_antenna_gain_dbi': 0,  # Mobile device antenna gain
    'sensitivity_floor_dbm': -110,  # Minimum detectable signal
}

# Noise Model Parameters
NOISE_CONFIG = {
    'log_normal_sigma_db': 4,  # Standard deviation for log-normal shadowing (reduced for cleaner viz)
    'use_noise': True,  # Enable/disable noise for testing
}

# Detection Algorithm Parameters
DETECTION_CONFIG = {
    'z_score_threshold': 3.5,  # One-sided z-score threshold
    'dbscan_eps_m': 300,  # DBSCAN clustering radius (meters)
    'dbscan_min_samples': 3,  # Minimum points per cluster
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
