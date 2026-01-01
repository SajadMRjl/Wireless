"""
Configuration file for wireless repeater detection simulation.
Contains all simulation parameters for Tehran mobile network.
"""

# Geographic Bounds (Tehran)
TEHRAN_BOUNDS = {
    'lat_min': 35.56,
    'lat_max': 35.80,
    'lon_min': 51.20,
    'lon_max': 51.60
}

# BTS (Base Transceiver Station) Parameters
BTS_CONFIG = {
    'count': 8,  # Number of BTS stations (5-10)
    'frequency_mhz': 1800,  # LTE Band 3
    'tx_power_dbm': 43,  # Typical macro cell transmit power
    'antenna_gain_dbi': 18,  # BTS antenna gain
    'antenna_height_m': 30,  # Typical cell tower height
    'min_separation_km': 1.5,  # Minimum distance between BTS
}

# Repeater Parameters
REPEATER_CONFIG = {
    'count': 2,  # Number of unauthorized repeaters (1-2)
    'gain_db': 70,  # Amplifier gain
    'min_distance_from_bts_km': 2.0,  # Minimum distance from serving BTS
    'max_distance_from_bts_km': 5.0,  # Maximum distance from serving BTS
}

# Drive Test Parameters
DRIVE_TEST_CONFIG = {
    'grid_spacing_m': 50,  # Spacing between measurement points
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
    'main_map_html': './output/tehran_network_map.html',
    'comparison_map_html': './output/detection_comparison.html',
    'anomaly_analysis_html': './output/anomaly_analysis.html',
}

# Random seed for reproducibility
RANDOM_SEED = 42
