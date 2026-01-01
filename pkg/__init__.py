"""
Wireless Repeater Detection Module

This package contains all the core functionality for simulating
and detecting unauthorized repeaters in mobile networks.
"""

__version__ = "1.0.0"

# Make key functions easily accessible
from .bts_generator import (
    generate_bts_stations,
    generate_repeaters,
    save_bts_to_csv,
    save_repeaters_to_csv,
    load_bts_from_csv,
    load_repeaters_from_csv
)

from .propagation import (
    friis_path_loss,
    calculate_received_power,
    add_log_normal_shadowing,
    combine_signal_paths,
    calculate_rssi_at_point
)

from .drive_test_simulator import (
    generate_measurement_grid,
    simulate_drive_test,
    save_measurements_to_csv,
    load_measurements_from_csv
)

from .detection import (
    detect_repeaters,
    validate_detection,
    build_expected_coverage_map
)

from .visualization import (
    create_main_detection_map,
    create_comparison_maps,
    plot_residual_histogram,
    plot_detection_metrics
)

__all__ = [
    # BTS Generation
    'generate_bts_stations',
    'generate_repeaters',
    'save_bts_to_csv',
    'save_repeaters_to_csv',
    'load_bts_from_csv',
    'load_repeaters_from_csv',

    # Propagation
    'friis_path_loss',
    'calculate_received_power',
    'add_log_normal_shadowing',
    'combine_signal_paths',
    'calculate_rssi_at_point',

    # Drive Test
    'generate_measurement_grid',
    'simulate_drive_test',
    'save_measurements_to_csv',
    'load_measurements_from_csv',

    # Detection
    'detect_repeaters',
    'validate_detection',
    'build_expected_coverage_map',

    # Visualization
    'create_main_detection_map',
    'create_comparison_maps',
    'plot_residual_histogram',
    'plot_detection_metrics',
]
