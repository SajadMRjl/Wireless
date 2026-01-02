"""
Signal propagation model using Friis transmission equation.
Implements path loss calculation, signal combination, and noise modeling.
"""

from typing import List, Dict, Any
import numpy as np
from geopy.distance import geodesic


def friis_path_loss(distance_km: float, frequency_mhz: float, tx_gain_dbi: float = 0, rx_gain_dbi: float = 0) -> float:
    """
    Calculate path loss using Friis transmission equation.

    Formula: PL(d) = 20*log10(d) + 20*log10(f) + 32.45 - Gt - Gr

    Where:
    - d is distance in km
    - f is frequency in MHz
    - Gt is transmit antenna gain in dBi
    - Gr is receive antenna gain in dBi
    - 32.45 is the constant for km and MHz units

    Args:
        distance_km: Distance between transmitter and receiver (km)
        frequency_mhz: Frequency in MHz
        tx_gain_dbi: Transmit antenna gain in dBi (default: 0)
        rx_gain_dbi: Receive antenna gain in dBi (default: 0)

    Returns:
        Path loss in dB
    """
    if distance_km <= 0:
        return 0  # No path loss at zero distance

    path_loss = (
        20 * np.log10(distance_km) +
        20 * np.log10(frequency_mhz) +
        32.45 -
        tx_gain_dbi -
        rx_gain_dbi
    )

    return path_loss


def calculate_received_power(tx_power_dbm: float, distance_km: float, frequency_mhz: float,
                             tx_gain_dbi: float = 0, rx_gain_dbi: float = 0) -> float:
    """
    Calculate received power using Friis equation.

    Formula: Pr(d) = Pt - PL(d)

    Where:
    - Pt is transmit power in dBm
    - PL(d) is path loss at distance d

    Args:
        tx_power_dbm: Transmit power in dBm
        distance_km: Distance in kilometers
        frequency_mhz: Frequency in MHz
        tx_gain_dbi: Transmit antenna gain in dBi
        rx_gain_dbi: Receive antenna gain in dBi

    Returns:
        Received power in dBm
    """
    path_loss = friis_path_loss(distance_km, frequency_mhz, tx_gain_dbi, rx_gain_dbi)
    received_power = tx_power_dbm - path_loss

    return received_power


def add_log_normal_shadowing(rssi_dbm: float, sigma_db: float = 8) -> float:
    """
    Add log-normal shadowing to model realistic signal variations.

    Log-normal shadowing models the random variation in received signal
    due to obstacles, terrain, and atmospheric conditions.

    Args:
        rssi_dbm: Received signal strength in dBm
        sigma_db: Standard deviation of shadowing in dB (default: 8 dB for urban)

    Returns:
        RSSI with shadowing noise in dBm
    """
    noise = np.random.normal(0, sigma_db)
    return rssi_dbm + noise


def combine_signal_paths(power_list_dbm: List[float]) -> float:
    """
    Combine multiple signal paths (e.g., direct + repeater paths).

    CRITICAL: Signals must be combined in linear power domain, not dB domain.
    This function converts dBm to linear power (mW), sums them, and converts
    back to dBm.

    Formula:
    P_total_mW = sum(10^(P_i/10) for each P_i in dBm)
    P_total_dBm = 10 * log10(P_total_mW)

    Args:
        power_list_dbm: List of received powers in dBm from different paths

    Returns:
        Combined power in dBm

    Example:
        Direct path: -70 dBm
        Repeater path: -65 dBm
        Combined: 10*log10(10^(-70/10) + 10^(-65/10)) = -63.46 dBm
    """
    if not power_list_dbm:
        return -np.inf  # No signal

    # Convert each dBm value to linear power (mW)
    power_linear = sum(10 ** (p_dbm / 10) for p_dbm in power_list_dbm)

    # Convert back to dBm
    if power_linear <= 0:
        return -np.inf

    power_combined_dbm = 10 * np.log10(power_linear)

    return power_combined_dbm


def calculate_rssi_at_point(
    point_lat: float,
    point_lon: float,
    bts_list: List[Dict[str, Any]],
    repeater_list: List[Dict[str, Any]],
    frequency_mhz: float,
    rx_gain_dbi: float = 0,
    add_noise: bool = True,
    sigma_noise: float = 8
) -> Dict[str, float]:
    """
    Calculate RSSI from all BTS at a measurement point.

    Considers both direct paths from BTS and indirect paths via repeaters.
    Implements dual-path propagation model.

    Args:
        point_lat, point_lon: Measurement point coordinates
        bts_list: List of BTS dictionaries
        repeater_list: List of repeater dictionaries
        frequency_mhz: Operating frequency
        rx_gain_dbi: Receiver antenna gain
        add_noise: Whether to add log-normal shadowing
        sigma_noise: Noise standard deviation in dB

    Returns:
        Dictionary: {bts_id: rssi_dbm} for all BTS
    """
    rssi_per_bts = {}

    for bts in bts_list:
        # Calculate direct path from BTS to measurement point
        dist_direct = geodesic((bts['lat'], bts['lon']), (point_lat, point_lon)).km

        rssi_direct = calculate_received_power(
            tx_power_dbm=bts['tx_power_dbm'],
            distance_km=dist_direct,
            frequency_mhz=frequency_mhz,
            tx_gain_dbi=bts['antenna_gain_dbi'],
            rx_gain_dbi=rx_gain_dbi
        )

        # Calculate indirect paths via repeaters serving this BTS
        rssi_indirect_paths = []

        for repeater in repeater_list:
            if repeater.get('serving_bts_id') == bts['id']:
                # Path: BTS -> Repeater
                dist_bts_to_rep = geodesic((bts['lat'], bts['lon']), (repeater['lat'], repeater['lon'])).km

                rssi_at_repeater = calculate_received_power(
                    tx_power_dbm=bts['tx_power_dbm'],
                    distance_km=dist_bts_to_rep,
                    frequency_mhz=frequency_mhz,
                    tx_gain_dbi=bts['antenna_gain_dbi'],
                    rx_gain_dbi=0  # Repeater input
                )

                # Repeater amplifies the signal
                rssi_repeater_output = rssi_at_repeater + repeater['gain_db']

                # Path: Repeater -> Measurement point
                dist_rep_to_point = geodesic((repeater['lat'], repeater['lon']), (point_lat, point_lon)).km

                # Calculate path loss from repeater to point
                path_loss_rep_to_point = friis_path_loss(
                    distance_km=dist_rep_to_point,
                    frequency_mhz=frequency_mhz,
                    tx_gain_dbi=0,  # Repeater antenna gain (assume 0)
                    rx_gain_dbi=rx_gain_dbi
                )

                rssi_via_repeater = rssi_repeater_output - path_loss_rep_to_point
                rssi_indirect_paths.append(rssi_via_repeater)

        # Combine all paths (direct + indirect) using logarithmic sum
        all_paths = [rssi_direct] + rssi_indirect_paths
        rssi_combined = combine_signal_paths(all_paths)

        # Add log-normal shadowing noise
        if add_noise:
            rssi_with_noise = add_log_normal_shadowing(rssi_combined, sigma_noise)
        else:
            rssi_with_noise = rssi_combined

        rssi_per_bts[bts['id']] = rssi_with_noise

    return rssi_per_bts
