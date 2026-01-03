"""
Signal propagation model for wireless network simulation.
Implements urban path loss, signal combination, and noise modeling.
"""

from typing import List, Dict, Any
import numpy as np
from geopy.distance import geodesic

# RSSI limits (applied once at the end, not repeated)
RSSI_MIN = -110  # Noise floor / sensitivity limit
RSSI_MAX = -50   # Maximum realistic received signal (very close to tower)


def clamp_rssi(rssi_dbm: float) -> float:
    """Clamp RSSI to realistic range. Single source of truth for limits."""
    return max(RSSI_MIN, min(RSSI_MAX, rssi_dbm))


def urban_path_loss(distance_km: float, frequency_mhz: float, hb_m: float = 30.0) -> float:
    """
    Calculate path loss using Cost-231 Hata urban model.
    
    Full formula: PL = 46.3 + 33.9*log10(f) - 13.82*log10(hb) + (44.9 - 6.55*log10(hb))*log10(d) + Cm
    
    Args:
        distance_km: Distance in km (minimum 0.01 km = 10m)
        frequency_mhz: Frequency in MHz (valid: 1500-2000 MHz)
        hb_m: Base station antenna height in meters (default: 30m)
    
    Returns:
        Path loss in dB (always positive)
    """
    distance_km = max(distance_km, 0.01)  # Minimum 10m
    
    # Cost-231 Hata model for urban macro cell
    # With height corrections included
    log_hb = np.log10(hb_m)
    log_f = np.log10(frequency_mhz)
    log_d = np.log10(distance_km)
    
    path_loss = (
        46.3 +
        33.9 * log_f -           # Frequency term
        13.82 * log_hb +         # BTS height correction (reduces loss)
        (44.9 - 6.55 * log_hb) * log_d +  # Distance term with height factor
        3.0                      # Metropolitan correction
    )
    
    return max(0, path_loss)


def calculate_received_power(
    tx_power_dbm: float,
    distance_km: float,
    frequency_mhz: float,
    tx_gain_dbi: float = 0,
    rx_gain_dbi: float = 0
) -> float:
    """
    Calculate received power (RSSI) at distance.
    
    Formula: Pr = Pt + Gt + Gr - PL
    
    Returns:
        Received power in dBm (NOT clamped - caller decides when to clamp)
    """
    path_loss = urban_path_loss(distance_km, frequency_mhz)
    received_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss
    return received_power


def add_log_normal_shadowing(rssi_dbm: float, sigma_db: float = 8) -> float:
    """Add log-normal shadowing noise (models buildings, terrain)."""
    noise = np.random.normal(0, sigma_db)
    return rssi_dbm + noise


def combine_signal_paths(power_list_dbm: List[float]) -> float:
    """
    Combine multiple signal paths in linear power domain.
    
    Signals add as power (mW), not as dB.
    """
    if not power_list_dbm:
        return RSSI_MIN
    
    # Filter out very weak signals that don't contribute
    valid_powers = [p for p in power_list_dbm if p > RSSI_MIN]
    if not valid_powers:
        return RSSI_MIN
    
    # Convert to linear, sum, convert back
    power_linear = sum(10 ** (p / 10) for p in valid_powers)
    return 10 * np.log10(power_linear)


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
    
    Considers direct path + any repeater paths, combines them properly.
    Clamping happens ONCE at the end.
    """
    rssi_per_bts = {}

    for bts in bts_list:
        signal_paths = []
        
        # === Direct path: BTS -> Point ===
        dist_direct = geodesic((bts['lat'], bts['lon']), (point_lat, point_lon)).km
        rssi_direct = calculate_received_power(
            tx_power_dbm=bts['tx_power_dbm'],
            distance_km=dist_direct,
            frequency_mhz=frequency_mhz,
            tx_gain_dbi=bts['antenna_gain_dbi'],
            rx_gain_dbi=rx_gain_dbi
        )
        signal_paths.append(rssi_direct)
        
        # === Indirect paths via repeaters ===
        for repeater in repeater_list:
            # Repeaters repeat signals from ALL BTS stations
            
            # BTS -> Repeater (repeater's input antenna assumed 0 dBi)
            dist_bts_rep = geodesic((bts['lat'], bts['lon']), (repeater['lat'], repeater['lon'])).km

            rssi_at_repeater = calculate_received_power(
                tx_power_dbm=bts['tx_power_dbm'],
                distance_km=dist_bts_rep,
                frequency_mhz=frequency_mhz,
                tx_gain_dbi=bts['antenna_gain_dbi'],
                rx_gain_dbi=repeater['gain_rx_db']  # Repeater input antenna
            )
            
            # Repeater -> Point (using lower antenna height for repeater)
            dist_rep_point = geodesic((repeater['lat'], repeater['lon']), (point_lat, point_lon)).km

            rssi_via_repeater = calculate_received_power(
                tx_power_dbm=rssi_at_repeater,
                distance_km=dist_rep_point,
                frequency_mhz=frequency_mhz,
                tx_gain_dbi=repeater['gain_tx_db'],
                rx_gain_dbi=rx_gain_dbi
            )

            if rssi_via_repeater < RSSI_MIN:
                continue

            signal_paths.append(rssi_via_repeater)
        
        # === Combine all paths ===
        rssi_combined = combine_signal_paths(signal_paths)
        
        # === Add noise ===
        if add_noise:
            rssi_combined = add_log_normal_shadowing(rssi_combined, sigma_noise)
        
        # === Single clamp at the end ===
        rssi_per_bts[bts['id']] = clamp_rssi(rssi_combined)

    return rssi_per_bts
