"""
Repeater detection algorithm using statistical anomaly detection.
Implements z-score based anomaly detection and DBSCAN spatial clustering.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
import config
from propagation import calculate_rssi_at_point


def build_expected_coverage_map(bts_list: List[Dict[str, Any]], measurement_points: List[Union[Dict[str, Any], Tuple[float, float]]], add_noise: bool = False) -> List[Dict[str, Any]]:
    """
    Build expected coverage map without repeaters.

    Calculates predicted RSSI at all measurement points assuming
    only direct paths from BTS (no repeaters).

    Args:
        bts_list: List of BTS dictionaries
        measurement_points: List of (lat, lon) tuples or measurement dicts
        add_noise: Whether to add noise to predictions (default: False)

    Returns:
        List of dictionaries with predicted RSSI for each point
    """
    print("Building expected coverage map (without repeaters)...")

    frequency_mhz = config.BTS_CONFIG['frequency_mhz']
    rx_gain_dbi = config.DRIVE_TEST_CONFIG['rx_antenna_gain_dbi']
    sigma_noise = config.NOISE_CONFIG['log_normal_sigma_db']

    predictions = []

    for point in measurement_points:
        # Extract coordinates
        if isinstance(point, dict):
            lat, lon = point['lat'], point['lon']
        else:
            lat, lon = point

        # Calculate expected RSSI (NO repeaters)
        rssi_per_bts = calculate_rssi_at_point(
            point_lat=lat,
            point_lon=lon,
            bts_list=bts_list,
            repeater_list=[],  # CRITICAL: No repeaters for expected coverage
            frequency_mhz=frequency_mhz,
            rx_gain_dbi=rx_gain_dbi,
            add_noise=add_noise,
            sigma_noise=sigma_noise
        )

        prediction = {
            'lat': lat,
            'lon': lon,
            **{f'predicted_rssi_{bts_id}': rssi
               for bts_id, rssi in rssi_per_bts.items()}
        }

        predictions.append(prediction)

    print(f"Generated predictions for {len(predictions)} points")
    return predictions


def calculate_residuals(measurements: List[Dict[str, Any]], predictions: List[Dict[str, Any]], bts_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate residuals (actual - predicted RSSI) for all measurements.

    Args:
        measurements: List of actual measurement dictionaries
        predictions: List of prediction dictionaries
        bts_list: List of BTS dictionaries

    Returns:
        List of dictionaries with residuals for each point and BTS
    """
    print("Calculating residuals (actual - predicted)...")

    residuals = []

    for meas, pred in zip(measurements, predictions):
        residual_dict = {
            'lat': meas['lat'],
            'lon': meas['lon']
        }

        for bts in bts_list:
            bts_id = bts['id']
            actual_key = f'rssi_{bts_id}'
            predicted_key = f'predicted_rssi_{bts_id}'

            if actual_key in meas and predicted_key in pred:
                actual_rssi = meas[actual_key]
                predicted_rssi = pred[predicted_key]
                residual = actual_rssi - predicted_rssi

                residual_dict[f'residual_{bts_id}'] = residual

        residuals.append(residual_dict)

    return residuals


def detect_anomalies_statistical(residuals: List[Dict[str, Any]], bts_list: List[Dict[str, Any]], z_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Detect anomalies using statistical z-score method.

    Calculates z-score for each residual and flags points with
    z-score above threshold as anomalous.

    Args:
        residuals: List of residual dictionaries
        bts_list: List of BTS dictionaries
        z_threshold: Z-score threshold (default: from config)

    Returns:
        List of anomaly dictionaries with z-scores
    """
    if z_threshold is None:
        z_threshold = config.DETECTION_CONFIG['z_score_threshold']

    print(f"Detecting anomalies using z-score threshold: {z_threshold}")

    # Collect all residuals across all points and BTS
    all_residuals = []
    for res_dict in residuals:
        for bts in bts_list:
            bts_id = bts['id']
            residual_key = f'residual_{bts_id}'
            if residual_key in res_dict:
                all_residuals.append(res_dict[residual_key])

    # Calculate mean and std of all residuals
    mean_residual = np.mean(all_residuals)
    std_residual = np.std(all_residuals)

    print(f"Residual statistics: mean={mean_residual:.2f} dB, std={std_residual:.2f} dB")

    # Calculate z-scores and detect anomalies
    anomalies = []

    for res_dict in residuals:
        lat = res_dict['lat']
        lon = res_dict['lon']

        max_z_score = -np.inf
        anomalous_bts_id = None

        for bts in bts_list:
            bts_id = bts['id']
            residual_key = f'residual_{bts_id}'

            if residual_key in res_dict:
                residual = res_dict[residual_key]

                # Calculate z-score
                z_score = (residual - mean_residual) / std_residual

                # Track maximum z-score for this point
                if z_score > max_z_score:
                    max_z_score = z_score
                    anomalous_bts_id = bts_id

        # If max z-score exceeds threshold, mark as anomaly
        if max_z_score > z_threshold:
            anomaly = {
                'lat': lat,
                'lon': lon,
                'z_score': max_z_score,
                'bts_id': anomalous_bts_id,
                'residual': res_dict[f'residual_{anomalous_bts_id}']
            }
            anomalies.append(anomaly)

    print(f"Detected {len(anomalies)} anomalous measurement points")
    return anomalies


def cluster_anomalies_dbscan(anomalies: List[Dict[str, Any]], eps_m: Optional[float] = None, min_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Cluster anomalous points spatially using DBSCAN.

    Groups nearby anomalies into clusters, filtering out isolated noise.

    Args:
        anomalies: List of anomaly dictionaries
        eps_m: Maximum distance between points in cluster (meters)
        min_samples: Minimum points to form a cluster

    Returns:
        List of cluster dictionaries with member points
    """
    if eps_m is None:
        eps_m = config.DETECTION_CONFIG['dbscan_eps_m']
    if min_samples is None:
        min_samples = config.DETECTION_CONFIG['dbscan_min_samples']

    print(f"Clustering anomalies with DBSCAN (eps={eps_m}m, min_samples={min_samples})...")

    if len(anomalies) < min_samples:
        print(f"Warning: Too few anomalies ({len(anomalies)}) for clustering")
        return []

    # Extract coordinates
    coords = np.array([[a['lat'], a['lon']] for a in anomalies])

    # Convert eps from meters to degrees (approximate)
    # At Tehran: 1 degree lat ≈ 111 km
    eps_degrees = (eps_m / 1000) / 111

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps_degrees, min_samples=min_samples)
    labels = dbscan.fit_predict(coords)

    # Group anomalies by cluster
    clusters = {}
    for anomaly, label in zip(anomalies, labels):
        if label == -1:
            # Noise point (not in any cluster)
            continue

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(anomaly)

    print(f"Found {len(clusters)} anomaly clusters")

    # Convert to list of cluster dictionaries
    cluster_list = []
    for cluster_id, members in clusters.items():
        cluster_dict = {
            'cluster_id': cluster_id,
            'num_members': len(members),
            'members': members
        }
        cluster_list.append(cluster_dict)

    return cluster_list


def localize_repeater_centroid(cluster: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Estimate repeater location using weighted centroid of anomaly cluster.

    Weights each anomaly point by its z-score magnitude.

    Args:
        cluster: Cluster dictionary with member anomalies

    Returns:
        Dictionary with estimated location and metrics
    """
    members = cluster['members']

    if not members:
        return None

    # Extract latitudes, longitudes, and weights (z-scores)
    lats = [m['lat'] for m in members]
    lons = [m['lon'] for m in members]
    weights = [m['z_score'] for m in members]

    # Calculate weighted centroid
    total_weight = sum(weights)
    weighted_lat = sum(lat * w for lat, w in zip(lats, weights)) / total_weight
    weighted_lon = sum(lon * w for lon, w in zip(lons, weights)) / total_weight

    # Calculate confidence metrics
    mean_z_score = np.mean(weights)
    max_z_score = np.max(weights)

    return {
        'lat': weighted_lat,
        'lon': weighted_lon,
        'confidence_mean_z': mean_z_score,
        'confidence_max_z': max_z_score,
        'cluster_size': len(members),
        'method': 'weighted_centroid'
    }


def detect_repeaters(measurements: List[Dict[str, Any]], bts_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main repeater detection pipeline.

    Combines all detection steps:
    1. Build expected coverage map (no repeaters)
    2. Calculate residuals
    3. Statistical anomaly detection
    4. Spatial clustering
    5. Repeater localization

    Args:
        measurements: List of measurement dictionaries
        bts_list: List of BTS dictionaries

    Returns:
        List of detected repeater dictionaries
    """
    print("\n" + "="*60)
    print("REPEATER DETECTION PIPELINE")
    print("="*60)

    # Step 1: Build expected coverage map
    predictions = build_expected_coverage_map(
        bts_list=bts_list,
        measurement_points=measurements,
        add_noise=False  # Don't add noise to predictions
    )

    # Step 2: Calculate residuals
    residuals = calculate_residuals(measurements, predictions, bts_list)

    # Step 3: Detect anomalies using z-score
    anomalies = detect_anomalies_statistical(residuals, bts_list)

    if not anomalies:
        print("\nNo anomalies detected. No repeaters found.")
        return []

    # Step 4: Cluster anomalies spatially
    clusters = cluster_anomalies_dbscan(anomalies)

    if not clusters:
        print("\nNo significant anomaly clusters found. No repeaters detected.")
        return []

    # Step 5: Localize repeater for each cluster
    print(f"\nLocalizing repeaters for {len(clusters)} clusters...")
    detected_repeaters = []

    for cluster in clusters:
        repeater_location = localize_repeater_centroid(cluster)

        if repeater_location:
            repeater_location['detection_id'] = f"DETECTED_{cluster['cluster_id']}"
            detected_repeaters.append(repeater_location)

            print(f"\nDetected Repeater #{cluster['cluster_id']}:")
            print(f"  Location: ({repeater_location['lat']:.6f}, " +
                  f"{repeater_location['lon']:.6f})")
            print(f"  Confidence (mean z-score): {repeater_location['confidence_mean_z']:.2f}")
            print(f"  Cluster size: {repeater_location['cluster_size']} points")

    print("\n" + "="*60)
    print(f"DETECTION COMPLETE: {len(detected_repeaters)} repeaters detected")
    print("="*60 + "\n")

    return detected_repeaters


def validate_detection(detected_repeaters: List[Dict[str, Any]], actual_repeaters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate detection results against ground truth.

    Calculates detection error (distance) between detected and actual
    repeater locations.

    Args:
        detected_repeaters: List of detected repeater dictionaries
        actual_repeaters: List of actual repeater dictionaries

    Returns:
        Validation metrics dictionary
    """
    from geopy.distance import geodesic

    print("\n" + "="*60)
    print("DETECTION VALIDATION")
    print("="*60)

    if not detected_repeaters:
        print("No repeaters detected!")
        return {
            'num_detected': 0,
            'num_actual': len(actual_repeaters),
            'detection_rate': 0.0
        }

    # Match each detected repeater to nearest actual repeater
    matches = []

    for detected in detected_repeaters:
        detected_loc = (detected['lat'], detected['lon'])

        min_distance_km = np.inf
        closest_actual = None

        for actual in actual_repeaters:
            actual_loc = (actual['lat'], actual['lon'])
            distance_km = geodesic(detected_loc, actual_loc).km

            if distance_km < min_distance_km:
                min_distance_km = distance_km
                closest_actual = actual

        match = {
            'detected': detected,
            'actual': closest_actual,
            'error_km': min_distance_km,
            'error_m': min_distance_km * 1000
        }
        matches.append(match)

        print(f"\nDetected Repeater: {detected['detection_id']}")
        print(f"  Detected at: ({detected['lat']:.6f}, {detected['lon']:.6f})")
        if closest_actual:
            print(f"  Actual at:   ({closest_actual['lat']:.6f}, " +
                  f"{closest_actual['lon']:.6f})")
            print(f"  Error: {min_distance_km*1000:.1f} meters")
            print(f"  Confidence: {detected['confidence_mean_z']:.2f}")

    # Calculate metrics
    errors_m = [m['error_m'] for m in matches]
    metrics = {
        'num_detected': len(detected_repeaters),
        'num_actual': len(actual_repeaters),
        'detection_rate': len(detected_repeaters) / len(actual_repeaters) if actual_repeaters else 0,
        'mean_error_m': np.mean(errors_m),
        'median_error_m': np.median(errors_m),
        'max_error_m': np.max(errors_m),
        'matches': matches
    }

    print(f"\n{'='*60}")
    print(f"Detection Rate: {metrics['detection_rate']*100:.1f}% " +
          f"({metrics['num_detected']}/{metrics['num_actual']})")
    print(f"Mean Error: {metrics['mean_error_m']:.1f} meters")
    print(f"Median Error: {metrics['median_error_m']:.1f} meters")
    print(f"{'='*60}\n")

    return metrics


if __name__ == '__main__':
    from bts_generator import load_bts_from_csv, load_repeaters_from_csv
    from drive_test_simulator import load_measurements_from_csv

    # Load data
    print("Loading data...")
    bts_stations = load_bts_from_csv()
    actual_repeaters = load_repeaters_from_csv()
    measurements = load_measurements_from_csv()

    # Run detection
    detected_repeaters = detect_repeaters(measurements, bts_stations)

    # Validate results
    if actual_repeaters:
        metrics = validate_detection(detected_repeaters, actual_repeaters)
