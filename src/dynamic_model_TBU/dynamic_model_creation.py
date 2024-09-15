import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

# Physical constants
SPEED_OF_LIGHT = 299792458  # m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
EARTH_RADIUS_KM = 6371  # Earth's mean radius in km

def read_satellite_parameters(file_path):
    params = {}
    ground_stations = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('Ground_station_'):
                    # Extract ground station data
                    _, data = line.split(':', 1)
                    station_info = [x.strip() for x in data.split(',')]
                    if len(station_info) != 6:
                        raise ValueError(f"Invalid ground station data: {line}")
                    ground_station = {
                        'Name': station_info[0],
                        'Latitude_deg': float(station_info[1]),
                        'Longitude_deg': float(station_info[2]),
                        'Altitude_m': float(station_info[3]),
                        'Downlink_data_rate_Gbps': float(station_info[4]),
                        'Tracking_accuracy_arcsec': float(station_info[5])
                    }
                    ground_stations.append(ground_station)
                elif ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()

    params['ground_stations'] = ground_stations
    return params

def calculate_boltzmann_constant_db():
    return 10 * np.log10(BOLTZMANN_CONSTANT)

def calculate_system_noise_temperature_db(T):
    return 10 * np.log10(T)

def calculate_slant_range(earth_radius_km, satellite_height_km, elevation_deg):
    earth_radius_m = earth_radius_km * 1000  # Convert to meters
    satellite_height_m = satellite_height_km * 1000  # Convert to meters
    elevation_rad = np.radians(elevation_deg)

    # Calculate slant range using spherical Earth model
    slant_range_m = np.sqrt((earth_radius_m + satellite_height_m)**2 -
                            (earth_radius_m * np.sin(elevation_rad))**2) - earth_radius_m * np.cos(elevation_rad)
    return slant_range_m / 1000  # Return in km

def calculate_free_space_loss(distance_m, wavelength_m):
    if distance_m <= 0 or wavelength_m <= 0:
        return float('inf')
    loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)
    return max(loss_db, 0)  # Ensure non-negative loss

def calculate_antenna_gain(aperture_diameter_m, wavelength_m):
    if aperture_diameter_m <= 0 or wavelength_m <= 0:
        return 0
    gain = (np.pi * aperture_diameter_m / wavelength_m)**2
    return max(10 * np.log10(gain), 0)  # Ensure non-negative gain

def calculate_pointing_loss(pointing_error_rad, divergence_angle_rad):
    if divergence_angle_rad <= 0:
        return float('inf')
    loss = (2 * pointing_error_rad / divergence_angle_rad)**2
    return max(-10 * np.log10(np.exp(-loss)), 0)  # Ensure non-negative loss

def calculate_turbulence_loss(Cn2, wavelength_m, path_length_m):
    if Cn2 <= 0 or wavelength_m <= 0 or path_length_m <= 0:
        return 0
    k = 2 * np.pi / wavelength_m
    sigma_R2 = 1.23 * Cn2 * (k ** (7 / 6)) * (path_length_m ** (11 / 6))
    loss_db = max(10 * np.log10(np.exp(-sigma_R2)), 0)  # Ensure non-negative loss
    return loss_db

def calculate_atmospheric_attenuation(elevation_deg, visibility_km):
    q = 1.6  # Size distribution coefficient for rural areas
    wavelength_microns = 1.55  # Typically used wavelength in microns
    visibility_m = visibility_km * 1000
    alpha = 3.91 / visibility_m * (wavelength_microns / 0.55) ** (-q)
    attenuation_db_per_km = alpha * 4.343  # Convert Neper to dB
    path_length_km = 1 / np.sin(np.radians(elevation_deg))
    total_attenuation_db = attenuation_db_per_km * path_length_km
    return max(total_attenuation_db, 0)

def calculate_cloud_attenuation(cloud_cover_fraction):
    if cloud_cover_fraction < 0 or cloud_cover_fraction > 1:
        return 0
    max_attenuation = 30  # dB, maximum attenuation for thick clouds
    return max(cloud_cover_fraction * max_attenuation, 0)  # Ensure non-negative attenuation

def calculate_data_rate(tx_power_w, link_margin_db, tx_gain_db, rx_gain_db, total_loss_db, boltzmann_db, noise_temp_db):
    if tx_power_w <= 0:
        raise ValueError("Transmit power must be greater than 0.")
    tx_power_dbw = 10 * np.log10(tx_power_w)
    rx_power_dbw = tx_power_dbw + tx_gain_db + rx_gain_db - total_loss_db - link_margin_db
    noise_power_dbw_per_hz = boltzmann_db + noise_temp_db
    snr_db = rx_power_dbw - noise_power_dbw_per_hz
    # Assume spectral efficiency of 1 bit/s/Hz
    achievable_data_rate_bps = 10 ** (snr_db / 10)
    return max(achievable_data_rate_bps, 0)

def create_pipeline(start_date_str, end_date_str, params, ground_station):
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Extract satellite parameters
    satellite_height_km = float(params['Altitude_km'])
    wavelength_nm = float(params['Operational_wavelength_nm'])
    wavelength_m = wavelength_nm * 1e-9  # Convert nm to m
    optical_tx_power_w = float(params['Optical_Tx_power_W'])
    tx_aperture_mm = float(params['Optical_aperture_mm'])
    tx_aperture_m = tx_aperture_mm / 1000  # Convert mm to m
    link_margin_db = float(params['Link_margin_dB'])
    data_rate_bps = float(params['Downlink_data_rate_Gbps']) * 1e9  # Convert Gbps to bps

    # Hardcoded divergence angle: 15 μrad
    divergence_angle_rad = 15e-6  # 15 μrad in radians

    # Find the correct ground station data
    ground_station_data = next((station for station in params['ground_stations'] if station['Name'] == ground_station), None)
    if ground_station_data is None:
        raise ValueError(f"Ground station {ground_station} not found in parameters")

    # Extract ground station-specific parameters
    rx_aperture_mm = float(params['Optical_aperture_mm'])  # Assuming same as satellite
    rx_aperture_m = rx_aperture_mm / 1000  # Convert mm to m
    tracking_accuracy_arcsec = float(ground_station_data['Tracking_accuracy_arcsec'])
    # Convert tracking accuracy from arcseconds to radians
    tracking_accuracy_rad = tracking_accuracy_arcsec * (np.pi / (180 * 3600))  # 1 arcsec = 1/3600 degrees

    ground_station_data_rate_bps = ground_station_data['Downlink_data_rate_Gbps'] * 1e9  # Convert Gbps to bps
    required_data_rate_bps = min(data_rate_bps, ground_station_data_rate_bps)

    # Read combined data for the ground station
    combined_data_path = os.path.join(PROJECT_ROOT,'IAC-2024', 'data', 'output', 'data_integrator', f"{ground_station}_combined_data.csv")
    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] < end_date)]

    network_performance_data = []
    k_B_db = calculate_boltzmann_constant_db()
    Ts_db = calculate_system_noise_temperature_db(290)  # Standard noise temperature

    # Iterate over passes
    for _, pass_data in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Processing {ground_station} passes", leave=False):
        elevation = pass_data['Max Elevation']
        cloud_cover_fraction = pass_data['Cloud Cover']  # Expected to be between 0 (clear) and 1 (overcast)
        visibility_km = pass_data['Visibility']  # Visibility in km
        Cn2 = pass_data['Turbulence']  # Refractive index structure parameter

        # Calculate slant range
        S_km = calculate_slant_range(EARTH_RADIUS_KM, satellite_height_km, elevation)  # in km
        S_m = S_km * 1000  # Convert to meters

        # Ensure slant range is positive
        if S_km <= 0:
            logger.warning(f"Calculated negative slant range for elevation {elevation} degrees.")
            S_km = abs(S_km)
            S_m = abs(S_m)

        # Calculate free space loss
        Ls_db = calculate_free_space_loss(S_m, wavelength_m)

        # Calculate antenna gains
        Gt_db = calculate_antenna_gain(tx_aperture_m, wavelength_m)
        Gr_db = calculate_antenna_gain(rx_aperture_m, wavelength_m)

        # Calculate pointing losses
        Lp_t_db = calculate_pointing_loss(tracking_accuracy_rad, divergence_angle_rad)
        Lp_r_db = Lp_t_db  # Assuming same pointing error at receiver
        Gt_net_db = Gt_db - Lp_t_db
        Gr_net_db = Gr_db - Lp_r_db

        # Calculate atmospheric attenuation
        atmospheric_attenuation_db = calculate_atmospheric_attenuation(elevation, visibility_km)

        # Calculate turbulence loss
        turb_loss_db = calculate_turbulence_loss(Cn2, wavelength_m, S_m)

        # Calculate cloud attenuation
        cloud_attenuation_db = calculate_cloud_attenuation(cloud_cover_fraction)

        # Total losses
        total_loss_db = Ls_db + atmospheric_attenuation_db + turb_loss_db + cloud_attenuation_db

        # Calculate data rate
        try:
            achievable_data_rate_bps = calculate_data_rate(
                optical_tx_power_w, link_margin_db, Gt_net_db, Gr_net_db, total_loss_db, k_B_db, Ts_db)
        except ValueError as e:
            logger.error(f"Error calculating data rate: {e}")
            achievable_data_rate_bps = 0

        # Check network availability
        network_available = (cloud_cover_fraction < 0.5) and (achievable_data_rate_bps >= required_data_rate_bps)

        network_performance_data.append({
            'start_time': pass_data['Start'],
            'end_time': pass_data['End'],
            'duration': pass_data['Duration'],
            'max_elevation': elevation,
            'slant_range_km': S_km,
            'cloud_cover_fraction': cloud_cover_fraction,
            'atmospheric_attenuation_db': atmospheric_attenuation_db,
            'turbulence_cn2': Cn2,
            'turbulence_loss_db': turb_loss_db,
            'total_loss_db': total_loss_db,
            'achievable_data_rate_bps': achievable_data_rate_bps,
            'required_data_rate_bps': required_data_rate_bps,
            'network_available': network_available
        })

    final_df = pd.DataFrame(network_performance_data)
    output_file = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'dynamic_model', f"{ground_station}_final_dynamic_model_dataframe.csv")
    final_df.to_csv(output_file, index=False)
    return final_df


if __name__ == "__main__":
    params = read_satellite_parameters(os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt'))
    start_date_str = '2024-06-01'
    end_date_str = '2024-06-05'

    for ground_station in tqdm([station['Name'] for station in params['ground_stations']], desc="Processing ground stations"):
        logger.info(f"Processing ground station: {ground_station}")
        try:
            final_df = create_pipeline(start_date_str, end_date_str, params, ground_station)
            logger.info(f"Results for {ground_station}:")
            logger.info(final_df.head())
        except Exception as e:
            logger.error(f"An error occurred while processing {ground_station}: {e}")

    logger.info("Processing complete. Check the output directory for results.")
