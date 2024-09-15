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
    """
    Read satellite parameters and ground station data from a file.

    Parameters:
    - file_path: Path to the satellite parameters file.

    Returns:
    - params: Dictionary containing satellite parameters.
    """
    params = {}
    ground_stations = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('Ground_station_'):
                    _, data = line.split(':', 1)
                    ground_stations.append([x.strip() for x in data.split(',')])
                elif ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()

    params['ground_stations'] = ground_stations
    return params

def calculate_boltzmann_constant_db():
    """
    Calculate Boltzmann constant in dBW/K/Hz.

    Returns:
    - Boltzmann constant in dB.
    """
    return 10 * np.log10(BOLTZMANN_CONSTANT)

def calculate_system_noise_temperature_db(T):
    """
    Calculate system noise temperature in dB.

    Parameters:
    - T: System noise temperature in Kelvin.

    Returns:
    - System noise temperature in dB.
    """
    return 10 * np.log10(T)

def calculate_slant_range(earth_radius_km, satellite_height_km, elevation_deg):
    """
    Calculate the slant range between satellite and ground station.

    Parameters:
    - earth_radius_km: Earth's radius in km.
    - satellite_height_km: Satellite's height above Earth's surface in km.
    - elevation_deg: Elevation angle in degrees.

    Returns:
    - slant_range_km: Slant range in km.
    """
    earth_radius_m = earth_radius_km * 1000  # Convert to meters
    satellite_height_m = satellite_height_km * 1000  # Convert to meters
    elevation_rad = np.radians(elevation_deg)

    Re_plus_h = earth_radius_m + satellite_height_m
    Re_cos_elev = earth_radius_m * np.cos(elevation_rad)

    slant_range_m = np.sqrt(Re_plus_h**2 - Re_cos_elev**2)
    return slant_range_m / 1000  # Return in km

def calculate_free_space_loss(distance_m, wavelength_m):
    """
    Calculate free space loss.

    Parameters:
    - distance_m: Distance between transmitter and receiver in meters.
    - wavelength_m: Wavelength in meters.

    Returns:
    - loss_db: Free space loss in dB.
    """
    if distance_m <= 0 or wavelength_m <= 0:
        return float('inf')
    loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)
    return max(loss_db, 0)  # Ensure non-negative loss

def calculate_antenna_gain(diameter_m, wavelength_m, efficiency):
    """
    Calculate antenna gain.

    Parameters:
    - diameter_m: Antenna diameter in meters.
    - wavelength_m: Wavelength in meters.
    - efficiency: Antenna efficiency (0 < efficiency <= 1).

    Returns:
    - gain_db: Antenna gain in dB.
    """
    if diameter_m <= 0 or wavelength_m <= 0 or not (0 < efficiency <= 1):
        return 0
    gain = efficiency * (np.pi * diameter_m / wavelength_m)**2
    return max(10 * np.log10(gain), 0)  # Ensure non-negative gain

def calculate_pointing_loss(pointing_error_rad, wavelength_m, aperture_diameter_m):
    """
    Calculate pointing loss due to antenna misalignment.

    Parameters:
    - pointing_error_rad: Pointing error in radians.
    - wavelength_m: Wavelength in meters.
    - aperture_diameter_m: Antenna aperture diameter in meters.

    Returns:
    - loss_db: Pointing loss in dB.
    """
    if pointing_error_rad <= 0 or wavelength_m <= 0 or aperture_diameter_m <= 0:
        return 0
    loss = (np.pi * pointing_error_rad * aperture_diameter_m / wavelength_m)**2
    return max(-10 * np.log10(np.exp(-loss)), 0)  # Ensure non-negative loss

def calculate_turbulence_loss(Cn2, wavelength_m, path_length_m):
    """
    Calculate turbulence loss.

    Parameters:
    - Cn2: Refractive index structure parameter (m^(-2/3)).
    - wavelength_m: Wavelength in meters.
    - path_length_m: Path length in meters.

    Returns:
    - loss_db: Turbulence loss in dB.
    """
    if Cn2 <= 0 or wavelength_m <= 0 or path_length_m <= 0:
        return 0
    k = 2 * np.pi / wavelength_m
    sigma_R2 = 1.23 * Cn2 * (k ** (7 / 6)) * (path_length_m ** (11 / 6))
    loss_db = max(10 * np.log10(np.exp(-sigma_R2)), 0)  # Ensure non-negative loss
    return loss_db

def calculate_atmospheric_attenuation(wavelength_nm, path_length_km):
    """
    Calculate atmospheric attenuation.

    Parameters:
    - wavelength_nm: Wavelength in nanometers.
    - path_length_km: Path length in kilometers.

    Returns:
    - attenuation_db: Atmospheric attenuation in dB.
    """
    # Attenuation values per km for different wavelengths (example values)
    fso_windows = {
        850: 0.15,
        1060: 0.1,
        1250: 0.08,
        1550: 0.05,
    }
    closest_window = min(fso_windows.keys(), key=lambda x: abs(x - wavelength_nm))
    attenuation_db_per_km = fso_windows[closest_window]
    total_attenuation_db = attenuation_db_per_km * path_length_km
    return max(total_attenuation_db, 0)  # Ensure non-negative attenuation

def calculate_cloud_attenuation(cloud_cover_fraction):
    """
    Calculate cloud attenuation based on cloud cover.

    Parameters:
    - cloud_cover_fraction: Cloud cover as a fraction (0 to 1).

    Returns:
    - attenuation_db: Cloud attenuation in dB.
    """
    if cloud_cover_fraction < 0 or cloud_cover_fraction > 1:
        return 0
    max_attenuation = 30  # dB, maximum attenuation for very thick clouds
    return max(cloud_cover_fraction * max_attenuation, 0)  # Ensure non-negative attenuation

def calculate_data_rate(tx_power_w, link_margin_db, tx_gain_db, rx_gain_db, total_loss_db, boltzmann_db, noise_temp_db, data_rate_bps):
    """
    Calculate Eb/N0 and achievable data rate based on link budget parameters.

    Parameters:
    - tx_power_w: Transmit power in watts.
    - link_margin_db: Link margin in dB.
    - tx_gain_db: Transmit antenna gain in dB.
    - rx_gain_db: Receive antenna gain in dB.
    - total_loss_db: Total link loss in dB.
    - boltzmann_db: Boltzmann constant in dBW/K/Hz.
    - noise_temp_db: System noise temperature in dB.
    - data_rate_bps: Data rate in bps.

    Returns:
    - eb_n0_db: Energy per bit to noise power spectral density ratio in dB.
    - achievable_data_rate_bps: Achievable data rate in bps.
    """
    if tx_power_w <= 0:
        raise ValueError("Transmit power must be greater than 0.")
    if data_rate_bps <= 0:
        raise ValueError("Data rate must be greater than 0.")
    tx_power_dbw = 10 * np.log10(tx_power_w)
    rx_power_dbw = tx_power_dbw + tx_gain_db + rx_gain_db - total_loss_db - link_margin_db
    bandwidth_hz = data_rate_bps  # Assuming bandwidth equals data rate
    noise_power_dbw = boltzmann_db + noise_temp_db + 10 * np.log10(bandwidth_hz)
    snr_db = rx_power_dbw - noise_power_dbw
    eb_n0_db = snr_db  # Since data_rate_bps / bandwidth_hz = 1
    snr_linear = 10 ** (snr_db / 10)
    if snr_linear <= 0:
        achievable_data_rate_bps = 0
    else:
        achievable_data_rate_bps = bandwidth_hz * np.log2(1 + snr_linear)
    return eb_n0_db, max(achievable_data_rate_bps, 0)

def create_pipeline(start_date_str, end_date_str, params, ground_station):
    """
    Create a data processing pipeline for satellite communication.

    Parameters:
    - start_date_str: Start date as a string (e.g., '2023-06-01').
    - end_date_str: End date as a string (e.g., '2023-06-05').
    - params: Dictionary containing satellite parameters.
    - ground_station: Name of the ground station.

    Returns:
    - final_df: Pandas DataFrame containing the final results.
    """
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Extract satellite parameters
    satellite_height_km = float(params['Altitude_km'])
    wavelength_nm = float(params['Operational_wavelength_nm'])
    wavelength_m = wavelength_nm * 1e-9  # Convert nm to m
    downlink_data_rate_gbps = float(params['Downlink_data_rate_Gbps'])
    optical_tx_power_w = float(params['Optical_Tx_power_W'])
    optical_aperture_mm = float(params['Optical_aperture_mm'])
    optical_aperture_m = optical_aperture_mm / 1000  # Convert mm to m
    link_margin_db = float(params['Link_margin_dB'])

    # Find the correct ground station data
    ground_station_data = next((station for station in params['ground_stations'] if station[0] == ground_station), None)
    if ground_station_data is None:
        raise ValueError(f"Ground station {ground_station} not found in parameters")

    # Extract ground station-specific tracking accuracy
    tracking_accuracy_arcsec = float(ground_station_data[5])
    # Convert tracking accuracy from arcseconds to radians
    tracking_accuracy_rad = tracking_accuracy_arcsec * (np.pi / (180 * 3600))  # 1 arcsec = 1/3600 degrees

    # Read combined data for the ground station
    combined_data_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator', f"{ground_station}_combined_data.csv")
    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] < end_date)]

    network_performance_data = []
    k_B_db = calculate_boltzmann_constant_db()
    Ts_db = calculate_system_noise_temperature_db(135)  # Assuming a system noise temperature of 135K

    # Iterate over passes
    for _, pass_data in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Processing {ground_station} passes", leave=False):
        elevation = pass_data['Max Elevation']
        cloud_cover_value = pass_data['Cloud Cover']  # Expected to be between 0 (overcast) and 2.0 (clear)
        Cn2 = pass_data['Turbulence']  # Assuming the 'Turbulence' column contains Cn^2 values

        # Adjust cloud cover fraction
        if cloud_cover_value < 0 or cloud_cover_value > 2.0:
            cloud_cover_value = max(0, min(cloud_cover_value, 2.0))
        cloud_cover_fraction = 1 - (cloud_cover_value / 2.0)  # Normalize and invert (0: clear, 1: overcast)

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
        efficiency = 0.55  # Assuming 0.55 efficiency
        Gr_db = calculate_antenna_gain(optical_aperture_m, wavelength_m, efficiency)
        Gt_db = Gr_db  # Assuming same antenna at transmitter and receiver

        # Calculate pointing losses
        Lp_r_db = calculate_pointing_loss(tracking_accuracy_rad, wavelength_m, optical_aperture_m)
        Gr_net_db = Gr_db - Lp_r_db
        Gt_net_db = Gt_db - Lp_r_db  # Assuming same pointing loss at transmitter

        # Calculate atmospheric attenuation
        atmospheric_attenuation_db = calculate_atmospheric_attenuation(wavelength_nm, S_km)

        # Calculate turbulence loss
        turb_loss_db = calculate_turbulence_loss(Cn2, wavelength_m, S_m)

        # Calculate cloud attenuation
        cloud_attenuation_db = calculate_cloud_attenuation(cloud_cover_fraction)

        # Total losses
        total_loss_db = Ls_db + atmospheric_attenuation_db + turb_loss_db + cloud_attenuation_db

        # Calculate data rate
        data_rate_bps = downlink_data_rate_gbps * 1e9  # Convert Gbps to bps
        try:
            eb_n0_db, achievable_data_rate_bps = calculate_data_rate(
                optical_tx_power_w, link_margin_db, Gt_net_db, Gr_net_db, total_loss_db, k_B_db, Ts_db, data_rate_bps)
        except ValueError as e:
            logger.error(f"Error calculating data rate: {e}")
            eb_n0_db = float('-inf')
            achievable_data_rate_bps = 0

        # Check network availability
        # Network is available if cloud cover is clear (value of 2.0) and data rate meets threshold
        network_available = (cloud_cover_value == 2.0) and (achievable_data_rate_bps >= data_rate_bps * 0.9)

        network_performance_data.append({
            'start_time': pass_data['Start'],
            'end_time': pass_data['End'],
            'duration': pass_data['Duration'],
            'max_elevation': elevation,
            'slant_range_km': S_km,
            'cloud_cover_value': cloud_cover_value,
            'atmospheric_attenuation_db': atmospheric_attenuation_db,
            'turbulence_cn2': Cn2,
            'turbulence_loss_db': turb_loss_db,
            'total_loss_db': total_loss_db,
            'data_rate_bps': achievable_data_rate_bps,
            'eb_n0_db': eb_n0_db,
            'network_available': network_available
        })

    final_df = pd.DataFrame(network_performance_data)
    output_file = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'dynamic_model', f"{ground_station}_final_dynamic_model_dataframe.csv")
    final_df.to_csv(output_file, index=False)
    return final_df

if __name__ == "__main__":
    params = read_satellite_parameters(os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt'))
    start_date_str = '2023-06-01'
    end_date_str = '2023-06-05'

    for ground_station in tqdm([station[0] for station in params['ground_stations']], desc="Processing ground stations"):
        logger.info(f"Processing ground station: {ground_station}")
        try:
            final_df = create_pipeline(start_date_str, end_date_str, params, ground_station)
            logger.info(f"Results for {ground_station}:")
            logger.info(final_df.head())
        except Exception as e:
            logger.error(f"An error occurred while processing {ground_station}: {e}")

    logger.info("Processing complete. Check the output directory for results.")
