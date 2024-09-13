import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

# Constants
SPEED_OF_LIGHT = 299792458  # m/s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

def read_satellite_parameters(file_path):
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

def calculate_boltzmann_constant():
    return 10 * np.log10(BOLTZMANN_CONSTANT)

def calculate_system_noise_temperature(T):
    return 10 * np.log10(T)

def calculate_slant_range(earth_radius_km, satellite_height_km, elevation_deg):
    earth_radius = earth_radius_km * 1000  # Convert to meters
    satellite_height = satellite_height_km * 1000  # Convert to meters
    elevation_rad = np.radians(elevation_deg)
    
    slant_range = np.sqrt(satellite_height**2 + 2*earth_radius*satellite_height + 
                          (earth_radius**2) * (np.cos(elevation_rad)**2)) - earth_radius * np.sin(elevation_rad)
    
    return slant_range / 1000  # Return in km

def calculate_free_space_loss(distance_m, frequency_hz):
    wavelength = SPEED_OF_LIGHT / frequency_hz
    if distance_m <= 0 or wavelength <= 0:
        return float('inf')
    loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength)
    return max(loss_db, 0)  # Ensure non-negative loss

def calculate_antenna_gain(diameter, wavelength, efficiency):
    gain = efficiency * (np.pi * diameter / wavelength)**2
    return max(10 * np.log10(gain), 0)  # Ensure non-negative gain

def calculate_pointing_loss(pointing_error_rad, wavelength, aperture_diameter):
    loss = (np.pi * pointing_error_rad * aperture_diameter / wavelength)**2
    return max(-10 * np.log10(np.exp(-loss)), 0)  # Ensure non-negative loss

def turbulence_loss(Cn2, wavelength, path_length):
    k = 2 * np.pi / wavelength
    sigma_R2 = 1.23 * Cn2 * (k ** (7 / 6)) * (path_length ** (11 / 6))
    loss_dB = max(10 * np.log10(np.exp(-sigma_R2)), 0)  # Ensure non-negative loss
    return loss_dB

def calculate_atmospheric_attenuation(wavelength_nm, path_length_km):
    fso_windows = {
        850: 0.2,
        1060: 0.2,
        1250: 0.2,
        1550: 0.2
    }
    closest_window = min(fso_windows.keys(), key=lambda x: abs(x - wavelength_nm))
    attenuation_db_per_km = fso_windows[closest_window]
    total_attenuation_db = attenuation_db_per_km * path_length_km
    return max(total_attenuation_db, 0)  # Ensure non-negative attenuation

def calculate_cloud_attenuation(cloud_cover):
    max_attenuation = 30  # dB, maximum attenuation for very thick clouds
    return max(cloud_cover * max_attenuation, 0)  # Ensure non-negative attenuation

def calculate_data_rate(tx_power_w, max_data_rate_gbps, link_margin_db, modulation_index, extinction_ratio, 
                        tx_gain_db, rx_gain_db, total_loss_db, boltzmann_db, noise_temp_db):
    tx_power_dbw = 10 * np.log10(tx_power_w)
    rx_power_dbw = tx_power_dbw + tx_gain_db + rx_gain_db - total_loss_db
    bandwidth = max_data_rate_gbps * 1e9  # Convert Gbps to Hz
    noise_power_dbw = boltzmann_db + noise_temp_db + 10 * np.log10(bandwidth)
    snr_db = rx_power_dbw - noise_power_dbw
    eb_n0_db = snr_db - 10 * np.log10(2 * modulation_index**2 * (extinction_ratio / (extinction_ratio + 1))**2)
    achievable_data_rate_bps = bandwidth * np.log2(1 + 10**((eb_n0_db - link_margin_db) / 10))
    return eb_n0_db, max(achievable_data_rate_bps, 0)  # Ensure non-negative data rate

def create_pipeline(start_date, end_date, params, ground_station):
    earth_radius_km = 6371  # Earth's mean radius in km
    satellite_height_km = float(params['Altitude_km'])
    wavelength = float(params['Operational_wavelength_nm']) * 1e-9  # Convert nm to m
    wavelength_nm = float(params['Operational_wavelength_nm'])
    downlink_data_rate_Gbps = float(params['Downlink_data_rate_Gbps'])
    optical_tx_power_W = float(params['Optical_Tx_power_W'])
    optical_aperture_m = float(params['Optical_aperture_mm']) / 1000  # Convert mm to m
    link_margin_dB = float(params['Link_margin_dB'])
    cloud_cover_threshold = float(params['Cloud_cover_percentage_threshold']) / 100  # Convert percentage to decimal

    # Find the correct ground station data
    ground_station_data = next((station for station in params['ground_stations'] if station[0] == ground_station), None)
    if ground_station_data is None:
        raise ValueError(f"Ground station {ground_station} not found in parameters")

    # Use the ground station-specific tracking accuracy
    tracking_accuracy_arcsec = float(ground_station_data[5])

    combined_data_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator', f"{ground_station}_combined_data.csv")
    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] < end_date)]

    network_performance_data = []
    k_B_dBi = calculate_boltzmann_constant()
    Ts_dBi = calculate_system_noise_temperature(135)  # Assuming a system noise temperature of 135K

    for _, pass_data in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Processing {ground_station} passes", leave=False):
        elevation = pass_data['Max Elevation']
        cloud_cover = pass_data['Cloud Cover']
        Cn2 = pass_data['Turbulence']  # Assuming the 'Turbulence' column contains Cn^2 values
        
        S = calculate_slant_range(earth_radius_km, satellite_height_km, elevation)  # in km
        S_m = S * 1000  # Convert to meters for other calculations
        
        Ls = calculate_free_space_loss(S_m, SPEED_OF_LIGHT / wavelength)
        Gr_db = calculate_antenna_gain(optical_aperture_m, wavelength, 0.55)  # Assuming 0.55 efficiency
        Lp_r = calculate_pointing_loss(tracking_accuracy_arcsec * 4.8481e-6, wavelength, optical_aperture_m)  # Convert arcsec to radians
        Gr_net = Gr_db + Lp_r
        G_t_net = calculate_antenna_gain(optical_aperture_m, wavelength, 0.55) - 1.96  # Including line loss

        atmospheric_attenuation = calculate_atmospheric_attenuation(wavelength_nm, S)
        turb_loss = turbulence_loss(Cn2, wavelength, S_m)
        cloud_attenuation = calculate_cloud_attenuation(cloud_cover)
        
        total_loss = max(Ls + atmospheric_attenuation + cloud_attenuation + turb_loss, 0)  # Ensure non-negative total loss

        eb_n0_db, data_rate_bps = calculate_data_rate(optical_tx_power_W, downlink_data_rate_Gbps, link_margin_dB, 
                                                      0.3, 0.3, G_t_net, Gr_net, total_loss, k_B_dBi, Ts_dBi)

        network_available = (cloud_cover < cloud_cover_threshold) and (data_rate_bps > downlink_data_rate_Gbps * 1e9 * 0.9)  # Assuming 90% of max data rate as threshold

        network_performance_data.append({
            'start_time': pass_data['Start'],
            'end_time': pass_data['End'],
            'duration': pass_data['Duration'],
            'max_elevation': elevation,
            'slant_range_km': S,
            'cloud_cover': cloud_cover,
            'atmospheric_attenuation_db': atmospheric_attenuation,
            'turbulence_cn2': Cn2,
            'turbulence_loss_db': turb_loss,
            'total_loss_db': total_loss,
            'data_rate_bps': data_rate_bps,
            'eb_n0_db': eb_n0_db,
            'network_available': network_available
        })

    final_df = pd.DataFrame(network_performance_data)
    output_file = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'dynamic_model', f"{ground_station}_final_dynamic_model_dataframe.csv")
    final_df.to_csv(output_file, index=False)
    return final_df

if __name__ == "__main__":
    params = read_satellite_parameters(os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt'))
    start_date = '2023-06-01'
    end_date = '2023-06-05'
    
    for ground_station in tqdm(params['ground_stations'], desc="Processing ground stations"):
        station_name = ground_station[0]
        tqdm.write(f"Processing ground station: {station_name}")
        final_df = create_pipeline(start_date, end_date, params, station_name)
        tqdm.write(f"Results for {station_name}:")
        tqdm.write(str(final_df.head()))
        tqdm.write("\n")

    print("Processing complete. Check the output directory for results.")