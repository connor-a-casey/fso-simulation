import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import matplotlib.ticker as mtick
from matplotlib.dates import DateFormatter

# Suppress detailed font-matching debug logs
import matplotlib
matplotlib.set_loglevel('WARNING')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    logger.info(f"Reading satellite parameters from: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('Ground_station_'):
                    _, data = line.split(':', 1)
                    station_info = [x.strip() for x in data.split(',')]
                    if len(station_info) != 6:
                        logger.warning(f"Invalid ground station data: {line}")
                        continue
                    ground_station = {
                        'Name': station_info[0],
                        'Latitude_deg': float(station_info[1]),
                        'Longitude_deg': float(station_info[2]),
                        'Altitude_m': float(station_info[3]),
                        'Downlink_data_rate_Gbps': float(station_info[4]),
                        'Tracking_accuracy_arcsec': float(station_info[5])
                    }
                    ground_stations.append(ground_station)
                    logger.debug(f"Added ground station: {ground_station['Name']}")
                else:
                    key, value = map(str.strip, line.split(':', 1))
                    params[key] = value

    params['ground_stations'] = ground_stations
    logger.info(f"Found {len(ground_stations)} ground stations")

    # Convert specific parameters to appropriate types and units
    float_params = [
        'Altitude_km', 'Inclination_deg', 'Orbit_repeat_cycle_days', 'Orbit_period_minutes',
        'Data_collection_rate_Tbit_per_day', 'Onboard_storage_Gbit', 'Downlink_data_rate_Gbps',
        'Optical_aperture_mm', 'Operational_wavelength_nm', 'Optical_Tx_power_W',
        'Link_margin_dB', 'System_noise_temperature_K', 'Pointing_accuracy_arcsec',
        'Cloud_cover_percentage_threshold', 'Acquisition_tracking_Az_deg', 'Acquisition_tracking_El_deg',
        'Bandwidth_GHz'
    ]

    # Remove '+/-' from parameters and convert to float
    for param in float_params:
        if param in params:
            try:
                params[param] = float(params[param].replace('+/-', '').strip())
                logger.debug(f"Converted {param} to float: {params[param]}")
            except ValueError:
                logger.error(f"Could not convert {param} to float. Value: {params[param]}")
                raise ValueError(f"Invalid value for parameter '{param}' in parameters file.")

    # Convert units
    if 'Optical_aperture_mm' in params:
        params['Optical_aperture_m'] = params['Optical_aperture_mm'] / 1000
    else:
        logger.error("Parameter 'Optical_aperture_mm' not found in parameters file.")
        raise KeyError("Missing parameter 'Optical_aperture_mm' in parameters file.")

    if 'Operational_wavelength_nm' in params:
        params['Operational_wavelength_m'] = params['Operational_wavelength_nm'] * 1e-9
    else:
        logger.error("Parameter 'Operational_wavelength_nm' not found in parameters file.")
        raise KeyError("Missing parameter 'Operational_wavelength_nm' in parameters file.")

    logger.info(f"Converted Optical aperture and wavelength to meters.")

    # Check for missing critical parameters
    required_params = [
        'Altitude_km', 'Downlink_data_rate_Gbps', 'Optical_Tx_power_W',
        'Optical_aperture_m', 'Operational_wavelength_m', 'Link_margin_dB',
        'System_noise_temperature_K', 'Pointing_accuracy_arcsec', 'Bandwidth_GHz'
    ]

    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        logger.error(f"Missing required parameters in parameters file: {missing_params}")
        raise KeyError(f"Missing required parameters: {missing_params}")

    # Ensure that ground stations are defined
    if not params['ground_stations']:
        logger.error("No ground stations defined in parameters file.")
        raise ValueError("At least one ground station must be defined in parameters file.")

    return params

def calculate_boltzmann_constant_db():
    return 10 * np.log10(BOLTZMANN_CONSTANT)

def calculate_system_noise_temperature_db(T):
    return 10 * np.log10(T)

def calculate_slant_range(earth_radius_km, satellite_height_km, elevation_deg):
    earth_radius_m = earth_radius_km * 1000
    satellite_height_m = satellite_height_km * 1000
    elevation_rad = np.radians(elevation_deg)
    
    if elevation_deg <= 0:
        raise ValueError("Elevation angle must be greater than 0 degrees.")
    
    slant_range_m = np.sqrt((earth_radius_m + satellite_height_m)**2 -
                            (earth_radius_m * np.sin(elevation_rad))**2) - earth_radius_m * np.cos(elevation_rad)
    
    return slant_range_m / 1000  # Return in km

def calculate_free_space_loss(distance_m, wavelength_m):
    if distance_m <= 0 or wavelength_m <= 0:
        return float('inf')
    
    loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)
    
    return max(loss_db, 0)

def calculate_antenna_gain(aperture_diameter_m, wavelength_m, efficiency=0.6):
    if aperture_diameter_m <= 0 or wavelength_m <= 0:
        return 0
    
    gain_linear = efficiency * (np.pi * aperture_diameter_m / wavelength_m) ** 2
    gain_db = 10 * np.log10(gain_linear)
    
    return max(gain_db, 0)

def calculate_pointing_loss(pointing_error_rad, divergence_angle_rad):
    if divergence_angle_rad <= 0:
        logger.warning(f"Invalid divergence angle: {divergence_angle_rad}")
        return 100  # Return a large but finite loss
    
    limited_error = min(pointing_error_rad, divergence_angle_rad)
    
    loss = (2 * limited_error / divergence_angle_rad) ** 2
    loss_db = -10 * np.log10(np.exp(-loss))
    
    max_loss_db = 40  # Adjust this value as needed
    return min(max_loss_db, loss_db)

def calculate_turbulence_loss_optical(Cn2, wavelength_m, path_length_m):
    # Simple model for turbulence-induced fading in optical links
    # Using a log-normal distribution for intensity fluctuations
    # Here we simulate turbulence loss as a random variable
    # For simplicity, we'll assume a mean turbulence loss of 3 dB with a standard deviation
    # Adjust these parameters based on actual turbulence models
    mean_turbulence_loss_db = 3
    std_turbulence_loss_db = 1
    turbulence_loss_db = np.random.normal(mean_turbulence_loss_db, std_turbulence_loss_db)
    return max(turbulence_loss_db, 0)  # Ensure loss is not negative

def calculate_atmospheric_attenuation(elevation_deg):
    if elevation_deg <= 0:
        return float('inf')
    
    attenuation_db_per_km = 0.1  # Adjusted for optical frequencies
    path_length_km = 1 / np.sin(np.radians(elevation_deg))
    total_attenuation_db = attenuation_db_per_km * path_length_km
    
    return max(total_attenuation_db, 0)

def calculate_cloud_attenuation(cloud_cover_fraction, params):
    cloud_cover_threshold = params.get('Cloud_cover_percentage_threshold', 100)  # Default to 100 if not specified
    if cloud_cover_fraction >= cloud_cover_threshold / 100.0:
        return 100  # High attenuation (link blocked)
    else:
        return 0  # No significant attenuation

def calculate_data_throughput(start_date_str, end_date_str, params):
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    logger.info(f"Calculating data throughput from {start_date_str} to {end_date_str}")

    combined_data_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator', 'combined_data.csv')

    if not os.path.exists(combined_data_path):
        logger.warning(f"Combined data file not found at {combined_data_path}")
        return pd.DataFrame(), pd.DataFrame(), None

    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df.columns = [col.replace(' ', '_').replace('/', '_') for col in combined_df.columns]

    logger.info(f"Read {len(combined_df)} rows from combined data file")
    logger.info(f"Columns in combined_df: {combined_df.columns}")

    expected_columns = [
        'Start', 'End', 'Duration', 'Max_Elevation',
        'Cloud_Cover', 'Turbulence', 'Day_Night', 'Ground_Station'
    ]

    missing_columns = [col for col in expected_columns if col not in combined_df.columns]
    if missing_columns:
        logger.error(f"Missing columns in the combined data: {missing_columns}")
        return pd.DataFrame(), pd.DataFrame(), None

    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] <= end_date)]
    logger.info(f"Filtered to {len(combined_df)} rows within date range")

    if combined_df.empty:
        logger.warning(f"No data available between {start_date_str} and {end_date_str}")
        return pd.DataFrame(), pd.DataFrame(), None

    satellite_data_rate_bps = params['Downlink_data_rate_Gbps'] * 1e9
    logger.info(f"Satellite data rate: {satellite_data_rate_bps} bps")

    bandwidth_hz = params['Bandwidth_GHz'] * 1e9
    logger.info(f"Bandwidth set to: {bandwidth_hz} Hz")

    total_data_transmitted_bits = 0
    maximum_possible_data_transmitted_bits = 0
    passes_data = []  # List to store per-pass data
    monthly_data = {}

    try:
        satellite_height_km = params['Altitude_km']
        wavelength_m = params['Operational_wavelength_m']
        optical_tx_power_w = params['Optical_Tx_power_W']
        tx_aperture_m = params['Optical_aperture_m']
        rx_aperture_m = params['Optical_aperture_m']
        link_margin_db = params['Link_margin_dB']
        satellite_pointing_accuracy_arcsec = params['Pointing_accuracy_arcsec']
        system_noise_temperature_k = params['System_noise_temperature_K']

        logger.info(f"Satellite parameters: (Height: {satellite_height_km} km, Wavelength: {wavelength_m} m, Tx Power: {optical_tx_power_w} W, Bandwidth: {bandwidth_hz} Hz)")

    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return pd.DataFrame(), pd.DataFrame(), None

    k_B_db = calculate_boltzmann_constant_db()
    Ts_db = calculate_system_noise_temperature_db(system_noise_temperature_k)
    logger.info(f"Boltzmann constant (dB): {k_B_db}, System noise temperature (dB): {Ts_db}")

    all_months = pd.date_range(start=start_date, end=end_date, freq='MS')

    for pass_data in tqdm(combined_df.itertuples(index=False), total=combined_df.shape[0], desc="Processing passes"):
        try:
            start_time = pass_data.Start
            end_time = pass_data.End
            duration_seconds = (end_time - start_time).total_seconds()

            ground_station_name = pass_data.Ground_Station
            ground_station_data = next((station for station in params['ground_stations'] if station['Name'] == ground_station_name), None)
            if ground_station_data is None:
                logger.warning(f"Ground station '{ground_station_name}' not found in parameters")
                continue

            gs_data_rate_bps = ground_station_data['Downlink_data_rate_Gbps'] * 1e9
            effective_data_rate_bps = min(satellite_data_rate_bps, gs_data_rate_bps)

            # Adjust the effective data rate based on real-world conditions
            cloud_cover_fraction = getattr(pass_data, 'Cloud_Cover', 0.0)
            Cn2 = getattr(pass_data, 'Turbulence', 1e-15)
            elevation = pass_data.Max_Elevation

            S_km = calculate_slant_range(EARTH_RADIUS_KM, satellite_height_km, elevation)
            S_m = S_km * 1000

            Ls_db = calculate_free_space_loss(S_m, wavelength_m)
            Gt_db = calculate_antenna_gain(tx_aperture_m, wavelength_m)
            Gr_db = calculate_antenna_gain(rx_aperture_m, wavelength_m)

            divergence_angle_rad = wavelength_m / (np.pi * tx_aperture_m)
            satellite_pointing_accuracy_rad = satellite_pointing_accuracy_arcsec * (np.pi / (180 * 3600))
            ground_pointing_accuracy_rad = ground_station_data['Tracking_accuracy_arcsec'] * (np.pi / (180 * 3600))
            total_pointing_error_rad = np.sqrt(satellite_pointing_accuracy_rad**2 + ground_pointing_accuracy_rad**2)
            Lp_db = calculate_pointing_loss(total_pointing_error_rad, divergence_angle_rad)

            atmospheric_attenuation_db = calculate_atmospheric_attenuation(elevation)
            cloud_attenuation_db = calculate_cloud_attenuation(cloud_cover_fraction, params)
            turb_loss_db = calculate_turbulence_loss_optical(Cn2, wavelength_m, S_m)

            total_loss_db = Ls_db + atmospheric_attenuation_db + cloud_attenuation_db + Lp_db + turb_loss_db

            optical_tx_power_dbw = 10 * np.log10(optical_tx_power_w)
            rx_power_dbw = optical_tx_power_dbw + Gt_db + Gr_db - total_loss_db - link_margin_db

            noise_power_dbw_total = k_B_db + Ts_db + 10 * np.log10(bandwidth_hz)

            snr_db = rx_power_dbw - noise_power_dbw_total
            snr_linear = 10 ** (snr_db / 10)

            achievable_data_rate_bps = min(bandwidth_hz * np.log2(1 + snr_linear), effective_data_rate_bps)

            data_transmitted_bits = max(achievable_data_rate_bps * duration_seconds, 0)
            max_data_transmitted_bits = effective_data_rate_bps * duration_seconds
            maximum_possible_data_transmitted_bits += max_data_transmitted_bits

            # Store per-pass data
            passes_data.append({
                'Start_Time': start_time,
                'End_Time': end_time,
                'Duration_seconds': duration_seconds,
                'Ground_Station': ground_station_name,
                'Achievable_Data_Rate_(Gbps)': achievable_data_rate_bps / 1e9,
                'Data_Transmitted_(Gbits)': data_transmitted_bits / 1e9
            })

            # Monthly data aggregation
            month = start_time.strftime('%Y-%m')
            if month not in monthly_data:
                monthly_data[month] = {
                    'Total_Data_Transmitted_(Gbits)': 0,
                    'Maximum_Possible_Data_(Gbits)': 0
                }
            monthly_data[month]['Total_Data_Transmitted_(Gbits)'] += data_transmitted_bits / 1e9
            monthly_data[month]['Maximum_Possible_Data_(Gbits)'] += max_data_transmitted_bits / 1e9

            total_data_transmitted_bits += data_transmitted_bits

        except Exception as e:
            logger.error(f"Error processing pass: {e}")
            logger.exception("Detailed error information:")

    overall_percentage_data_throughput = (total_data_transmitted_bits / maximum_possible_data_transmitted_bits * 100
                                          if maximum_possible_data_transmitted_bits > 0 else 0.0)

    total_data_transmitted_Gbits = total_data_transmitted_bits / 1e9

    logger.info(f"Total data transmitted: {total_data_transmitted_Gbits} Gbits")
    logger.info(f"Overall percentage data throughput: {overall_percentage_data_throughput}%")

    # Update the dataframes and outputs
    overall_df = pd.DataFrame({
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Total_Data_Transmitted_(Gbits)': [total_data_transmitted_Gbits],
        'Percentage_Data_Throughput_(%)': [overall_percentage_data_throughput]
    })

    # Prepare monthly DataFrame
    if monthly_data:
        monthly_df = pd.DataFrame.from_dict(monthly_data, orient='index').reset_index()
        monthly_df.rename(columns={'index': 'Month'}, inplace=True)
        monthly_df['Percentage_Data_Throughput_(%)'] = (
            monthly_df['Total_Data_Transmitted_(Gbits)'] / monthly_df['Maximum_Possible_Data_(Gbits)'] * 100
        ).fillna(0)
    else:
        monthly_df = pd.DataFrame(columns=[
            'Month', 'Total_Data_Transmitted_(Gbits)', 'Maximum_Possible_Data_(Gbits)', 'Percentage_Data_Throughput_(%)'
        ])

    # Save outputs to CSV files and return the updated dataframes
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'dynamic_analysis', 'data_throughput')
    os.makedirs(output_directory, exist_ok=True)

    # Save per-pass data
    passes_output_file = os.path.join(output_directory, "per_pass_throughput.csv")
    passes_df = pd.DataFrame(passes_data)
    passes_df.to_csv(passes_output_file, index=False)
    logger.info(f"Per-pass throughput data saved to: {passes_output_file}")

    monthly_output_file = os.path.join(output_directory, "monthly_data_throughput.csv")
    monthly_df.to_csv(monthly_output_file, index=False)
    logger.info(f"Monthly data saved to: {monthly_output_file}")

    overall_output_file = os.path.join(output_directory, "overall_network_throughput.csv")
    overall_df.to_csv(overall_output_file, index=False)
    logger.info(f"Overall network throughput saved to: {overall_output_file}")

    return monthly_df, overall_df, output_directory

def plot_monthly_data(monthly_df, output_directory):
    # Ensure the 'Month' column exists
    if 'Month' not in monthly_df.columns:
        logger.error("The 'Month' column is missing from the DataFrame.")
        raise KeyError("'Month' column not found in the DataFrame.")

    # Check if the DataFrame is empty
    if monthly_df.empty:
        logger.warning("Monthly DataFrame is empty. No data to plot.")
        return

    # Sort the DataFrame by 'Month' to ensure correct order
    monthly_df['Month'] = pd.to_datetime(monthly_df['Month'])
    monthly_df = monthly_df.sort_values('Month')

    # Convert 'Month' back to string format for plotting
    monthly_df['Month'] = monthly_df['Month'].dt.strftime('%Y-%m')

    # Extract the month strings for x-axis labels
    x_labels = monthly_df['Month'].astype(str)

    # Create numerical x positions
    x_positions = np.arange(len(x_labels))

    # Convert data to numpy arrays for plotting
    total_data_transmitted = monthly_df['Total_Data_Transmitted_(Gbits)'].to_numpy()
    percentage_throughput = monthly_df['Percentage_Data_Throughput_(%)'].to_numpy()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the total data transmitted on the primary y-axis
    ax1.bar(x_positions, total_data_transmitted, color='black', label='Total Data Transmitted (Gbits)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Data Transmitted (Gbits)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')

    # Add comma formatting to the primary y-axis
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    # Create a secondary y-axis for the throughput percentage
    ax2 = ax1.twinx()
    ax2.plot(x_positions, percentage_throughput, color='blue', marker='o', label='Percentage Data Throughput (%)')
    ax2.set_ylabel('Percentage Data Throughput (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Format the secondary y-axis as percentage
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set the title and legend
    plt.title('Monthly Data Transmission and Throughput')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Adjust layout and save the plot
    fig.tight_layout()
    os.makedirs(output_directory, exist_ok=True)
    plot_file = os.path.join(output_directory, "monthly_data_throughput_plot.png")
    plt.savefig(plot_file)
    plt.close(fig)  # Close the figure to free up memory
    logger.info(f"Monthly data and throughput plot saved to: {plot_file}")

def plot_per_pass_data_monthly(passes_df, output_directory):
    if passes_df.empty:
        logger.warning("Per-pass DataFrame is empty. No data to plot.")
        return

    # Add a 'Month' column to the DataFrame
    passes_df['Month'] = passes_df['Start_Time'].dt.to_period('M')

    # Calculate the average data transmitted per pass for each month
    monthly_avg = passes_df.groupby('Month')['Data_Transmitted_(Gbits)'].mean()

    # Save the monthly average data to a CSV file
    per_pass_monthly_output_file = os.path.join(output_directory, "per_pass_monthly_data.csv")
    monthly_avg_df = monthly_avg.reset_index()
    monthly_avg_df.columns = ['Month', 'Average_Data_Transmitted_per_Pass_(Gbits)']
    monthly_avg_df.to_csv(per_pass_monthly_output_file, index=False)
    logger.info(f"Per-pass monthly data saved to: {per_pass_monthly_output_file}")

    # Convert the index to datetime for plotting
    months = monthly_avg.index.to_timestamp().to_pydatetime()
    avg_data_transmitted = monthly_avg.values

    # Create the line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(months, avg_data_transmitted, marker='o', linestyle='-', color='blue')

    # Customize the plot
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Data Transmitted per Pass (Gbits)')
    ax.set_title('Monthly Average Data Transmitted per Pass')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_directory, "per_pass_data_monthly_plot.png")
    plt.savefig(plot_file)
    plt.close(fig)
    logger.info(f"Per-pass data monthly plot saved to: {plot_file}")

def main():
    start_date_str = '2023-06-01'
    end_date_str = '2024-06-01'

    params_file_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
    if not os.path.exists(params_file_path):
        logger.error(f"Parameters file not found at {params_file_path}")
        return

    try:
        params = read_satellite_parameters(params_file_path)

        # Print key parameters
        logger.info(f"Satellite Altitude: {params['Altitude_km']} km")
        logger.info(f"Optical Tx Power: {params['Optical_Tx_power_W']} W")
        logger.info(f"Downlink Data Rate: {params['Downlink_data_rate_Gbps']} Gbps")
        logger.info(f"Number of Ground Stations: {len(params['ground_stations'])}")
        logger.info(f"System Noise Temperature: {params['System_noise_temperature_K']} K")
        logger.info(f"Bandwidth: {params['Bandwidth_GHz']} GHz")

        monthly_df, overall_df, output_directory = calculate_data_throughput(start_date_str, end_date_str, params)

        if not monthly_df.empty and output_directory:
            plot_monthly_data(monthly_df, output_directory)
            logger.info("Monthly data and throughput plot saved successfully.")
        else:
            logger.warning("No data to plot.")

        # Load per-pass data
        passes_output_file = os.path.join(output_directory, "per_pass_throughput.csv")
        if os.path.exists(passes_output_file):
            passes_df = pd.read_csv(passes_output_file, parse_dates=['Start_Time', 'End_Time'])
            plot_per_pass_data_monthly(passes_df, output_directory)
            logger.info("Per-pass data monthly plot saved successfully.")
        else:
            logger.warning(f"Per-pass throughput data file not found at {passes_output_file}")

        if not overall_df.empty:
            logger.info(f"Overall Network Throughput:\n{overall_df.to_string(index=False)}")
        else:
            logger.warning("Overall network throughput data is empty.")

        logger.info("Processing complete. Check the output directory for results.")
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    main()
