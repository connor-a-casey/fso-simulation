import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

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

def create_pipeline(start_date_str, end_date_str, params, ground_station):
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Extract satellite parameters
    data_rate_bps = float(params['Downlink_data_rate_Gbps']) * 1e9  # Convert Gbps to bps

    # Find the correct ground station data
    ground_station_data = next((station for station in params['ground_stations'] if station['Name'] == ground_station), None)
    if ground_station_data is None:
        raise ValueError(f"Ground station {ground_station} not found in parameters")

    # Read combined data for the ground station
    combined_data_path = os.path.join(PROJECT_ROOT,'IAC-2024', 'data', 'output', 'data_integrator', f"{ground_station}_combined_data.csv")
    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] <= end_date)]

    if combined_df.empty:
        logger.warning(f"No data available for {ground_station} between {start_date_str} and {end_date_str}")
        # Create an empty final dataframe with zero values
        final_df = pd.DataFrame({
            'Ground Station': [ground_station],
            'Start Date': [start_date_str],
            'End Date': [end_date_str],
            'Network Availability (%)': [0.0],
            'Total Data Transmitted (Gbits)': [0.0]
        })
        return final_df

    # Initialize total time and available time
    total_time_seconds = (end_date - start_date).total_seconds()
    network_available_time_seconds = 0
    total_data_transmitted_bits = 0

    # Iterate over passes
    for _, pass_data in combined_df.iterrows():
        # Get duration in seconds
        duration_seconds = (pass_data['End'] - pass_data['Start']).total_seconds()
        cloud_cover_fraction = pass_data['Cloud Cover'] / 8  # Convert oktas to fraction between 0 and 1

        # Determine network availability
        network_available = (cloud_cover_fraction < 0.5)

        if network_available:
            network_available_time_seconds += duration_seconds

            # Get the ground station data rate
            gs_data_rate_bps = ground_station_data['Downlink_data_rate_Gbps'] * 1e9  # Convert Gbps to bps

            # Use the minimum of satellite and ground station data rates
            effective_data_rate_bps = min(data_rate_bps, gs_data_rate_bps)

            data_transmitted_bits = effective_data_rate_bps * duration_seconds
            total_data_transmitted_bits += data_transmitted_bits

    # Calculate Network Availability (%)
    network_availability_percentage = (network_available_time_seconds / total_time_seconds) * 100

    # Convert total data transmitted to appropriate units, e.g., Gbits
    total_data_transmitted_Gbits = total_data_transmitted_bits / 1e9

    # Create a final dataframe with the results
    final_df = pd.DataFrame({
        'Ground Station': [ground_station],
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Network Availability (%)': [network_availability_percentage],
        'Total Data Transmitted (Gbits)': [total_data_transmitted_Gbits]
    })

    # Save the final dataframe to a CSV file
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'static_analysis')
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f"{ground_station}_static_analysis_results.csv")
    final_df.to_csv(output_file, index=False)
    return final_df

if __name__ == "__main__":
    # Update the start and end dates to match your data
    start_date_str = '2023-06-01'
    end_date_str = '2023-06-05'

    params_file_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
    if not os.path.exists(params_file_path):
        logger.error(f"Parameters file not found at {params_file_path}")
        exit(1)

    params = read_satellite_parameters(params_file_path)

    ground_station_names = [station['Name'] for station in params['ground_stations']]

    for ground_station in tqdm(ground_station_names, desc="Processing ground stations"):
        logger.info(f"Processing ground station: {ground_station}")
        try:
            final_df = create_pipeline(start_date_str, end_date_str, params, ground_station)
            logger.info(f"Results for {ground_station}:")
            logger.info(final_df)
        except Exception as e:
            logger.error(f"An error occurred while processing {ground_station}: {e}")

    logger.info("Processing complete. Check the output directory for results.")
