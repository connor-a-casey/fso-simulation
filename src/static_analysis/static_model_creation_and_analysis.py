import os
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm 

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

def calculate_data_throughput(start_date_str, end_date_str, params):
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Total duration in seconds
    total_time_seconds = (end_date - start_date).total_seconds()

    # Initialize total data transmitted
    total_data_transmitted_bits = 0
    maximum_possible_data_transmitted_bits = 0

    # Paths
    combined_data_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator', 'combined_data.csv')

    # Read combined data
    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] <= end_date)]

    if combined_df.empty:
        logger.warning(f"No data available between {start_date_str} and {end_date_str}")
        # Return zero metrics if no data
        final_df = pd.DataFrame({
            'Start Date': [start_date_str],
            'End Date': [end_date_str],
            'Total Data Transmitted (Gbits)': [0.0],
            'Percentage Data Throughput (%)': [0.0]
        })
        return final_df

    # Get satellite data rate
    satellite_data_rate_bps = float(params['Downlink_data_rate_Gbps']) * 1e9  # Convert Gbps to bps

    # Iterate over passes for all ground stations with a progress bar
    for _, pass_data in tqdm(combined_df.iterrows(), total=combined_df.shape[0], desc="Processing passes"):
        # Get start and end times
        start_time = pass_data['Start']
        end_time = pass_data['End']
        duration_seconds = (end_time - start_time).total_seconds()

        ground_station_name = pass_data['Ground Station']
        # Find the corresponding ground station's data rate
        ground_station_data = next((station for station in params['ground_stations'] if station['Name'] == ground_station_name), None)
        if ground_station_data is None:
            raise ValueError(f"Ground station {ground_station_name} not found in parameters")

        # Get the ground station data rate
        gs_data_rate_bps = ground_station_data['Downlink_data_rate_Gbps'] * 1e9  # Convert Gbps to bps

        # Use the minimum of satellite and ground station data rates
        effective_data_rate_bps = min(satellite_data_rate_bps, gs_data_rate_bps)

        # Calculate data transmitted during this pass
        data_transmitted_bits = effective_data_rate_bps * duration_seconds
        total_data_transmitted_bits += data_transmitted_bits

        # Calculate maximum possible data transmitted during this pass
        max_data_transmitted_bits = satellite_data_rate_bps * duration_seconds
        maximum_possible_data_transmitted_bits += max_data_transmitted_bits

    # Calculate percentage data throughput
    if maximum_possible_data_transmitted_bits > 0:
        percentage_data_throughput = (total_data_transmitted_bits / maximum_possible_data_transmitted_bits) * 100
    else:
        percentage_data_throughput = 0.0

    # Convert total data transmitted to Gbits
    total_data_transmitted_Gbits = total_data_transmitted_bits / 1e9

    # Create a final dataframe with the results
    final_df = pd.DataFrame({
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Total Data Transmitted (Gbits)': [total_data_transmitted_Gbits],
        'Percentage Data Throughput (%)': [percentage_data_throughput]
    })

    # Save the final dataframe to a CSV file
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'static_analysis')
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "data_throughput_results.csv")
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

    try:
        final_df = calculate_data_throughput(start_date_str, end_date_str, params)
        logger.info("Data Throughput Results:")
        logger.info(final_df)
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")

    logger.info("Processing complete. Check the output directory for results.")
