import os
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm 
import glob

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
    
    # Extract cloud cover threshold from parameters
    if 'Cloud_cover_percentage_threshold' in params:
        params['Cloud_cover_threshold'] = float(params['Cloud_cover_percentage_threshold']) / 100.0 * 2.0
        # This assumes 0% cloud cover is 0, and 100% cloud cover is 2.0
    else:
        logger.warning("Cloud cover threshold not found in parameters. Using default value of 1.0.")
        params['Cloud_cover_threshold'] = 1.0  # Default value if not specified

    return params

def parse_passes_txt_file(file_path):
    passes = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Start:'):
                parts = line.split(',')
                start_time_str = parts[0].split('Start: ')[1].strip()
                end_time_str = parts[1].split('End: ')[1].strip()
                max_elevation_str = parts[3].split('Max Elevation: ')[1].strip().replace(' degrees', '')

                start_time = pd.to_datetime(start_time_str)
                end_time = pd.to_datetime(end_time_str)
                max_elevation = float(max_elevation_str)

                passes.append({
                    'Start': start_time,
                    'End': end_time,
                    'Max Elevation (degrees)': max_elevation
                })

    return pd.DataFrame(passes)

def merge_intervals(intervals):
    """
    Merge overlapping intervals.
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end:
            # Overlapping intervals, merge them
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            # Disjoint intervals, add to list
            merged.append(current)

    return merged

def calculate_network_availability(start_date_str, end_date_str, params):
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Total duration in seconds
    total_time_seconds = (end_date - start_date).total_seconds()

    # Path to the directory containing cloud cover data files
    cloud_data_dir = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover')

    # Get all CSV files in the cloud_data_dir
    cloud_files = glob.glob(os.path.join(cloud_data_dir, '*_df.csv'))

    if not cloud_files:
        raise FileNotFoundError(f"No cloud cover data files found in {cloud_data_dir}")

    # Read and process cloud cover data for each file
    df_list = []
    for file in cloud_files:
        station_name = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file, sep=',', parse_dates=['time'])
        df = df[['time', 'cloud_cover']]
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by='time', inplace=True)
        df = df.rename(columns={'cloud_cover': f'cloud_{station_name}'})
        df_list.append(df)

    # Merge all cloud cover data
    df_merged = pd.concat(df_list, axis=1)
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()].copy()

    # Filter data for the specified date range
    df_merged = df_merged[(df_merged['time'] >= start_date) & (df_merged['time'] <= end_date)]

    # Calculate availability for all OGS
    cloud_columns = [col for col in df_merged.columns if col.startswith('cloud_')]
    condition_all_ogs = df_merged[cloud_columns].apply(lambda x: x < 1.5).any(axis=1)
    availability_all_ogs = (condition_all_ogs.sum() / len(df_merged)) * 100

    # Create a final dataframe with the results
    final_df = pd.DataFrame({
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Network Availability All OGS (%)': [availability_all_ogs]
    })

    # Save the final dataframe to a CSV file
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'static_analysis')
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "network_availability_results.csv")
    final_df.to_csv(output_file, index=False)

    return final_df

if __name__ == "__main__":
    # Update the start and end dates to match your data
    start_date_str = '2023-06-01'
    end_date_str = '2024-06-01'

    params_file_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
    if not os.path.exists(params_file_path):
        logger.error(f"Parameters file not found at {params_file_path}")
        exit(1)

    params = read_satellite_parameters(params_file_path)

    try:
        final_df = calculate_network_availability(start_date_str, end_date_str, params)
        logger.info("Network Availability Results:")
        logger.info(final_df)
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")

    logger.info("Processing complete. Check the output directory for results.")