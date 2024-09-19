import os
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm  # For progress bar

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

    # Paths
    passes_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'satellite_passes')
    cloud_cover_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover')

    # Read ground station list
    ground_stations = params['ground_stations']

    available_intervals = []

    # Iterate over each ground station
    for ground_station in ground_stations:
        gs_name = ground_station['Name']
        logger.info(f"Processing ground station: {gs_name}")

        # Read satellite pass data for the ground station from a .txt file
        passes_file = os.path.join(passes_directory, f"{gs_name}_passes.txt")
        if not os.path.exists(passes_file):
            logger.warning(f"No passes file found for {gs_name}, skipping.")
            continue

        passes_df = parse_passes_txt_file(passes_file)
        passes_df = passes_df[(passes_df['Start'] >= start_date) & (passes_df['End'] <= end_date)]

        if passes_df.empty:
            logger.info(f"No passes within date range for {gs_name}, skipping.")
            continue

        # Adjusted cloud cover file name to match the naming convention
        cloud_cover_file = os.path.join(cloud_cover_directory, f"{gs_name}_eumetsat_{start_date_str}_{end_date_str}_detailed_df.csv")
        if not os.path.exists(cloud_cover_file):
            logger.warning(f"No cloud cover file found for {gs_name}, skipping.")
            continue

        cloud_df = pd.read_csv(cloud_cover_file, parse_dates=['time'])

        # Iterate over each pass with a progress bar
        for _, pass_data in tqdm(passes_df.iterrows(), total=passes_df.shape[0], desc=f"Processing passes for {gs_name}"):
            pass_start = pass_data['Start']
            pass_end = pass_data['End']

            # Get cloud cover data during the pass
            mask = (cloud_df['time'] >= pass_start) & (cloud_df['time'] <= pass_end)
            cloud_pass_df = cloud_df.loc[mask]

            if cloud_pass_df.empty:
                # No cloud data during this pass, assume worst case (completely cloudy)
                pass_available = False
            else:
                # Determine if pass is available based on cloud cover threshold
                cloud_cover_threshold = params['Cloud_cover_threshold']
                pass_available = (cloud_pass_df['cloud_cover'] <= cloud_cover_threshold).all()

            if pass_available:
                # Add the available interval
                available_intervals.append((pass_start, pass_end))

    # Merge overlapping intervals across all ground stations
    merged_intervals = merge_intervals(available_intervals)

    # Calculate total network available time
    total_network_available_time_seconds = sum((end - start).total_seconds() for start, end in merged_intervals)

    # Calculate network availability percentage
    network_availability_percentage = (total_network_available_time_seconds / total_time_seconds) * 100

    # Ensure network availability is between 0% and 100%
    network_availability_percentage = max(0.0, min(network_availability_percentage, 100.0))

    # Create a final dataframe with the results
    final_df = pd.DataFrame({
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Network Availability (%)': [network_availability_percentage]
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
    end_date_str = '2023-06-05'

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
