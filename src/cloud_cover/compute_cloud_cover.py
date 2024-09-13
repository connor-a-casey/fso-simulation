import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import glob

def read_ground_station_parameters(filename):
    ground_stations = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Ground_station"):
                parts = line.split(":")[1].strip().split(',')
                name = parts[0].strip()
                latitude = float(parts[1].strip())
                longitude = float(parts[2].strip())
                altitude = int(parts[3].strip())
                downlink_data_rate = float(parts[4].strip())
                tracking_accuracy = int(parts[5].strip())
                ground_stations[name] = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'altitude': altitude,
                    'downlink_data_rate': downlink_data_rate,
                    'tracking_accuracy': tracking_accuracy
                }
    return ground_stations

def find_file(directory, filename):
    file_path = os.path.join(directory, filename)
    return file_path if os.path.exists(file_path) else None

def load_data(ground_station_name, project_root):
    los_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'satellite_passes', f"{ground_station_name}_passes.txt")
    cloud_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'cloud_cover', f"{ground_station_name}_eumetsat_2023-06-01_2023-06-05_detailed_df.csv")
    turbulence_day_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'turbulence', f"{ground_station_name}_day.txt")
    turbulence_night_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'turbulence', f"{ground_station_name}_night.txt")

    files_to_check = [
        (los_file, "Line of Sight"),
        (cloud_file, "Cloud Cover"),
        (turbulence_day_file, "Day Turbulence"),
        (turbulence_night_file, "Night Turbulence")
    ]

    missing_files = []
    for file_path, desc in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(f"{desc} file ({file_path})")
        else:
            print(f"Found {desc} file: {file_path}")

    if missing_files:
        print(f"Warning: The following files for {ground_station_name} are missing: {', '.join(missing_files)}")
        return None, None, None, None

    try:
        # Load Line of Sight (LoS) data
        los_data = []
        with open(los_file, 'r') as f:
            for line in tqdm(f, desc=f"Loading LoS data for {ground_station_name}", leave=False):
                parts = line.split(', ')
                start_time = datetime.strptime(parts[0].split(': ')[1], '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(parts[1].split(': ')[1], '%Y-%m-%d %H:%M:%S')
                duration = parts[2].split(': ')[1]
                max_elevation = float(parts[3].split(': ')[1].split()[0])
                los_data.append({
                    'Start': start_time,
                    'End': end_time,
                    'Duration': duration,
                    'Max Elevation': max_elevation
                })

        # Load cloud cover data
        cloud_data = pd.read_csv(cloud_file, parse_dates=['time'])
        print(f"Loaded cloud data. Shape: {cloud_data.shape}")

        # Load turbulence data (day and night)
        def read_turbulence_file(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                turbulence_data = []
                for line in tqdm(lines, desc=f"Loading turbulence data from {os.path.basename(file_path)}", leave=False):
                    parts = line.strip().split(', ')
                    time = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
                    turbulence = float(parts[3])
                    turbulence_data.append({'time': time, 'turbulence': turbulence})
            return pd.DataFrame(turbulence_data)

        turbulence_day = read_turbulence_file(turbulence_day_file)
        turbulence_night = read_turbulence_file(turbulence_night_file)
        
        print(f"Loaded turbulence data. Day shape: {turbulence_day.shape}, Night shape: {turbulence_night.shape}")
        
        return los_data, cloud_data, turbulence_day, turbulence_night
    
    except Exception as e:
        print(f"Error loading data for {ground_station_name}: {str(e)}")
        return None, None, None, None

def calculate_weighted_cloud_cover(start_time, end_time, cloud_data):
    # Find the closest data points before and after the start and end times
    before_start = cloud_data[cloud_data['time'] <= start_time].iloc[-1] if not cloud_data[cloud_data['time'] <= start_time].empty else None
    after_start = cloud_data[cloud_data['time'] > start_time].iloc[0] if not cloud_data[cloud_data['time'] > start_time].empty else None
    before_end = cloud_data[cloud_data['time'] <= end_time].iloc[-1] if not cloud_data[cloud_data['time'] <= end_time].empty else None
    after_end = cloud_data[cloud_data['time'] > end_time].iloc[0] if not cloud_data[cloud_data['time'] > end_time].empty else None

    # Calculate weights and weighted cloud cover
    total_duration = (end_time - start_time).total_seconds()
    weighted_cloud_cover = 0

    if before_start and after_start:
        start_weight = (after_start['time'] - start_time).total_seconds() / (after_start['time'] - before_start['time']).total_seconds()
        weighted_cloud_cover += start_weight * before_start['cloud_cover'] + (1 - start_weight) * after_start['cloud_cover']
    elif before_start:
        weighted_cloud_cover += before_start['cloud_cover']
    elif after_start:
        weighted_cloud_cover += after_start['cloud_cover']

    if before_end and after_end:
        end_weight = (after_end['time'] - end_time).total_seconds() / (after_end['time'] - before_end['time']).total_seconds()
        weighted_cloud_cover += end_weight * before_end['cloud_cover'] + (1 - end_weight) * after_end['cloud_cover']
    elif before_end:
        weighted_cloud_cover += before_end['cloud_cover']
    elif after_end:
        weighted_cloud_cover += after_end['cloud_cover']

    # Average the start and end weighted cloud covers
    return weighted_cloud_cover / 2

def combine_data(los_data, cloud_data, turbulence_day, turbulence_night, ground_station_name, output_dir):
    if los_data is None or cloud_data.empty or turbulence_day.empty or turbulence_night.empty:
        print(f"Skipping ground station {ground_station_name} due to missing data.")
        return

    combined_data = []

    for los_entry in tqdm(los_data, desc=f"Processing passes for {ground_station_name}", leave=False):
        start_time = los_entry['Start']
        end_time = los_entry['End']
        
        # Calculate weighted cloud cover
        cloud_cover = calculate_weighted_cloud_cover(start_time, end_time, cloud_data)
        
        # Get turbulence (check both day and night files)
        turbulence_day_values = turbulence_day[(turbulence_day['time'] >= start_time) & (turbulence_day['time'] <= end_time)]
        turbulence_night_values = turbulence_night[(turbulence_night['time'] >= start_time) & (turbulence_night['time'] <= end_time)]
        
        if not turbulence_day_values.empty and turbulence_night_values.empty:
            turbulence = turbulence_day_values['turbulence'].mean()
            day_or_night = 'day'
        elif turbulence_day_values.empty and not turbulence_night_values.empty:
            turbulence = turbulence_night_values['turbulence'].mean()
            day_or_night = 'night'
        else:
            # If we have both day and night values, or neither, we'll use both and note it
            turbulence = pd.concat([turbulence_day_values, turbulence_night_values])['turbulence'].mean()
            day_or_night = 'mixed'

        combined_data.append({
            'Start': start_time,
            'End': end_time,
            'Duration': los_entry['Duration'],
            'Max Elevation': los_entry['Max Elevation'],
            'Cloud Cover': cloud_cover,
            'Turbulence': turbulence,
            'Day/Night': day_or_night
        })

    combined_df = pd.DataFrame(combined_data)
    output_filename = os.path.join(output_dir, f"{ground_station_name}_combined_data.csv")
    combined_df.to_csv(output_filename, index=False)
    print(f"Combined data saved to {output_filename}")

# Main execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PARAMS_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator')

print(f"Current directory: {CURRENT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Parameters file: {PARAMS_FILE}")
print(f"Output directory: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ground_stations = read_ground_station_parameters(PARAMS_FILE)

for ground_station_name in tqdm(ground_stations.keys(), desc="Processing Ground Stations"):
    print(f"\nProcessing ground station: {ground_station_name}")
    los_data, cloud_data, turbulence_day, turbulence_night = load_data(ground_station_name, PROJECT_ROOT)
    if los_data is not None:
        combine_data(los_data, cloud_data, turbulence_day, turbulence_night, ground_station_name, OUTPUT_DIR)
    else:
        print(f"Skipping data combination for {ground_station_name} due to missing data.")

print("Processing complete. Check the output directory for results.")