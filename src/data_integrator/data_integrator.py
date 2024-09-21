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
    cloud_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'cloud_cover', f"{ground_station_name}_eumetsat_2023-06-01_2024-06-02_detailed_df.csv")
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
    before_start = cloud_data[cloud_data['time'] <= start_time].iloc[-1] if not cloud_data[cloud_data['time'] <= start_time].empty else None
    after_start = cloud_data[cloud_data['time'] > start_time].iloc[0] if not cloud_data[cloud_data['time'] > start_time].empty else None
    before_end = cloud_data[cloud_data['time'] <= end_time].iloc[-1] if not cloud_data[cloud_data['time'] <= end_time].empty else None
    after_end = cloud_data[cloud_data['time'] > end_time].iloc[0] if not cloud_data[cloud_data['time'] > end_time].empty else None

    weighted_cloud_cover = 0
    count = 0

    if before_start is not None and after_start is not None:
        start_weight = (after_start['time'] - start_time).total_seconds() / (after_start['time'] - before_start['time']).total_seconds()
        weighted_cloud_cover += start_weight * before_start['cloud_cover'] + (1 - start_weight) * after_start['cloud_cover']
        count += 1
    elif before_start is not None:
        weighted_cloud_cover += before_start['cloud_cover']
        count += 1
    elif after_start is not None:
        weighted_cloud_cover += after_start['cloud_cover']
        count += 1

    if before_end is not None and after_end is not None:
        end_weight = (after_end['time'] - end_time).total_seconds() / (after_end['time'] - before_end['time']).total_seconds()
        weighted_cloud_cover += end_weight * before_end['cloud_cover'] + (1 - end_weight) * after_end['cloud_cover']
        count += 1
    elif before_end is not None:
        weighted_cloud_cover += before_end['cloud_cover']
        count += 1
    elif after_end is not None:
        weighted_cloud_cover += after_end['cloud_cover']
        count += 1

    return weighted_cloud_cover / count if count > 0 else np.nan

def combine_data(los_data, cloud_data, turbulence_day, turbulence_night, ground_station_name, combined_data_list):
    if not los_data or cloud_data.empty:
        print(f"Skipping ground station {ground_station_name} due to insufficient data.")
        return

    for los_entry in tqdm(los_data, desc=f"Processing passes for {ground_station_name}", leave=False):
        start_time = los_entry['Start']
        end_time = los_entry['End']
        
        cloud_cover = calculate_weighted_cloud_cover(start_time, end_time, cloud_data)
        
        if not turbulence_day.empty:
            turbulence_day_values = turbulence_day[(turbulence_day['time'] >= start_time) & (turbulence_day['time'] <= end_time)]
        else:
            turbulence_day_values = pd.DataFrame()
        
        if not turbulence_night.empty:
            turbulence_night_values = turbulence_night[(turbulence_night['time'] >= start_time) & (turbulence_night['time'] <= end_time)]
        else:
            turbulence_night_values = pd.DataFrame()
        
        if not turbulence_day_values.empty and turbulence_night_values.empty:
            turbulence = turbulence_day_values['turbulence'].mean()
            day_or_night = 'day'
        elif turbulence_day_values.empty and not turbulence_night_values.empty:
            turbulence = turbulence_night_values['turbulence'].mean()
            day_or_night = 'night'
        elif not turbulence_day_values.empty and not turbulence_night_values.empty:
            turbulence = pd.concat([turbulence_day_values, turbulence_night_values])['turbulence'].mean()
            day_or_night = 'mixed'
        else:
            turbulence = np.nan
            day_or_night = 'unknown'

        combined_data_list.append({
            'Start': start_time,
            'End': end_time,
            'Duration': los_entry['Duration'],
            'Max Elevation': los_entry['Max Elevation'],
            'Cloud Cover': cloud_cover,
            'Turbulence': turbulence,
            'Day/Night': day_or_night,
            'Ground Station': ground_station_name
        })

# Main execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PARAMS_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator', 'combined_data.csv')

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

ground_stations = read_ground_station_parameters(PARAMS_FILE)

combined_data_list = []

for ground_station_name in tqdm(ground_stations.keys(), desc="Processing Ground Stations"):
    print(f"\nProcessing ground station: {ground_station_name}")
    los_data, cloud_data, turbulence_day, turbulence_night = load_data(ground_station_name, PROJECT_ROOT)
    combine_data(los_data, cloud_data, turbulence_day, turbulence_night, ground_station_name, combined_data_list)

combined_df = pd.DataFrame(combined_data_list)
combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"Combined data saved to {OUTPUT_FILE}")
