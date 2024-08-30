import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import re

# Function to read ground station parameters from the satelliteParameters.txt file
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

# Function to load data from specific files based on the ground station name
def load_data(ground_station_name, project_root):
    # Construct absolute paths using the project root
    los_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'satellite_passes', f"{ground_station_name}_passes.txt")
    cloud_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'cloud_cover', f"{ground_station_name}_eumetsat_2023-06-01_2023-06-05_df.csv")
    turbulence_day_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'turbulence', f"{ground_station_name}_day.txt")
    turbulence_night_file = os.path.join(project_root, 'IAC-2024', 'data', 'output', 'turbulence', f"{ground_station_name}_night.txt")

    # Check if files exist before loading
    if not os.path.exists(los_file):
        print(f"Warning: LoS file {los_file} not found.")
        return None, None, None, None

    if not os.path.exists(cloud_file):
        print(f"Warning: Cloud cover file {cloud_file} not found.")
        return None, None, None, None

    if not os.path.exists(turbulence_day_file):
        print(f"Warning: Turbulence day file {turbulence_day_file} not found.")
        return None, None, None, None

    if not os.path.exists(turbulence_night_file):
        print(f"Warning: Turbulence night file {turbulence_night_file} not found.")
        return None, None, None, None
    
    # Load Line of Sight (LoS) data
    los_data = []
    with open(los_file, 'r') as f:
        for line in f:
            if "Start" in line and "End" in line:
                times = line.split(", ")
                start_time_str = times[0].split(": ", 1)[1].strip()
                end_time_str = times[1].split(": ", 1)[1].strip()
                
                # Convert string to datetime
                start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
                
                # Append parsed data
                los_data.append({'Start': start_time, 'End': end_time})

    # Load cloud cover data and parse 'time' as date and time
    cloud_data = pd.read_csv(cloud_file, parse_dates=['time'])

    # Load turbulence data (day and night) with numeric extraction
    with open(turbulence_day_file, 'r') as f:
        day_data = f.read().strip()
        turbulence_day = float(re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', day_data).group())

    with open(turbulence_night_file, 'r') as f:
        night_data = f.read().strip()
        turbulence_night = float(re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', night_data).group())
    
    return los_data, cloud_data, turbulence_day, turbulence_night

# Function to combine LoS, cloud data, and turbulence data
def combine_data(los_data, cloud_data, turbulence_data_day, turbulence_data_night, ground_station_name, output_dir):
    if los_data is None or cloud_data is None or turbulence_data_day is None or turbulence_data_night is None:
        print(f"Skipping ground station {ground_station_name} due to missing data.")
        return

    combined_data = []

    for los_entry in los_data:
        start_time = los_entry['Start']
        end_time = los_entry['End']
        day_or_night = 'day' if start_time.hour >= 6 and end_time.hour < 18 else 'night'

        # Select appropriate turbulence strength based on time of day
        turbulence_strength = turbulence_data_day if day_or_night == 'day' else turbulence_data_night

        # Find corresponding cloud cover by matching closest timestamp
        closest_cloud_cover = cloud_data.iloc[(cloud_data['time'] - start_time).abs().argsort()[:1]]
        if not closest_cloud_cover.empty:
            cloud_cover = closest_cloud_cover['cloud_cover'].values[0]
        else:
            cloud_cover = None

        combined_data.append({
            'Start': start_time,
            'End': end_time,
            'Cloud Cover': cloud_cover,
            'Turbulence Strength': turbulence_strength,
        })

    combined_df = pd.DataFrame(combined_data)
    output_filename = os.path.join(output_dir, f"{ground_station_name}_combined_data.csv")
    combined_df.to_csv(output_filename, index=False)
    print(f"Combined data saved to {output_filename}")

# Specify the parameters file and output directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PARAMS_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read ground station parameters
ground_stations = read_ground_station_parameters(PARAMS_FILE)

# Combine data for each ground station with a progress bar
for ground_station_name in tqdm(ground_stations.keys(), desc="Processing Ground Stations"):
    los_data, cloud_data, turbulence_data_day, turbulence_data_night = load_data(ground_station_name, PROJECT_ROOT)
    combine_data(los_data, cloud_data, turbulence_data_day, turbulence_data_night, ground_station_name, OUTPUT_DIR)
