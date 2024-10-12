import os
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, wgs84
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PARAMS_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
TLE_FILE = os.path.join(CURRENT_DIR, 'terra.tle')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'satellite_passes')

def read_parameters(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split(':', 1)
                params[key.strip()] = value.strip()
    return params

def extract_ground_stations(params):
    ground_stations = []
    for i in range(1, 8):
        key = f'Ground_station_{i}'
        if key in params:
            station_data = params[key].split(',')
            ground_stations.append({
                'name': station_data[0].strip(),
                'location': wgs84.latlon(
                    latitude_degrees=float(station_data[1]),
                    longitude_degrees=float(station_data[2]),
                    elevation_m=float(station_data[3])
                ),
                'downlink_data_rate_Gbps': float(station_data[4].strip())
            })
    return ground_stations

def load_satellite(tle_file):
    with open(tle_file, 'r') as f:
        tle_lines = f.readlines()
    satellite = EarthSatellite(tle_lines[1].strip(), tle_lines[2].strip(), "Terra", load.timescale())
    return satellite

def compute_passes():
    params = read_parameters(PARAMS_FILE)
    ground_stations = extract_ground_stations(params)
    satellite = load_satellite(TLE_FILE)

    ts = load.timescale()
    start_time = ts.utc(2008, 1, 1)
    end_time = ts.utc(2013, 1, 1)  

    elevation_threshold = 20  # degrees
    min_pass_duration = timedelta(minutes=1)  # Only consider passes lasting longer than 1 minute

    ground_station_passes = {gs['name']: [] for gs in ground_stations}

    for gs in tqdm(ground_stations, desc="Processing ground stations"):
        observer = gs['location']
        t, events = satellite.find_events(observer, start_time, end_time, altitude_degrees=elevation_threshold)

        current_pass = None
        for ti, event in zip(t, events):
            if event == 0:  # Rise
                current_pass = {'start': ti.utc_datetime()}
            elif event == 1:  # Culminate
                if current_pass is not None:
                    topocentric = (satellite - observer).at(ti)
                    alt, az, distance = topocentric.altaz()
                    current_pass['max_el'] = alt.degrees
            elif event == 2:  # Set
                if current_pass is not None:
                    current_pass['end'] = ti.utc_datetime()
                    current_pass['duration'] = current_pass['end'] - current_pass['start']
                    if current_pass['duration'] >= min_pass_duration:
                        ground_station_passes[gs['name']].append(current_pass)
                    current_pass = None

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write passes to separate files for each ground station
    for gs_name, passes in ground_station_passes.items():
        file_path = os.path.join(OUTPUT_DIR, f'{gs_name}_passes.txt')
        with open(file_path, 'w') as f:
            for pass_data in passes:
                duration = pass_data['duration']
                f.write(
                    f"Start: {pass_data['start'].strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"End: {pass_data['end'].strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Duration: {duration}, "
                    f"Max Elevation: {pass_data['max_el']:.2f} degrees\n"
                )

if __name__ == "__main__":
    compute_passes()