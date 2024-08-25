import os
from datetime import datetime, timedelta
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import ICRS, CartesianRepresentation, EarthLocation, AltAz
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs84
from tqdm import tqdm  # Import tqdm for the progress bar

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PARAMS_FILE = os.path.join(CURRENT_DIR, 'satelliteParameters.txt')
TLE_FILE = os.path.join(CURRENT_DIR, 'terra.tle')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'satellite_passes')

def read_parameters(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split(':')
                params[key.strip()] = value.strip()
    return params

def compute_passes():
    # Read parameters
    params = read_parameters(PARAMS_FILE)
    
    # Satellite parameters
    altitude_km = float(params['Altitude_km'])
    inclination_deg = float(params['Inclination_deg'])
    data_collection_rate_Tbit_per_day = float(params['Data_collection_rate_Tbit_per_day'])
    onboard_storage_Gbit = float(params['Onboard_storage_Gbit'])
    downlink_data_rate_Gbps = float(params['Downlink_data_rate_Gbps'])

    # Ground station parameters
    ground_stations = [
        {
            'name': 'Tenerife',
            'location': EarthLocation(lat=28.299525*u.deg, lon=-16.5101*u.deg, height=2393*u.m),
            'downlink_data_rate_Gbps': float(params['Ground_station_1'].split(',')[4])
        },
        {
            'name': 'Nemea',
            'location': EarthLocation(lat=37.8167*u.deg, lon=22.7167*u.deg, height=600*u.m),
            'downlink_data_rate_Gbps': float(params['Ground_station_2'].split(',')[4])
        },
        {
            'name': 'Trauen',
            'location': EarthLocation(lat=52.9167*u.deg, lon=9.6167*u.deg, height=110*u.m),
            'downlink_data_rate_Gbps': float(params['Ground_station_3'].split(',')[4])
        }
    ]

    # Cloud and atmospheric parameters
    cloud_cover_percentage_threshold = float(params['Cloud_cover_percentage_threshold'])
    turbulence_strength_threshold = params['Turbulence_strength_threshold']
    link_margin_dB = float(params['Link_margin_dB'])

    # Initialize scenario
    start_time = datetime(2020, 5, 1, 11, 36, 0)
    stop_time = start_time + timedelta(days=365)
    sample_time = timedelta(seconds=10)  # Reduced sample time to increase granularity

    # Read TLE and initialize satellite
    with open(TLE_FILE, 'r') as f:
        tle_lines = f.readlines()
    satellite = Satrec.twoline2rv(tle_lines[1], tle_lines[2])

    ground_station_passes = {gs['name']: [] for gs in ground_stations}

    total_iterations = (stop_time - start_time) // sample_time  # Calculate total number of iterations

    # Initialize the progress bar
    with tqdm(total=total_iterations) as pbar:
        current_time = start_time
        while current_time <= stop_time:
            jd, fr = Time(current_time).jd1, Time(current_time).jd2
            e, r, v = satellite.sgp4(jd, fr)
            if e != 0:
                print(f"SGP4 error at {current_time}: error code {e}")
                continue

            # Create a CartesianRepresentation object
            cart = CartesianRepresentation(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km)
            sat_pos = ICRS(cart)

            for gs in ground_stations:
                alt_az = sat_pos.transform_to(AltAz(obstime=Time(current_time), location=gs['location']))
                el = alt_az.alt.deg  # Get elevation (altitude) in degrees

                if el > 20:  # Satellite is visible and above 20° elevation
                    ground_station_passes[gs['name']].append(current_time.strftime('%Y-%m-%d %H:%M:%S'))

            current_time += sample_time
            pbar.update(1)  # Update the progress bar after each iteration

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write passes to text files
    for gs_name, passes in ground_station_passes.items():
        file_path = os.path.join(OUTPUT_DIR, f'{gs_name}_passes.txt')
        with open(file_path, 'w') as f:
            f.write('\n'.join(passes))

    print("Satellite passes computation completed and written to text files.")

if __name__ == "__main__":
    compute_passes()
