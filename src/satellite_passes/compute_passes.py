import os
from datetime import datetime, timedelta
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import ICRS, CartesianRepresentation, EarthLocation, AltAz
from sgp4.api import Satrec
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
    for i in range(1, 4):
        key = f'Ground_station_{i}'
        if key in params:
            station_data = params[key].split(',')
            ground_stations.append({
                'name': station_data[0].strip(),
                'location': EarthLocation(
                    lat=float(station_data[1]) * u.deg,
                    lon=float(station_data[2]) * u.deg,
                    height=float(station_data[3]) * u.m
                ),
                'downlink_data_rate_Gbps': float(station_data[4].strip())
            })
    return ground_stations


def load_satellite(tle_file):
    with open(tle_file, 'r') as f:
        tle_lines = f.readlines()
    return Satrec.twoline2rv(tle_lines[1], tle_lines[2])


def compute_passes():
    params = read_parameters(PARAMS_FILE)
    ground_stations = extract_ground_stations(params)
    satellite = load_satellite(TLE_FILE)

    start_time = datetime(2023, 6, 1, 0, 0, 0)
    stop_time = start_time + timedelta(days=4)
    sample_time = timedelta(minutes=1)

    ground_station_passes = {gs['name']: [] for gs in ground_stations}
    current_passes = {gs['name']: None for gs in ground_stations}

    total_iterations = int((stop_time - start_time) / sample_time)

    with tqdm(total=total_iterations) as pbar:
        current_time = start_time
        while current_time <= stop_time:
            jd, fr = Time(current_time).jd1, Time(current_time).jd2
            e, r, v = satellite.sgp4(jd, fr)

            if e != 0:
                print(f"SGP4 error at {current_time}: error code {e}")
                current_time += sample_time
                pbar.update(1)
                continue

            cart = CartesianRepresentation(x=r[0] * u.km, y=r[1] * u.km, z=r[2] * u.km)
            sat_pos = ICRS(cart)

            for gs in ground_stations:
                alt_az = sat_pos.transform_to(AltAz(obstime=Time(current_time), location=gs['location']))
                el = alt_az.alt.deg

                if el > 20:  # Satellite is visible and above 20° elevation
                    if current_passes[gs['name']] is None:
                        current_passes[gs['name']] = {'start': current_time}
                else:
                    if current_passes[gs['name']] is not None:
                        current_passes[gs['name']]['end'] = current_time - sample_time
                        ground_station_passes[gs['name']].append(current_passes[gs['name']])
                        current_passes[gs['name']] = None

            current_time += sample_time
            pbar.update(1)

    # Close any open passes
    for gs_name, current_pass in current_passes.items():
        if current_pass is not None:
            current_pass['end'] = stop_time
            ground_station_passes[gs_name].append(current_pass)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write passes to separate files for each ground station
    for gs_name, passes in ground_station_passes.items():
        file_path = os.path.join(OUTPUT_DIR, f'{gs_name}_passes.txt')
        with open(file_path, 'w') as f:
            for pass_data in passes:
                f.write(
                    f"Start: {pass_data['start'].strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"End: {pass_data['end'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

    print("Satellite passes computation completed and written to text files.")


if __name__ == "__main__":
    compute_passes()
