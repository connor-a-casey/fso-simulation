import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'turbulence')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to read FITS file and get turbulence data
def read_turbulence_data(FITS_FILE, LAT, LON):
    with fits.open(FITS_FILE) as HDUL:
        DATA = HDUL[0].data

        # Define the geographical bounds
        LAT_MIN = -90.0
        LAT_MAX = 90.0
        LON_MIN = -180.0
        LON_MAX = 180.0
        
        # Get the resolution of the data
        LAT_RES = (LAT_MAX - LAT_MIN) / DATA.shape[0]
        LON_RES = (LON_MAX - LON_MIN) / DATA.shape[1]

        # Find the nearest grid point to the requested latitude and longitude
        LAT_IDX = int((LAT - LAT_MIN) / LAT_RES)
        LON_IDX = int((LON - LON_MIN) / LON_RES)

        # Ensure the indices are within the data bounds
        LAT_IDX = np.clip(LAT_IDX, 0, DATA.shape[0] - 1)
        LON_IDX = np.clip(LON_IDX, 0, DATA.shape[1] - 1)

        # Get the turbulence strength value at the specified location
        TURBULENCE_STRENGTH = DATA[LAT_IDX, LON_IDX]

        return TURBULENCE_STRENGTH

# Function to parse satellite parameters from a text file
def parse_satellite_parameters(FILE_PATH):
    SATELLITE_PARAMS = {}
    GROUND_STATIONS = []

    with open(FILE_PATH, 'r') as FILE:
        LINES = FILE.readlines()
        for LINE in LINES:
            if ':' in LINE:
                KEY, VALUE = LINE.split(':', 1)
                KEY = KEY.strip()
                VALUE = VALUE.strip()
                
                # Check if the line is for ground stations
                if KEY.startswith('Ground_station'):
                    STATION_DATA = VALUE.split(',')
                    STATION_INFO = {
                        'NAME': STATION_DATA[0].strip(),
                        'LATITUDE_DEG': float(STATION_DATA[1].strip()),
                        'LONGITUDE_DEG': float(STATION_DATA[2].strip()),
                        'ALTITUDE_M': float(STATION_DATA[3].strip()),
                        'DOWNLINK_DATA_RATE_GBPS': float(STATION_DATA[4].strip()),
                        'TRACKING_ACCURACY_ARCSEC': float(STATION_DATA[5].strip())
                    }
                    GROUND_STATIONS.append(STATION_INFO)
                else:
                    SATELLITE_PARAMS[KEY] = VALUE

    return SATELLITE_PARAMS, GROUND_STATIONS

# Function to write turbulence data to output files for each ground station
def write_turbulence_data_to_files(DAY_FITS_FILE, NIGHT_FITS_FILE, GROUND_STATIONS):
    for STATION in tqdm(GROUND_STATIONS, desc="Processing Ground Stations"):
        LATITUDE = STATION['LATITUDE_DEG']
        LONGITUDE = STATION['LONGITUDE_DEG']
        STATION_NAME = STATION['NAME'].replace(" ", "_")  # Replace spaces with underscores for the file name
        
        # Calculate turbulence strength for day and night
        TURBULENCE_STRENGTH_DAY = read_turbulence_data(DAY_FITS_FILE, LATITUDE, LONGITUDE)
        TURBULENCE_STRENGTH_NIGHT = read_turbulence_data(NIGHT_FITS_FILE, LATITUDE, LONGITUDE)
        
        # Write day data to a file named after the ground station
        DAY_FILE_NAME = os.path.join(OUTPUT_DIR, f"{STATION_NAME}_day.txt")
        with open(DAY_FILE_NAME, 'w') as DAY_FILE:
            DAY_FILE.write(f"Turbulence strength during the day at {STATION['NAME']} ({LATITUDE}, {LONGITUDE}): {TURBULENCE_STRENGTH_DAY}\n")
        
        # Write night data to a file named after the ground station
        NIGHT_FILE_NAME = os.path.join(OUTPUT_DIR, f"{STATION_NAME}_night.txt")
        with open(NIGHT_FILE_NAME, 'w') as NIGHT_FILE:
            NIGHT_FILE.write(f"Turbulence strength during the night at {STATION['NAME']} ({LATITUDE}, {LONGITUDE}): {TURBULENCE_STRENGTH_NIGHT}\n")

# Example usage
FITS_FILE_DAY = os.path.join(CURRENT_DIR, 'cn2_ml_sfc_day_noTwilight.fits')
FITS_FILE_NIGHT = os.path.join(CURRENT_DIR, 'cn2_ml_sfc_night_noTwilight.fits')
SATELLITE_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input','satelliteParameters.txt')

# Parse the satellite and ground station parameters
SATELLITE_PARAMS, GROUND_STATIONS = parse_satellite_parameters(SATELLITE_PARAMS_FILE)

# Write turbulence data to the output files
write_turbulence_data_to_files(FITS_FILE_DAY, FITS_FILE_NIGHT, GROUND_STATIONS)
