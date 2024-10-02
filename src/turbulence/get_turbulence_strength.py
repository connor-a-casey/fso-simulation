import os
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm
from astropy.io import fits

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'turbulence')
PASS_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'satellite_passes')
PARAMS_FILE_PATH = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')

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
        LAT_IDX = max(0, min(LAT_IDX, DATA.shape[0] - 1))
        LON_IDX = max(0, min(LON_IDX, DATA.shape[1] - 1))

        # Get the turbulence strength value at the specified location
        TURBULENCE_STRENGTH = DATA[LAT_IDX, LON_IDX]

        return TURBULENCE_STRENGTH

# Function to parse the satellite passes files and get LoS times
def parse_all_los_times(directory, station_names):
    import pytz  # Ensure pytz is imported
    los_times = {}

    # Use tqdm to show progress bar while parsing LoS times
    for station_name in tqdm(station_names, desc="Parsing LoS times"):
        filename = f"{station_name}_passes.txt"
        file_path = os.path.join(directory, filename)
        los_times[station_name] = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("Start"):
                        parts = line.split(", ")
                        start_time_str = parts[0].split(": ")[1]
                        end_time_str = parts[1].split(": ")[1]
                        
                        # Convert the times to datetime objects and make them timezone-aware (UTC)
                        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
                        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
                        
                        los_times[station_name].append((start_time, end_time))
    return los_times

# Function to determine if it's day or night based on latitude, longitude, and current time
def is_daytime(LAT, LON, CURRENT_TIME):
    from astral import LocationInfo
    from astral.sun import sun

    location = LocationInfo(latitude=LAT, longitude=LON)
    
    # Calculate sunrise and sunset times for the location on the given date
    s = sun(location.observer, date=CURRENT_TIME.date())
    
    # Ensure all times are timezone-aware (convert sunrise and sunset to UTC)
    sunrise = s['sunrise'].astimezone(pytz.utc)
    sunset = s['sunset'].astimezone(pytz.utc)
    
    # Convert CURRENT_TIME to UTC (if it isn't already)
    CURRENT_TIME_UTC = CURRENT_TIME.astimezone(pytz.utc)
    
    # Return True if the current time is between sunrise and sunset (daytime)
    return sunrise <= CURRENT_TIME_UTC <= sunset

# Function to write turbulence data for each ground station to separate day and night files
def write_turbulence_data_to_files(DAY_FITS_FILE, NIGHT_FITS_FILE, GROUND_STATIONS, los_times):
    # Loop through each ground station with a tqdm progress bar
    for STATION in tqdm(GROUND_STATIONS, desc="Processing Ground Stations"):
        LATITUDE = STATION['LATITUDE_DEG']
        LONGITUDE = STATION['LONGITUDE_DEG']
        STATION_NAME = STATION['NAME']

        # Get LoS times for the current station
        station_los_times = los_times.get(STATION_NAME, [])

        # Create file names for day and night
        DAY_FILE_NAME = os.path.join(OUTPUT_DIR, f"{STATION_NAME}_day.txt")
        NIGHT_FILE_NAME = os.path.join(OUTPUT_DIR, f"{STATION_NAME}_night.txt")

        # Open files for writing day and night data
        with open(DAY_FILE_NAME, 'w') as DAY_FILE, open(NIGHT_FILE_NAME, 'w') as NIGHT_FILE:
            # Loop through the LoS times for this station
            for start_time, end_time in station_los_times:
                # Loop through every minute during the LoS window
                current_time = start_time
                while current_time <= end_time:
                    # Determine if it's day or night
                    if is_daytime(LATITUDE, LONGITUDE, current_time):
                        FITS_FILE = DAY_FITS_FILE
                        FILE = DAY_FILE
                        period = "day"
                    else:
                        FITS_FILE = NIGHT_FITS_FILE
                        FILE = NIGHT_FILE
                        period = "night"

                    # Calculate turbulence strength for the specific time
                    TURBULENCE_STRENGTH = read_turbulence_data(FITS_FILE, LATITUDE, LONGITUDE)

                    # Debugging: Print which period is being processed
                    print(f"Processing {period} data for {STATION_NAME} on {current_time}")

                    # Write the data to the appropriate file (day or night)
                    FILE.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}, {LATITUDE}, {LONGITUDE}, {TURBULENCE_STRENGTH}\n")

                    # Move to the next minute
                    current_time += timedelta(minutes=1)

# Function to parse satellite parameters from a text file and extract station names
def parse_satellite_parameters(FILE_PATH):
    GROUND_STATIONS = []
    with open(FILE_PATH, 'r') as FILE:
        LINES = FILE.readlines()
        for LINE in LINES:
            if LINE.startswith("Ground_station"):
                _, VALUE = LINE.split(":", 1)
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
    return GROUND_STATIONS

# Main function to call all other functions and allow updating the date range easily
def main():
    # Using the "cn2_ml_sfc_day_noTwilight.fits" file for day and "cn2_ml_sfc_night_noTwilight.fits" for night
    FITS_FILE_DAY = os.path.join(CURRENT_DIR, 'cn2_ml_sfc_day_noTwilight.fits')
    FITS_FILE_NIGHT = os.path.join(CURRENT_DIR, 'cn2_ml_sfc_night_noTwilight.fits')
    SATELLITE_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')

    # Parse the satellite and ground station parameters
    GROUND_STATIONS = parse_satellite_parameters(SATELLITE_PARAMS_FILE)
    station_names = [station['NAME'].replace(" ", "_") for station in GROUND_STATIONS]

    # Parse all LoS times from the satellite_passes directory with tqdm progress
    los_times = parse_all_los_times(PASS_DIR, station_names)

    # Write turbulence data to the output files based on LoS and day/night classification
    write_turbulence_data_to_files(FITS_FILE_DAY, FITS_FILE_NIGHT, GROUND_STATIONS, los_times)

# Entry point for the script
if __name__ == "__main__":
    main()
