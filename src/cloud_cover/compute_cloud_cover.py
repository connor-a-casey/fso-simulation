import os
import base64
import requests
import warnings
from dotenv import load_dotenv
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import shutil
import time
import eumdac
from tqdm import tqdm
import concurrent.futures
import numpy as np 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
TOKEN_URL = "https://api.eumetsat.int/token"
MAX_RETRIES = 10
RAW_GRIB_STAGING = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover','cloud_staging')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover')



def read_satellite_parameters(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split(':', 1)
                params[key.strip()] = value.strip()
    # Convert start and end time to datetime objects
    params['Start_time'] = datetime.strptime(params['Start_time'], '%Y-%m-%d')
    params['End_time'] = datetime.strptime(params['End_time'], '%Y-%m-%d')
    return params


def get_eumetsat_api_token():
    credentials = f"{CONSUMER_KEY}:{CONSUMER_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    try:
        response = requests.post(TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        token = eumdac.AccessToken((CONSUMER_KEY, CONSUMER_SECRET))
        print(f"Successfully obtained token. Expires in {token_data['expires_in']} seconds.")
        return token
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def get_collection(token, collection_id):
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection_id)
    try:
        print(selected_collection.search_options)
    except eumdac.datastore.DataStoreError as error:
        print(f"Error related to the data store: '{error.msg}'")
    except eumdac.collection.CollectionError as error:
        print(f"Error related to the collection: '{error.msg}'")
    except requests.exceptions.RequestException as error:
        print(f"Unexpected error: {error}")
    return selected_collection, datastore


def get_product_list(selected_collection, start_date, end_date):
    query = {'dtstart': start_date.isoformat(), 'dtend': end_date.isoformat()}
    if 'type' in selected_collection.search_options and selected_collection.search_options['type']:
        query['type'] = 'MSGCLMK'

    products = selected_collection.search(**query)
    try:
        print(f'Found Datasets: {len(products)} datasets for the given time range.')
    except eumdac.collection.CollectionError as error:
        print(f"Error related to the collection: '{error.msg}'")
    except requests.exceptions.RequestException as error:
        print(f"Unexpected error: {error}")

    product_list = []
    for product in products:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                product_list.append(str(product))
                break
            except eumdac.product.ProductError as error:
                print(f"Error related to the product: '{error.msg}'")
            except requests.exceptions.RequestException as error:
                print(f"Unexpected error: {error}")
    return product_list


def read_ogs_data(file_path):
    ogs_locations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Ground_station"):
                parts = line.split(":")[1].strip().split(", ")
                name = parts[0]
                latitude = float(parts[1])
                longitude = float(parts[2])
                ogs_locations.append({'ogs_code': name, 'lon/lat': (longitude, latitude)})
    return pd.DataFrame(ogs_locations)


def get_station_bbox(ogs_data):
    scan_radius = 0.1  # Adjust this as needed for appropriate bounding box size
    station_bbox_list = []
    for index, row in ogs_data.iterrows():
        station_code = row['ogs_code']
        coord = row['lon/lat']
        lat = coord[1]
        lon = coord[0]
        if lon < 0:
            lon += 360  # Adjust negative longitude to 0-360 degrees
        lat_min = lat - scan_radius
        lat_max = lat + scan_radius
        lon_min = lon - scan_radius
        lon_max = lon + scan_radius
        bbox = [lat_min, lat_max, lon_min, lon_max]
        station_bbox_list.append({
            'ogs_code': station_code,
            'coord': (lon, lat),
            'bbox': bbox
        })
    return pd.DataFrame(station_bbox_list)


def remove_index_file(grib_file_path):
    index_file_path = grib_file_path + '.idx'
    if os.path.exists(index_file_path):
        os.remove(index_file_path)
        print(f"Removed index file: {index_file_path}")


def download_products(product_id, collection_id, datastore):
    selected_product = datastore.get_product(product_id=product_id, collection_id=collection_id)
    retries = 0
    grib_file_name = product_id + '.grb'
    grib_file_path = os.path.join(RAW_GRIB_STAGING, grib_file_name)

    while retries < MAX_RETRIES:
        try:
            with selected_product.open(entry=grib_file_name) as fsrc, \
                    open(grib_file_path, mode='wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
                break
        except eumdac.product.ProductError as error:
            if error.extra_info['status'] >= 500:
                time.sleep(retries * 10)
                print(f"Attempt {retries}/{MAX_RETRIES}: {error}")
                retries += 1
            else:
                print(f"Error related to the product '{selected_product}' while trying to download it: '{error.msg}'")
        except requests.exceptions.ConnectionError as error:
            time.sleep(retries * 10)
            print(f"Attempt {retries}/{MAX_RETRIES}: {error}")
            retries += 1
        except requests.exceptions.RequestException as error:
            if error.extra_info['status'] >= 500:
                time.sleep(retries * 10)
                print(f"Attempt {retries}/{MAX_RETRIES}: {error}")
                retries += 1
            else:
                print(f"Unexpected error: {error}")

    return grib_file_path


def process_grib_file(grib_file_path, station_bbox_df):
    try:
        # Remove existing index file
        remove_index_file(grib_file_path)

        # Open the GRIB file
        dataset = xr.open_dataset(grib_file_path, engine='cfgrib')

        # Extract time, latitude, longitude, and cloud cover data
        time_values = dataset['time'].values
        latitudes = dataset['latitude'].values
        longitudes = dataset['longitude'].values

        # Adjust dataset longitudes to 0-360 degrees if necessary
        if longitudes.min() < 0:
            longitudes = np.where(longitudes < 0, longitudes + 360, longitudes)

        cloud_data = dataset['p260537'].values  # Assuming 'p260537' is the correct variable for cloud cover

        results = []

        # Ensure time_values is iterable (likely a 1D array)
        if time_values.ndim == 0:
            # If time is a single value, wrap it in a list
            time_values = [time_values]
        
        for _, station in station_bbox_df.iterrows():
            station_code = station['ogs_code']
            bbox = station['bbox']

            # Filter data within the bounding box
            lat_mask = (latitudes >= bbox[0]) & (latitudes <= bbox[1])
            lon_mask = (longitudes >= bbox[2]) & (longitudes <= bbox[3])
            cloud_cover_within_bbox = cloud_data[lat_mask & lon_mask]

            if cloud_cover_within_bbox.size > 0:
                # Calculate the average cloud cover within the bounding box for each timestamp
                for t_idx, timestamp in enumerate(time_values):
                    avg_cloud_cover = cloud_cover_within_bbox.mean()  # Take the mean value
                    results.append({
                        'station_code': station_code,
                        'time': timestamp,  # Use the time value for each step
                        'latitude': station['coord'][1],  # Use the station's coordinates
                        'longitude': station['coord'][0],
                        'cloud_cover': avg_cloud_cover
                    })
            else:
                print(f"Station {station_code}: No data within bounding box")

        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error processing GRIB file: {e}")
        return pd.DataFrame()



def process_single_product(product_id, collection_id, datastore, station_bbox_df):
    try:
        grib_file_path = download_products(product_id, collection_id, datastore)
        df = process_grib_file(grib_file_path, station_bbox_df)
        return df
    except Exception as e:
        print(f"Error processing product {product_id}: {e}")
        return pd.DataFrame()


def main():
    # Suppress specific warning
    warnings.filterwarnings("ignore", message="dlopen")
    warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

    # Load environment variables
    load_dotenv()

    os.makedirs(RAW_GRIB_STAGING, exist_ok=True)

    params = read_satellite_parameters(os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt'))
    collection_id = 'EO:EUM:DAT:MSG:CLM'
    start_date = params['Start_time']
    end_date = params['End_time']
    
    token = get_eumetsat_api_token()
    selected_collection, datastore = get_collection(token, collection_id)

    ogs_data = read_ogs_data(os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt'))
    station_bbox_df = get_station_bbox(ogs_data)

    # Initialize a dictionary to store weather data for each station
    station_weather_dataframes = {row['ogs_code']: pd.DataFrame(columns=['time', 'latitude', 'longitude', 'cloud_cover']) for _, row in station_bbox_df.iterrows()}

    current_date = start_date
    total_products_processed = 0

    while current_date <= end_date:
        print(f"Processing data for {current_date.strftime('%Y-%m-%d')}")
        product_list = get_product_list(selected_collection, current_date, current_date + timedelta(days=1))

        # Use tqdm to track progress across the total number of datasets
        with tqdm(total=len(product_list), desc="Processing datasets") as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for product_id in product_list:
                    futures.append(executor.submit(process_single_product, product_id, collection_id, datastore, station_bbox_df))

                for future in concurrent.futures.as_completed(futures):
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        for station_code in station_weather_dataframes.keys():
                            station_df = result_df[result_df['station_code'] == station_code]
                            if not station_df.empty:
                                station_weather_dataframes[station_code] = pd.concat([station_weather_dataframes[station_code], station_df], ignore_index=True)
                    # Update the progress bar after each dataset is processed
                    pbar.update(1)

        current_date += timedelta(days=1)

    for station_code, df in station_weather_dataframes.items():
        # Ensure the correct order of columns
        df = df[['time', 'latitude', 'longitude', 'cloud_cover']]

        # Save each DataFrame to a CSV file, using a comma as the separator
        df.to_csv(f"{OUTPUT_DIR}/{station_code}_eumetsat_{start_date.date()}_{end_date.date()}_detailed_df.csv", sep=',', index=False)

if __name__ == "__main__":
    main()