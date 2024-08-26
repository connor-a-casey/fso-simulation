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

# Ignore warnings related to the environment setup
warnings.filterwarnings("ignore", message="dlopen")

# Load environment variables
load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
TOKEN_URL = "https://api.eumetsat.int/token"
MAX_RETRIES = 10
RAW_GRIB_STAGING = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover', 'staging')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover')

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
        query['type'] = 'MSGCLMK'  # Example value, adjust accordingly

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
    scan_radius = 0.1
    station_bbox_list = []
    for index, row in ogs_data.iterrows():
        station_code = row['ogs_code']
        coord = row['lon/lat']
        lat_min = coord[1] - scan_radius
        lat_max = coord[1] + scan_radius
        lon_min = coord[0] - scan_radius
        lon_max = coord[0] + scan_radius
        bbox = [lat_min, lat_max, lon_min, lon_max]
        station_bbox_list.append({
            'ogs_code': station_code,
            'coord': coord,
            'bbox': bbox
        })
    return pd.DataFrame(station_bbox_list)

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
    
    return grib_file_name

def process_grib_file(grib_file_path, station_bbox_df):
    try:
        dataset = xr.open_dataset(grib_file_path, engine='cfgrib')
        cloud_data = dataset['tcc']  # Example: Total cloud cover ('tcc') might be the relevant variable
        results = []

        for _, station in station_bbox_df.iterrows():
            station_code = station['ogs_code']
            bbox = station['bbox']
            # Extract the cloud cover data for the station's bounding box
            cloud_cover = cloud_data.sel(latitude=slice(bbox[0], bbox[1]), longitude=slice(bbox[2], bbox[3])).mean().item()
            results.append({'station_code': station_code, 'cloud_cover': cloud_cover})

        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error processing GRIB file: {e}")
        return pd.DataFrame()

def main():
    collection_id = 'EO:EUM:DAT:MSG:CLM'
    start_date = datetime(2023, 6, 1)  # Set this to the desired day for testing
    end_date = datetime(2023, 6, 1)    # Set the end date to the same as the start date

    token = get_eumetsat_api_token()
    selected_collection, datastore = get_collection(token, collection_id)

    ogs_data = read_ogs_data(os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt'))
    station_bbox_df = get_station_bbox(ogs_data)

    station_weather_dataframes = {row['ogs_code']: pd.DataFrame(columns=['time', 'cloud_cover']) for _, row in station_bbox_df.iterrows()}

    current_date = start_date
    while current_date <= end_date:  # Process only one day
        print(f"Processing data for {current_date.strftime('%Y-%m-%d')}")
        product_list = get_product_list(selected_collection, current_date, current_date)  # Only query for this single day

        for i, product_id in enumerate(tqdm(product_list, desc=f"Processing products for {current_date.strftime('%Y-%m-%d')}")):
            print(f"Parsing GRIB: {i+1}/{len(product_list)}")
            try:
                grib_file_name = download_products(product_id, collection_id, datastore)
                grib_file_path = os.path.join(RAW_GRIB_STAGING, grib_file_name)
                df = process_grib_file(grib_file_path, station_bbox_df)
                
                for _, row in df.iterrows():
                    station_code = row['station_code']
                    station_weather_dataframes[station_code] = station_weather_dataframes[station_code].append(
                        {'time': current_date, 'cloud_cover': row['cloud_cover']}, ignore_index=True)

            except Exception as e:
                print(f"An error occurred: {type(e).__name__} - {e}")
                continue

        current_date += timedelta(days=1)

    for station_code, df in station_weather_dataframes.items():
        df.to_csv(f"{OUTPUT_DIR}/{station_code}_eumetsat_{start_date.date()}_{end_date.date()}_df.csv", sep='\t')

if __name__ == "__main__":
    main()