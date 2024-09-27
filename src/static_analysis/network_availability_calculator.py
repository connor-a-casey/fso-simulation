import os
import pandas as pd
from datetime import datetime
import logging
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

def read_satellite_parameters(file_path):
    params = {}
    ground_stations = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('Ground_station_'):
                    _, data = line.split(':', 1)
                    station_info = [x.strip() for x in data.split(',')]
                    if len(station_info) != 6:
                        raise ValueError(f"Invalid ground station data: {line}")
                    ground_station = {
                        'Name': station_info[0],
                        'Latitude_deg': float(station_info[1]),
                        'Longitude_deg': float(station_info[2]),
                        'Altitude_m': float(station_info[3]),
                        'Downlink_data_rate_Gbps': float(station_info[4]),
                        'Tracking_accuracy_arcsec': float(station_info[5])
                    }
                    ground_stations.append(ground_station)
                elif ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()

    params['ground_stations'] = ground_stations
    
    if 'Cloud_cover_percentage_threshold' in params:
        params['Cloud_cover_threshold'] = float(params['Cloud_cover_percentage_threshold']) / 100.0 * 2.0
    else:
        logger.warning("Cloud cover threshold not found in parameters. Using default value of 1.0.")
        params['Cloud_cover_threshold'] = 1.0

    return params

def calculate_network_availability(start_date_str, end_date_str, params):
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    cloud_data_dir = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'cloud_cover')
    cloud_files = glob.glob(os.path.join(cloud_data_dir, '*_df.csv'))

    if not cloud_files:
        raise FileNotFoundError(f"No cloud cover data files found in {cloud_data_dir}")

    df_list = []
    for file in cloud_files:
        station_name = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file, sep=',', parse_dates=['time'])
        df = df[['time', 'cloud_cover']]
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by='time', inplace=True)
        df = df.rename(columns={'cloud_cover': f'cloud_{station_name}'})
        df_list.append(df)

    df_merged = pd.concat(df_list, axis=1)
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()].copy()
    df_merged = df_merged[(df_merged['time'] >= start_date) & (df_merged['time'] <= end_date)]

    cloud_columns = [col for col in df_merged.columns if col.startswith('cloud_')]
    condition_all_ogs = df_merged[cloud_columns].apply(lambda x: x < 1.5).any(axis=1)
    df_merged['Availability'] = condition_all_ogs.astype(int)

    df_merged['Month'] = df_merged['time'].dt.to_period('M')
    monthly_availability = df_merged.groupby('Month')['Availability'].mean() * 100

    full_month_range = pd.period_range(start='2023-06', end='2024-05', freq='M')
    monthly_availability = monthly_availability.reindex(full_month_range, fill_value=0)

    # Plot the monthly average network availability
    plt.figure(figsize=(12, 6))
    
    # Convert period index to datetime for proper date formatting
    x_values = [period.to_timestamp() for period in monthly_availability.index]
    plt.plot(x_values, monthly_availability.values, marker='o', linestyle='-', color='black')
    
    plt.title(f'Average Network Availability by Month ({start_date_str} to {end_date_str})')
    plt.xlabel('Month')
    plt.ylabel('Network Availability (%)')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Set y-axis range
    plt.ylim(0, 100)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the plot to the output directory
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'static_analysis', 'network_availability')
    os.makedirs(output_directory, exist_ok=True)
    plot_file = os.path.join(output_directory, "network_availability_plot.png")
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)

    monthly_availability_df = monthly_availability.reset_index()
    monthly_availability_df.columns = ['Month', 'Network Availability (%)']
    monthly_availability_file = os.path.join(output_directory, "monthly_network_availability.csv")
    monthly_availability_df.to_csv(monthly_availability_file, index=False)

    final_df = pd.DataFrame({
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Average Network Availability All OGS (%)': [monthly_availability.mean()]
    })

    output_file = os.path.join(output_directory, "network_availability_results_fixed.csv")
    final_df.to_csv(output_file, index=False)

    return final_df

if __name__ == "__main__":
    start_date_str = '2023-06-01'
    end_date_str = '2024-06-01'

    params_file_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
    if not os.path.exists(params_file_path):
        logger.error(f"Parameters file not found at {params_file_path}")
        exit(1)

    params = read_satellite_parameters(params_file_path)

    try:
        final_df = calculate_network_availability(start_date_str, end_date_str, params)
        logger.info("Network Availability Results:")
        logger.info(final_df)
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")

    logger.info("Processing complete. Check the output directory for results.")