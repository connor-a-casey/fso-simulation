import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.ticker as mtick  # Importing ticker for formatting

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
        for line in f.read().splitlines():
            if line and not line.startswith('#'):
                if line.startswith('Ground_station_'):
                    # Extract ground station data
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
    return params

def calculate_data_throughput(start_date_str, end_date_str, params):
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Paths
    combined_data_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'data_integrator', 'combined_data.csv')

    # Read combined data
    combined_df = pd.read_csv(combined_data_path, parse_dates=['Start', 'End'])
    combined_df = combined_df[(combined_df['Start'] >= start_date) & (combined_df['End'] <= end_date)]

    if combined_df.empty:
        logger.warning(f"No data available between {start_date_str} and {end_date_str}")
        return pd.DataFrame(), pd.DataFrame()

    # Get satellite data rate
    satellite_data_rate_bps = float(params['Downlink_data_rate_Gbps']) * 1e9  # Convert Gbps to bps

    # Initialize variables for overall totals
    total_data_transmitted_bits = 0
    maximum_possible_data_transmitted_bits = 0

    # Prepare a dictionary to hold monthly data
    monthly_data = {}

    # Iterate over passes for all ground stations with a progress bar
    for _, pass_data in tqdm(combined_df.iterrows(), total=combined_df.shape[0], desc="Processing passes"):
        # Get start and end times
        start_time = pass_data['Start']
        end_time = pass_data['End']
        duration_seconds = (end_time - start_time).total_seconds()

        ground_station_name = pass_data['Ground Station']
        # Find the corresponding ground station's data rate
        ground_station_data = next((station for station in params['ground_stations'] if station['Name'] == ground_station_name), None)
        if ground_station_data is None:
            raise ValueError(f"Ground station {ground_station_name} not found in parameters")

        # Get the ground station data rate
        gs_data_rate_bps = ground_station_data['Downlink_data_rate_Gbps'] * 1e9  # Convert Gbps to bps

        # Use the minimum of satellite and ground station data rates
        effective_data_rate_bps = min(satellite_data_rate_bps, gs_data_rate_bps)

        # Calculate data transmitted during this pass
        data_transmitted_bits = effective_data_rate_bps * duration_seconds
        max_data_transmitted_bits = satellite_data_rate_bps * duration_seconds
        maximum_possible_data_transmitted_bits += max_data_transmitted_bits

        # Determine the month for this pass
        month = start_time.strftime('%Y-%m')

        # Initialize the dictionary for the month if not already done
        if month not in monthly_data:
            monthly_data[month] = {
                'Total Data Transmitted (Gbits)': 0,
                'Maximum Possible Data (Gbits)': 0
            }

        # Update monthly totals
        monthly_data[month]['Total Data Transmitted (Gbits)'] += data_transmitted_bits / 1e9  # Convert to Gbits
        monthly_data[month]['Maximum Possible Data (Gbits)'] += max_data_transmitted_bits / 1e9  # Convert to Gbits

        # Accumulate the overall totals
        total_data_transmitted_bits += data_transmitted_bits

    # Calculate overall network throughput
    total_data_transmitted_Gbits = total_data_transmitted_bits / 1e9
    overall_percentage_data_throughput = (total_data_transmitted_bits / maximum_possible_data_transmitted_bits) * 100 if maximum_possible_data_transmitted_bits > 0 else 0.0

    # Convert the monthly dictionary to a DataFrame
    monthly_df = pd.DataFrame.from_dict(monthly_data, orient='index')
    monthly_df.index.name = 'Month'  # Name the index to 'Month' for proper reset
    monthly_df['Percentage Data Throughput (%)'] = (monthly_df['Total Data Transmitted (Gbits)'] / monthly_df['Maximum Possible Data (Gbits)']) * 100

    # Save the monthly data to a CSV file
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'static_analysis', 'data_throughput')
    os.makedirs(output_directory, exist_ok=True)
    monthly_output_file = os.path.join(output_directory, "monthly_data_throughput.csv")
    monthly_df.to_csv(monthly_output_file, index=True, index_label='Month')

    # Create a separate DataFrame for overall network throughput
    overall_df = pd.DataFrame({
        'Start Date': [start_date_str],
        'End Date': [end_date_str],
        'Total Data Transmitted (Gbits)': [total_data_transmitted_Gbits],
        'Percentage Data Throughput (%)': [overall_percentage_data_throughput]
    })

    # Save the overall network throughput to a separate CSV file
    overall_output_file = os.path.join(output_directory, "overall_network_throughput.csv")
    overall_df.to_csv(overall_output_file, index=False)

    logger.info(f"Monthly data saved to: {monthly_output_file}")
    logger.info(f"Overall network throughput saved to: {overall_output_file}")

    return monthly_df, overall_df

def plot_monthly_data(monthly_df):
    # Reset the index to get 'Month' as a column
    monthly_df = monthly_df.reset_index()

    # Ensure the 'Month' column exists
    if 'Month' not in monthly_df.columns:
        logger.error("The 'Month' column is missing from the DataFrame after reset_index.")
        raise KeyError("'Month' column not found in the DataFrame.")

    # Extract the month strings for x-axis labels
    x_labels = monthly_df['Month'].astype(str)

    # Create numerical x positions
    x_positions = np.arange(len(x_labels))

    # Convert data to numpy arrays for plotting
    total_data_transmitted = monthly_df['Total Data Transmitted (Gbits)'].to_numpy()
    percentage_throughput = monthly_df['Percentage Data Throughput (%)'].to_numpy()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the total data transmitted on the primary y-axis
    ax1.bar(x_positions, total_data_transmitted, color='black', label='Total Data Transmitted (Gbits)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Data Transmitted (Gbits)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')

    # Add comma formatting to the primary y-axis
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    # Create a secondary y-axis for the throughput percentage
    ax2 = ax1.twinx()
    ax2.plot(x_positions, percentage_throughput, color='blue', marker='o', label='Percentage Data Throughput (%)')
    ax2.set_ylabel('Percentage Data Throughput (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Optionally, format the secondary y-axis (percentages typically don't need commas, but for consistency)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Set the title and legend
    plt.title('Monthly Data Transmission and Throughput')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Adjust layout and save the plot
    fig.tight_layout()
    output_directory = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'output', 'static_analysis', 'data_throughput')
    plot_file = os.path.join(output_directory, "monthly_data_throughput_plot.png")
    plt.savefig(plot_file)
    plt.close(fig)  # Close the figure to free up memory
    logger.info(f"Monthly data and throughput plot saved to: {plot_file}")

if __name__ == "__main__":
    # Update the start and end dates to match your data
    start_date_str = '2023-06-01'
    end_date_str = '2024-06-01'

    params_file_path = os.path.join(PROJECT_ROOT, 'IAC-2024', 'data', 'input', 'satelliteParameters.txt')
    if not os.path.exists(params_file_path):
        logger.error(f"Parameters file not found at {params_file_path}")
        exit(1)

    params = read_satellite_parameters(params_file_path)

    try:
        monthly_df, overall_df = calculate_data_throughput(start_date_str, end_date_str, params)
        if not monthly_df.empty:
            plot_monthly_data(monthly_df)
            logger.info("Monthly data and throughput plot saved successfully.")
        else:
            logger.warning("No data to plot.")
        
        logger.info(f"Overall Network Throughput:\n{overall_df.to_string(index=False)}")
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")

    logger.info("Processing complete. Check the output directory for results.")
