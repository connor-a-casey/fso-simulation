#!/bin/bash

# Run compute_passes.py
echo "Running compute_passes.py..."
python src/satellite_passes/compute_passes.py
if [ $? -ne 0 ]; then
    echo "Error: compute_passes.py failed to execute."
    exit 1
fi

# Run get_turbulence_strength.py
echo "Running get_turbulence_strength.py..."
python src/turbulence/get_turbulence_strength.py
if [ $? -ne 0 ]; then
    echo "Error: get_turbulence_strength.py failed to execute."
    exit 1
fi

# Run compute_cloud_cover.py in the conda environment
echo "Activating conda environment 'cfgrib_env'..."
source activate cfgrib_env || conda activate cfgrib_env
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the conda environment 'cfgrib_env'."
    exit 1
fi

echo "Running compute_cloud_cover.py..."
python src/cloud_cover/compute_cloud_cover.py
if [ $? -ne 0 ]; then
    echo "Error: compute_cloud_cover.py failed to execute."
    conda deactivate
    exit 1
fi

echo "Deactivating conda environment..."
conda deactivate

# Run data_integrator.py
echo "Running data_integrator.py..."
python src/data_integrator/data_integrator.py
if [ $? -ne 0 ]; then
    echo "Error: data_integrator.py failed to execute."
    exit 1
fi

# Run network_availability_calculator.py
echo "Running network_availability_calculator.py..."
python src/dynamic_analysis/network_availability_calculator.py
if [ $? -ne 0 ]; then
    echo "Error: network_availability_calculator.py failed to execute."
    exit 1
fi

# Run data_throughput_calculator.py
echo "Running data_throughput_calculator.py..."
python src/dynamic_analysis/data_throughput_calculator.py
if [ $? -ne 0 ]; then
    echo "Error: data_throughput_calculator.py failed to execute."
    exit 1
fi

echo "All scripts executed successfully."
