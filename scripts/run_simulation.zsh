#!/bin/zsh

# display warning message
echo "Warning: Running this script will delete any existing data in the data/output folder. Do you want to proceed? (y/n)"

while true; do
    echo "enter choice: y/n "
    read choice
    case "$choice" in
        y|Y ) 
            echo "proceeding with the action..."
            # place the code for the action here
            break
            ;;
        n|N )
            echo "action canceled."
            exit 0
            ;;
        * )
            echo "invalid input. please enter 'y' or 'n'."
            ;;
    esac
done


# Run compute_passes.py
echo "Running compute_passes.py..."
python3 src/satellite_passes/compute_passes.py
if [ $? -ne 0 ]; then
    echo "Error: compute_passes.py failed to execute."
    exit 1
fi
# Run get_turbulence_strength.py
echo "Running get_turbulence_strength.py..."
python3 src/turbulence/get_turbulence_strength.py
if [ $? -ne 0 ]; then
    echo "Error: get_turbulence_strength.py failed to execute."
    exit 1
fi

# Run compute_cloud_cover.py
echo "Running compute_cloud_cover.py..."
python3 src/cloud_cover/compute_cloud_cover.py
if [ $? -ne 0 ]; then
    echo "Error: compute_cloud_cover.py failed to execute."
    conda deactivate
    exit 1
fi


# Run data_integrator.py
echo "Running data_integrator.py..."
python3 src/data_integrator/data_integrator.py
if [ $? -ne 0 ]; then
    echo "Error: data_integrator.py failed to execute."
    exit 1
fi

# Run network_availability_calculator.py
echo "Running network_availability_calculator.py..."
python3 src/dynamic_analysis/network_availability_calculator.py
if [ $? -ne 0 ]; then
    echo "Error: network_availability_calculator.py failed to execute."
    exit 1
fi

# Run data_throughput_calculator.py
echo "Running data_throughput_calculator.py..."
python3 src/dynamic_analysis/data_throughput_calculator.py
if [ $? -ne 0 ]; then
    echo "Error: data_throughput_calculator.py failed to execute."
    exit 1
fi

echo "All scripts executed successfully. See the data/output folders for respective results."
