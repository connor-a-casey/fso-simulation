<div align="center">
    <h1>
        <img src="assets/header.jpg">
    </h1>
</div>

# IAC 2024 Space Communications and Quantum Symposium

This repository contains resources related to the **75th International Astronautical Congress (IAC) 2024** symposium focused on Space Communications and Quantum Technologies.

## Presentation Information
**Title:** Advancing Free-Space Optical Communication System Architecture: Performance Analysis of Diverse Optical Ground Station Network Configurations

**Authors:** 
- Mr. Connor Casey: University of Massachusetts Amherst, United States
- Mr. Eugene Rotherham: University College London (UCL), United Kingdom
- Ms. Eva Fernandez Rodriguez: Netherlands Organisation for Applied Scientific Research (TNO), The Netherlands
- Ms. Karen Wendy Vidaurre Torrez: Kyushu Institute of Technology, Japan
- Mr. Maren Mashor: National Space Research and Development Agency (NASRDA), Nigeria
- Mr. Isaac Pike: University College London (UCL), United Kingdom

**About the Presentation:**
This presentation explores the frontier of space-based optical and quantum communications. It provides an in-depth analysis of various configurations of optical ground station networks to enhance communication systems in space.

## Repository Structure

The repository is organized into two main folders:

1. `src`: Contains the source code for each component of the model.
2. `data`: Stores input data and output results.

### Source Code (`src`)

The `src` folder contains subfolders for each component of the model:

1. satellite_passes
2. turbulence
3. cloud_cover
4. data_integrator
5. dynamic_analysis

Each subfolder contains the scripts necessary for that particular component of the model.

### Data (`data`)

The `data` folder is divided into two subfolders:

1. `input`: Contains the `satelliteParameters.txt` file with essential parameters for the model.
2. `output`: Stores the results from each component of the model in separate subfolders:
   - satellite_passes
   - turbulence
   - cloud_cover
   - data_integrator
   - dynamic_analysis

## Data Flow and Execution Order

The model components are executed in the following order:

1. satellite_passes
2. turbulence
3. cloud_cover
4. data_integrator
5. dynamic_analysis

### Key Points:

- Each component in the `src` folder reads data from the `satelliteParameters.txt` file in the `data/input` folder.
- The `satellite_passes` component also uses a `.tle` (Two-Line Element) file for satellite orbit data.
- The `cloud_cover` component interacts with an API gateway (EUMETSAT) for retrieving cloud cover data.
- The `data_integrator` component pulls data from the `satellite_passes`, `turbulence`, and `cloud_cover` components.
- The `dynamic_analysis` component uses data from both the `satelliteParameters.txt` file and the `data_integrator` component.

## Software Architecture
<div align="center">
    <h1>
        <img src="assets/software_architecture_vF.svg">
    </h1>
</div>

This diagram illustrates the repository structure, execution order, and data flow between components.

## Usage

### Reproducing the Simulation Environment Using Docker

To ensure a consistent and reproducible environment, we've containerized the simulation using Docker. Follow the steps below to set up and run the simulation environment.

#### Prerequisites:

- Install [Docker](https://www.docker.com/products/docker-desktop) on your system.
- Ensure Docker is running.

#### Steps to Pull and Run the Docker Container:

1. **Pull the Docker image from Docker Hub:**

    ```bash
    docker pull cocasey/fso-simulation:1.0.1
    ```

2. **Run the container interactively:**

    ```bash
    docker run -it cocasey/fso-simulation:1.0.1
    ```

    This command will start the container and open an interactive shell.

3. **Navigate to the project directory inside the container:**

    ```bash
    cd /path/to/project
    ```

4. **Set up the environment variables:**

    Create a `.env` file in the project root directory inside the container and add your EUMETSAT API keys:

    ```bash
    echo "CONSUMER_KEY=your_consumer_key_here" >> .env
    echo "CONSUMER_SECRET=your_consumer_secret_here" >> .env
    ```

5. **Add your TLE file:**

    Place your `.tle` file (e.g., `terra.tle`) into the appropriate directory inside the container:

    ```bash
    cp /host/path/to/yourfile.tle /container/path/to/satellite_passes/
    ```

    *(You may need to mount a volume or use `docker cp` to transfer files from the host to the container.)*

6. **Fill in necessary parameters:**

    Edit the `satelliteParameters.txt` file located in the `data/input` folder to include your specific parameters.

7. **Run the simulation script:**

    Execute the `run_simulation.zsh` script located in the `scripts` folder from the main directory:

    ```bash
    cd scripts
    ./scripts/run_simulation.zsh
    ```

    *Ensure the script has execute permissions. If not, you can make it executable with:*

    ```bash
    chmod +x run_simulation.zsh
    ```

8. **Check the output:**

    The results will be stored in the `data/output/dynamic_analysis` folder within the container.

#### Exiting the Container:

- To exit the interactive session, type `exit` or press `Ctrl+D`.

#### Optional: Saving Your Work

- If you need to save data generated within the container to your host machine, consider using Docker volumes or the `docker cp` command.

#### Additional Notes:

- Ensure that any changes made inside the container are saved or exported if needed, as they will not persist after the container is stopped unless volumes are used.

### Traditional Setup (Without Docker)

If you prefer to run the simulation without Docker, follow these steps:

1. **Install Dependencies:**

    - Python (version 3.9 or higher)
    - Use `pip` to install required Python packages:

      ```bash
      pip install -r requirements.txt
      ```

2. **Prepare the Environment:**

    - Add your TLE file (e.g., `terra.tle`) into the same folder as the `satellite_passes` component.
    - Create a `.env` file in the project root directory and add your EUMETSAT API keys:

      ```
      CONSUMER_KEY=your_consumer_key_here
      CONSUMER_SECRET=your_consumer_secret_here
      ```

    - Fill in all necessary parameters in the `satelliteParameters.txt` file located in the `data/input` folder.

3. **Execute the Simulation Script:**

    Run the `run_simulation.zsh` script located in the `scripts` folder from the main directory:

    ```bash
    cd scripts
    ./scripts/run_simulation.zsh
    ```

    *Ensure the script has execute permissions. If not, you can make it executable with:*

    ```bash
    chmod +x run_simulation.zsh
    ```

4. **Check the Output:**

    Check the `data/output/dynamic_analysis` folder for results from each component.

## Dependencies

- Satellite TLE data file (`.tle`)
- Access to the EUMETSAT API gateway for cloud cover data
- Docker (if using the Docker setup)
- Python (version 3.9 or higher)
- Required Python packages listed in `requirements.txt`

## Repository Contents:
- **Paper:** [Link to Published Paper](https://arxiv.org/abs/2410.23470)
- **Datasets:** [Link to Datasets](https://www.dropbox.com/scl/fo/k7yug64a3rukr89p8xlmq/AH92crjcwRxoaDdJj1JSdmM?rlkey=i7putlvrf36inva68dfwipo6y&st=e0bcoyof&dl=0)
- **Data Analysis:** [Link to Notebook](https://www.dropbox.com/scl/fo/b1rnfzf3o8bz4iplv2ql4/AL59S5MLe3bO77BnmGWxis0?rlkey=fm22cwnophj40uggofkulb1x7&st=ienm03r6&dl=0)

## Contact Information:
For inquiries or collaborations related to this presentation, please reach out to Connor Casey via the provided email addresse in AUTHORS.txt

We appreciate your interest in our work!

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

![CC BY-NC-ND License](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)

### Summary of License Terms:

- **Attribution (BY):** You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial (NC):** You may not use the material for commercial purposes.
- **NoDerivatives (ND):** If you remix, transform, or build upon the material, you may not distribute the modified material.

For full license details, please refer to the [Creative Commons BY-NC-ND 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

