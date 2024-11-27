# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the working directory in the container
WORKDIR /app

# Command to run your application
CMD ["./scripts/run_simulation.zsh"]
