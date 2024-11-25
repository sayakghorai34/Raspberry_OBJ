FROM debian:bookworm

# Install base dependencies
RUN apt update && apt install -y --no-install-recommends gnupg
RUN echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

# Install Python and required system packages
RUN apt update && apt -y upgrade
RUN apt update && apt install -y --no-install-recommends \
        python3-pip \
        python3-picamera2 \
        python3-opencv \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . /app/

# Install Python dependencies
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Install the package in development mode
#RUN pip3 install --break-system-packages -e .

# Expose the port used by the server
EXPOSE 8000

# Run the server
# CMD ["python3", "-m", "src.main.scripts.run_server", "--port", "8000"]
