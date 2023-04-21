#!/bin/bash
# Assuming using the gnome-terminal (Ubuntu)
CONDA_ENV=articulated

conda activate $CONDA_ENV

echo "Syncing NTP Server"
./scripts/sync_ntp.sh

echo "Launching Recorders"
gnome-terminal -t "Terminal 1" -- bash -c "bash ./scripts/start_daemon.sh; exec bash;"

echo "Launching IMU Console"
gnome-terminal -t "Terminal 2" -- bash -c "bash -c 'python -m tcpbroker'; exec bash;"

echo "Launching Experiment console"
gnome-terminal -t "Terminal 3" -- bash -c "bash -c 'python src/experiment.py --config=./config/config.json'; exec bash;"