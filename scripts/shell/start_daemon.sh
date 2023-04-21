#!/bin/bash
python -m tcpbroker -P 2>&1 &
python -m realsense_remote serve | bash