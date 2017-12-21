#!/usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
python3 download_sw.py
python3 generate_filtered_datasets.py
python3 format_data.py

