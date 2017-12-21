#!/usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

python3 build_fasttext_data.py
python3 build_model.py
python3 build_dictionary.py
python3 create_glove_vocab.py