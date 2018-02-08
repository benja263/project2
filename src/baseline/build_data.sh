#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

./build_vocab.sh
./cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
python3 glove_solution.py
python3 build_train_feature_matrix.py
python3 build_test_feature_matrix.py