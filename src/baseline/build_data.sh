#!/bin/bash

./build_vocab.sh
./cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
python3 glove_solution.py
python3 build_train_feature.py
python3 build_test_feature.py