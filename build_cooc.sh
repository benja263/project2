#!/bin/bash

./src/preprocessing/build_vocab.sh
./src/preprocessing/cut_vocab.sh
python3 src/preprocessing/pickle_vocab.py
python3 src/preprocessing/cooc.py
python3 src/preprocessing/glove_solution.py