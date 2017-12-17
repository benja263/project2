#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

# Use the relative path to build_vocab.sh instead of the relative path to the caller
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cat ../../data/raw/train_pos_full.txt ../../data/raw/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
python3 parse_test_dataset.py
