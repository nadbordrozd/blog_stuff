#!/usr/bin/env bash
bash get_data.sh

python prepare_input_files.py ../data/austen 'austen.txt' ../data/austen_clean
awk '{ print $0 > "../data/austen_clean/austen"++i".txt" }' RS='\n\n\n' ../data/austen_clean/austen.txt
python prepare_input_files.py ../data/scikit-learn/ '*.py' ../data/sklearn_clean

