#!/usr/bin/env bash
mkdir -p ../data/austen
aws s3 cp s3://neuralnets/blstm_blog/data/austen.txt ../data/austen/austen.txt

cd ../data/
git clone https://github.com/scikit-learn/scikit-learn.git
