#!/bin/bash

# Download BERT model
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip

mkdir ./bert

# Unzip the file
unzip uncased_L-12_H-768_A-12.zip -d ./bert/base


# https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip (tiny)
# https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip (mini)
# https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip (small)
# https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip (medium)
# https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip (base)
