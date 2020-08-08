#!/bin/bash

# Download BERT model
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip

mkdir ./models
mkdir ./models/bert
mkdir ./models/bert/uncased_L-12_H-768_A-12

# Unzip the file
unzip uncased_L-12_H-768_A-12.zip -d ./models/bert/uncased_L-12_H-768_A-12
