#!/bin/bash

# download and unzip dataset
wget https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-train-image-only.zip
wget https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-test.zip

mkdir clear100
unzip clear100-train-image-only.zip
unzip clear100-test.zip
rm clear100-train-image-only.zip
rm clear100-test.zip
mv train_image_only clear100
mv test clear100
echo done
