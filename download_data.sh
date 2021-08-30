#!/bin/bash

while getopts u:p: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        p) password=${OPTARG};;
    esac
done

mkdir ./data
cd ./data

echo "downloading files..."

if [ -f "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip" ]; then
    echo "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip already downloaded"
else
    wget ftp://$username:$password@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip
fi


if [ -f "ISPRS_semantic_labeling_Vaihingen.zip" ]; then
    echo "ISPRS_semantic_labeling_Vaihingen.zip exists"
else
    wget ftp://$username:$password@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip
fi

echo "files downloading"
echo "unzipping files..."
unzip ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip -d labels
unzip ISPRS_semantic_labeling_Vaihingen.zip -d ./images
echo "done!"





