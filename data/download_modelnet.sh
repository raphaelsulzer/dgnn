#!/bin/bash
echo "Start downloading ModelNet10..."
cd data
wget https://zenodo.org/record/5940164/files/ModelNet10.zip?download=1
unzip ModelNet10.zip
rm ModelNet10.zip
echo "Done!"