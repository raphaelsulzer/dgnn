#!/bin/bash
cd ../data
echo "Start downloading ..."
wget https://drive.google.com/file/d/1dNn91OUSxRKWabmkTu1b2SwGaGl0SeG1/view?usp=sharing
unzip reconbench.zip
rm reconbench.zip
echo "Done!"