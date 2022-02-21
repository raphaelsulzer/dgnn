#!/bin/bash
echo "Start downloading Berger et al dataset..."
cd data
gdown --id 18usEYyY0A1KqbVdbwu7QDA2rH-UNRdsj
unzip reconbench.zip
rm reconbench.zip
echo "Done!"