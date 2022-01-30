#!/bin/bash
echo "Start downloading ..."
cd data
gdown --id 18usEYyY0A1KqbVdbwu7QDA2rH-UNRdsj
unzip reconbench.zip
rm reconbench.zip
echo "Done!"