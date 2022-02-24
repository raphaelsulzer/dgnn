scan="/home/rsulzer/"
labatut="/home/adminlocal/PhD/cpp/mesh-tools/build/release/labatut"

wdir="/home/adminlocal/PhD/data/synthetic_room"

$scan -w $wdir -i "00000866.off" --export all --gclosed 0 --cameras 50 --points 50000 --noise 0.0025 --outliers 0.01

$labatut -w $wdir -i "00000866" -s npz --gco angle-5.0

