feat="/home/rsulzer/cpp/mesh-tools/build/release/feat"
#labatut="/home/adminlocal/PhD/cpp/mesh-tools/build/release/labatut"

wdir="/home/rsulzer/data2/yanis_ETH3D/courtyard"
$feat -w $wdir -i "scan/courtyard" -s "npz" -e ""

wdir="/home/rsulzer/data2/yanis_ETH3D/pipes"
$feat -w $wdir -i "scan/pipes" -s "npz" -e ""

wdir="/home/rsulzer/data2/yanis_ETH3D/terrace"
$feat -w $wdir -i "scan/terrace" -s "npz" -e ""
