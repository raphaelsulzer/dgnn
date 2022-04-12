from scan import scan
from feat import feat
from gt import gt
import os
import argparse
import shutil

path = "/home/adminlocal/PhD/data/synthetic_room"
path = "/linkhome/rech/genlgm01/uku93eu/data/synthetic_room_dataset/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='reconbench evaluation')
    parser.add_argument('-c','--category', type=str, default=None,
                        help='Indicate the category class')
    args = parser.parse_args()

    path = os.path.join(path,args.category)

    files = os.listdir(path)
    # files = ["00000007"]
    for f in files:
        print('\n',f)
        try:
            scan(path,f)
            feat(path,f)
            shutil.rmtree(os.path.join(path,f,"gt"),ignore_errors=True)
        except Exception as e:
            print(e)
