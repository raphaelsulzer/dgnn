from scan import scan
from feat import feat
from gt import gt
import os

path = "/home/adminlocal/PhD/data/synthetic_room"


if __name__ == "__main__":

    path = "/home/adminlocal/PhD/data/synthetic_room"
    files = os.listdir(path)
    files = ["00000007"]
    for f in files:
        print('\n',f)
        try:
            # scan(path,f)
            feat(path,f)
            # gt(path,f)
        except Exception as e:
            print(e)
