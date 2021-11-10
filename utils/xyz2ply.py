import argparse
import numpy as np
import point_cloud_utils as pcu

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("xyz_path", type=str)
    args = argparser.parse_args()

    if not args.xyz_path.endswith(".xyz"):
        raise ValueError("Input file must end in .xyz")

    pts = []
    with open(args.xyz_path) as f:
        for line in f:
            pts.append([float(c) for c in line.strip().split()])

    pts = np.array(pts)
    out_path = args.xyz_path[:-len("xyz")] + "ply"
    print(out_path)
    pcu.write_ply(out_path, pts, np.zeros([0, 3], dtype=np.int32),
                  np.zeros([0, 3], dtype=pts.dtype), np.zeros([0, 3], dtype=pts.dtype))


if __name__ == "__main__":
    main()