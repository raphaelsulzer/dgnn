import argparse
import trimesh
import os
import numpy as np
from tqdm import tqdm

def main(args):

    sfile = os.path.join(args.in_dir,args.f)


    pc = trimesh.load(sfile)

    sx = pc.metadata['ply_raw']['vertex']['data']['sx']
    sy = pc.metadata['ply_raw']['vertex']['data']['sy']
    sz = pc.metadata['ply_raw']['vertex']['data']['sz']
    sensor_pos = np.concatenate((sx, sy, sz), axis=1)

    points = pc.vertices.astype(np.float64)
    sensor_pos = sensor_pos.astype(np.float64)

    assert(pc.vertices.shape[0]==sensor_pos.shape[0])


    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_file = os.path.join(args.out_dir,args.f.split('.')[0]+".npz")
    np.savez(out_file,points=points,normals=sensor_pos,sensor_position=sensor_pos)

    a=5




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('--user_dir', type=str, default="/mnt/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="ShapeNetManifoldPlus/",
                        help='working directory which should include the different scene folders.')

    args = parser.parse_args()

    categories = os.listdir(os.path.join(args.user_dir,args.data_dir))
    scan_confs = [0,1,2,3,4]

    for i,c in enumerate(categories):
        if c.startswith('.'):
            continue
        print("\n############## Processing {}/{} ############\n".format(i+1,len(categories)))
        args.in_dir = os.path.join(args.user_dir, args.data_dir, c, "scans", "with_sensor")
        files = os.listdir(args.in_dir)
        for f in files:
            args.out_dir = os.path.join(args.user_dir,args.data_dir,c,"3_scans")
            args.f = f
            main(args)