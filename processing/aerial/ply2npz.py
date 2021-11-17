import argparse
import trimesh
import os
import numpy as np


def main(args):

    nfile = os.path.join(args.in_dir,f)
    sfile = os.path.join(args.in_dir,'..',"sensor",f)


    pc = trimesh.load(sfile)

    sx = pc.metadata['ply_raw']['vertex']['data']['scalar_x_origin']
    sy = pc.metadata['ply_raw']['vertex']['data']['scalar_y_origin']
    sz = pc.metadata['ply_raw']['vertex']['data']['scalar_z_origin']
    sensor_pos = np.concatenate((sx, sy, sz), axis=1)
    sensor_pos = sensor_pos.astype(np.float64)

    pc = trimesh.load(nfile)
    nx = pc.metadata['ply_raw']['vertex']['data']['nx']
    ny = pc.metadata['ply_raw']['vertex']['data']['ny']
    nz = pc.metadata['ply_raw']['vertex']['data']['nz']
    normals = np.concatenate((nx, ny, nz), axis=1)
    normals = normals.astype(np.float64)

    points = pc.vertices.astype(np.float64)

    assert(normals.shape[0]==sensor_pos.shape[0])
    assert(pc.vertices.shape[0]==sensor_pos.shape[0])

    out_file = os.path.join(args.in_dir,f[:-4]+".npz")
    np.savez(out_file,points=points,normals=normals,sensor_position=sensor_pos)

    a=5




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/yanis/aerial/scan",
                        help='working directory which should include the different scene folders.')

    args = parser.parse_args()



    ## train
    folder = os.path.join(args.user_dir,args.data_dir,"normal")
    for f in os.listdir(folder):
        args.f = f
        args.in_dir = folder
        main(args)


    # folder = os.path.join(args.user_dir,args.data_dir,"test")
    # for f in os.listdir(folder):
    #     args.f = f
    #     args.in_dir = os.path.join(folder,f)
    #     main(args)