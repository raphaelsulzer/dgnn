import argparse
import trimesh
import os
import numpy as np


def main(args):

    sfile = os.path.join(args.user_dir,args.data_dir,"scans","with_sensor",args.model+"_"+args.scan_conf+".ply")
    nfile = os.path.join(args.user_dir,args.data_dir,"scans","with_normals",args.model+"_"+args.scan_conf+".ply")


    pc = trimesh.load(sfile)

    sx = pc.metadata['ply_raw']['vertex']['data']['sx']
    sy = pc.metadata['ply_raw']['vertex']['data']['sy']
    sz = pc.metadata['ply_raw']['vertex']['data']['sz']
    sensor_pos = np.concatenate((sx, sy, sz), axis=1)

    pc = trimesh.load(nfile)
    nx = pc.metadata['ply_raw']['vertex']['data']['nx']
    ny = pc.metadata['ply_raw']['vertex']['data']['ny']
    nz = pc.metadata['ply_raw']['vertex']['data']['nz']
    normals = np.concatenate((nx, ny, nz), axis=1)

    points = pc.vertices.astype(np.float64)
    normals = normals.astype(np.float64)
    sensor_pos = sensor_pos.astype(np.float64)

    assert(normals.shape[0]==sensor_pos.shape[0])
    assert(pc.vertices.shape[0]==sensor_pos.shape[0])

    out_folder = os.path.join(args.user_dir,args.data_dir,"3_scan")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_file = os.path.join(args.user_dir,args.data_dir,"3_scan",args.model+"_"+args.scan_conf+".npz")
    np.savez(out_file,points=points,normals=normals,sensor_position=sensor_pos)

    a=5




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('--user_dir', type=str, default="/mnt/raphael/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="reconbench/",
                        help='working directory which should include the different scene folders.')

    args = parser.parse_args()

    models = ["anchor", "daratech", "dc", "lordquas", "gargoyle"]
    scan_confs = [0,1,2,3,4]
    for s in scan_confs:
        for args.model in models:
            args.scan_conf = str(s)
            main(args)