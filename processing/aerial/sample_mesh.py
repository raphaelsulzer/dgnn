import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..', 'utils'))
from libmesh import check_mesh_contains
import re


def main(args):
    input_files = glob.glob(os.path.join(args.gt_folder, '*.off'))
    # try:
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)
    # except:
    #     pass


def process_path(in_path, args):

    in_file = os.path.basename(in_path)

    modelname = re.split(r'[_.]+', in_file)[1]

    # check if there is a scan with this configuration
    # scan_file = os.path.join(args.scan_folder,
    #                             modelname + '_' + str(args.config) +  '.npz')
    # if not os.path.exists(scan_file):
    #     return

    mesh = trimesh.load(in_path, process=False)

    # Determine bounding box

    # Standard bounding boux
    loc = np.zeros(3)
    scale = args.scale

    # Export various modalities
    # this is the input point cloud for reconstruction, noise is added in the data loader before training
    if args.pointcloud_folder is not None:
        export_pointcloud(mesh, modelname, loc, scale, args)


    # this is for training, it's sampled points in the bounding box that are used as occupancy samples during training
    if args.points_folder is not None:
        export_points(mesh, modelname, loc, scale, args)




def export_pointcloud(mesh, modelname, loc, scale, args):


    # filename = os.path.join(args.pointcloud_folder, modelname.split('_')[1], 'pointcloud.npz')
    filename = os.path.join(args.pointcloud_folder, modelname, 'pointcloud.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return
    else:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    # should be as simple as replacing the points and normals array of this the following two lines with the
    # ones read from the already existing sampling
    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]
    points = points.astype(dtype)
    normals = normals.astype(dtype)
    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)

    #####################################################################
    #### read from an existing sampling (for making ConvONet files) #####
    #####################################################################

    # existing_sampling_file = os.path.join(args.scan_folder,
    #                         modelname + '_' + str(args.config) + '.npz')
    # content = np.load(existing_sampling_file)

    # points = content['points'].astype(dtype)
    # normals = content['normals'].astype(dtype)
    # gt_normals = content['gt_normals'].astype(dtype)
    # sensors = content['sensor_position'].astype(dtype)

    # np.savez(filename, points=points, normals=normals, gt_normals=gt_normals, sensors=sensors, loc=loc, scale=scale)
    # np.savez(filename, points=points, normals=normals, sensors=sensors, loc=loc, scale=scale)


def export_points(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return

    # filename = os.path.join(args.points_folder, modelname.split('_')[1], 'points.npz')
    filename = os.path.join(args.points_folder, modelname, 'points.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return
    else:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = args.scale + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    occupancies = check_mesh_contains(mesh, points)

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)

    if args.packbits:
        occupancies = np.packbits(occupancies)

    print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale)



if __name__ == '__main__':
    ''' This script samples points in the bounding box and on the surface of a mesh and saves them 
    in a points.npz and pointcloud.npz file'''
    ''' Useful for evaluating a reconstruction '''





    parser = argparse.ArgumentParser('Sample a watertight mesh.')

    parser.add_argument('--dataset_dir', type=str, default="/mnt/raphael/ModelNet10/",
                        help='Path to the dataset.')

    parser.add_argument('--category', type=str, default=None,
                        help='Process specific category class only')

    parser.add_argument('--gt_folder', type=str,
                        help='Path to input watertight meshes.')
    parser.add_argument('--n_proc', type=int, default=0,
                        help='Number of processes to use.')

    parser.add_argument('--scale', type=float, default=1.0,
                        help='scale of ground truth = bounding box')

    parser.add_argument('--bbox_padding', type=float, default=0.,
                        help='Padding for bounding box')


    parser.add_argument('--pointcloud_folder', type=str,
                        help='Output path for point cloud.')
    parser.add_argument('--scan_folder', type=str,
                        help='Output path for point cloud.')
    parser.add_argument('--pointcloud_size', type=int, default=100000,
                        help='Size of point cloud.')

    parser.add_argument('--points_folder', type=str,
                        help='Output path for points.')
    parser.add_argument('--points_size', type=int, default=100000,
                        help='Size of points.')
    parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                        help='Ratio of points to sample uniformly'
                             'in bounding box.')
    parser.add_argument('--points_sigma', type=float, default=0.05,
                        help='Standard deviation of gaussian noise added to points'
                             'samples on the surfaces.')
    parser.add_argument('--points_padding', type=float, default=0.1,
                        help='Additional padding applied to the uniformly'
                             'sampled points on both sides (in total).')

    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite output.')
    parser.add_argument('--float16', action='store_true',
                        help='Whether to use half precision.')
    parser.add_argument('--packbits', action='store_true',
                        help='Whether to save truth values as bit array.')

    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0,1,2,3,4],
                        help='which config file to load')

    args = parser.parse_args()


    args.packbits = True

    if args.category is not None:
        categories = [args.category]
    else:
        categories = os.listdir(args.dataset_dir)



    for i,c in enumerate(categories):
        if c.startswith('.'):
            continue
        print("\n############## Processing {} - {}/{} ############\n".format(c,i+1,len(categories)))



        # args.scan_folder = os.path.join(args.dataset_dir, c, "3_scan", str(config))    # where the scans are
        # args.gt_folder = os.path.join(args.dataset_dir, c, "2_watertight")             # where the meshes are
        # args.points_folder = os.path.join(args.dataset_dir, c, "convonet",str(config))
        # args.pointcloud_folder = os.path.join(args.dataset_dir, c, "convonet",str(config))
        args.scan_folder = os.path.join(args.dataset_dir, c, "3_scans")    # where the scans are
        args.gt_folder = os.path.join(args.dataset_dir, c, "2_watertight")             # where the meshes are
        args.points_folder = os.path.join(args.dataset_dir, c, "eval")
        args.pointcloud_folder = os.path.join(args.dataset_dir, c, "eval")

        main(args)


            # ### train
            # # save a train.lst file here
            # args.train_folder = os.path.join(args.dataset_dir, c, "train")
            # if not os.path.exists(args.points_folder):
            #     os.makedirs(args.points_folder)
            # if not os.path.exists(args.pointcloud_folder):
            #     os.makedirs(args.pointcloud_folder)
            # # save a train_str(conf).lst file here
            # file = open(os.path.join(args.points_folder,"train.lst"), "w")
            # for f in os.listdir(args.train_folder):
            #     if f.startswith('.'):
            #         continue
            #     file.write(f.split('.')[0].split('_')[1] + "\n")
            # file.close()
            #
            # ### test
            # # save a test.lst file here
            # args.test_folder = os.path.join(args.dataset_dir, c, "test")
            # if not os.path.exists(args.points_folder):
            #     os.makedirs(args.points_folder)
            # if not os.path.exists(args.pointcloud_folder):
            #     os.makedirs(args.pointcloud_folder)
            # file = open(os.path.join(args.points_folder,"test.lst"), "w")
            # for f in os.listdir(args.test_folder):
            #     if f.startswith('.'):
            #         continue
            #     file.write(f.split('.')[0].split('_')[1] + "\n")
            # file.close()


