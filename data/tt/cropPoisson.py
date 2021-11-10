import open3d as o3d
import argparse
import os

# read Poisson ply

def crop(args):


    mesh_file = os.path.join(args.data_dir,args.scene,args.scene+"_poisson.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # cropfile = os.path.join(args.data_dir,args.scene,args.scene+".json")
    # crop_vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    #
    # cropped_mesh = crop_vol.crop_triangle_mesh(mesh)

    pc_file = os.path.join(args.data_dir,args.scene,args.scene+".ply")
    pc = o3d.io.read_point_cloud(pc_file)
    bb = pc.get_oriented_bounding_box()
    bb.scale(1.1,bb.get_center())

    cropped_mesh = mesh.crop(bb)

    out_file = os.path.join(args.data_dir,args.scene,args.scene+"_poisson.off")
    o3d.io.write_triangle_mesh(out_file,cropped_mesh)

    # bb = o3d.geometry.TriangleMesh(cropped_mesh.get_oriented_bounding_box().TriangleMesh)
    # out_file = os.path.join(args.data_dir, args.scene, args.scene + "_bb.off")
    # o3d.io.write_triangle_mesh(out_file, bb)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='crop and save poisson as off')

    parser.add_argument('-d', '--data_dir', type=str, default="/home/adminlocal/PhD/data/TanksAndTemples/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs = '+', type=str, default=["Truck"],
                        help='on which scene to execute pipeline.')

    args = parser.parse_args()

    if(args.scenes[0] == 'all'):
        args.scenes = []
        # for scene in os.listdir(args.user_dir+args.data_dir):
        for scene in os.listdir(args.data_dir):
            if (not scene[0] == "x"):
                args.scenes+=[scene]


    for i,scene in enumerate(args.scenes):
        args.scene=scene
        crop(args)
