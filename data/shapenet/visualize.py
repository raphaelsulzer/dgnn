import argparse, subprocess, sys, os
import open3d as o3d

# visualize
def dump(args):

    # args.set_view = True



    recon_mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    recon_mesh.compute_vertex_normals()
    recon_mesh.vertex_colors = recon_mesh.vertex_normals

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600,height=600, visible=True)
    # vis.create_window()
    vis.add_geometry(recon_mesh)

    # if(not args.set_view):
    vis.get_render_option().mesh_show_wireframe=False
    vis.get_render_option().light_on=True
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
    # vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    vis.get_render_option().mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    # view_file = os.path.join(args.user_dir,args.data_dir,"views",args.scene+"_viewpoint.json")

    vis.run()


    # # set viewfile
    # if(args.set_view):
    # vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # print("write view file: ", view_file)
    # o3d.io.write_pinhole_camera_parameters(view_file, param)
    # else:
    #     load viewfile
    #     ctr = vis.get_view_control()
    #     param = o3d.io.read_pinhole_camera_parameters(view_file)
    #     ctr.convert_from_pinhole_camera_parameters(param)
    #     vis.run()

    # make image
    image_dir = os.path.join(args.user_dir, args.data_dir, "..", "images")
    image_file = os.path.join(image_dir,args.mesh_file.split('/')[-1][:-4]+".png")
    print("save to: ", image_file)
    vis.capture_screen_image(image_file, do_render=True)
    vis.close()


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('-d', '--data_dir', type=str, default="data/ShapeNet/import/",
                        help='working directory which should include the different scene folders.')

    args = parser.parse_args()


    for f in os.listdir(os.path.join(args.user_dir,args.data_dir)):

        args.mesh_file = os.path.join(args.user_dir,args.data_dir,f)
        dump(args)



