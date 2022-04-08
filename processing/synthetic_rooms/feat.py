import subprocess, os

mesh_tools = "/home/adminlocal/PhD/cpp/mesh-tools/build/debug/feat"

def feat(path,f):

    # outfile = os.path.join(args.wdir,"dgnn",args.overwrite):
    #     print("exists!")
    #     return

    wdir = os.path.join(path,f)

    # extract features from mpu
    command = [mesh_tools,
               "-w", wdir,
               "-i", "scan",
               "-o", f,
               "-g", "mesh.off",
               "--gtype", "srd",
               "--pcp", "10",
               "-s", "npz",
               '-e', ""]
    print("\nrun: ", *command)
    p = subprocess.Popen(command)
    p.wait()


if __name__ == "__main__":

    path = "/home/adminlocal/PhD/data/synthetic_room"
    files = os.listdir(path)
    files = ["00000007"]
    for f in files:
        try:
            feat(path,f)
        except Exception as e:
            print(e)
