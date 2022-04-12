import subprocess, os

mesh_tools = "/home/adminlocal/PhD/cpp/mesh-tools/build/release/feat"
mesh_tools = "/linkhome/rech/genlgm01/uku93eu/code/feat"

def feat(path,f):

    wdir = os.path.join(path,f)

    # extract features from mpu
    command = [mesh_tools,
               "-w", wdir,
               "-i", "scan/9",
               "-o", f,
               "-g", "mesh.off",
               "--gtype", "srd",
               "--pcp", "100",
               "-s", "npz",
               '-e', ""]
    print("\nrun: ", *command)
    p = subprocess.Popen(command)
    p.wait()


if __name__ == "__main__":

    path = "/home/adminlocal/PhD/data/synthetic_room"
    files = os.listdir(path)
    # files = ["00000007"]
    for f in files:
        try:
            feat(path,f)
        except Exception as e:
            print(e)
