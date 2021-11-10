import argparse, subprocess, sys, os





def sure(args):

    working_dir = os.path.join(args.user_dir,args.data_dir)
    args.scene_conf=args.scene+"_"+str(args.conf)

    if(args.sure_method.split(',')[0]=="rt"):
        folder='labatu/'
    else:
        folder='clf/'+args.sure_method.split(',')[0]+'/'
        if('sv' in args.gco):
            folder+='sv/'
        elif('angle' in args.gco):
            folder+='angle/'
        elif('cc' in args.gco):
            folder+='cc/'
        else:
            folder+='no/'

    outfolder = os.path.join(working_dir,'reconstructions',folder)
    if(not os.path.exists(outfolder)):
        os.makedirs(outfolder)

    command = [args.user_dir + args.sure_dir + "/sure",
               "-w", working_dir,
               "-i", "scans/with_sensor/"+args.scene_conf,
               "-m", args.sure_method,
               "-s", "lidar",
               "-e", args.export_options,
               "--omanifold", "0",
               "--eval", "1"]
    if(args.gco.split('-')[0] != "no"):
        command += ["--gco", args.gco]

    print("run command: "+str(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    # exit the whole programm if this step didn't work
    if (p.returncode):
        sys.exit(1)
    # get the stdout output and save it in an array
    # from here: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
    output = []
    for line in iter(p.stdout.readline, b''):
        print(line.decode("utf-8")[:-1])
        output.append(line.decode("utf-8")[:-1])
    output = output[-2:]

    return output








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reconbench evaluation')

    # choose method:
    parser.add_argument('--mode', type=str, default="clf",
                        help='choose a mode. choices=["download","clf","poisson","onet"]')

    parser.add_argument('--user_dir', type=str, default="/home/adminlocal/PhD/",
                        help='the user folder, or PhD folder.')
    parser.add_argument('--data_dir', type=str, default="data/reconbench/",
                        help='working directory which should include the different scene folders.')
    parser.add_argument('-s', '--scenes', nargs='+', type=str, default=["anchor"],
                        help='on which scene to execute pipeline.')
    parser.add_argument('-c', '--confs', nargs='+', type=int, default=[0],
                        help='which config file to load')

    # Sure options
    parser.add_argument('--sure_dir', type=str, default="cpp/surfaceReconstruction/build/release",
                        help='Indicate the sure build directory, pointing to .../build/release folder starting from user_dir')
    parser.add_argument('-m','--sure_method', type=str, default="rt,labatu,1,-1,32",
                        help='the reconstruction method, default: rt,1,labatu')
    parser.add_argument('-p', type=str, default="",
                        help='which prediction. e.g. 9575')
    parser.add_argument('--gco', type=str, default="angle-5.0",
                        help='graph cut optimization type,weight. default: area,1.0')
    parser.add_argument('-e', '--export_options', type=str, default="i",
                        help='graph cut optimization type,weight. default: area,1.0')

    # Poisson options
    parser.add_argument('--poisson_dir', type=str, default="cpp/PoissonReconOri/Bin/Linux",
                        help='Indicate the poisson build directory, starting from user_dir')
    parser.add_argument('--depth', type=int, default=6,
                        help='Poisson depth')
    parser.add_argument('--trim', type=int, default=5,
                        help='Poisson trimming value')

    # whether to download the predictino files
    parser.add_argument('-d', '--download_prediction', type=int, default=0,
                        help='download the predictino')
    # wether to upload results
    parser.add_argument('-u', '--upload', type=int, default=1,
                        help='upload the results to google spreadsheet')

    # eval options
    parser.add_argument('--n_sample_points', type=int, default=100000, help='how many points to sample for IoU and Chamfer')

    args = parser.parse_args()

    args.mode = args.mode.split('+')


    args.confs = [0,1,2,3,4]

    args.scenes = ['anchor','gargoyle','dc','daratech','lordquas']


    sum = 0
    num = 0
    for scene in args.scenes:
        args.scene = scene
        for i,conf in enumerate(args.confs):
            args.conf = conf

            sn=sure(args)
            sum+=float(sn[0])
            num+=float(sn[1])


    print(sum/num)

