import os,sys,shutil
import numpy as np

def main(c):
    test_folder = os.path.join(data_path, c, "test")
    test_list = os.listdir(test_folder)
    test_list = [item.split('_')[1].split('.')[0] for item in test_list]
    test_file = os.path.join(data_path, c, "test.lst")
    np.savetxt(test_file, test_list, delimiter="\n", fmt="%s")
    shutil.rmtree(test_folder)

    train_folder = os.path.join(data_path, c, "train")
    train_list = os.listdir(train_folder)
    train_list = [item.split('_')[1].split('.')[0] for item in train_list]
    train_file = os.path.join(data_path, c, "train.lst")
    np.savetxt(train_file, train_list, delimiter="\n", fmt="%s")
    shutil.rmtree(train_folder)

if __name__ == "__main__":

    data_path = "/mnt/raphael/ModelNet10"

    classes = os.listdir(data_path)


    for c in classes:
        main(c)
