from subprocess import call
import os
from shutil import rmtree

import numpy as np
from settings import CAFFE_ROOT, GPU


def create_lmdbs(split_num, cwd):
    trainfile = os.path.join(cwd, "../splits/train{0}.txt".format(split_num))
    testfile = os.path.join(cwd, "../splits/test{0}.txt".format(split_num))
    # create lmdb
    lmdb_tool_path = os.path.join(CAFFE_ROOT, "tools/convert_imageset")
    train_lmdb_path = os.path.join(cwd, "../images/trainlmdb")
    test_lmdb_path = os.path.join(cwd, "../images/testlmdb")
    # delete old lmdbs if they exist
    [rmtree(p) for p in [train_lmdb_path, test_lmdb_path] if os.path.isdir(p)]
    call([lmdb_tool_path, "../images/", trainfile, train_lmdb_path])
    call([lmdb_tool_path, "../images/", testfile, test_lmdb_path])


def prepare_solver_prototxt(storage_dir):
    solvertemplate_path = "../caffe_data/solvertemplate.prototxt"
    solvername = os.path.basename(solvertemplate_path)
    solverpath = storage_dir + solvername
    call(["cp", solvertemplate_path, storage_dir])
    solverfile = open(solverpath, 'a')
    solverfile.write("snapshot_prefix: " + "\"" + storage_dir + "snapshots/\"")
    solverfile.close()
    return solverpath


def train_split(split_num, cwd, solverpath):
    caffe_path = os.path.join(CAFFE_ROOT, "tools/caffe")
    relative_model_path = ("models/bvlc_reference_caffenet/")
    weights_path = os.path.join(CAFFE_ROOT, relative_model_path,
                                "bvlc_reference_caffenet.caffemodel")

    # fine tune from bvlc reference caffe model
    call([caffe_path, "train", "-solver", solverpath,
          "-weights", weights_path, "-gpu", str(GPU)])


def evaluate_results(split):
    # create the confusion matrix and link the misclassified images
    call(["./CaffeNetAnalysis/CaffeNetAnalysisMain",
          "../splits/test{0}.txt".format(split),
          os.path.join(os.path.abspath("../images/"), ""),
          "../caffe_data/deploy.prototxt",
          "../results/{0}/snapshots/_iter_10000.caffemodel".format(split),
          "../classid.txt",
          "../results/{0}/".format(split), str(GPU)])


def export_np_mat_with_header(mean_mat, std_dev_mat, file_to_copy_header,
                              export_filename, skip_header=0, skip_footer=0):
    with open(file_to_copy_header, 'r') as f:
        lines = f.readlines()
        header = lines[:skip_header]
        lines = lines[skip_header:len(lines) - skip_footer]
        first_column = [l.split(';')[0] for l in lines]
    with open(export_filename, 'w') as f:
        if header:
            f.writelines(header)  # copy header
        for i in range(len(first_column)):
            if len(mean_mat.shape) == 1:
                mean_stddev = [("{:.3f}".format(mean_mat[i]),
                                "{:.3f}".format(std_dev_mat[i]))]
            else:
                mean_stddev = zip(map("{:.3f}".format, mean_mat[i]),
                                  map("{:.3f}".format, std_dev_mat[i]))
            mean_mat_line = "; ".join([" +- ".join(m) for m in mean_stddev])
            f.write(first_column[i] + "; " + mean_mat_line + "; \n")


def evaluate_mean():
    confusion_mats = []
    accuracy_mats = []
    for i in range(5):
        confusion_mat_path = "../results/{0}/confusion_matrix.csv".format(i)
        accuracy_mat_path = "../results/{0}/accuracy.csv".format(i)
        confusion_mats.append(np.genfromtxt(confusion_mat_path,
                                            usecols=list(range(1, 26)),
                                            delimiter=';', skip_header=1,
                                            skip_footer=1))
        accuracy_mats.append(np.genfromtxt(accuracy_mat_path,
                                           usecols=(1,),
                                           delimiter=';'))
    export_np_mat_with_header(np.mean(confusion_mats, axis=0),
                              np.std(confusion_mats, axis=0),
                              "../results/0/confusion_matrix.csv",
                              "../results/mean_confusion_matrix.csv", 1, 1)
    export_np_mat_with_header(np.mean(accuracy_mats, axis=0),
                              np.std(accuracy_mats, axis=0),
                              "../results/0/accuracy.csv",
                              "../results/mean_accuracy_matrix.csv")


def check_if_training_files_exist(storage_dir):
    if not os.path.isdir(storage_dir):
        return
    print("It seems there already exist files from a previous"
          "training. Delete all files in folder results? y/n")
    while True:
        inp = str(input())
        if inp.lower() == 'y':
            for i in range(5):
                rmtree("../results/".format(i))
            break
        elif inp.lower() == 'n':
            exit()


if __name__ == "__main__":
    for i in range(5):  # train the 5 splits
        cwd = os.getcwd()
        storage_dir = "../results/{0}/snapshots/".format(i)
        check_if_training_files_exist(storage_dir)
        os.makedirs(storage_dir)
        solverpath = prepare_solver_prototxt("../results/{0}/".format(i))
        create_lmdbs(i, cwd)
        train_split(i, cwd, solverpath)
        evaluate_results(i)
    evaluate_mean()
