from subprocess import call
import os
from settings import CUDA_DIR, CAFFE_ROOT


def compile_analysis_tool():
    os.chdir("CaffeNetAnalysis")
    link = "LIBS += -L" + os.path.join(CAFFE_ROOT, "build/lib") \
           + " -L" + os.path.join(CUDA_DIR, "lib") + "\n"
    include = "INCDIRS = -I " + os.path.join(CUDA_DIR, "include") \
              + " -I " + os.path.join(CAFFE_ROOT, "include") + "\n"
    with open("Makefile_template", 'r') as f:
        makefile_template = f.readlines()
    makefile_template.insert(5, link)
    makefile_template.insert(6, include)
    with open("Makefile", 'w') as f:
        f.writelines(makefile_template)
    call(['make'])
    os.chdir("..")


def download_caffemodel_ifneeded():
    # Requires bvlc_reference.caffemodel; download if it doesn't exist
    relative_model_path = ("models/bvlc_reference_caffenet/")
    model_path = os.path.join(CAFFE_ROOT, relative_model_path)
    weights_path = os.path.join(model_path,
                                "bvlc_reference_caffenet.caffemodel")
    if not os.path.isfile(weights_path):
        downloadtool = os.path.join(CAFFE_ROOT,
                                    "scripts/download_model_binary.py")
        call(["python2", downloadtool, model_path])


if __name__ == "__main__":
    compile_analysis_tool()
    download_caffemodel_ifneeded()
