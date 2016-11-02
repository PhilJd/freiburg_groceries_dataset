/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */
#include <vector>
#include <string>
#include <utility>  // pair
#include <stdlib.h>  // atoi 
#include "./CaffeController.h"
#include "./CaffeNetAnalysis.h"



void PrintUsage() {
    printf("\n USAGE: CaffenetAnalysisMain path/to/TEST.TXT img/source/dir path/to/DEPLOY.PROTOTXT path/to/WEIGHTS.TXT path/to/CLASSID.TXT path/to/CONFUSIONMATRIX_OUTDIR\n optionalGPUId");
}

int main(int argc, char *argv[]) {
    if (argc < 7 || argc > 8) {
      PrintUsage();
      return 1;
    }
    std::string ground_truth_filepath(argv[1]);
    std::string image_dir(argv[2]);
    std::string deploy_filepath(argv[3]);
    std::string weights_filepath(argv[4]);
    std::string idtoclass_filepath(argv[5]);
    std::string confusion_outpath(argv[6]);
    CaffeController controller;
    controller.SetDeployConfig(deploy_filepath);
    controller.LoadWeights(weights_filepath);
    controller.InitIdToClassDict(idtoclass_filepath);
    if (argc == 8) {
        controller.SetMode(CONTROLLER_GPU);
        controller.SetDeviceNumber(atoi(argv[7]));
    } else {
        controller.SetMode(CONTROLLER_CPU);
    }
    if (!controller.IsInitialized()) {
        std::exit(0);
    }
    CaffeNetAnalysis::Analyze(controller, ground_truth_filepath, image_dir,
                              confusion_outpath);
}

