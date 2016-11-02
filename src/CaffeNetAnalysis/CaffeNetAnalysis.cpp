/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#include "./CaffeNetAnalysis.h"
#include <vector>
#include <string>
#include <fstream>  // ifstream
#include <cassert>
#include "opencv2/highgui/highgui.hpp"  // cv::imread
#include "opencv2/imgproc/imgproc.hpp"  // cvtColor
#include "ConfusionMatrix.h"
#include "LinkWronglyClassified.h"

// ____________________________________________________________________________
void CaffeNetAnalysis::Analyze(const CaffeController& controller,
                               const std::string& ground_truth_filepath,
                               const std::string& image_dir,
                               const std::string& output_path) {
    std::vector<ClassAndProbability> ground_truth, netresult, batch;
    std::vector<cv::Mat> imgs;
    std::vector<std::string> ground_truth_imgpaths;
    LoadImagesAndGroundTruth(&imgs, &ground_truth, &ground_truth_imgpaths,
                             ground_truth_filepath, image_dir, controller);
    netresult.reserve(ground_truth.size());
    std::vector<cv::Mat>::iterator it = imgs.begin();
    // use batch size of 50
    for (size_t i = 0; i < imgs.size(); i += 50) {
        size_t step = i + 50 < imgs.size() ? 50 : imgs.size() - i;
        batch = controller.ForwardPass(
                  std::vector<cv::Mat>(it + i, it + i + step));
        netresult.insert(netresult.end(), batch.begin(), batch.end());
    }
    assert(netresult.size() == ground_truth.size());
    std::map<int, std::string> id_to_class;
    for (auto& g : ground_truth) {
        id_to_class[g.classid] = g.classname;
    }
    printf("creating confusion matrix\n");
    ConfusionMatrix::WriteCsvAndPng(netresult, ground_truth,
                                    id_to_class, output_path);
    printf("linking wrongly classified images\n");
    LinkWronglyClassified::Link(netresult, ground_truth, ground_truth_imgpaths,
                               id_to_class, output_path);
}

// ____________________________________________________________________________
void CaffeNetAnalysis::LoadImagesAndGroundTruth(
        std::vector<cv::Mat>* imgs_out,
        std::vector<ClassAndProbability>* ground_truth_out,
        std::vector<std::string>* ground_truth_imgpaths_out,
        const std::string& ground_truth_filepath,
        const std::string& image_dir,
        const CaffeController& controller) {
    std::ifstream inputFile(ground_truth_filepath);
    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int id;
        std::string image_path;
        iss >> image_path >> id;
        image_path = image_dir + image_path;
        cv::Mat img = cv::imread(image_path);
        if (img.cols == 0 && img.rows == 0) {
            printf("Error loading: %s\n", image_path.c_str());
            continue;
        }
        imgs_out->push_back(img);
        ground_truth_imgpaths_out->push_back(image_path);
        std::string cname = controller.ClassnameFromId(id);
        ground_truth_out->push_back(ClassAndProbability { cname, id, 100.0 });
    }
}