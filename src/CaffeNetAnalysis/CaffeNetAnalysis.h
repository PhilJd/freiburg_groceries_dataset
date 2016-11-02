/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#include <string>
#include <vector>
#include "opencv2/core/core.hpp"  // cv::Mat
#include "./CaffeController.h"


class CaffeNetAnalysis {
 public:
    /// Compares the ground truth to the results obtained by running
    /// the images through the net specified in the controller.
    /// Each AnalysisModule defines a comparision.
    /// The info message is formatted according to OutputStyle.
    static void Analyze(const CaffeController& controller,
                        const std::string& ground_truth_filepath,
                        const std::string& image_dir,
                        const std::string& output_path);

 private:
    /// Loads the images into imgs_out and fills the vector ground_truth_out
    ///  with the class specified in the file
    static void LoadImagesAndGroundTruth(
        std::vector<cv::Mat>* imgs_out,
        std::vector<ClassAndProbability>* ground_truth_out,
        std::vector<std::string>* ground_truth_imgpaths_out,
        const std::string& ground_truth_filepath,
        const std::string& image_dir,
        const CaffeController& controller);
};
