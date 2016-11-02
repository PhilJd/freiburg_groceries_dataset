/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include <vector>
#include <string>
#include <map>
#include "./CaffeController.h"  // ClassAndProbability

class ConfusionMatrix {
 public:
    static void WriteCsvAndPng(
    	const std::vector<ClassAndProbability>& netresult,
        const std::vector<ClassAndProbability>& ground_truth,
        const std::map<int, std::string>& id_to_class,
        const std::string& output_csv_dirpath);

    static cv::Mat PlotMatrix(
    	              const std::vector<std::vector<float> >& confusion_matrix,
                      const size_t img_dim,
                      const std::map<int, std::string>& id_to_class);
};

#endif  // CONFUSIONMATRIX_H
