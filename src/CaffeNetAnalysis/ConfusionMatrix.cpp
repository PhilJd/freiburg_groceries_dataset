/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#include "./ConfusionMatrix.h"
#include <vector>
#include <string>
#include <map>
#include <numeric>  // accumulate
#include <algorithm>  // max
#include <fstream> 
#include "opencv2/core/core.hpp"  // cv::Mat
#include "opencv2/highgui/highgui.hpp"  // imshow
#include "opencv2/imgproc/imgproc.hpp"  // rotationMatrix2D

// ____________________________________________________________________________
void ConfusionMatrix::WriteCsvAndPng(
        const std::vector<ClassAndProbability>& netresult,
        const std::vector<ClassAndProbability>& ground_truth,
        const std::map<int, std::string>& id_to_class,
        const std::string& output_csv_dirpath) {
    int nr_of_classes = id_to_class.rbegin()->first + 1;
    std::vector<std::vector<float> > confusion_matrix;
    confusion_matrix.resize(nr_of_classes);
    std::ofstream confusion_csv, accuracy_csv;
    accuracy_csv.open(output_csv_dirpath + "accuracy.csv");
    confusion_csv.open(output_csv_dirpath + "confusion_matrix.csv");
    // header
    confusion_csv << "; ";
    for (size_t i = 0; i < confusion_matrix.size(); ++i) {
        confusion_matrix[i].resize(nr_of_classes, 0);
        if (id_to_class.find(i) == id_to_class.end()) {
            continue;
        }
        confusion_csv << id_to_class.at(i) << "; ";
    }
    confusion_csv << "\n";
    // create confusion matrix
    float num_correct = 0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        int correct_class = ground_truth[i].classid;
        int pred_class = netresult[i].classid;
        confusion_matrix[correct_class][pred_class]++;
        if (correct_class == pred_class) {
            num_correct++;
        }
    }
    // write matrix to csv and png
    for (int i = 0; i < nr_of_classes; ++i) {
        if (id_to_class.find(i) == id_to_class.end()) {
            continue;
        }
        float sum = std::accumulate(confusion_matrix[i].begin(),
                                    confusion_matrix[i].end(), 0);
        confusion_csv << i << ":" << id_to_class.at(i) << "; ";
        for (int j = 0; j < nr_of_classes; ++j) {
            confusion_matrix[i][j] = confusion_matrix[i][j] / sum;
            confusion_csv << confusion_matrix[i][j] << "; ";
        }
        accuracy_csv << i << ":" << id_to_class.at(i) << "; "
                     << confusion_matrix[i][i] << "; \n";
        confusion_csv << "     sum:" << sum << "\n";
    }
    confusion_csv << "overall accuracy:; " << num_correct / ground_truth.size() << "; \n";
    accuracy_csv << "overall accuracy:; " << num_correct / ground_truth.size() << "; \n";
    accuracy_csv.close();
    confusion_csv.close();
    cv::Mat cv_confusionmat = PlotMatrix(confusion_matrix, 900, id_to_class);
    cv::imwrite(output_csv_dirpath + "confusion_matrix.png", cv_confusionmat);
}


// ____________________________________________________________________________
void rotate(const cv::Mat& src, double angle, cv::Mat* dst) {
    int len = std::max(src.cols, src.rows);
    cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, *dst, r, cv::Size(len, len));
}

// ____________________________________________________________________________
cv::Mat ConfusionMatrix::PlotMatrix(
        const std::vector<std::vector<float> >& confusion_matrix,
        const size_t img_dim,
        const std::map<int, std::string>& id_to_class) {
    cv::Mat plot(img_dim, img_dim, CV_8UC1, 255);
    // int square_size = (img_dim - 100) / id_to_class.size();  // 100pxfortext
    int square_size = (img_dim) / id_to_class.size();
    for (size_t y = 0; y < confusion_matrix.size(); ++y) {
        for (size_t x = 0; x < confusion_matrix.size(); ++x) {
            /* cv::Point2i p1(100 + x * square_size, 100 + y * square_size);
            cv::Point2i p2(100 + (x + 1) * square_size,
                           100 + (y + 1) * square_size); */
            cv::Point2i p1(x * square_size, y * square_size);
            cv::Point2i p2((x + 1) * square_size,
                           (y + 1) * square_size);
            cv::rectangle(plot, p1, p2, 255 * confusion_matrix[y][x], -1);
        }
    }
    return cv::Scalar::all(255) - plot;
}
