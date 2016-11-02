/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#include "./LinkWronglyClassified.h"
#include <vector>
#include <string>
#include <map>
#include <boost/filesystem.hpp>
#include <unistd.h>  // symlink

// ____________________________________________________________________________
void LinkWronglyClassified::Link(
        const std::vector<ClassAndProbability>& netresult,
        const std::vector<ClassAndProbability>& ground_truth,
        const std::vector<std::string>& ground_truth_imgpaths,
        const std::map<int, std::string>& id_to_class,
        const std::string& parent_link_path) {
    CreateFolderStructure(parent_link_path, id_to_class);
    int nr_of_classes = id_to_class.rbegin()->first + 1;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        int correct_class = ground_truth[i].classid;
        int pred_class = netresult[i].classid;
        if (correct_class == pred_class) {
            continue;
        }
        std::string link_path_str = parent_link_path + "misclassified/"
                                    + id_to_class.at(correct_class) + "/"
                                    + id_to_class.at(pred_class)
                                    + std::to_string(i) + ".png";
        boost::filesystem::path ground_truth_path(ground_truth_imgpaths[i]);
        ground_truth_path = boost::filesystem::canonical(ground_truth_path);
        symlink(ground_truth_path.string().c_str(), link_path_str.c_str());
    }
}

// ____________________________________________________________________________
void LinkWronglyClassified::CreateFolderStructure(
        const std::string& parent_link_path,
        const std::map<int, std::string>& id_to_class) {
    for (auto& x : id_to_class) {
        std::string class_dir = parent_link_path + "misclassified/" + x.second;
        boost::filesystem::path p(class_dir);
        if (!boost::filesystem::is_directory(p)) {
            boost::filesystem::create_directories(p);
        }
    }
}