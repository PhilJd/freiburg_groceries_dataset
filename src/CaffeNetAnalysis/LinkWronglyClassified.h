/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#ifndef LINKWRONGLYCLASSIFIED_H
#define LINKWRONGLYCLASSIFIED_H

#include <vector>
#include <string>
#include <map>
#include "./CaffeController.h"  // ClassAndProbability

class LinkWronglyClassified {
 public:
    static void Link(
        const std::vector<ClassAndProbability>& netresult,
        const std::vector<ClassAndProbability>& ground_truth,
        const std::vector<std::string>& ground_truth_imgpaths,
        const std::map<int, std::string>& id_to_class,
        const std::string& parent_link_path);

 private:
 	static void CreateFolderStructure(
        const std::string& parent_link_path,
        const std::map<int, std::string>& id_to_class);
};

#endif  // LINKWRONGLYCLASSIFIED_H
