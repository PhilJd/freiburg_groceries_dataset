/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#include "CaffeController.h"
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"  // cv::resize
#include "caffe/layers/memory_data_layer.hpp"  // memory layer
#include "caffe/caffe.hpp"  // includes blob and io

CaffeController::CaffeController() {
    _initialized = false;
}

CaffeController::CaffeController(const std::string& deploy_path,
                                 const std::string& weights_path,
                                 const std::string& idtoclass_path) {
    _initialized = false;  // If setup works var is true after constructor
    SetDeployConfig(deploy_path);
    LoadWeights(weights_path);
    InitIdToClassDict(idtoclass_path);
}

// ____________________________________________________________________________
void CaffeController::SetMode(CaffeMode mode) {
    if (mode == CONTROLLER_CPU) {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    } else if (mode == CONTROLLER_GPU) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    }
}

// ____________________________________________________________________________
void CaffeController::SetDeviceNumber(const int device_id) {
    if (caffe::Caffe::mode() == caffe::Caffe::GPU) {
        caffe::Caffe::SetDevice(device_id);
    }
}

// ____________________________________________________________________________
cr::Error CaffeController::SetDeployConfig(const std::string& deploy_fpath) {
    // TO DO: error handling
    _net = unique_ptr_caffenet(
                   new caffe::Net<float>(deploy_fpath, caffe::TEST));
    return cr::SUCCESS;
}

// ____________________________________________________________________________
cr::Error CaffeController::LoadWeights(const std::string& weight_fpath) {
    if (!_net) {  // ptr not initialized
        return cr::NET_NOT_INITIALIZED;
    }
    // TO DO: error handling
    _net->CopyTrainedLayersFrom(weight_fpath);
    _initialized = true;
    return cr::SUCCESS;
}

// ____________________________________________________________________________
std::vector<ClassAndProbability> CaffeController::ForwardPass(
                                    const std::vector<cv::Mat>& images) const {
    std::vector<ClassAndProbability> result;
    if (!_initialized || images.size() == 0) {
        return result;
    }
    // layer_by_name returns a pointer to a standard layer, in order to access
    // AddMatVector() conversion to the specific layer type is needed
    boost::shared_ptr<caffe::MemoryDataLayer<float> > memory_data_layer =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >
                                                (_net->layer_by_name("data"));
    if (memory_data_layer == NULL) {
        return result;
    }
    memory_data_layer->set_batch_size(images.size());
    cv::Size data_size(memory_data_layer->width(), memory_data_layer->height());
    std::vector<caffe::Datum> datumvec = MatToDatumVec(images, data_size);
    // if compilation fails here, caffe most likely didn't define
    // USE_OPENCV; uncomment or define it in "caffe/util/io.hpp" to fix this
    memory_data_layer->AddDatumVector(datumvec);
    // Currently I don't need the resultblobs, this is only for forward passing
    std::vector<caffe::Blob<float>*> resultblobs = _net->Forward();
    const boost::shared_ptr<caffe::Blob<float> >& argmaxLayer =
            _net->blob_by_name("argmax");
    const float* argmax = argmaxLayer->cpu_data();
    for (int i = 0; i < argmaxLayer->num() * 2; i += 2) {
        // classid is argmax[i]; probabilty is argmax[i + 1]
        result.push_back(
            ClassAndProbability {
                id_to_class.at(static_cast<int>(argmax[i])),
                static_cast<int>(argmax[i]), argmax[i + 1] });
    }
    return result;
}

// ____________________________________________________________________________
void CaffeController::InitIdToClassDict(const std::string& classfile) {
    std::ifstream inputFile(classfile);
    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream ss(line);
        int key;
        std::string value;
        ss >> value >> key;
        id_to_class[key] = value;
    }
}

// ____________________________________________________________________________
std::string CaffeController::ClassnameFromId(int id) const {
    return id_to_class.find(id)->second;
}

// ____________________________________________________________________________
std::vector<caffe::Datum> CaffeController::MatToDatumVec(
         const std::vector<cv::Mat>& images, const cv::Size& data_dims) const {
    std::vector<caffe::Datum> datumvec;
    for (auto& mat : images) {
        caffe::Datum datum;
        cv::Mat scaled;
        cv::resize(mat, scaled, data_dims);
        caffe::CVMatToDatum(scaled, &datum);
        datumvec.push_back(datum);
    }
    return datumvec;
}
