/* Copyright Philipp Jund, 2016
 * jundp@informatik.uni-freiburg.de */

#ifndef CAFFECONTROLLER_H
#define CAFFECONTROLLER_H
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "opencv2/core/core.hpp"
#include "caffe/caffe.hpp"

/* The CaffeController provides an interface for setting
 *  caffe parameters and for managing one caffe net
 */


namespace cr {

enum Error {
    DEPLOY_FILE_UNREADABLE,
    WEIGHTS_FILE_UNREADABLE,
    NET_NOT_INITIALIZED,
    GPU_DEVICE_NOT_FOUND,
    SUCCESS
};

}  // namespace cr


typedef std::unique_ptr<caffe::Net<float> > unique_ptr_caffenet;

struct ClassAndProbability {
    std::string classname;
    int classid;
    float probability;
};

enum CaffeMode {
    CONTROLLER_GPU,
    CONTROLLER_CPU
};

class CaffeController {
 public:
    CaffeController();
    CaffeController(const std::string& deploy_path,
                    const std::string& weights_path,
                    const std::string& idtoclass_path);

    /// Sets the global Caffe parameters
    static void SetMode(CaffeMode mode);
    static void SetDeviceNumber(const int device_id);

    /// Functions to specify the net. Deploy config file must be loaded before
    /// the weights can be loaded!
    cr::Error SetDeployConfig(const std::string& deploy_fpath);
    cr::Error LoadWeights(const std::string& weight_fpath);
    void InitIdToClassDict(const std::string& classfile);

    /// Executes one forward pass through the net in test mode, so no
    /// modifications/training of the net
    std::vector<ClassAndProbability> ForwardPass(
                                     const std::vector<cv::Mat>& images) const;

    /// Returns true if the net was loaded successfully
    bool IsInitialized() { return _initialized; }

    /// Returns the classname for an id, depending on the currently active net
    std::string ClassnameFromId(int id) const;

 private:
    unique_ptr_caffenet _net;  // pointer to net the controller manages
    std::map<int, std::string> id_to_class;
    bool _initialized;

    /// Converts a vector of mats to datum; currently necessary as I didn't get
    /// caffe's AddMatVector function to work; If this works in the future,
    /// remove this function and use
    /// std::vector<int> vec;
    /// memory_data_layer->AddMatVector(images, vec);
    std::vector<caffe::Datum> MatToDatumVec(const std::vector<cv::Mat>& images,
                                            const cv::Size& data_dims) const;
};

#endif  // CAFFECONTROLLER_H
