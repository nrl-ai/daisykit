#ifndef DAISYKIT_FLOWS_FACE_DETECTOR_FLOW_H_
#define DAISYKIT_FLOWS_FACE_DETECTOR_FLOW_H_

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

#include <daisykitsdk/common/types.h>
#include <daisykitsdk/models/face_detector.h>
#include <daisykitsdk/models/facial_landmark_estimator.h>
#include <daisykitsdk/utils/img_proc/img_utils.h>
#include <daisykitsdk/utils/visualizer/viz_utils.h>
#include <daisykitsdk/thirdparties/json.hpp>

namespace daisykit {
namespace flows {
class FaceDetectorFlow {
 public:
  FaceDetectorFlow(const std::string& config_str);
#ifdef __ANDROID__
  FaceDetectorFlow(AAssetManager* mgr, const std::string& config_str);
#endif
  ~FaceDetectorFlow();
  void Process(cv::Mat& rgb);
  void DrawResult(cv::Mat& rgb);

 private:
  std::vector<common::Face> faces_;
  std::mutex faces_lock_;

  bool with_landmark_ = false;
  models::FaceDetector* face_detector_;
  models::FacialLandmarkEstimator* facial_landmark_estimator_;
};

}  // namespace flows
}  // namespace daisykit

#endif