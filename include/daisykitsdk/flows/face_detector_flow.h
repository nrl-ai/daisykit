// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DAISYKIT_FLOWS_FACE_DETECTOR_FLOW_H_
#define DAISYKIT_FLOWS_FACE_DETECTOR_FLOW_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/face_detector.h"
#include "daisykitsdk/models/facial_landmark_estimator.h"

#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace flows {
class FaceDetectorFlow {
 public:
  FaceDetectorFlow(const std::string& config_str);
#ifdef __ANDROID__
  FaceDetectorFlow(AAssetManager* mgr, const std::string& config_str);
#endif
  ~FaceDetectorFlow();
  std::vector<types::Face> Process(const cv::Mat& rgb);
  void DrawResult(cv::Mat& rgb, std::vector<types::Face>& faces);

 private:
  bool with_landmark_ = false;
  models::FaceDetector* face_detector_;
  models::FacialLandmarkEstimator* facial_landmark_estimator_;
};

}  // namespace flows
}  // namespace daisykit

#endif
