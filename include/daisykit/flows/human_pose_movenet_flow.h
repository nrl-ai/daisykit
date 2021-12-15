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

#ifndef DAISYKIT_FLOWS_PUSHUP_COUNTER_FLOW_H_
#define DAISYKIT_FLOWS_PUSHUP_COUNTER_FLOW_H_

#include "daisykit/common/types.h"
#include "daisykit/models/body_detector.h"
#include "daisykit/models/pose_detector_movenet.h"

#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace flows {
class HumanPoseMoveNetFlow {
 public:
  HumanPoseMoveNetFlow(const std::string& config_str);
#ifdef __ANDROID__
  HumanPoseMoveNetFlow(AAssetManager* mgr, const std::string& config_str);
#endif

  std::vector<types::ObjectWithKeypoints> Process(cv::Mat& rgb);
  void DrawResult(cv::Mat& rgb, std::vector<types::ObjectWithKeypoints>& poses);

 private:
  models::BodyDetector* body_detector_;
  models::PoseDetectorMoveNet* pose_detector_;
};

}  // namespace flows
}  // namespace daisykit

#endif
