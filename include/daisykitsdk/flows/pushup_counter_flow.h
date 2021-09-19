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

#include "daisykitsdk/models/action_classifier.h"
#include "daisykitsdk/models/body_detector.h"
#include "daisykitsdk/models/pose_detector.h"
#include "daisykitsdk/processors/fitness/pushup_analyzer.h"

#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace flows {
class PushupCounterFlow {
 public:
  PushupCounterFlow(const std::string& config_str);
#ifdef __ANDROID__
  PushupCounterFlow(AAssetManager* mgr, const std::string& config_str);
#endif
  void Process(cv::Mat& rgb);
  int NumPushups();
  void DrawResult(cv::Mat& rgb);

 private:
  std::vector<types::Object> bodies_;
  std::mutex bodies_lock_;

  std::vector<std::vector<types::Keypoint>> keypoints_;
  std::mutex keypoints_lock_;

  std::atomic<int> num_pushups_;
  std::atomic<float> is_pushup_score_;

  models::BodyDetector* body_detector_;
  models::PoseDetector* pose_detector_;
  models::ActionClassifier* action_classifier_;
  processors::PushupAnalyzer* pushup_analyzer_;
};

}  // namespace flows
}  // namespace daisykit

#endif