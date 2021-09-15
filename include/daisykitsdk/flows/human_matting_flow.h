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

#ifndef DAISYKIT_FLOWS_HUMAN_MATTING_FLOW_H_
#define DAISYKIT_FLOWS_HUMAN_MATTING_FLOW_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/human_matting.h"
#include "daisykitsdk/thirdparties/json.hpp"
#include "daisykitsdk/utils/img_proc/img_utils.h"
#include "daisykitsdk/utils/visualizers/base_visualizer.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace flows {
class HumanMattingFlow {
 public:
  HumanMattingFlow(const std::string& config_str,
                   const cv::Mat& default_background);
#ifdef __ANDROID__
  HumanMattingFlow(AAssetManager* mgr, const std::string& config_str,
                   const cv::Mat& default_background);
#endif
  ~HumanMattingFlow();
  void Process(cv::Mat& rgb);
  void DrawResult(cv::Mat& rgb);

 private:
  cv::Mat mask_;
  std::mutex mask_lock_;
  cv::Mat background_;

  models::HumanMatting* human_matting_model_;
};

}  // namespace flows
}  // namespace daisykit

#endif