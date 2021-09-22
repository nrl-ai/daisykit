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

#ifndef DAISYKIT_FLOWS_BACKGROUND_MATTING_FLOW_H_
#define DAISYKIT_FLOWS_BACKGROUND_MATTING_FLOW_H_

#include "daisykitsdk/models/background_matting.h"

#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace flows {
class BackgroundMattingFlow {
 public:
  BackgroundMattingFlow(const std::string& config_str,
                        const cv::Mat& default_background);
#ifdef __ANDROID__
  BackgroundMattingFlow(AAssetManager* mgr, const std::string& config_str,
                        const cv::Mat& default_background);
#endif
  ~BackgroundMattingFlow();
  cv::Mat Process(const cv::Mat& rgb);
  void DrawResult(cv::Mat& rgb, const cv::Mat& mask);

 private:
  cv::Mat background_;
  models::BackgroundMatting* background_matting_model_;
};

}  // namespace flows
}  // namespace daisykit

#endif
