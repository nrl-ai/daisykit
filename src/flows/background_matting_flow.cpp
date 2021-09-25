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

#include "daisykitsdk/flows/background_matting_flow.h"
#include "daisykitsdk/thirdparties/json.hpp"

namespace daisykit {
namespace flows {

BackgroundMattingFlow::BackgroundMattingFlow(
    const std::string& config_str, const cv::Mat& default_background) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  background_matting_model_ = new models::BackgroundMatting(
      config["background_matting_model"]["model"],
      config["background_matting_model"]["weights"]);
  background_ = default_background.clone();
}

#ifdef __ANDROID__
BackgroundMattingFlow::BackgroundMattingFlow(
    AAssetManager* mgr, const std::string& config_str,
    const cv::Mat& default_background) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  background_matting_model_ = new models::BackgroundMatting(
      mgr, config["background_matting_model"]["model"],
      config["background_matting_model"]["weights"]);
  background_ = default_background.clone();
}
#endif

BackgroundMattingFlow::~BackgroundMattingFlow() {
  delete background_matting_model_;
  background_matting_model_ = nullptr;
}

cv::Mat BackgroundMattingFlow::Process(const cv::Mat& rgb) {
  cv::Mat mask;
  background_matting_model_->Segmentation(rgb, mask);
  return mask;
}

void BackgroundMattingFlow::DrawResult(cv::Mat& rgb, const cv::Mat& mask) {
  background_matting_model_->BindWithBackground(rgb, background_, mask);
}

}  // namespace flows
}  // namespace daisykit
