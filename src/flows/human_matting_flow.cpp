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

#include "daisykitsdk/flows/human_matting_flow.h"

namespace daisykit {
namespace flows {

HumanMattingFlow::HumanMattingFlow(const std::string& config_str,
                                   const cv::Mat& default_background) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  human_matting_model_ =
      new models::HumanMatting(config["human_matting_model"]["model"],
                               config["human_matting_model"]["weights"]);
  background_ = default_background.clone();
}

#ifdef __ANDROID__
HumanMattingFlow::HumanMattingFlow(AAssetManager* mgr,
                                   const std::string& config_str,
                                   const cv::Mat& default_background) {
  nlohmann::json config = nlohmann::json::parse(config_str);
  human_matting_model_ =
      new models::HumanMatting(mgr, config["human_matting_model"]["model"],
                               config["human_matting_model"]["weights"]);
  background_ = default_background.clone();
}
#endif

HumanMattingFlow::~HumanMattingFlow() {
  delete human_matting_model_;
  human_matting_model_ = nullptr;
}

void HumanMattingFlow::Process(cv::Mat& rgb) {
  cv::Mat mask;
  human_matting_model_->Segmentation(rgb, mask);

  {
    const std::lock_guard<std::mutex> lock(mask_lock_);
    mask_ = mask;
  }
}

void HumanMattingFlow::DrawResult(cv::Mat& rgb) {
  {
    const std::lock_guard<std::mutex> lock(mask_lock_);
    human_matting_model_->BindWithBackground(rgb, background_, mask_);
  }
}

}  // namespace flows
}  // namespace daisykit