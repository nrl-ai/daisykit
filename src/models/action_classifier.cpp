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

#include "daisykitsdk/models/action_classifier.h"
#include "daisykitsdk/processors/image_processors/img_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace daisykit {
namespace models {

ActionClassifier::ActionClassifier(const std::string& param_file,
                                   const std::string& weight_file,
                                   bool smooth) {
  _smooth = smooth;
  LoadModel(param_file, weight_file);
}

void ActionClassifier::LoadModel(const std::string& param_file,
                                 const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(param_file.c_str());
  int ret_model = model_->load_model(weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}

#ifdef __ANDROID__
ActionClassifier::ActionClassifier(AAssetManager* mgr,
                                   const std::string& param_file,
                                   const std::string& weight_file,
                                   bool smooth) {
  _smooth = smooth;
  LoadModel(mgr, param_file, weight_file);
}

void ActionClassifier::LoadModel(AAssetManager* mgr,
                                 const std::string& param_file,
                                 const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(mgr, param_file.c_str());
  int ret_model = model_->load_model(mgr, weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}
#endif

types::Action ActionClassifier::Classify(cv::Mat& image, float& confidence) {
  cv::Mat rgb = image.clone();
  rgb = processors::ImgUtils::SquarePadding(rgb, input_width_).clone();
  ncnn::Mat in =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols,
                                    rgb.rows, input_width_, input_height_);

  ncnn::Mat out;
  {
    ncnn::MutexLockGuard g(lock_);
    ncnn::Extractor ex = model_->create_extractor();
    ex.input("input_1_blob", in);
    ex.extract("dense_Softmax_blob", out);
  }

  out = out.reshape(out.w * out.h * out.c);
  confidence = out[1];
  bool is_pushup = confidence > 0.9;

  if (!_smooth) {
    return is_pushup ? types::Action::kPushup : types::Action::kUnknown;
  }

  // Check and recognize pushup
  long long int current_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (is_pushup) {
    last_pushup_time_ = current_time;
  }

  // Return smoothed result
  if (current_time - last_pushup_time_ < 2000) {
    return types::Action::kPushup;
  }

  return types::Action::kUnknown;
}

types::Action ActionClassifier::Classify(cv::Mat& image) {
  float confidence;
  return Classify(image, confidence);
}

}  // namespace models
}  // namespace daisykit