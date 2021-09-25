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

ActionClassifier::ActionClassifier(const char* param_buffer,
                                   const unsigned char* weight_buffer,
                                   bool smooth, int input_width,
                                   int input_height, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {
  smooth_ = smooth;
}

ActionClassifier::ActionClassifier(const std::string& param_file,
                                   const std::string& weight_file, bool smooth,
                                   int input_width, int input_height,
                                   bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  smooth_ = smooth;
}

void ActionClassifier::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  cv::Mat rgb = processors::ImgUtils::SquarePadding(image, InputWidth());
  net_input =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols,
                                    rgb.rows, InputWidth(), InputHeight());
}

int ActionClassifier::Classify(const cv::Mat& image, types::Action& action,
                               float& confidence) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Model inference
  ncnn::Mat out;
  int result = Predict(in, out, "input_1_blob", "dense_Softmax_blob");
  if (result != 0) {
    return result;
  }

  // Postprocess
  out = out.reshape(out.w * out.h * out.c);
  confidence = out[1];
  bool is_pushup = confidence > 0.9;

  if (!smooth_) {
    action = is_pushup ? types::Action::kPushup : types::Action::kUnknown;
    return 0;
  }

  // Check and recognize push-up
  long long int current_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (is_pushup) {
    last_pushup_time_ = current_time;
  }

  // Return smoothed result
  if (current_time - last_pushup_time_ < 2000) {
    action = types::Action::kPushup;
    return 0;
  }

  action = types::Action::kUnknown;
  return 0;
}

}  // namespace models
}  // namespace daisykit
