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

#ifndef DAISYKIT_MODELS_ACTION_CLASSIFIER_H_
#define DAISYKIT_MODELS_ACTION_CLASSIFIER_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/image_model.h"
#include "daisykitsdk/models/ncnn_model.h"

#include <opencv2/opencv.hpp>

namespace daisykit {
namespace models {

/// Action classifier. Currently only recognizes push-up or not push-up.
class ActionClassifier : public NCNNModel, public ImageModel {
 public:
  ActionClassifier(const char* param_buffer, const unsigned char* weight_buffer,
                   bool smooth = true, int input_width = 224,
                   int input_height = 224, bool use_gpu = false);

  ActionClassifier(const std::string& param_file,
                   const std::string& weight_file, bool smooth = true,
                   int input_width = 224, int input_height = 224,
                   bool use_gpu = false);

  /// Classify actions.
  /// Return 0 on success, otherwise return error code.
  int Classify(const cv::Mat& image, types::Action& action, float& confidence);

 private:
  /// Preprocess image data to obtain net input.
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;
  bool smooth_ = true;  // Smooth the result or not
  long long int last_pushup_time_ = 0;
};

}  // namespace models
}  // namespace daisykit

#endif
