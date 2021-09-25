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

#ifndef DAISYKIT_MODELS_IMAGE_MODEL_H_
#define DAISYKIT_MODELS_IMAGE_MODEL_H_

#include <net.h>
#include <opencv2/opencv.hpp>

namespace daisykit {
namespace models {

/// Model interface with image input.
class ImageModel {
 public:
  ImageModel(int input_width, int input_height);

 protected:
  /// Preprocess function for image model.
  /// Get an cv::Mat in and return a net input.
  virtual void Preprocess(const cv::Mat& img, ncnn::Mat& net_input) = 0;
  /// Get net input width
  int InputWidth();
  /// Get net input height
  int InputHeight();

 private:
  int input_width_;   /// Net input width
  int input_height_;  /// Net input height
};

}  // namespace models
}  // namespace daisykit

#endif
