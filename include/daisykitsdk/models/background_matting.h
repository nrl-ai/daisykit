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

#ifndef DAISYKIT_MODELS_BACKGROUND_MATTING_H_
#define DAISYKIT_MODELS_BACKGROUND_MATTING_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/base_model.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace daisykit {
namespace models {

class BackgroundMatting : public BaseModel<cv::Mat, cv::Mat> {
 public:
  BackgroundMatting(const char* param_buffer,
                    const unsigned char* weight_buffer, int input_width = 256,
                    int input_height = 256);

  // Will be deleted when IO module is supported. Keep for old compatibility.
  BackgroundMatting(const std::string& param_file,
                    const std::string& weight_file, int input_width = 256,
                    int input_height = 256);

  // Override abstract Predict
  /// Get the mask of foreground.
  virtual cv::Mat Predict(const cv::Mat& image);

  /// Get the mask of foreground. Predict like this can share the same
  /// allocation => save memory
  void Predict(cv::Mat& image, cv::Mat& mask);

  /// Bind the segmented foreground with defined background.
  void BindWithBackground(cv::Mat& rgb, const cv::Mat& bg, const cv::Mat& mask);

 private:
  int input_width_;
  int input_height_;
};

}  // namespace models
}  // namespace daisykit

#endif
