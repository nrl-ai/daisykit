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
#include "daisykitsdk/models/image_model.h"
#include "daisykitsdk/models/ncnn_model.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace daisykit {
namespace models {

/// Background matting model.
/// Used to segment human body from background, for background replacement, like
/// in video call software such as Google Meet, Zoom.
class BackgroundMatting : public NCNNModel, public ImageModel {
 public:
  BackgroundMatting(const char* param_buffer,
                    const unsigned char* weight_buffer, int input_width = 256,
                    int input_height = 256, bool use_gpu = false);

  BackgroundMatting(const std::string& param_file,
                    const std::string& weight_file, int input_width = 256,
                    int input_height = 256, bool use_gpu = false);

  /// Get forground probability mask.
  /// Return 0 on success, otherwise return error code.
  int Segmentation(const cv::Mat& image, cv::Mat& mask);

  /// Bind the segmented foreground with defined background.
  void BindWithBackground(cv::Mat& rgb, const cv::Mat& background,
                          const cv::Mat& mask);

 private:
  /// Preprocess image data to obtain net input.
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;
};

}  // namespace models
}  // namespace daisykit

#endif
