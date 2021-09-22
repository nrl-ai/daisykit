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

#ifndef DAISYKIT_MODELS_FACE_DETECTOR_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/base_model.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace daisykit {
namespace models {

class FaceDetector
    : public BaseModel<cv::Mat, std::vector<daisykit::types::Face>> {
 public:
  FaceDetector(const char* param_buffer, const unsigned char* weight_buffer,
               int input_width = 320, int input_height = 320,
               float score_threshold = 0.7, float iou_threshold = 0.5);

  // Will be deleted when IO module is supported. Keep for old compatibility.
  FaceDetector(const std::string& param_file, const std::string& weight_file,
               int input_width = 320, int input_height = 320,
               float score_threshold = 0.7, float iou_threshold = 0.5);

  // Override abstract Predict.
  /// Predict faces in an image.
  virtual std::vector<daisykit::types::Face> Predict(const cv::Mat& image);

 private:
  int input_width_;
  int input_height_;

  int num_anchors_;

  float score_threshold_;
  float iou_threshold_;
};

}  // namespace models
}  // namespace daisykit

#endif
