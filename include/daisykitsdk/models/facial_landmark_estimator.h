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

#ifndef DAISYKIT_MODELS_FACIAL_LANDMARK_ESTIMATOR_H_
#define DAISYKIT_MODELS_FACIAL_LANDMARK_ESTIMATOR_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/base_model.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace daisykit {
namespace models {

class FacialLandmarkEstimator
    : public BaseModel<cv::Mat, std::vector<daisykit::types::Keypoint>> {
 public:
  FacialLandmarkEstimator(const char* param_buffer,
                          const unsigned char* weight_buffer,
                          int input_width = 112, int input_height = 112);

  // Will be deleted when IO module is supported. Keep for old compatibility.
  FacialLandmarkEstimator(const std::string& param_file,
                          const std::string& weight_file, int input_width = 112,
                          int input_height = 112);

  // Override abstract Predict
  /// Predict keypoints for a single object.
  virtual std::vector<daisykit::types::Keypoint> Predict(const cv::Mat& image);

  /// Predict keypoints for a single object with modifiable offset.
  /// (Supports for predict multi objects in an image).
  std::vector<daisykit::types::Keypoint> Predict(const cv::Mat& image,
                                                 float offset_x = 0,
                                                 float offset_y = 0);

  /// Detect keypoints for multiple objects.
  void PredictMulti(const cv::Mat& image,
                    std::vector<daisykit::types::Face>& objects);

 private:
  int input_width_;
  int input_height_;
};

}  // namespace models
}  // namespace daisykit

#endif
