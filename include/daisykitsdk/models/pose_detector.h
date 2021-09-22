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

#ifndef DAISYKIT_MODELS_POSE_DETECTOR_H_
#define DAISYKIT_MODELS_POSE_DETECTOR_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/base_model.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace daisykit {
namespace models {

class PoseDetector : public BaseModel<cv::Mat, std::vector<types::Keypoint>> {
 public:
  PoseDetector(const char* param_buffer, const unsigned char* weight_buffer,
               int input_width = 256, int input_height = 256);

  // Will be deleted when IO module is supported. Keep for old compatibility.
  PoseDetector(const std::string& param_file, const std::string& weight_file,
               int input_width = 256, int input_height = 256);

  // Override abstract Predict
  /// Predict keypoints for a single object.
  virtual std::vector<types::Keypoint> Predict(const cv::Mat& image);

  /// Predict keypoints for a single object with modifiable offset.
  /// (Supports for predict multi objects in an image).
  std::vector<types::Keypoint> Predict(cv::Mat& image, float offset_x = 0,
                                       float offset_y = 0);

  /// Predict keypoints for multiple objects.
  std::vector<std::vector<types::Keypoint>> PredictMulti(
      cv::Mat& image, const std::vector<types::Object>& objects);

  /// Draw keypoints and their joints.
  void DrawKeypoints(const cv::Mat& image,
                     const std::vector<types::Keypoint>& keypoints);

 private:
  int input_width_;
  int input_height_;
};

}  // namespace models
}  // namespace daisykit

#endif
