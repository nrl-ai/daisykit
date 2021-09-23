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

#ifndef DAISYKIT_MODELS_FACE_DETECTOR_WITH_LANDMARK_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_WITH_LANDMARK_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/base_model.h"
#include "daisykitsdk/models/face_recognition/anchor_generator.h"
#include "daisykitsdk/models/face_recognition/config.h"
#include "daisykitsdk/models/face_recognition/face_alignment.h"
#include "daisykitsdk/models/face_recognition/tools.h"
#include "opencv2/opencv.hpp"

namespace daisykit {
namespace models {

class FaceDetectorWithLandMark
    : public BaseModel<cv::Mat, std::vector<daisykit::types::Det>> {
 public:
  FaceDetectorWithLandMark(const std::string& param_file,
                           const std::string& bin_path_file,
                           int input_width = 320, int input_height = 320,
                           float score_threshold = 0.7,
                           float iou_threshold = 0.5);

  virtual std::vector<daisykit::types::Det> Predict(cv::Mat& img);

 private:
  int input_width_;
  int input_height_;
  float score_threshold_;
  float iou_threshold_;
};

}  // namespace models
}  // namespace daisykit
#endif
