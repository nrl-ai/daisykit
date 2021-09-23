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

#ifndef DAISYKIT_MODELS_FACE_EXTRACTOR_H_
#define DAISYKIT_MODELS_FACE_EXTRACTOR_H_

#include "daisykitsdk/common/types.h"
#include "daisykitsdk/models/base_model.h"
#include <opencv2/opencv.hpp>

namespace daisykit {
namespace models {

class FaceExtractor
    : public BaseModel<cv::Mat, std::vector<daisykit::types::Feature>> {
public:
  FaceExtractor(const std::string &param_file, const std::string &weight_file);
  virtual std::vector<daisykit::types::Feature> Predict(cv::Mat &image);

private:
  int input_width_;
  int input_height_;
};

} // namespace models
} // namespace daisykit

#endif
