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

#include <opencv2/opencv.hpp>
#include "daisykit/common/types.h"
#include "daisykit/models/image_model.h"
#include "daisykit/models/ncnn_model.h"

namespace daisykit {
namespace models {

class FaceExtractor : public NCNNModel, public ImageModel {
 public:
  FaceExtractor(const std::string& param_file, const std::string& weight_file,
                int input_size = 112, bool use_gpu = false);
  FaceExtractor(const char* param_buffer, const unsigned char* weight_buffer,
                int input_size = 112, bool use_gpu = false);

#ifdef __ANDROID__
  FaceExtractor(AAssetManager* mgr, const std::string& param_file,
                const std::string& weight_file, int input_size = 112,
                bool use_gpu = false);
#endif

  void Predict(std::vector<daisykit::types::FaceExtended>& faces);

 private:
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;
  float Norm(std::vector<float> const& u);
  void L2Norm(std::vector<float>& v);

 private:
  std::string input_name_ = "input";
  std::string output_name_ = "output";
};

}  // namespace models
}  // namespace daisykit

#endif
