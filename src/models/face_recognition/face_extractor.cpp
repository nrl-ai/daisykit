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

#include "daisykit/models/face_recognition/face_extractor.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace daisykit {
namespace models {

FaceExtractor::FaceExtractor(const std::string& param_file,
                             const std::string& weight_file, int input_size,
                             bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_size, input_size) {}

FaceExtractor::FaceExtractor(const char* param_buffer,
                             const unsigned char* weight_buffer, int input_size,
                             bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_size, input_size) {}

#ifdef __ANDROID__
FaceExtractor::FaceExtractor(AAssetManager* mgr, const std::string& param_path,
                             const std::string& bin_path, int input_size,
                             bool use_gpu)
    : NCNNModel(mgr, param_path, bin_path, use_gpu),
      ImageModel(input_size, input_size) {}
#endif

float FaceExtractor::Norm(std::vector<float> const& u) {
  float accum = 0.;
  for (int i = 0; i < u.size(); ++i) {
    accum += u[i] * u[i];
  }
  return std::sqrt(accum);
}

void FaceExtractor::L2Norm(std::vector<float>& v) {
  float k = Norm(v);
  std::transform(v.begin(), v.end(), v.begin(),
                 [k](float& c) { return c / k; });
}

void FaceExtractor::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  int img_w = image.cols;
  int img_h = image.rows;
  net_input =
      ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
  net_input.substract_mean_normalize(mean_vals, norm_vals);
}

void FaceExtractor::Predict(std::vector<daisykit::types::FaceExtended>& faces) {
  for (auto& face : faces) {
    std::vector<float> feature;
    feature.resize(512);
    ncnn::Mat in;
    ncnn::Mat out;
    Preprocess(face.aligned_face, in);
    int res = Infer(in, out, input_name_, output_name_);
    if (res != 0) {
      std::cout << "Inference failed" << std::endl;
      return;
    }
    for (int j = 0; j < 512; j++) {
      feature[j] = out[j];
    }
    L2Norm(feature);
    face.feature = feature;
  }
}

}  // namespace models
}  // namespace daisykit
