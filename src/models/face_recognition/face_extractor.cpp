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

#include "daisykitsdk/models/face_recognition/face_extractor.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace daisykit {
namespace models {

FaceExtractor::FaceExtractor(const std::string& param_file,
                             const std::string& weight_file) {
  LoadModel(param_file, weight_file);
}

void preprocess(cv::Mat& image, ncnn::Mat& in) {
  int img_w = image.cols;
  int img_h = image.rows;
  in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};

  in.substract_mean_normalize(mean_vals, norm_vals);
}

std::vector<types::Feature> FaceExtractor::Predict(cv::Mat& image) {
  float fnum[512];
  float* prob = fnum;
  std::vector<types::Feature> features;
  types::Feature feature;
  ncnn::Mat input;
  ncnn::Mat out;

  preprocess(image, input);

  ncnn::MutexLockGuard g(lock_);
  auto ex = model_.create_extractor();
  ex.input("input", input);
  ex.extract("output", out);

  for (int j = 0; j < 512; j++) {
    fnum[j] = out[j];
  }

  cv::Mat out_m(512, 1, CV_32FC1, prob);
  cv::normalize(out_m, feature.f);
  features.push_back(feature);
  return features;
}

}  // namespace models
}  // namespace daisykit
