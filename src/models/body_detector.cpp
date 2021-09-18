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

#include "daisykitsdk/models/body_detector.h"
#include "daisykitsdk/processors/image_processors/img_utils.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

BodyDetector::BodyDetector(const char* param_buffer,
                           const unsigned char* weight_buffer,
                           const int& input_width, const int& input_height) {
  LoadModel(param_buffer, weight_buffer);
  input_width_ = input_width;
  input_height_ = input_height;
}

// Will be deleted when IO module is supported. Keep for old compatibility.
BodyDetector::BodyDetector(const std::string& param_file,
                           const std::string& weight_file,
                           const int& input_width, const int& input_height) {
  LoadModel(param_file, weight_file);
  input_width_ = input_width;
  input_height_ = input_height;
}

std::vector<types::Object> BodyDetector::Predict(cv::Mat& image) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();
  int img_w = rgb.cols;
  int img_h = rgb.rows;

  ncnn::Mat in =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w,
                                    img_h, input_width_, input_height_);

  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = model_.create_extractor();
  ex.input("data", in);
  ncnn::Mat out;
  ex.extract("output", out);

  std::vector<types::Object> objects;
  for (int i = 0; i < out.h; i++) {
    types::Object object;
    float x1, y1, x2, y2, score, label;
    float pw, ph, cx, cy;
    const float* values = out.row(i);

    x1 = values[2] * img_w;
    y1 = values[3] * img_h;
    x2 = values[4] * img_w;
    y2 = values[5] * img_h;

    pw = x2 - x1;
    ph = y2 - y1;
    cx = x1 + 0.5 * pw;
    cy = y1 + 0.5 * ph;

    x1 = cx - 0.7 * pw;
    y1 = cy - 0.6 * ph;
    x2 = cx + 0.7 * pw;
    y2 = cy + 0.6 * ph;

    object.confidence = values[1];
    object.class_id = values[0];
    object.x = x1;
    object.y = y1;
    object.w = x2 - x1;
    object.h = y2 - y1;

    objects.push_back(object);
  }

  return objects;
}

}  // namespace models
}  // namespace daisykit