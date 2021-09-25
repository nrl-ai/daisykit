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

#include <string>
#include <vector>

namespace daisykit {
namespace models {

BodyDetector::BodyDetector(const char* param_buffer,
                           const unsigned char* weight_buffer, int input_width,
                           int input_height, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {}

BodyDetector::BodyDetector(const std::string& param_file,
                           const std::string& weight_file, int input_width,
                           int input_height, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {}

void BodyDetector::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();

  net_input =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols,
                                    rgb.rows, InputWidth(), InputHeight());

  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  net_input.substract_mean_normalize(mean_vals, norm_vals);
}

int BodyDetector::Detect(const cv::Mat& image,
                         std::vector<daisykit::types::Object>& objects) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Inference
  ncnn::Mat out;
  int result = Predict(in, out, "data", "output");
  if (result != 0) {
    return result;
  }

  // Postprocess
  int img_width = image.cols;
  int img_height = image.rows;
  int count = out.h;
  objects.resize(count);
  for (int i = 0; i < count; i++) {
    types::Object object;
    float x1, y1, x2, y2, score, label;
    float pw, ph, cx, cy;
    const float* values = out.row(i);

    x1 = values[2] * img_width;
    y1 = values[3] * img_height;
    x2 = values[4] * img_width;
    y2 = values[5] * img_height;

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

    objects[i] = object;
  }
}

}  // namespace models
}  // namespace daisykit
