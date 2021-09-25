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

#include "daisykitsdk/models/facial_landmark_estimator.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

FacialLandmarkEstimator::FacialLandmarkEstimator(
    const char* param_buffer, const unsigned char* weight_buffer,
    int input_width, int input_height, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {}

FacialLandmarkEstimator::FacialLandmarkEstimator(const std::string& param_file,
                                                 const std::string& weight_file,
                                                 int input_width,
                                                 int input_height, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {}

void FacialLandmarkEstimator::Preprocess(const cv::Mat& image,
                                         ncnn::Mat& net_input) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();
  net_input =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols,
                                    rgb.rows, InputWidth(), InputHeight());
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  net_input.substract_mean_normalize(0, norm_vals);
}

int FacialLandmarkEstimator::Detect(const cv::Mat& image,
                                    std::vector<types::Keypoint>& keypoints,
                                    float offset_x, float offset_y) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Inference
  ncnn::Mat out;
  int result = Predict(in, out, "input_1", "415");
  if (result != 0) {
    return result;
  }

  // Postprocess
  int img_width = image.cols;
  int img_height = image.rows;
  keypoints.clear();
  for (int i = 0; i < (int)(out.w / 2); ++i) {
    float x = out[i * 2];
    float y = out[i * 2 + 1];
    types::Keypoint keypoint;
    keypoint.x = x * img_width + offset_x;
    keypoint.y = y * img_height + offset_y;
    keypoint.confidence = 1.0;
    keypoints.push_back(keypoint);
  }
  return 0;
}

int FacialLandmarkEstimator::DetectMulti(const cv::Mat& image,
                                         std::vector<types::Face>& faces) {
  int num_errors = 0;
  int img_width = image.cols;
  int img_height = image.rows;
  int x1, y1, x2, y2;

  int padding = 20;
  for (size_t i = 0; i < faces.size(); ++i) {
    x1 = faces[i].x - padding;
    y1 = faces[i].y - padding;
    x2 = faces[i].x + faces[i].w + padding;
    y2 = faces[i].y + faces[i].h + padding;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;
    if (x1 > img_width) x1 = img_width;
    if (y1 > img_height) y1 = img_height;
    if (x2 > img_width) x2 = img_width;
    if (y2 > img_height) y2 = img_height;

    cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    if (Detect(roi, faces[i].landmark, x1, y1) != 0) {
      ++num_errors;
    }
  }

  return num_errors;
}

}  // namespace models
}  // namespace daisykit
