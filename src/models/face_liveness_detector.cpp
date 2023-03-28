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

#include "daisykit/models/face_liveness_detector.h"
#include "daisykit/common/types/face_extended.h"
#include "daisykit/processors/image_processors/img_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace daisykit {
namespace models {
FaceLivenessDetector::FaceLivenessDetector(const char* param_buffer,
                                           const unsigned char* weight_buffer,
                                           int input_width, int input_height,
                                           bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {}

FaceLivenessDetector::FaceLivenessDetector(const std::string& param_file,
                                           const std::string& weight_file,
                                           int input_width, int input_height,
                                           bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {}

#if __ANDROID__
LivenessDetector::LivenessDetector(AAssetManager* mgr,
                                   const std::string& param_file,
                                   const std::string& weight_file)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {}
#endif

cv::Rect FaceLivenessDetector::CalculateBox(types::FaceExtended& face_box,
                                            int w, int h) {
  float scale_ = 4.0;
  float scale = std::min(scale_, std::min((w - 1) / (float)face_box.w,
                                          (h - 1) / (float)face_box.h));
  int box_center_x = face_box.w / 2 + face_box.x;
  int box_center_y = face_box.h / 2 + face_box.y;

  int new_width = static_cast<int>(face_box.w * scale);
  int new_height = static_cast<int>(face_box.h * scale);

  int left_top_x = box_center_x - new_width / 2;
  int left_top_y = box_center_y - new_height / 2;
  int right_bottom_x = box_center_x + new_width / 2;
  int right_bottom_y = box_center_y + new_height / 2;

  if (left_top_x < 0) {
    right_bottom_x -= left_top_x;
    left_top_x = 0;
  }

  if (left_top_y < 0) {
    right_bottom_y -= left_top_y;
    left_top_y = 0;
  }

  if (right_bottom_x >= w) {
    int s = right_bottom_x - w + 1;
    left_top_x -= s;
    right_bottom_x -= s;
  }

  if (right_bottom_y >= h) {
    int s = right_bottom_y - h + 1;
    left_top_y -= s;
    right_bottom_y -= s;
  }

  return cv::Rect(left_top_x, left_top_y, new_width, new_height);
}

void FaceLivenessDetector::Preprocess(const cv::Mat& image,
                                      ncnn::Mat& net_input) {
  net_input = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR,
                                     image.cols, image.rows);
}

int FaceLivenessDetector::Predict(const cv::Mat& image,
                                  types::FaceExtended& faces) {
  float liveness_score = 0.f;
  cv::Mat roi;
  cv::Rect rect = CalculateBox(faces, image.cols, image.rows);
  cv::resize(image(rect), roi, cv::Size(80, 80));

  // Preprocess
  ncnn::Mat in;
  Preprocess(roi, in);

  ncnn::Mat out;
  int result = Infer(in, out, "data", "softmax");
  if (result != 0) return 0;

  // Post process
  liveness_score += out.row(0)[1];
  faces.liveness_score = liveness_score;
  // Model Inference

  return 0;
}
}  // namespace models
}  // namespace daisykit
