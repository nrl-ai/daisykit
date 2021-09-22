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
    int input_width, int input_height) {
  LoadModel(param_buffer, weight_buffer);
  input_width_ = input_width;
  input_height_ = input_height;
}

// Will be deleted when IO module is supported. Keep for old compatibility.
FacialLandmarkEstimator::FacialLandmarkEstimator(const std::string& param_file,
                                                 const std::string& weight_file,
                                                 int input_width,
                                                 int input_height) {
  LoadModel(param_file, weight_file);
  input_width_ = input_width;
  input_height_ = input_height;
}

std::vector<types::Keypoint> FacialLandmarkEstimator::Predict(
    const cv::Mat& image) {
  std::vector<types::Keypoint> keypoints;
  int w = image.cols;
  int h = image.rows;
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB,
                                               image.cols, image.rows,
                                               input_width_, input_height_);
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(0, norm_vals);
  ncnn::MutexLockGuard g(lock_);
  ncnn::Extractor ex = model_.create_extractor();
  ncnn::Mat out;
  ex.input("input_1", in);
  ex.extract("415", out);
  keypoints.clear();
  for (int i = 0; i < (int)(out.w / 2); ++i) {
    float x = out[i * 2];
    float y = out[i * 2 + 1];
    types::Keypoint keypoint;
    keypoint.x = x * w;
    keypoint.y = y * h;
    keypoint.confidence = 1.0;
    keypoints.push_back(keypoint);
  }
  return keypoints;
}

std::vector<types::Keypoint> FacialLandmarkEstimator::Predict(
    const cv::Mat& image, float offset_x, float offset_y) {
  std::vector<types::Keypoint> keypoints;
  int w = image.cols;
  int h = image.rows;
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB,
                                               image.cols, image.rows,
                                               input_width_, input_height_);
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(0, norm_vals);
  ncnn::MutexLockGuard g(lock_);
  ncnn::Extractor ex = model_.create_extractor();
  ncnn::Mat out;
  ex.input("input_1", in);
  ex.extract("415", out);
  keypoints.clear();
  for (int i = 0; i < (int)(out.w / 2); ++i) {
    float x = out[i * 2];
    float y = out[i * 2 + 1];
    types::Keypoint keypoint;
    keypoint.x = x * w + offset_x;
    keypoint.y = y * h + offset_y;
    keypoint.confidence = 1.0;
    keypoints.push_back(keypoint);
  }
  return keypoints;
}

void FacialLandmarkEstimator::PredictMulti(const cv::Mat& image,
                                           std::vector<types::Face>& objects) {
  int img_w = image.cols;
  int img_h = image.rows;
  int x1, y1, x2, y2;

  int padding = 20;
  for (size_t i = 0; i < objects.size(); ++i) {
    x1 = objects[i].x - padding;
    y1 = objects[i].y - padding;
    x2 = objects[i].x + objects[i].w + padding;
    y2 = objects[i].y + objects[i].h + padding;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;
    if (x1 > img_w) x1 = img_w;
    if (y1 > img_h) y1 = img_h;
    if (x2 > img_w) x2 = img_w;
    if (y2 > img_h) y2 = img_h;

    cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    objects[i].landmark = Predict(roi, x1, y1);
  }
}

}  // namespace models
}  // namespace daisykit
