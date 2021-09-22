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

#include "daisykitsdk/models/pose_detector.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

PoseDetector::PoseDetector(const char* param_buffer,
                           const unsigned char* weight_buffer, int input_width,
                           int input_height) {
  LoadModel(param_buffer, weight_buffer);
  input_height_ = input_height;
  input_width_ = input_width;
}

// Will be deleted when IO module is supported. Keep for old compatibility.
PoseDetector::PoseDetector(const std::string& param_file,
                           const std::string& weight_file, int input_width,
                           int input_height) {
  LoadModel(param_file, weight_file);
  input_height_ = input_height;
  input_width_ = input_width;
}

// Detect keypoints for single object
std::vector<types::Keypoint> PoseDetector::Predict(const cv::Mat& image) {
  std::vector<types::Keypoint> keypoints;
  int w = image.cols;
  int h = image.rows;
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB,
                                               image.cols, image.rows,
                                               input_width_, input_height_);
  const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
  const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f,
                              1 / 0.225f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);
  ncnn::MutexLockGuard g(lock_);
  ncnn::Extractor ex = model_.create_extractor();
  ex.input("data", in);
  ncnn::Mat out;
  ex.extract("hybridsequential0_conv7_fwd", out);
  keypoints.clear();
  for (int p = 0; p < out.c; p++) {
    const ncnn::Mat m = out.channel(p);
    float max_prob = 0.f;
    int max_x = 0;
    int max_y = 0;
    for (int y = 0; y < out.h; y++) {
      const float* ptr = m.row(y);
      for (int x = 0; x < out.w; x++) {
        float prob = ptr[x];
        if (prob > max_prob) {
          max_prob = prob;
          max_x = x;
          max_y = y;
        }
      }
    }

    types::Keypoint keypoint;
    keypoint.x = max_x * w / (float)out.w;
    keypoint.y = max_y * h / (float)out.h;
    keypoint.confidence = max_prob;
    keypoints.push_back(keypoint);
  }
  return keypoints;
}

// Detect keypoints for single object
std::vector<types::Keypoint> PoseDetector::Predict(cv::Mat& image,
                                                   float offset_x,
                                                   float offset_y) {
  std::vector<types::Keypoint> keypoints;
  int w = image.cols;
  int h = image.rows;
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB,
                                               image.cols, image.rows,
                                               input_width_, input_height_);
  const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
  const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f,
                              1 / 0.225f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);
  ncnn::MutexLockGuard g(lock_);
  ncnn::Extractor ex = model_.create_extractor();
  ex.input("data", in);
  ncnn::Mat out;
  ex.extract("hybridsequential0_conv7_fwd", out);
  keypoints.clear();
  for (int p = 0; p < out.c; p++) {
    const ncnn::Mat m = out.channel(p);
    float max_prob = 0.f;
    int max_x = 0;
    int max_y = 0;
    for (int y = 0; y < out.h; y++) {
      const float* ptr = m.row(y);
      for (int x = 0; x < out.w; x++) {
        float prob = ptr[x];
        if (prob > max_prob) {
          max_prob = prob;
          max_x = x;
          max_y = y;
        }
      }
    }

    types::Keypoint keypoint;
    keypoint.x = max_x * w / (float)out.w + offset_x;
    keypoint.y = max_y * h / (float)out.h + offset_y;
    keypoint.confidence = max_prob;
    keypoints.push_back(keypoint);
  }
  return keypoints;
}

// Detect keypoints for multiple objects
std::vector<std::vector<types::Keypoint>> PoseDetector::PredictMulti(
    cv::Mat& image, const std::vector<types::Object>& objects) {
  int img_w = image.cols;
  int img_h = image.rows;
  int x1, y1, x2, y2;

  std::vector<std::vector<types::Keypoint>> keypoints;
  for (size_t i = 0; i < objects.size(); ++i) {
    x1 = objects[i].x;
    y1 = objects[i].y;
    x2 = objects[i].x + objects[i].w;
    y2 = objects[i].y + objects[i].h;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;
    if (x1 > img_w) x1 = img_w;
    if (y1 > img_h) y1 = img_h;
    if (x2 > img_w) x2 = img_w;
    if (y2 > img_h) y2 = img_h;

    cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    std::vector<types::Keypoint> keypoints_single = Predict(roi, x1, y1);
    keypoints.push_back(keypoints_single);
  }

  return keypoints;
}

// Draw pose
void PoseDetector::DrawKeypoints(
    const cv::Mat& image, const std::vector<types::Keypoint>& keypoints) {
  float threshold = 0.2;
  // draw bone
  static const int joint_pairs[16][2] = {
      {0, 1},   {1, 3},   {0, 2},   {2, 4},  {5, 6},  {5, 7},
      {7, 9},   {6, 8},   {8, 10},  {5, 11}, {6, 12}, {11, 12},
      {11, 13}, {12, 14}, {13, 15}, {14, 16}};
  for (int i = 0; i < 16; i++) {
    const types::Keypoint& p1 = keypoints[joint_pairs[i][0]];
    const types::Keypoint& p2 = keypoints[joint_pairs[i][1]];
    if (p1.confidence < threshold || p2.confidence < threshold) continue;
    cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y),
             cv::Scalar(255, 0, 0), 2);
  }
  // draw joint
  for (size_t i = 0; i < keypoints.size(); i++) {
    const types::Keypoint& keypoint = keypoints[i];
    if (keypoint.confidence < threshold) continue;
    cv::circle(image, cv::Point(keypoint.x, keypoint.y), 3,
               cv::Scalar(0, 255, 0), -1);
  }
}

}  // namespace models
}  // namespace daisykit
