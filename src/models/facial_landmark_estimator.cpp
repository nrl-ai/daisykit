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
#include "daisykitsdk/utils/img_proc/img_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace daisykit::common;
using namespace daisykit::models;

FacialLandmarkEstimator::FacialLandmarkEstimator(
    const std::string& param_file, const std::string& weight_file) {
  LoadModel(param_file, weight_file);
}

void FacialLandmarkEstimator::LoadModel(const std::string& param_file,
                                        const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(param_file.c_str());
  int ret_model = model_->load_model(weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}

#ifdef __ANDROID__
FacialLandmarkEstimator::FacialLandmarkEstimator(
    AAssetManager* mgr, const std::string& param_file,
    const std::string& weight_file) {
  LoadModel(mgr, param_file, weight_file);
}

void FacialLandmarkEstimator::LoadModel(AAssetManager* mgr,
                                        const std::string& param_file,
                                        const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(mgr, param_file.c_str());
  int ret_model = model_->load_model(mgr, weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}
#endif

// Detect keypoints for single object
std::vector<Keypoint> FacialLandmarkEstimator::Detect(cv::Mat& image,
                                                      float offset_x,
                                                      float offset_y) {
  std::vector<Keypoint> keypoints;
  int w = image.cols;
  int h = image.rows;
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB,
                                               image.cols, image.rows,
                                               input_width_, input_height_);
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in.substract_mean_normalize(0, norm_vals);
  ncnn::MutexLockGuard g(lock_);
  ncnn::Extractor ex = model_->create_extractor();
  ncnn::Mat out;
  ex.input("input_1", in);
  ex.extract("415", out);
  keypoints.clear();
  for (int i = 0; i < (int)(out.w / 2); ++i) {
    float x = out[i * 2];
    float y = out[i * 2 + 1];
    Keypoint keypoint;
    keypoint.x = x * w + offset_x;
    keypoint.y = y * h + offset_y;
    keypoint.prob = 1.0;
    keypoints.push_back(keypoint);
  }
  return keypoints;
}

// Detect keypoints for multiple objects
void FacialLandmarkEstimator::DetectMulti(cv::Mat& image,
                                          std::vector<Face>& objects) {
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
    objects[i].landmark = Detect(roi, x1, y1);
  }
}

// Draw pose
void FacialLandmarkEstimator::DrawKeypoints(
    const cv::Mat& image, const std::vector<Keypoint>& keypoints) {
  float threshold = 0.2;
  // draw joint
  for (size_t i = 0; i < keypoints.size(); i++) {
    const Keypoint& keypoint = keypoints[i];
    // fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y,
    // keypoint.prob);
    if (keypoint.prob < threshold) continue;
    cv::circle(image, cv::Point(keypoint.x, keypoint.y), 2,
               cv::Scalar(255, 0, 0), -1);
  }
}