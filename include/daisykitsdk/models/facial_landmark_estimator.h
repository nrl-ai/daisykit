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

#ifndef DAISYKIT_MODELS_FACIAL_LANDMARK_ESTIMATOR_H_
#define DAISYKIT_MODELS_FACIAL_LANDMARK_ESTIMATOR_H_

#include <daisykitsdk/common/types.h>

#include <opencv2/opencv.hpp>
#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

// Ncnn
#include <benchmark.h>
#include <cpu.h>
#include <datareader.h>
#include <gpu.h>
#include <net.h>
#include <platform.h>
namespace daisykit {
namespace models {

class FacialLandmarkEstimator {
 public:
  FacialLandmarkEstimator(const std::string& param_file,
                          const std::string& weight_file);
  void LoadModel(const std::string& param_file, const std::string& weight_file);
#ifdef __ANDROID__
  FacialLandmarkEstimator(AAssetManager* mgr, const std::string& param_file,
                          const std::string& weight_file);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif
  // Detect keypoints for single object
  std::vector<daisykit::types::Keypoint> Detect(cv::Mat& image,
                                                float offset_x = 0,
                                                float offset_y = 0);
  // Detect keypoints for multiple objects
  void DetectMulti(cv::Mat& image, std::vector<daisykit::types::Face>& objects);
  // Draw pose
  void DrawKeypoints(const cv::Mat& image,
                     const std::vector<daisykit::types::Keypoint>& keypoints);

 private:
  const int input_width_ = 112;
  const int input_height_ = 112;
  ncnn::Net* model_ = 0;
  ncnn::Mutex lock_;
};

}  // namespace models
}  // namespace daisykit

#endif