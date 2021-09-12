#ifndef DAISYKIT_MODELS_FACIAL_LANDMARK_ESTIMATOR_H_
#define DAISYKIT_MODELS_FACIAL_LANDMARK_ESTIMATOR_H_

#include "daisykitsdk/common/types.h"

#include <benchmark.h>
#include <cpu.h>
#include <datareader.h>
#include <gpu.h>
#include <net.h>
#include <platform.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace models {

class FacialLandmarkEstimator {
 public:
  FacialLandmarkEstimator(const std::string &param_file,
                          const std::string &weight_file);
  void LoadModel(const std::string &param_file, const std::string &weight_file);
#ifdef __ANDROID__
  FacialLandmarkEstimator(AAssetManager *mgr, const std::string &param_file,
                          const std::string &weight_file);
  void LoadModel(AAssetManager *mgr, const std::string &param_file,
                 const std::string &weight_file);
#endif
  // Detect keypoints for single object
  std::vector<daisykit::common::Keypoint> Detect(cv::Mat &image,
                                                 float offset_x = 0,
                                                 float offset_y = 0);
  // Detect keypoints for multiple objects
  void DetectMulti(cv::Mat &image,
                   std::vector<daisykit::common::Face> &objects);
  // Draw pose
  void DrawKeypoints(const cv::Mat &image,
                     const std::vector<daisykit::common::Keypoint> &keypoints);

 private:
  const int input_width_ = 112;
  const int input_height_ = 112;
  ncnn::Net *model_ = 0;
  ncnn::Mutex lock_;
};

}  // namespace models
}  // namespace daisykit

#endif