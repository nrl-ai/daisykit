#ifndef DAISYKIT_MODELS_HUMAN_MATTING_H_
#define DAISYKIT_MODELS_HUMAN_MATTING_H_

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

class HumanMatting {
 public:
  HumanMatting(const std::string& param_file, const std::string& weight_file);
  void LoadModel(const std::string& param_file, const std::string& weight_file);
#ifdef __ANDROID__
  HumanMatting(AAssetManager* mgr, const std::string& param_file,
               const std::string& weight_file);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif

  void Segmentation(cv::Mat& image, cv::Mat& mask);
  void BindWithBackground(cv::Mat& rgb, const cv::Mat& bg, const cv::Mat& mask);

 private:
  ncnn::Net* model_ = 0;
  int input_width_ = 256;
  int input_height_ = 256;
};

}  // namespace models
}  // namespace daisykit

#endif