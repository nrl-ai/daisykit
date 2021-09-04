#ifndef DAISYKIT_MODELS_ACTION_CLASSIFIER_H_
#define DAISYKIT_MODELS_ACTION_CLASSIFIER_H_

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <benchmark.h>
#include <cpu.h>
#include <datareader.h>
#include <gpu.h>
#include <net.h>
#include <platform.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON

#include "daisykitsdk/common/types.h"

namespace daisykit {
namespace models {
class ActionClassifier {
 private:
  const int _input_width = 224;
  const int _input_height = 224;
  ncnn::Mutex _lock;
  ncnn::Net* _model = 0;
  bool _smooth = true;  // Smooth the result or not
  long long int _last_pushup_time = 0;

 public:
  ActionClassifier(const std::string& param_file,
                   const std::string& weight_file, bool smooth = true);
  void load_model(const std::string& param_file,
                  const std::string& weight_file);
#ifdef __ANDROID__
  ActionClassifier(AAssetManager* mgr, const std::string& param_file,
                   const std::string& weight_file, bool smooth = true);
  void load_model(AAssetManager* mgr, const std::string& param_file,
                  const std::string& weight_file);
#endif
  daisykit::common::Action classify(cv::Mat& image);
  daisykit::common::Action classify(cv::Mat& image, float& confidence);

 private:
  cv::Mat square_padding(const cv::Mat& img, int target_width = 500);
};

}  // namespace models
}  // namespace daisykit

#endif