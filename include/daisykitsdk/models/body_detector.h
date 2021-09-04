#ifndef DAISYKIT_MODELS_BODY_DETECTOR_H_
#define DAISYKIT_MODELS_BODY_DETECTOR_H_

#include <stdio.h>
#include <algorithm>
#include <chrono>
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

#include <daisykitsdk/common/types.h>

namespace daisykit {
namespace models {

class BodyDetector {
 public:
  BodyDetector(const std::string& param_file, const std::string& weight_file);
  void LoadModel(const std::string& param_file, const std::string& weight_file);
#ifdef __ANDROID__
  BodyDetector(AAssetManager* mgr, const std::string& param_file,
               const std::string& weight_file);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif

  std::vector<daisykit::common::Object> Detect(cv::Mat& image);

 private:
  const int input_width_ = 320;
  const int input_height_ = 320;
  ncnn::Mutex lock_;
  ncnn::Net* model_ = 0;
};

}  // namespace models
}  // namespace daisykit

#endif