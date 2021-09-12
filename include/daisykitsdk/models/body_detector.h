#ifndef DAISYKIT_MODELS_BODY_DETECTOR_H_
#define DAISYKIT_MODELS_BODY_DETECTOR_H_

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