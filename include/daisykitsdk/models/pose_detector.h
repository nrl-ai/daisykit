#ifndef DAISYKIT_MODELS_POSE_DETECTOR_H_
#define DAISYKIT_MODELS_POSE_DETECTOR_H_

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

#include "daisykitsdk/common/types.h"

namespace daisykit {
namespace models {

class PoseDetector {
 private:
  const int _input_width = 192;
  const int _input_height = 256;
  ncnn::Mutex _lock;
  ncnn::Net* _model = 0;

 public:
  PoseDetector(const std::string& param_file, const std::string& weight_file);
  void load_model(const std::string& param_file,
                  const std::string& weight_file);
#ifdef __ANDROID__
  PoseDetector(AAssetManager* mgr, const std::string& param_file,
               const std::string& weight_file);
  void load_model(AAssetManager* mgr, const std::string& param_file,
                  const std::string& weight_file);
#endif
  // Detect keypoints for single object
  std::vector<daisykit::common::Keypoint> detect(cv::Mat& image, float offset_x = 0,
                               float offset_y = 0);
  // Detect keypoints for multiple objects
  std::vector<std::vector<daisykit::common::Keypoint>> detect_multi(
      cv::Mat& image, const std::vector<daisykit::common::Object>& objects);
  // Draw pose
  void draw_pose(const cv::Mat& image, const std::vector<daisykit::common::Keypoint>& keypoints);
};

}  // namespace models
}  // namespace daisykit

#endif