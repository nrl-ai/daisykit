#ifndef POSE_DETECTOR_BLAZEPOSE_
#define POSE_DETECTOR_BLAZEPOSE_

#include <stdio.h>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>

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

class PoseDetectorBlazepose {
 private:
  const int _input_width = 256;
  const int _input_height = 256;
  ncnn::Mutex _lock;
  ncnn::Net* _model = 0;

 public:
  PoseDetectorBlazepose(const std::string& param_file, const std::string& weight_file);
  void load_model(const std::string& param_file,
                  const std::string& weight_file);
#ifdef __ANDROID__
  PoseDetectorBlazepose(AAssetManager* mgr, const std::string& param_file,
               const std::string& weight_file);
  void load_model(AAssetManager* mgr, const std::string& param_file,
                  const std::string& weight_file);
#endif
  // Detect keypoints for single object
  std::vector<Keypoint> detect(cv::Mat& image, float offset_x = 0,
                               float offset_y = 0);
  // Detect keypoints for multiple objects
  std::vector<std::vector<Keypoint>> detect_multi(
      cv::Mat& image, const std::vector<Object>& objects);
  // Draw pose
  void draw_pose(const cv::Mat& image, const std::vector<Keypoint>& keypoints);
};

#endif