#ifndef DAISYKIT_MODELS_FACE_DETECTOR_WITH_MASK_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_WITH_MASK_H_

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

class FaceDetectorWithMask {
 public:
  FaceDetectorWithMask(const std::string& param_file,
                       const std::string& weight_file, int input_width = 320,
                       int input_height = 320, float score_threshold = 0.7,
                       float iou_threshold = 0.5);
  void LoadModel(const std::string& param_file,
                 const std::string& weight_file5);
#ifdef __ANDROID__
  FaceDetectorWithMask(AAssetManager* mgr, const std::string& param_file,
                       const std::string& weight_file, int input_width = 320,
                       int input_height = 320, float score_threshold = 0.7,
                       float iou_threshold = 0.5);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif

  std::vector<daisykit::common::Face> Detect(cv::Mat& image);

 private:
  void InitParams(int input_width = 320, int input_height = 320,
                  float score_threshold = 0.7, float iou_threshold = 0.5);

  void Nms(std::vector<common::Face>& input, std::vector<common::Face>& output);

  ncnn::Net* model_ = 0;

  int input_width_ = 320;
  int input_height_ = 320;
  std::vector<int> w_h_list_;

  int num_anchors_;

  float score_threshold_;
  float iou_threshold_;
};

}  // namespace models
}  // namespace daisykit

#endif