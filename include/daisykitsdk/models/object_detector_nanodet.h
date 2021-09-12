#ifndef DAISYKIT_MODELS_OBJECT_DETECTOR_NANODET_H_
#define DAISYKIT_MODELS_OBJECT_DETECTOR_NANODET_H_

#include "daisykitsdk/common/types.h"

#include <omp.h>
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

class ObjectDetectorNanodet {
 public:
  ObjectDetectorNanodet(const std::string& param_file,
                        const std::string& weight_file);
  void LoadModel(const std::string& param_file, const std::string& weight_file);
#ifdef __ANDROID__
  ObjectDetectorNanodet(AAssetManager* mgr, const std::string& param_file,
                        const std::string& weight_file);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif

  std::vector<daisykit::common::Object> Detect(cv::Mat& image);

 private:
  static float IntersectionArea(const common::Object& a,
                                const common::Object& b);
  static void QsortDescentInplace(std::vector<common::Object>& objects,
                                  int left, int right);
  static void QsortDescentInplace(std::vector<common::Object>& objects);
  static void NmsSortedBboxes(const std::vector<common::Object>& objects,
                              std::vector<int>& picked, float nms_threshold);
  static void GenerateProposals(const ncnn::Mat& cls_pred,
                                const ncnn::Mat& dis_pred, int stride,
                                const ncnn::Mat& in_pad, float prob_threshold,
                                std::vector<common::Object>& objects);

  const int input_width_ = 320;
  const int input_height_ = 320;
  ncnn::Mutex lock_;
  ncnn::Net* model_ = 0;
};

}  // namespace models
}  // namespace daisykit

#endif