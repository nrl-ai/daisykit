#ifndef DAISYKIT_MODELS_FACE_DETECTOR_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_H_

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

#include <opencv2/opencv.hpp>

#include <benchmark.h>
#include <cpu.h>
#include <datareader.h>
#include <gpu.h>
#include <net.h>
#include <platform.h>

#include <daisykitsdk/common/types.h>

namespace daisykit {
namespace models {

enum NmsMethod {
  kHardNms = 1,
  kBlendingNms = 2 /* mix nms was been proposaled in paper blaze
                                 face, aims to minimize the temporal jitter*/
};

class FaceDetector {
 public:
  FaceDetector(const std::string& param_file, const std::string& weight_file,
               int input_width = 320, int input_height = 240,
               float score_threshold = 0.7, float iou_threshold = 0.5);
  void LoadModel(const std::string& param_file, const std::string& weight_file5);
#ifdef __ANDROID__
  FaceDetector(AAssetManager* mgr, const std::string& param_file,
               const std::string& weight_file, int input_width = 320,
               int input_height = 240, float score_threshold = 0.7,
               float iou_threshold = 0.5);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif

  std::vector<daisykit::common::Face> Detect(cv::Mat& image);

 private:
  void InitParams(int input_width = 320, int input_height = 240,
                  float score_threshold = 0.7, float iou_threshold = 0.5);
  void GenerateBBox(std::vector<common::Face>& bbox_collection,
                    ncnn::Mat scores, ncnn::Mat boxes, float score_threshold,
                    int num_anchors, int image_width, int image_height);

  void Nms(std::vector<common::Face>& input, std::vector<common::Face>& output,
           int type = 2);

  ncnn::Net* model_ = 0;

  const int kNumFeaturemaps = 4;

  int input_width_ = 320;
  int input_height_ = 240;
  std::vector<int> w_h_list_;

  int num_anchors_;

  float score_threshold_;
  float iou_threshold_;

  const float mean_vals_[3] = {127, 127, 127};
  const float norm_vals_[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

  const float center_variance_ = 0.1;
  const float size_variance_ = 0.2;
  const std::vector<std::vector<float>> min_boxes_ = {{10.0f, 16.0f, 24.0f},
                                                      {32.0f, 48.0f},
                                                      {64.0f, 96.0f},
                                                      {128.0f, 192.0f, 256.0f}};
  const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
  std::vector<std::vector<float>> featuremap_size_;
  std::vector<std::vector<float>> shrinkage_size_;
  std::vector<std::vector<float>> priors_ = {};
};

}  // namespace models
}  // namespace daisykit

#endif