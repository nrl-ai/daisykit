// Copyright 2021 The DaisyKit Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DAISYKIT_MODELS_FACE_DETECTOR_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_H_
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
  void LoadModel(const std::string& param_file,
                 const std::string& weight_file5);
#ifdef __ANDROID__
  FaceDetector(AAssetManager* mgr, const std::string& param_file,
               const std::string& weight_file, int input_width = 320,
               int input_height = 240, float score_threshold = 0.7,
               float iou_threshold = 0.5);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif

  std::vector<daisykit::types::Face> Detect(cv::Mat& image);

 private:
  void InitParams(int input_width = 320, int input_height = 240,
                  float score_threshold = 0.7, float iou_threshold = 0.5);
  void GenerateBBox(std::vector<types::Face>& bbox_collection, ncnn::Mat scores,
                    ncnn::Mat boxes, float score_threshold, int num_anchors,
                    int image_width, int image_height);

  void Nms(std::vector<types::Face>& input, std::vector<types::Face>& output,
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