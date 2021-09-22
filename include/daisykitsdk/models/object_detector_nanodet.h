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

#ifndef DAISYKIT_MODELS_OBJECT_DETECTOR_NANODET_H_
#define DAISYKIT_MODELS_OBJECT_DETECTOR_NANODET_H_

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
#include <omp.h>
#include <platform.h>

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

  std::vector<daisykit::types::Object> Detect(cv::Mat& image);

 private:
  static float IntersectionArea(const types::Object& a, const types::Object& b);
  static void QsortDescentInplace(std::vector<types::Object>& objects, int left,
                                  int right);
  static void QsortDescentInplace(std::vector<types::Object>& objects);
  static void NmsSortedBboxes(const std::vector<types::Object>& objects,
                              std::vector<int>& picked, float nms_threshold);
  static void GenerateProposals(const ncnn::Mat& cls_pred,
                                const ncnn::Mat& dis_pred, int stride,
                                const ncnn::Mat& in_pad, float prob_threshold,
                                std::vector<types::Object>& objects);

  const int input_width_ = 320;
  const int input_height_ = 320;
  ncnn::Mutex lock_;
  ncnn::Net* model_ = 0;
};

}  // namespace models
}  // namespace daisykit

#endif
