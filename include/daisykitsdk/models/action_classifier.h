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

#ifndef DAISYKIT_MODELS_ACTION_CLASSIFIER_H_
#define DAISYKIT_MODELS_ACTION_CLASSIFIER_H_

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
class ActionClassifier {
 public:
  ActionClassifier(const std::string& param_file,
                   const std::string& weight_file, bool smooth = true);
  void LoadModel(const std::string& param_file, const std::string& weight_file);
#ifdef __ANDROID__
  ActionClassifier(AAssetManager* mgr, const std::string& param_file,
                   const std::string& weight_file, bool smooth = true);
  void LoadModel(AAssetManager* mgr, const std::string& param_file,
                 const std::string& weight_file);
#endif
  daisykit::types::Action Classify(cv::Mat& image);
  daisykit::types::Action Classify(cv::Mat& image, float& confidence);

 private:
  const int input_width_ = 224;
  const int input_height_ = 224;
  ncnn::Mutex lock_;
  ncnn::Net* model_ = 0;
  bool _smooth = true;  // Smooth the result or not
  long long int last_pushup_time_ = 0;
};

}  // namespace models
}  // namespace daisykit

#endif
