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

#ifndef DAISYKIT_MODELS_HAND_POSE_DETECTOR_H_
#define DAISYKIT_MODELS_HAND_POSE_DETECTOR_H_

#include "daisykit/common/types.h"
#include "daisykit/models/image_model.h"
#include "daisykit/models/ncnn_model.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#ifdef __ANDROID__
#include <android/asset_manager_jni.h>
#endif

namespace daisykit {
namespace models {

/// Hand pose model.
class HandPoseDetector : public NCNNModel, public ImageModel {
 public:
  HandPoseDetector(const char* param_buffer, const unsigned char* weight_buffer,
                   int input_size = 224, bool use_gpu = false);

  HandPoseDetector(const std::string& param_file,
                   const std::string& weight_file, int input_size = 224,
                   bool use_gpu = false);

#ifdef __ANDROID__
  HandPoseDetector(AAssetManager* mgr, const std::string& param_file,
                   const std::string& weight_file, int input_size = 224,
                   bool use_gpu = false);
#endif

  /// Detect keypoints for a single face.
  /// This function adds offset_x and offset_y to the keypoints.
  /// Return 0 on success, otherwise return error code.
  int Predict(const cv::Mat& image, std::vector<types::KeypointXYZ>& keypoints,
              float& lr_score, float offset_x = 0, float offset_y = 0);

  /// Detect keypoints for multiple faces.
  /// Modify faces vector to add landmark info. Return 0 on success, otherwise
  /// return the number of inference errors.
  int PredictMulti(const cv::Mat& image,
                   const std::vector<types::Object>& objects,
                   std::vector<std::vector<types::KeypointXYZ>>& poses,
                   std::vector<float>& lr_scores);

  /// Draw keypoints and their joints.
  void DrawHandPoses(
      cv::Mat& image,
      const std::vector<types::ObjectWithKeypointsXYZ>& keypoints);

 private:
  /// Preprocess image data to obtain net input.
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;

  /// Cache scale from preprocess step
  /// For restoring on postprocess
  float scale_;
  int hpad_;
  int wpad_;
};

}  // namespace models
}  // namespace daisykit

#endif
