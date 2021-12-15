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

#ifndef DAISYKIT_MODELS_OBJECT_DETECTOR_YOLOX_H_
#define DAISYKIT_MODELS_OBJECT_DETECTOR_YOLOX_H_

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

/// Object detection model
class ObjectDetectorYOLOX : public NCNNModel, public ImageModel {
 public:
  ObjectDetectorYOLOX(const char* param_buffer,
                      const unsigned char* weight_buffer,
                      float score_threshold = 0.7, float iou_threshold = 0.5,
                      int input_width = 416, int input_height = 416,
                      bool use_gpu = false);

  ObjectDetectorYOLOX(const std::string& param_file,
                      const std::string& weight_file,
                      float score_threshold = 0.7, float iou_threshold = 0.5,
                      int input_width = 416, int input_height = 416,
                      bool use_gpu = false);

#ifdef __ANDROID__
  ObjectDetectorYOLOX(AAssetManager* mgr, const std::string& param_file,
                      const std::string& weight_file,
                      float score_threshold = 0.7, float iou_threshold = 0.5,
                      int input_width = 416, int input_height = 416,
                      bool use_gpu = false);
#endif

  /// Detect faces in an image.
  /// Return 0 on success, otherwise return error code.
  int Predict(const cv::Mat& image,
              std::vector<daisykit::types::Object>& hands);

  /// Set class names.
  void SetClassNames(const std::vector<std::string>& class_names);

  /// Get class names.
  std::vector<std::string>& GetClassNames();

 private:
  /// Preprocess image data to obtain net input.
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;

  /// Score threshold for bounding box. Currently we ignore to set this value
  /// because the current model has an internal of 0.25 inside and that's
  /// enough.
  float score_threshold_;

  /// IoU threshold for NMS. Currently we ignore this value because our current
  /// model doesn't need NMS.
  float iou_threshold_;

  /// Cache scale from preprocess step
  /// For restoring on postprocess
  float scale_;

  /// Class names
  std::vector<std::string> class_names_;
};

}  // namespace models
}  // namespace daisykit

#endif
