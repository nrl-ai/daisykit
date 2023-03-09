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

#ifndef DAISYKIT_MODELS_FACE_DETECTOR_RETINA_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_RETINA_H_

#include "daisykit/common/types.h"
#include "daisykit/models/face_recognition/anchor_cfg.h"
#include "daisykit/models/face_recognition/anchor_generator.h"
#include "daisykit/models/face_recognition/tools.h"
#include "daisykit/models/image_model.h"
#include "daisykit/models/ncnn_model.h"
#include "opencv2/opencv.hpp"

namespace daisykit {
namespace models {

class FaceDetectorRetina : public NCNNModel, public ImageModel {
 public:
  FaceDetectorRetina(const std::string& param_file,
                     const std::string& bin_path_file, int input_width = 320,
                     int input_height = 320, float score_threshold = 0.7,
                     float iou_threshold = 0.5, bool use_gpu = false);

  FaceDetectorRetina(const char* param_buffer,
                     const unsigned char* weight_buffer, int input_width = 320,
                     int input_height = 320, float score_threshold = 0.7,
                     float iou_threshold = 0.5, bool use_gpu = false);
#ifdef __ANDROID__
  FaceDetectorWithLandMark(AAssetManager* mgr, const std::string& param_file,
                           const std::string& bin_path_file,
                           int input_width = 320, int input_height = 320,
                           float score_threshold = 0.7,
                           float iou_threshold = 0.5, bool use_gpu = false);
#endif

  std::vector<daisykit::types::FaceDet> Predict(cv::Mat& img);
  void DrawFaceDets(cv::Mat& img, std::vector<daisykit::types::FaceDet>& dets);

 private:
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;
  void Init();
  int input_width_;
  int input_height_;
  float score_threshold_;
  float iou_threshold_;
  float ratio_x_;
  float ratio_y_;
  std::vector<std::string> output_names_;
  std::vector<AnchorGenerator> ac_;
  float pixel_mean_[3] = {0, 0, 0};
  float pixel_std_[3] = {1, 1, 1};
  float cls_threshold_ = 0.8;
  float nms_threshold_ = 0.4;
};

}  // namespace models
}  // namespace daisykit
#endif
