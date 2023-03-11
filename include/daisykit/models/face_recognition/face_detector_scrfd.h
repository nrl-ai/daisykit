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

#ifndef DAISYKIT_MODELS_FACE_DETECTOR_SCRFD_H_
#define DAISYKIT_MODELS_FACE_DETECTOR_SCRFD_H_

#include "daisykit/common/types.h"
#include "daisykit/models/image_model.h"
#include "daisykit/models/ncnn_model.h"
#include "opencv2/opencv.hpp"

namespace daisykit {
namespace models {

template <typename FaceT>
class FaceDetectorSCRFD : public NCNNModel, public ImageModel {
 public:
  FaceDetectorSCRFD(const std::string& param_file,
                    const std::string& bin_path_file, int input_size = 640,
                    float score_threshold = 0.7, float iou_threshold = 0.5,
                    bool use_gpu = false);

  FaceDetectorSCRFD(const char* param_buffer,
                    const unsigned char* weight_buffer, int input_size = 640,
                    float score_threshold = 0.7, float iou_threshold = 0.5,
                    bool use_gpu = false);
#ifdef __ANDROID__
  FaceDetectorSCRFD(AAssetManager* mgr, const std::string& param_file,
                    const std::string& bin_path_file, int input_size = 640,
                    float score_threshold = 0.7, float iou_threshold = 0.5,
                    bool use_gpu = false);
#endif

  /// Detect faces in an image.
  /// Return 0 on success, otherwise return error code.
  int Predict(const cv::Mat& image, std::vector<FaceT>& faces);

 private:
  void Preprocess(const cv::Mat& image, ncnn::Mat& net_input) override;
  float IntersectionArea(const FaceT& a, const FaceT& b);
  void QsortDescentInplace(std::vector<FaceT>& faceobjects, int left,
                           int right);
  void QsortDescentInplace(std::vector<FaceT>& faceobjects);
  void NmsSortedBboxes(std::vector<FaceT>& faceobjects,
                       std::vector<int>& picked, float nms_threshold);
  ncnn::Mat GenerateAnchors(int base_size, const ncnn::Mat& ratios,
                            const ncnn::Mat& scales);
  void GenerateProposals(const ncnn::Mat& anchors, int feat_stride,
                         const ncnn::Mat& score_blob,
                         const ncnn::Mat& bbox_blob, const ncnn::Mat& kps_blob,
                         float prob_threshold, std::vector<FaceT>& faces);

 private:
  int input_size_;
  bool is_landmark_ = true;
  float score_threshold_;
  float iou_threshold_;
  int hpad_ = 0;
  int wpad_ = 0;
  float scale_ = 1.0;
  std::vector<std::string> output_names_ = {"score_8",  "bbox_8",  "kps_8",
                                            "score_16", "bbox_16", "kps_16",
                                            "score_32", "bbox_32", "kps_32"};
  std::string input_name_ = "input.1";
};

}  // namespace models
}  // namespace daisykit
#endif
