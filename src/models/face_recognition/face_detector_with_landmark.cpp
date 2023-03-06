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
// limitations under the License

#include "daisykit/models/face_recognition/face_detector_with_landmark.h"

namespace daisykit {
namespace models {

FaceDetectorWithLandmark::FaceDetectorWithLandmark(
    const char* param_buffer, const unsigned char* weight_buffer,
    int input_width, int input_height, float score_threshold,
    float iou_threshold, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  Init();
}

FaceDetectorWithLandmark::FaceDetectorWithLandmark(
    const std::string& param_file, const std::string& weight_file,
    int input_width, int input_height, float score_threshold,
    float iou_threshold, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  Init();
}

void FaceDetectorWithLandmark::Init() {
  std::vector<int> feat_stride_fpn_ = {32, 16, 8};
  std::map<int, AnchorCfg> anchor_cfg_ = {
      {32, AnchorCfg(std::vector<float>{32, 16}, std::vector<float>{1}, 16)},
      {16, AnchorCfg(std::vector<float>{8, 4}, std::vector<float>{1}, 16)},
      {8, AnchorCfg(std::vector<float>{2, 1}, std::vector<float>{1}, 16)}};

  for (int i = 0; i < feat_stride_fpn_.size(); ++i) {
    int stride = feat_stride_fpn_[i];
    ac_.push_back(AnchorGenerator());
    ac_[i].Init(stride, anchor_cfg_[stride], false);
  }

  for (int i = 0; i < feat_stride_fpn_.size(); ++i) {
    char clsname[100];
    sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", feat_stride_fpn_[i]);
    char regname[100];
    sprintf(regname, "face_rpn_bbox_pred_stride%d", feat_stride_fpn_[i]);
    char ptsname[100];
    sprintf(ptsname, "face_rpn_landmark_pred_stride%d", feat_stride_fpn_[i]);
    output_names_.push_back(clsname);
    output_names_.push_back(regname);
    output_names_.push_back(ptsname);
  }
}

#if __ANDROID__
FaceDetectorWithLandmark::FaceDetectorWithLandmark(
    AAssetManager* mgr, const std::string& param_file,
    const std::string& weight_file, int input_width, int input_height,
    float score_threshold, float iou_threshold, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
}
#endif

void FaceDetectorWithLandmark::Preprocess(const cv::Mat& image,
                                          ncnn::Mat& net_input) {
  int width = InputWidth();
  int height = InputHeight();
  net_input =
      ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB,
                                    image.cols, image.rows, width, height);
  ratio_x_ = float(image.cols) / width;
  ratio_y_ = float(image.rows) / height;
  net_input.substract_mean_normalize(pixel_mean_, pixel_std_);
}

std::vector<daisykit::types::FaceDet> FaceDetectorWithLandmark::Predict(
    cv::Mat& img) {
  std::vector<daisykit::types::FaceDet> dets;
  ncnn::Mat in;
  std::map<std::string, ncnn::Mat> out;
  std::vector<Anchor> proposals;
  proposals.clear();

  Preprocess(img, in);
  int res = Infer(in, out, "data", output_names_);
  if (res != 0) {
    std::cout << "Inference failed" << std::endl;
    return dets;
  }

  for (int i = 0; i < output_names_.size(); i += 3) {
    ncnn::Mat cls = (ncnn::Mat)out[output_names_[i]];
    ncnn::Mat reg = (ncnn::Mat)out[output_names_[i + 1]];
    ncnn::Mat pts = (ncnn::Mat)out[output_names_[i + 2]];
    ac_[i / 3].FilterAnchor(cls, reg, pts, proposals, cls_threshold_);
  }

  std::vector<Anchor> result;

  NMS_CPU(proposals, nms_threshold_, result);

  for (int i = 0; i < result.size(); i++) {
    daisykit::types::FaceDet det;
    cv::Rect rect(
        (int)result[i].finalbox_.x * ratio_x_,
        (int)result[i].finalbox_.y * ratio_y_,
        (int)(result[i].finalbox_.width - (int)result[i].finalbox_.x) *
            ratio_x_,
        (int)(result[i].finalbox_.height - (int)result[i].finalbox_.y) *
            ratio_y_);
    det.boxes = rect;
    for (int j = 0; j < result[i].pts_.size(); ++j) {
      det.landmark.x[j] = (int)result[i].pts_[j].x * ratio_x_;
      det.landmark.y[j] = (int)result[i].pts_[j].y * ratio_y_;
    }
    int a =
        GetAlignedFace(img, (float*)&det.landmark, 5, 112, det.face_aligned);
    dets.push_back(det);
  }
  return dets;
}

void FaceDetectorWithLandmark::DrawFaceDets(
    cv::Mat& img, std::vector<daisykit::types::FaceDet>& dets) {
  for (int i = 0; i < dets.size(); i++) {
    cv::rectangle(img, dets[i].boxes, cv::Scalar(0, 255, 0), 2);
    for (int j = 0; j < 5; j++) {
      cv::circle(img, cv::Point(dets[i].landmark.x[j], dets[i].landmark.y[j]),
                 2, cv::Scalar(0, 0, 255), 2);
    }
  }
}

}  // namespace models
}  // namespace daisykit
