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

#include "daisykitsdk/models/face_recognition/face_detector_with_landmark.h"

namespace daisykit {
namespace models {

// Will be deleted when IO module is supported. Keep for old compatibility.
FaceDetectorWithLandMark::FaceDetectorWithLandMark(
    const std::string& param_file, const std::string& weight_file,
    int input_width, int input_height, float score_threshold,
    float iou_threshold) {
  LoadModel(param_file, weight_file);
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_width_ = input_width;
  input_height_ = input_height;
}

std::vector<daisykit::types::Det> FaceDetectorWithLandMark::Predict(
    cv::Mat& img) {
  std::vector<daisykit::types::Det> dets;
  ncnn::Mat input = ncnn::Mat::from_pixels_resize(
      img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, input_width_,
      input_height_);
  float ratio_x = float(img.cols) / input_width_;
  float ratio_y = float(img.rows) / input_height_;
  input.substract_mean_normalize(pixel_mean, pixel_std);
  ncnn::Extractor ex = model_.create_extractor();
  ex.input("data", input);

  std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
  for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    int stride = _feat_stride_fpn[i];
    ac[i].Init(stride, anchor_cfg[stride], false);
  }

  std::vector<Anchor> proposals;
  proposals.clear();

  for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    ncnn::Mat cls;
    ncnn::Mat reg;
    ncnn::Mat pts;

    char clsname[100];
    sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
    char regname[100];
    sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
    char ptsname[100];
    sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
    ex.extract(clsname, cls);
    ex.extract(regname, reg);
    ex.extract(ptsname, pts);
    ac[i].FilterAnchor(cls, reg, pts, proposals);
  }

  std::vector<Anchor> result;

  nms_cpu(proposals, iou_threshold_, result);

  for (int i = 0; i < result.size(); i++) {
    daisykit::types::Det det;
    cv::Rect rect((int)result[i].finalbox.x * ratio_x,
                  (int)result[i].finalbox.y * ratio_y,
                  (int)result[i].finalbox.width * ratio_x,
                  (int)result[i].finalbox.height * ratio_y);
    det.boxes = rect;
    for (int j = 0; j < result[i].pts.size(); ++j) {
      det.landmark.x[j] = (int)result[i].pts[j].x * ratio_x;
      det.landmark.y[j] = (int)result[i].pts[j].y * ratio_y;
    }
    int a =
        get_aligned_face(img, (float*)&det.landmark, 5, 112, det.face_aligned);
    dets.push_back(det);
  }
  return dets;
}

}  // namespace models
}  // namespace daisykit
