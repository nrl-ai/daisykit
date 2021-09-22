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

#include "daisykitsdk/models/face_detector.h"
#include "daisykitsdk/processors/image_processors/img_utils.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

FaceDetector::FaceDetector(const char* param_buffer,
                           const unsigned char* weight_buffer, int input_width,
                           int input_height, float score_threshold,
                           float iou_threshold) {
  LoadModel(param_buffer, weight_buffer);
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_width_ = input_width;
  input_height_ = input_height;
}

// Will be deleted when IO module is supported. Keep for old compatibility.
FaceDetector::FaceDetector(const std::string& param_file,
                           const std::string& weight_file, int input_width,
                           int input_height, float score_threshold,
                           float iou_threshold) {
  LoadModel(param_file, weight_file);
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
  input_width_ = input_width;
  input_height_ = input_height;
}

std::vector<types::Face> FaceDetector::Predict(const cv::Mat& image) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();

  // letterbox pad to multiple of 32
  int im_width = rgb.cols;
  int im_height = rgb.rows;

  int w = im_width;
  int h = im_height;

  float scale = 1.f;

  if (w > h) {
    scale = (float)input_width_ / w;
    w = input_width_;
    h = h * scale;
  } else {
    scale = (float)input_height_ / h;
    h = input_height_;
    w = w * scale;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,
                                               im_width, im_height, w, h);

  // Pad to target_size rectangle
  // yolo/utils/datasets.py letterbox
  int wpad = (w + 31) / 32 * 32 - w;
  int hpad = (h + 31) / 32 * 32 - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in_pad.substract_mean_normalize(0, norm_vals);

  ncnn::Extractor ex = model_.create_extractor();

  ex.input("data", in_pad);
  ncnn::Mat out;
  ex.extract("output", out);

  int count = out.h;

  std::vector<types::Face> objects;
  objects.resize(count);
  for (int i = 0; i < count; i++) {
    int label;
    float x1, y1, x2, y2, score;
    float pw, ph, cx, cy;
    const float* values = out.row(i);

    x1 = values[2] * im_width;
    y1 = values[3] * im_height;
    x2 = values[4] * im_width;
    y2 = values[5] * im_height;

    score = values[1];
    label = values[0];

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;

    if (x1 > im_width) x1 = im_width;
    if (y1 > im_height) y1 = im_height;
    if (x2 > im_width) x2 = im_width;
    if (y2 > im_height) y2 = im_height;

    objects[i].wearing_mask_prob = label == 2 ? 1.0 : 0.0;
    objects[i].confidence = score;
    objects[i].x = x1;
    objects[i].y = y1;
    objects[i].w = x2 - x1;
    objects[i].h = y2 - y1;
  }

  return objects;
}

}  // namespace models
}  // namespace daisykit
