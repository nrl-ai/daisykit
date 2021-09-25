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
                           const unsigned char* weight_buffer,
                           float score_threshold, float iou_threshold,
                           int input_width, int input_height, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
}

FaceDetector::FaceDetector(const std::string& param_file,
                           const std::string& weight_file,
                           float score_threshold, float iou_threshold,
                           int input_width, int input_height, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  score_threshold_ = score_threshold;
  iou_threshold_ = iou_threshold;
}

void FaceDetector::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();

  // Letterbox pad to multiple of 32
  int img_width = rgb.cols;
  int img_height = rgb.rows;

  int w = img_width;
  int h = img_height;

  float scale = 1.f;

  if (w > h) {
    scale = (float)InputWidth() / w;
    w = InputWidth();
    h = h * scale;
  } else {
    scale = (float)InputHeight() / h;
    h = InputHeight();
    w = w * scale;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,
                                               img_width, img_height, w, h);

  // Pad to target_size rectangle
  // yolo/utils/datasets.py letterbox
  int wpad = (w + 31) / 32 * 32 - w;
  int hpad = (h + 31) / 32 * 32 - h;
  ncnn::copy_make_border(in, net_input, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  net_input.substract_mean_normalize(0, norm_vals);
}

int FaceDetector::Detect(const cv::Mat& image,
                         std::vector<daisykit::types::Face>& faces) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Inference
  ncnn::Mat out;
  int result = Predict(in, out, "data", "output");
  if (result != 0) {
    std::cout << "WTF" << std::endl;
    return result;
  }

  // Postprocess
  int img_width = image.cols;
  int img_height = image.rows;
  int count = out.h;
  faces.resize(count);
  for (int i = 0; i < count; i++) {
    int label;
    float x1, y1, x2, y2, score;
    float pw, ph, cx, cy;
    const float* values = out.row(i);

    x1 = values[2] * img_width;
    y1 = values[3] * img_height;
    x2 = values[4] * img_width;
    y2 = values[5] * img_height;

    score = values[1];
    label = values[0];

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;

    if (x1 > img_width) x1 = img_width;
    if (y1 > img_height) y1 = img_height;
    if (x2 > img_width) x2 = img_width;
    if (y2 > img_height) y2 = img_height;

    faces[i].wearing_mask_prob = label == 2 ? 1.0 : 0.0;
    faces[i].confidence = score;
    faces[i].x = x1;
    faces[i].y = y1;
    faces[i].w = x2 - x1;
    faces[i].h = y2 - y1;
  }

  return 0;
}

}  // namespace models
}  // namespace daisykit
