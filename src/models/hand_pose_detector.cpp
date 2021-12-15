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

#include "daisykit/models/hand_pose_detector.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

HandPoseDetector::HandPoseDetector(const char* param_buffer,
                                   const unsigned char* weight_buffer,
                                   int input_size, bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu), ImageModel(input_size) {}

HandPoseDetector::HandPoseDetector(const std::string& param_file,
                                   const std::string& weight_file,
                                   int input_size, bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu), ImageModel(input_size) {}

#if __ANDROID__
HandPoseDetector::HandPoseDetector(AAssetManager* mgr,
                                   const std::string& param_file,
                                   const std::string& weight_file,
                                   int input_size, bool use_gpu)
    : NCNNModel(mgr, param_file, weight_file, use_gpu),
      ImageModel(input_size) {}
#endif

void HandPoseDetector::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();

  int target_size = InputWidth();  // input width = input height
  int w = rgb.cols;
  int h = rgb.rows;
  scale_ = 1.f;
  if (w > h) {
    scale_ = (float)target_size / w;
    w = target_size;
    h = h * scale_;
  } else {
    scale_ = (float)target_size / h;
    h = target_size;
    w = w * scale_;
  }

  net_input = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,
                                            rgb.cols, rgb.rows, w, h);
  wpad_ = target_size - w;
  hpad_ = target_size - h;
  ncnn::Mat net_input_pad;
  ncnn::copy_make_border(net_input, net_input_pad, hpad_ / 2, hpad_ - hpad_ / 2,
                         wpad_ / 2, wpad_ - wpad_ / 2, ncnn::BORDER_CONSTANT,
                         0.f);
  net_input = net_input_pad;

  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  net_input.substract_mean_normalize(NULL, norm_vals);
}

int HandPoseDetector::Predict(const cv::Mat& image,
                              std::vector<types::KeypointXYZ>& keypoints,
                              float& lr_score, float offset_x, float offset_y) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Inference
  std::map<std::string, ncnn::Mat> out;
  int result = Infer(in, out, "input", {"points", "score"});
  if (result != 0) {
    return result;
  }

  // Postprocess
  keypoints.clear();
  float* points_data = (float*)out["points"].data;
  float* score_data = (float*)out["score"].data;
  lr_score = score_data[0];
  for (int i = 0; i < 21; ++i) {
    types::KeypointXYZ pt;
    pt.x = (points_data[i * 3] - (wpad_ / 2)) / scale_ + offset_x;
    pt.y = (points_data[i * 3 + 1] - (hpad_ / 2)) / scale_ + offset_y;
    pt.z = points_data[i * 3 + 2];
    keypoints.push_back(pt);
  }
  return 0;
}

int HandPoseDetector::PredictMulti(
    const cv::Mat& image, const std::vector<types::Object>& objects,
    std::vector<std::vector<types::KeypointXYZ>>& poses,
    std::vector<float>& lr_scores) {
  int num_errors = 0;
  int img_width = image.cols;
  int img_height = image.rows;
  int x1, y1, x2, y2;

  int padding = 20;
  poses.clear();
  float lr_score;
  lr_scores.clear();
  for (size_t i = 0; i < objects.size(); ++i) {
    x1 = objects[i].x - padding;
    y1 = objects[i].y - padding;
    x2 = objects[i].x + objects[i].w + padding;
    y2 = objects[i].y + objects[i].h + padding;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;
    if (x1 > img_width) x1 = img_width;
    if (y1 > img_height) y1 = img_height;
    if (x2 > img_width) x2 = img_width;
    if (y2 > img_height) y2 = img_height;

    std::vector<types::KeypointXYZ> keypoints;
    cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    if (Predict(roi, keypoints, lr_score, x1, y1) != 0) {
      ++num_errors;
    }
    poses.push_back(keypoints);
    lr_scores.push_back(lr_score);
  }

  return num_errors;
}

void HandPoseDetector::DrawHandPoses(
    cv::Mat& image, const std::vector<types::ObjectWithKeypointsXYZ>& objects) {
  static const char* class_names[] = {"left_hand", "right_hand"};
  static const unsigned char colors[19][3] = {
      {54, 67, 244},  {99, 30, 233},   {176, 39, 156}, {183, 58, 103},
      {181, 81, 63},  {243, 150, 33},  {244, 169, 3},  {212, 188, 0},
      {136, 150, 0},  {80, 175, 76},   {74, 195, 139}, {57, 220, 205},
      {59, 235, 255}, {7, 193, 255},   {0, 152, 255},  {34, 87, 255},
      {72, 85, 121},  {158, 158, 158}, {139, 125, 96}};

  int color_index = 0;

  for (size_t i = 0; i < objects.size(); i++) {
    const types::ObjectWithKeypointsXYZ& obj = objects[i];

    const unsigned char* color = colors[color_index % 19];
    color_index++;

    cv::Scalar cc(color[0], color[1], color[2]);

    cv::rectangle(image, cv::Rect(obj.x, obj.y, obj.w, obj.h), cc, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[obj.class_id], obj.confidence * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = obj.x;
    int y = obj.y - label_size.height - baseLine;
    if (y < 0) y = 0;
    if (x + label_size.width > image.cols) x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        cc, -1);

    cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381)
                            ? cv::Scalar(0, 0, 0)
                            : cv::Scalar(255, 255, 255);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    // draw hand pose
    {
      cv::Scalar color1(10, 215, 255);
      cv::Scalar color2(255, 115, 55);
      cv::Scalar color3(5, 255, 55);
      cv::Scalar color4(25, 15, 255);
      cv::Scalar color5(225, 15, 55);
      for (size_t j = 0; j < 21; j++) {
        cv::circle(image, cv::Point2i(obj.keypoints[j].x, obj.keypoints[j].y),
                   4, cv::Scalar(255, 0, 0), -1);
        if (j < 4) {
          cv::line(image, cv::Point2i(obj.keypoints[j].x, obj.keypoints[j].y),
                   cv::Point2i(obj.keypoints[j + 1].x, obj.keypoints[j + 1].y),
                   color1, 2, 8);
        }
        if (j < 8 && j > 4) {
          cv::line(image, cv::Point2i(obj.keypoints[j].x, obj.keypoints[j].y),
                   cv::Point2i(obj.keypoints[j + 1].x, obj.keypoints[j + 1].y),
                   color2, 2, 8);
        }
        if (j < 12 && j > 8) {
          cv::line(image, cv::Point2i(obj.keypoints[j].x, obj.keypoints[j].y),
                   cv::Point2i(obj.keypoints[j + 1].x, obj.keypoints[j + 1].y),
                   color3, 2, 8);
        }
        if (j < 16 && j > 12) {
          cv::line(image, cv::Point2i(obj.keypoints[j].x, obj.keypoints[j].y),
                   cv::Point2i(obj.keypoints[j + 1].x, obj.keypoints[j + 1].y),
                   color4, 2, 8);
        }
        if (j < 20 && j > 16) {
          cv::line(image, cv::Point2i(obj.keypoints[j].x, obj.keypoints[j].y),
                   cv::Point2i(obj.keypoints[j + 1].x, obj.keypoints[j + 1].y),
                   color5, 2, 8);
        }
      }
      cv::line(image, cv::Point2i(obj.keypoints[0].x, obj.keypoints[0].y),
               cv::Point2i(obj.keypoints[5].x, obj.keypoints[5].y), color2, 2,
               8);
      cv::line(image, cv::Point2i(obj.keypoints[0].x, obj.keypoints[0].y),
               cv::Point2i(obj.keypoints[9].x, obj.keypoints[9].y), color3, 2,
               8);
      cv::line(image, cv::Point2i(obj.keypoints[0].x, obj.keypoints[0].y),
               cv::Point2i(obj.keypoints[13].x, obj.keypoints[13].y), color4, 2,
               8);
      cv::line(image, cv::Point2i(obj.keypoints[0].x, obj.keypoints[0].y),
               cv::Point2i(obj.keypoints[17].x, obj.keypoints[17].y), color5, 2,
               8);
    }
  }
}

}  // namespace models
}  // namespace daisykit
