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

#include "daisykit/models/pose_detector_movenet.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

PoseDetectorMoveNet::PoseDetectorMoveNet(const char* param_buffer,
                                         const unsigned char* weight_buffer,
                                         int input_width, int input_height,
                                         bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {
  assert(input_width ==
         input_height);  // Current MoveNet has input size 192x192 or 256x256
  PrepareFeatureKeypointsParams(input_width);
}

PoseDetectorMoveNet::PoseDetectorMoveNet(const std::string& param_file,
                                         const std::string& weight_file,
                                         int input_width, int input_height,
                                         bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  assert(input_width ==
         input_height);  // Current MoveNet has input size 192x192 or 256x256
  PrepareFeatureKeypointsParams(input_width);
}

#if __ANDROID__
PoseDetectorMoveNet::PoseDetectorMoveNet(AAssetManager* mgr,
                                         const std::string& param_file,
                                         const std::string& weight_file,
                                         int input_width, int input_height,
                                         bool use_gpu)
    : NCNNModel(mgr, param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {
  assert(input_width ==
         input_height);  // Current MoveNet has input size 192x192 or 256x256

  PrepareFeatureKeypointsParams(input_width);
}
#endif

void PoseDetectorMoveNet::PrepareFeatureKeypointsParams(float input_size) {
  if (input_size == 192) {
    feature_size_ = 48;
    kpt_scale_ = 0.02083333395421505;
  } else {
    feature_size_ = 64;
    kpt_scale_ = 0.015625;
  }
  for (int i = 0; i < feature_size_; i++) {
    std::vector<float> x, y;
    for (int j = 0; j < feature_size_; j++) {
      x.push_back(j);
      y.push_back(i);
    }
    dist_y_.push_back(y);
    dist_x_.push_back(x);
  }
}

void PoseDetectorMoveNet::Preprocess(const cv::Mat& image,
                                     ncnn::Mat& net_input) {
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

  // Pad to target size
  int wpad = InputWidth() - w;
  int hpad = InputHeight() - h;
  ncnn::copy_make_border(in, net_input, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
  net_input.substract_mean_normalize(mean_vals, norm_vals);

  // Cache params for postprocessing
  scale_ = scale;
  hpad_ = hpad;
  wpad_ = wpad;
}

int PoseDetectorMoveNet::Predict(const cv::Mat& image,
                                 std::vector<types::Keypoint>& keypoints,
                                 float offset_x, float offset_y) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Inference
  std::map<std::string, ncnn::Mat> out;
  int result =
      Infer(in, out, "input", {"regress", "center", "heatmap", "offset"});
  if (result != 0) {
    return result;
  }

  // Postprocess
  float* center_data = (float*)out["center"].data;
  float* heatmap_data = (float*)out["heatmap"].data;
  float* offset_data = (float*)out["offset"].data;

  int top_index = 0;
  float top_score = 0;

  top_index = int(argmax(center_data, center_data + out["center"].h));
  top_score = *std::max_element(center_data, center_data + out["center"].h);

  int ct_y = (top_index / feature_size_);
  int ct_x = top_index - ct_y * feature_size_;

  std::vector<float> y_regress(kNumJoints), x_regress(kNumJoints);
  float* regress_data = (float*)out["regress"].channel(ct_y).row(ct_x);
  for (size_t i = 0; i < kNumJoints; i++) {
    y_regress[i] = regress_data[i] + (float)ct_y;
    x_regress[i] = regress_data[i + kNumJoints] + (float)ct_x;
  }

  ncnn::Mat kpt_scores =
      ncnn::Mat(feature_size_ * feature_size_, kNumJoints, sizeof(float));
  float* scores_data = (float*)kpt_scores.data;
  for (int i = 0; i < feature_size_; i++) {
    for (int j = 0; j < feature_size_; j++) {
      std::vector<float> score;
      for (int c = 0; c < kNumJoints; c++) {
        float y =
            (dist_y_[i][j] - y_regress[c]) * (dist_y_[i][j] - y_regress[c]);
        float x =
            (dist_x_[i][j] - x_regress[c]) * (dist_x_[i][j] - x_regress[c]);
        float dist_weight = std::sqrt(y + x) + 1.8;
        scores_data[c * feature_size_ * feature_size_ + i * feature_size_ + j] =
            heatmap_data[i * feature_size_ * kNumJoints + j * kNumJoints + c] /
            dist_weight;
      }
    }
  }

  std::vector<int> kpts_ys, kpts_xs;
  for (int i = 0; i < kNumJoints; i++) {
    top_index = 0;
    top_score = 0;
    top_index =
        int(argmax(scores_data + feature_size_ * feature_size_ * i,
                   scores_data + feature_size_ * feature_size_ * (i + 1)));
    top_score = *std::max_element(
        scores_data + feature_size_ * feature_size_ * i,
        scores_data + feature_size_ * feature_size_ * (i + 1));

    int top_y = (top_index / feature_size_);
    int top_x = top_index - top_y * feature_size_;
    kpts_ys.push_back(top_y);
    kpts_xs.push_back(top_x);
  }

  keypoints.clear();
  for (int i = 0; i < kNumJoints; i++) {
    float kpt_offset_x =
        offset_data[kpts_ys[i] * feature_size_ * kNumJoints * 2 +
                    kpts_xs[i] * kNumJoints * 2 + i * 2];
    float kpt_offset_y =
        offset_data[kpts_ys[i] * feature_size_ * kNumJoints * 2 +
                    kpts_xs[i] * kNumJoints * 2 + i * 2 + 1];

    float x = (kpts_xs[i] + kpt_offset_y) * kpt_scale_ * InputWidth();
    float y = (kpts_ys[i] + kpt_offset_x) * kpt_scale_ * InputHeight();

    types::Keypoint kpt;
    kpt.x = (x - (wpad_ / 2)) / scale_ + offset_x;
    kpt.y = (y - (hpad_ / 2)) / scale_ + offset_y;
    kpt.confidence = heatmap_data[kpts_ys[i] * feature_size_ * kNumJoints +
                                  kpts_xs[i] * kNumJoints + i];
    keypoints.push_back(kpt);
  }

  return 0;
}

int PoseDetectorMoveNet::PredictMulti(
    const cv::Mat& image, const std::vector<types::Object>& objects,
    std::vector<std::vector<types::Keypoint>>& poses) {
  int num_errors = 0;
  int img_width = image.cols;
  int img_height = image.rows;
  int x1, y1, x2, y2;

  poses.clear();
  for (size_t i = 0; i < objects.size(); ++i) {
    x1 = objects[i].x;
    y1 = objects[i].y;
    x2 = objects[i].x + objects[i].w;
    y2 = objects[i].y + objects[i].h;
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 < 0) x2 = 0;
    if (y2 < 0) y2 = 0;
    if (x1 >= img_width) x1 = img_width - 1;
    if (y1 >= img_height) y1 = img_height - 1;
    if (x2 >= img_width) x2 = img_width - 1;
    if (y2 >= img_height) y2 = img_height - 1;

    std::vector<types::Keypoint> keypoints_single;
    if (x2 - x1 > 20 && y2 - y1 > 20) {
      cv::Mat roi =
          image(cv::Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))).clone();
      int rows = roi.rows;
      int cols = roi.cols;
      int empty = roi.empty();
      if (Predict(roi, keypoints_single, x1, y1) != 0) {
        ++num_errors;
      }
    }
    poses.push_back(keypoints_single);
  }

  return num_errors;
}

// Draw pose
void PoseDetectorMoveNet::DrawKeypoints(
    cv::Mat& image, const std::vector<types::Keypoint>& keypoints) {
  float threshold = 0.3;
  int skele_index[][2] = {{0, 1},  {0, 2},   {1, 3},  {2, 4},   {0, 5},
                          {0, 6},  {5, 6},   {5, 7},  {7, 9},   {6, 8},
                          {8, 10}, {11, 12}, {5, 11}, {11, 13}, {13, 15},
                          {6, 12}, {12, 14}, {14, 16}};
  int color_index[][3] = {
      {255, 0, 0}, {0, 0, 255}, {255, 0, 0}, {0, 0, 255}, {255, 0, 0},
      {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 0, 0}, {0, 0, 255},
      {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 0, 0}, {255, 0, 0},
      {0, 0, 255}, {0, 0, 255}, {0, 0, 255},
  };

  for (int i = 0; i < 18; i++) {
    if (keypoints[skele_index[i][0]].confidence > threshold &&
        keypoints[skele_index[i][1]].confidence > threshold)
      cv::line(
          image,
          cv::Point(keypoints[skele_index[i][0]].x,
                    keypoints[skele_index[i][0]].y),
          cv::Point(keypoints[skele_index[i][1]].x,
                    keypoints[skele_index[i][1]].y),
          cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]),
          2);
  }
  for (int i = 0; i < 17; i++) {
    if (keypoints[i].confidence > threshold)
      cv::circle(image, cv::Point(keypoints[i].x, keypoints[i].y), 3,
                 cv::Scalar(100, 255, 150), -1);
  }
}

}  // namespace models
}  // namespace daisykit
