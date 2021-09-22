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

#include "daisykitsdk/models/background_matting.h"
#include "daisykitsdk/processors/image_processors/img_utils.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

BackgroundMatting::BackgroundMatting(const char* param_buffer,
                                     const unsigned char* weight_buffer,
                                     int input_width, int input_height) {
  LoadModel(param_buffer, weight_buffer);
  input_width_ = input_width;
  input_height_ = input_height;
}

// Will be deleted when IO module is supported. Keep for old compatibility.
BackgroundMatting::BackgroundMatting(const std::string& param_file,
                                     const std::string& weight_file,
                                     int input_width, int input_height) {
  LoadModel(param_file, weight_file);
  input_width_ = input_width;
  input_height_ = input_height;
}

cv::Mat BackgroundMatting::Predict(const cv::Mat& image) {
  cv::Mat rgb = image.clone();
  const int w = rgb.cols;
  const int h = rgb.rows;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      rgb.data, ncnn::Mat::PIXEL_RGB2BGR, w, h, input_width_, input_height_);

  ncnn::Extractor ex = model_.create_extractor();
  ex.input("input_blob1", in);
  const float mean_vals[3] = {104.f, 112.f, 121.f};
  const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Mat out;
  ex.extract("sigmoid_blob1", out);

  const float denorm_vals[3] = {255.f, 255.f, 255.f};
  out.substract_mean_normalize(0, denorm_vals);

  cv::Mat mask = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
  out.to_pixels_resize(mask.data, ncnn::Mat::PIXEL_GRAY, w, h);
  return mask;
}

void BackgroundMatting::Predict(cv::Mat& image, cv::Mat& mask) {
  cv::Mat rgb = image.clone();
  const int w = rgb.cols;
  const int h = rgb.rows;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      rgb.data, ncnn::Mat::PIXEL_RGB2BGR, w, h, input_width_, input_height_);

  ncnn::Extractor ex = model_.create_extractor();
  ex.input("input_blob1", in);
  const float mean_vals[3] = {104.f, 112.f, 121.f};
  const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Mat out;
  ex.extract("sigmoid_blob1", out);

  const float denorm_vals[3] = {255.f, 255.f, 255.f};
  out.substract_mean_normalize(0, denorm_vals);

  mask = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);
  out.to_pixels_resize(mask.data, ncnn::Mat::PIXEL_GRAY, w, h);
}

void BackgroundMatting::BindWithBackground(cv::Mat& rgb, const cv::Mat& bg,
                                           const cv::Mat& mask) {
  const int w = rgb.cols;
  const int h = rgb.rows;

  for (int y = 0; y < h; y++) {
    uchar* rgbptr = rgb.ptr<uchar>(y);
    const uchar* bgptr = bg.ptr<const uchar>(y);
    const uchar* mptr = mask.ptr<const uchar>(y);

    for (int x = 0; x < w; x++) {
      const uchar alpha = mptr[0];

      rgbptr[0] = cv::saturate_cast<uchar>(
          (rgbptr[0] * alpha + bgptr[0] * (255 - alpha)) / 255);
      rgbptr[1] = cv::saturate_cast<uchar>(
          (rgbptr[1] * alpha + bgptr[1] * (255 - alpha)) / 255);
      rgbptr[2] = cv::saturate_cast<uchar>(
          (rgbptr[2] * alpha + bgptr[2] * (255 - alpha)) / 255);

      rgbptr += 3;
      bgptr += 3;
      mptr += 1;
    }
  }
}

}  // namespace models
}  // namespace daisykit
