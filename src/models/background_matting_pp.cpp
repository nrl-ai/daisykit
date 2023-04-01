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

#include "daisykit/models/background_matting.h"
#include "daisykit/processors/image_processors/img_utils.h"

#include <string>
#include <vector>

namespace daisykit {
namespace models {

BackgroundMatting::BackgroundMatting(const char* param_buffer,
                                     const unsigned char* weight_buffer,
                                     int input_width, int input_height,
                                     bool use_gpu)
    : NCNNModel(param_buffer, weight_buffer, use_gpu),
      ImageModel(input_width, input_height) {}

BackgroundMatting::BackgroundMatting(const std::string& param_file,
                                     const std::string& weight_file,
                                     int input_width, int input_height,
                                     bool use_gpu)
    : NCNNModel(param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {}

#if __ANDROID__
BackgroundMatting::BackgroundMatting(AAssetManager* mgr,
                                     const std::string& param_file,
                                     const std::string& weight_file,
                                     int input_width, int input_height,
                                     bool use_gpu)
    : NCNNModel(mgr, param_file, weight_file, use_gpu),
      ImageModel(input_width, input_height) {}
#endif

void BackgroundMatting::Preprocess(const cv::Mat& image, ncnn::Mat& net_input) {
  // Clone the original cv::Mat to ensure continuous address for memory
  cv::Mat rgb = image.clone();

  net_input =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols,
                                    rgb.rows, InputWidth(), InputHeight());

  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
  net_input.substract_mean_normalize(mean_vals, norm_vals);
}

int BackgroundMatting::Predict(const cv::Mat& image, cv::Mat& mask) {
  // Preprocess
  ncnn::Mat in;
  Preprocess(image, in);

  // Inference
  ncnn::Mat out;
  int result = Infer(in, out, "x", "save_infer_model/scale_0.tmp_1");
  if (result != 0) {
    return result;
  }

  // Postprocess
  // Resize to original size
  // out has shape (h, w, c)
  // Get max value on channel 2
  mask = cv::Mat(out.h, out.w, CV_8UC1);
  for (int y = 0; y < out.h; y++) {
    for (int x = 0; x < out.w; x++) {
      mask.at<uchar>(y, x) =
          out.channel(0).row(y)[x] > out.channel(1).row(y)[x] ? 0 : 255;
    }
  }
  cv::imshow("mask", mask);
  cv::waitKey(1);
  cv::resize(mask, mask, cv::Size(image.cols, image.rows), 0, 0,
             cv::INTER_CUBIC);

  return 0;
}

void BackgroundMatting::BindWithBackground(cv::Mat& rgb,
                                           const cv::Mat& background,
                                           const cv::Mat& mask) {
  const int w = rgb.cols;
  const int h = rgb.rows;

  for (int y = 0; y < h; y++) {
    uchar* rgbptr = rgb.ptr<uchar>(y);
    const uchar* bgptr = background.ptr<const uchar>(y);
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
