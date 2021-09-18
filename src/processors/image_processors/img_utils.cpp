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

#include "daisykitsdk/processors/image_processors/img_utils.h"

namespace daisykit {
namespace processors {

cv::Mat ImgUtils::SquarePadding(const cv::Mat& img, int target_width) {
  int width = img.cols, height = img.rows;

  cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

  int max_dim = (width >= height) ? width : height;
  float scale = ((float)target_width) / max_dim;
  cv::Rect roi;
  if (width >= height) {
    roi.width = target_width;
    roi.x = 0;
    roi.height = height * scale;
    roi.y = (target_width - roi.height) / 2;
  } else {
    roi.y = 0;
    roi.height = target_width;
    roi.width = width * scale;
    roi.x = (target_width - roi.width) / 2;
  }

  cv::resize(img, square(roi), roi.size());

  return square;
}

}  // namespace processors
}  // namespace daisykit