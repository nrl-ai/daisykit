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

#ifndef DAISYKIT_MODELS_YOLOX_UTILS_H_
#define DAISYKIT_MODELS_YOLOX_UTILS_H_

#include <net.h>
#include <opencv2/core/core.hpp>
#include "daisykitsdk/common/types.h"

namespace daisykit {
namespace models {

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer {
 public:
  YoloV5Focus() { one_blob_only = true; }

  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                      const ncnn::Option& opt) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = w / 2;
    int outh = h / 2;
    int outc = channels * 4;

    top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
    if (top_blob.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++) {
      const float* ptr =
          bottom_blob.channel(p % channels).row((p / channels) % 2) +
          ((p / channels) / 2);
      float* outptr = top_blob.channel(p);

      for (int i = 0; i < outh; i++) {
        for (int j = 0; j < outw; j++) {
          *outptr = *ptr;

          outptr += 1;
          ptr += 2;
        }

        ptr += w;
      }
    }

    return 0;
  }
};

float intersection_area(const types::Object& a, const types::Object& b);
void qsort_descent_inplace(std::vector<types::Object>& faceobjects, int left,
                           int right);
void qsort_descent_inplace(std::vector<types::Object>& objects);
void nms_sorted_bboxes(const std::vector<types::Object>& faceobjects,
                       std::vector<int>& picked, float nms_threshold);
void generate_grids_and_stride(const int target_w, const int target_h,
                               std::vector<int>& strides,
                               std::vector<GridAndStride>& grid_strides);
void generate_yolox_proposals(std::vector<GridAndStride> grid_strides,
                              const ncnn::Mat& feat_blob, float prob_threshold,
                              std::vector<types::Object>& objects);

}  // namespace models
}  // namespace daisykit

#endif
