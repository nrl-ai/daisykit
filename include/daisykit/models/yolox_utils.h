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
#include "daisykit/common/types.h"

namespace daisykit {
namespace models {

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
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
