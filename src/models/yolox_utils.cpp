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
// Most code in this file was taken from NCNN library
// https://opensource.org/licenses/BSD-3-Clause

#include "daisykit/models/yolox_utils.h"
#include "daisykit/common/types.h"

#include <iostream>

namespace daisykit {
namespace models {

float intersection_area(const types::Object& a, const types::Object& b) {
  float l1_x = a.x;
  float l1_y = a.y;
  float r1_x = a.x + a.w;
  float r1_y = a.y + a.h;
  float l2_x = b.x;
  float l2_y = b.y;
  float r2_x = b.x + b.w;
  float r2_y = b.y + b.h;

  // Area of 1st Rectangle
  float area1 = abs(l1_x - r1_x) * abs(l1_y - r1_y);

  // Area of 2nd Rectangle
  float area2 = abs(l2_x - r2_x) * abs(l2_y - r2_y);

  // Length of intersecting part i.e
  // start from max(l1.x, l2.x) of
  // x-coordinate and end at min(r1.x,
  // r2.x) x-coordinate by subtracting
  // start from end we get required
  // lengths
  float x_dist = std::min(r1_x, r2_x) - std::max(l1_x, l2_x);
  float y_dist = (std::min(r1_y, r2_y) - std::max(l1_y, l2_y));
  float areaI = 0;
  if (x_dist > 0 && y_dist > 0) {
    areaI = x_dist * y_dist;
  }

  return (area1 + area2 - areaI);
}

void qsort_descent_inplace(std::vector<types::Object>& objects, int left,
                           int right) {
  int i = left;
  int j = right;
  float p = objects[(left + right) / 2].confidence;

  while (i <= j) {
    while (objects[i].confidence > p) i++;

    while (objects[j].confidence < p) j--;

    if (i <= j) {
      // swap
      std::swap(objects[i], objects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j) qsort_descent_inplace(objects, left, j);
    }
#pragma omp section
    {
      if (i < right) qsort_descent_inplace(objects, i, right);
    }
  }
}

void qsort_descent_inplace(std::vector<types::Object>& objects) {
  if (objects.empty()) return;

  qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<types::Object>& objects,
                       std::vector<int>& picked, float nms_threshold) {
  picked.clear();

  const int n = objects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = objects[i].w * objects[i].h;
  }

  for (int i = 0; i < n; i++) {
    const types::Object& a = objects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const types::Object& b = objects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }

    if (keep) picked.push_back(i);
  }
}

void generate_grids_and_stride(const int target_w, const int target_h,
                               std::vector<int>& strides,
                               std::vector<GridAndStride>& grid_strides) {
  for (auto stride : strides) {
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        GridAndStride gs;
        gs.grid0 = g0;
        gs.grid1 = g1;
        gs.stride = stride;
        grid_strides.push_back(gs);
      }
    }
  }
}

void generate_yolox_proposals(std::vector<GridAndStride> grid_strides,
                              const ncnn::Mat& feat_blob, float prob_threshold,
                              std::vector<types::Object>& objects) {
  const int num_grid = feat_blob.h;

  const int num_class = feat_blob.w - 5;

  const int num_anchors = grid_strides.size();

  const float* feat_ptr = feat_blob.channel(0);
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    // yolox/models/yolo_head.py decode logic
    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    float x_center = (feat_ptr[0] + grid0) * stride;
    float y_center = (feat_ptr[1] + grid1) * stride;
    float w = exp(feat_ptr[2]) * stride;
    float h = exp(feat_ptr[3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_ptr[4];
    for (int class_idx = 0; class_idx < num_class; class_idx++) {
      float box_cls_score = feat_ptr[5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        types::Object obj;
        obj.x = x0;
        obj.y = y0;
        obj.w = w;
        obj.h = h;
        obj.class_id = class_idx;
        obj.confidence = box_prob;

        objects.push_back(obj);
      }

    }  // class loop
    feat_ptr += feat_blob.w;

  }  // point anchor loop
}

}  // namespace models
}  // namespace daisykit
