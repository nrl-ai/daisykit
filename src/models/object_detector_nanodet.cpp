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

#include "daisykitsdk/models/object_detector_nanodet.h"
#include "daisykitsdk/processors/image_processors/img_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace daisykit {
namespace models {

ObjectDetectorNanodet::ObjectDetectorNanodet(const std::string& param_file,
                                             const std::string& weight_file) {
  LoadModel(param_file, weight_file);
}

void ObjectDetectorNanodet::LoadModel(const std::string& param_file,
                                      const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(param_file.c_str());
  int ret_model = model_->load_model(weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}

#ifdef __ANDROID__
ObjectDetectorNanodet::ObjectDetectorNanodet(AAssetManager* mgr,
                                             const std::string& param_file,
                                             const std::string& weight_file) {
  LoadModel(mgr, param_file, weight_file);
}

void ObjectDetectorNanodet::LoadModel(AAssetManager* mgr,
                                      const std::string& param_file,
                                      const std::string& weight_file) {
  if (model_) {
    delete model_;
    model_ = nullptr;
  }
  model_ = new ncnn::Net;
  int ret_param = model_->load_param(mgr, param_file.c_str());
  int ret_model = model_->load_model(mgr, weight_file.c_str());
  if (ret_param != 0 || ret_model != 0) {
    exit(1);
  }
}
#endif

std::vector<types::Object> ObjectDetectorNanodet::Detect(cv::Mat& image) {
  cv::Mat rgb = image.clone();
  int img_w = rgb.cols;
  int img_h = rgb.rows;

  const float prob_threshold = 0.4f;
  const float nms_threshold = 0.5f;

  // pad to multiple of 32
  int w = img_w;
  int h = img_h;
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

  ncnn::Mat in =
      ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, img_w,
                                    img_h, input_width_, input_height_);

  // pad to target_size rectangle
  int wpad = 320 - w;  //(w + 31) / 32 * 32 - w;
  int hpad = 320 - h;  //(h + 31) / 32 * 32 - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

  const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
  const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
  in_pad.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = model_->create_extractor();

  ex.input("input.1", in_pad);

  std::vector<types::Object> proposals;

  // stride 8
  {
    ncnn::Mat cls_pred;
    ncnn::Mat dis_pred;
    ex.extract("cls_pred_stride_8", cls_pred);
    ex.extract("dis_pred_stride_8", dis_pred);

    std::vector<types::Object> objects8;
    GenerateProposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, objects8);

    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
  }

  // stride 16
  {
    ncnn::Mat cls_pred;
    ncnn::Mat dis_pred;
    ex.extract("cls_pred_stride_16", cls_pred);
    ex.extract("dis_pred_stride_16", dis_pred);

    std::vector<types::Object> objects16;
    GenerateProposals(cls_pred, dis_pred, 16, in_pad, prob_threshold,
                      objects16);

    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
  }

  // stride 32
  {
    ncnn::Mat cls_pred;
    ncnn::Mat dis_pred;
    ex.extract("cls_pred_stride_32", cls_pred);
    ex.extract("dis_pred_stride_32", dis_pred);

    std::vector<types::Object> objects32;
    GenerateProposals(cls_pred, dis_pred, 32, in_pad, prob_threshold,
                      objects32);

    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
  }

  // sort all proposals by score from highest to lowest
  QsortDescentInplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  NmsSortedBboxes(proposals, picked, nms_threshold);

  int count = picked.size();

  std::vector<types::Object> objects;
  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].x - (wpad / 2)) / scale;
    float y0 = (objects[i].y - (hpad / 2)) / scale;
    float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
    float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    objects[i].x = x0;
    objects[i].y = y0;
    objects[i].w = x1 - x0;
    objects[i].h = y1 - y0;
  }

  return objects;
}

float ObjectDetectorNanodet::IntersectionArea(const Object& a,
                                              const Object& b) {
  cv::Rect_<float> inter =
      cv::Rect(a.x, a.y, a.w, a.h) & cv::Rect(b.x, b.y, b.w, b.h);
  return inter.area();
}

void ObjectDetectorNanodet::QsortDescentInplace(
    std::vector<types::Object>& objects, int left, int right) {
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
      if (left < j) QsortDescentInplace(objects, left, j);
    }
#pragma omp section
    {
      if (i < right) QsortDescentInplace(objects, i, right);
    }
  }
}

void ObjectDetectorNanodet::QsortDescentInplace(
    std::vector<types::Object>& objects) {
  if (objects.empty()) return;
  QsortDescentInplace(objects, 0, objects.size() - 1);
}

void ObjectDetectorNanodet::NmsSortedBboxes(
    const std::vector<types::Object>& objects, std::vector<int>& picked,
    float nms_threshold) {
  picked.clear();

  const int n = objects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = objects[i].w * objects[i].h;
  }

  for (int i = 0; i < n; i++) {
    const Object& a = objects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object& b = objects[picked[j]];

      // intersection over union
      float inter_area = IntersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }

    if (keep) picked.push_back(i);
  }
}

void ObjectDetectorNanodet::GenerateProposals(
    const ncnn::Mat& cls_pred, const ncnn::Mat& dis_pred, int stride,
    const ncnn::Mat& in_pad, float prob_threshold,
    std::vector<types::Object>& objects) {
  const int num_grid = cls_pred.h;

  int num_grid_x;
  int num_grid_y;
  if (in_pad.w > in_pad.h) {
    num_grid_x = in_pad.w / stride;
    num_grid_y = num_grid / num_grid_x;
  } else {
    num_grid_y = in_pad.h / stride;
    num_grid_x = num_grid / num_grid_y;
  }

  const int num_class = cls_pred.w;
  const int reg_max_1 = dis_pred.w / 4;

  for (int i = 0; i < num_grid_y; i++) {
    for (int j = 0; j < num_grid_x; j++) {
      const int idx = i * num_grid_x + j;

      const float* scores = cls_pred.row(idx);

      // find label with max score
      int label = -1;
      float score = -FLT_MAX;
      for (int k = 0; k < num_class; k++) {
        if (scores[k] > score) {
          label = k;
          score = scores[k];
        }
      }

      if (score >= prob_threshold) {
        ncnn::Mat bbox_pred(reg_max_1, 4, (void*)dis_pred.row(idx));
        {
          ncnn::Layer* softmax = ncnn::create_layer("Softmax");

          ncnn::ParamDict pd;
          pd.set(0, 1);  // axis
          pd.set(1, 1);
          softmax->load_param(pd);

          ncnn::Option opt;
          opt.num_threads = 1;
          opt.use_packing_layout = false;

          softmax->create_pipeline(opt);
          softmax->forward_inplace(bbox_pred, opt);
          softmax->destroy_pipeline(opt);

          delete softmax;
        }

        float pred_ltrb[4];
        for (int k = 0; k < 4; k++) {
          float dis = 0.f;
          const float* dis_after_sm = bbox_pred.row(k);
          for (int l = 0; l < reg_max_1; l++) {
            dis += l * dis_after_sm[l];
          }

          pred_ltrb[k] = dis * stride;
        }

        float pb_cx = (j + 0.5f) * stride;
        float pb_cy = (i + 0.5f) * stride;

        float x0 = pb_cx - pred_ltrb[0];
        float y0 = pb_cy - pred_ltrb[1];
        float x1 = pb_cx + pred_ltrb[2];
        float y1 = pb_cy + pred_ltrb[3];

        Object obj;
        obj.x = x0;
        obj.y = y0;
        obj.w = x1 - x0;
        obj.h = y1 - y0;
        obj.class_id = label;
        obj.confidence = score;

        objects.push_back(obj);
      }
    }
  }
}

}  // namespace models
}  // namespace daisykit